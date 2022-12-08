import glob
import itertools
import os
import numpy as np
import pandas as pd
import sys
import zarr.hierarchy

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as im_utils
import micro_dl.utils.io_utils as io_utils
import micro_dl.utils.mp_utils as mp_utils
from micro_dl.torch_unet.utils.io import show_progress_bar


def generate_normalization_metadata(
    zarr_dir,
    num_workers=4,
    channel_ids=-1,
    grid_spacing=32,
):
    """
    Generate pixel intensity metadata to be later used in on-the-fly normalization
    during training and inference. Sampling is used for efficient estimation of median
    and interquartile range for intensity values on both a dataset and field-of-view
    level.

    Normalization values are recorded in the image-level metadata in the corresponding
    position of each zarr_dir store. Format of metadata is as follows:
    {
        channel_idx : {
            dataset_statistics: dataset level normalization values (positive float),
            fov_statistics: field-of-view level normalization values (positive float)
        },
        .
        .
        .
    }

    :param str zarr_dir: path to zarr store directory containing dataset.
    :param int num_workers: number of cpu workers for multiprocessing, defaults to 4
    :param list/int channel_ids: indices of channels to process in dataset arrays,
                                    by default calculates all
    :param int grid_spacing: distance between points in sampling grid
    """
    modifier = io_utils.HCSZarrModifier(
        zarr_file=zarr_dir,
        enable_creation=True,
        overwrite_ok=True,
    )

    if channel_ids == -1:
        channel_ids = range(modifier.channels)
    elif isinstance(channel_ids, int):
        channel_ids = [channel_ids]

    # get arguments for multiprocessed grid sampling
    mp_grid_sampler_args = []
    for position in modifier.position_map:
        mp_grid_sampler_args.append([position, True, zarr_dir, grid_spacing])

    # sample values and use them to get normalization statistics
    for channel in channel_ids:
        show_progress_bar(
            dataloader=channel_ids,
            current=channel,
            process="sampling channel values",
        )
        channel_name = modifier.channel_names[channel]
        this_channels_args = tuple([args + [channel] for args in mp_grid_sampler_args])

        # NOTE: Doing sequential mp with pool execution creates synchronization
        #      points between each step. This could be detrimental to performance
        fov_sample_values = mp_utils.mp_sample_im_pixels(
            this_channels_args, num_workers
        )
        dataset_sample_values = np.stack(fov_sample_values, 0)

        fov_level_statistics = mp_utils.mp_get_val_stats(fov_sample_values, num_workers)
        dataset_level_statistics = mp_utils.get_val_stats(dataset_sample_values)

        for position in modifier.position_map:
            show_progress_bar(
                dataloader=channel_ids,
                current=channel,
                process="calculating statistics",
            )
            position_statistics = {
                "fov_statistics": fov_level_statistics[position],
                "dataset_statistics": dataset_level_statistics,
            }
            channel_position_statistics = {
                channel_name: position_statistics,
            }

            modifier.write_meta_field(
                position=position,
                metadata=channel_position_statistics,
                field_name="normalization",
            )


def compute_zscore_params(
    frames_meta, ints_meta, input_dir, normalize_im, min_fraction=0.99
):
    """
    Get zscore median and interquartile range

    :param pd.DataFrame frames_meta: Dataframe containing all metadata
    :param pd.DataFrame ints_meta: Metadata containing intensity statistics
        each z-slice and foreground fraction for masks
    :param str input_dir: Directory containing images
    :param None or str normalize_im: normalization scheme for input images
    :param float min_fraction: Minimum foreground fraction (in case of masks)
        for computing intensity statistics.

    :return pd.DataFrame frames_meta: Dataframe containing all metadata
    :return pd.DataFrame ints_meta: Metadata containing intensity statistics
        each z-slice
    """

    assert normalize_im in [
        None,
        "slice",
        "volume",
        "dataset",
    ], 'normalize_im must be None or "slice" or "volume" or "dataset"'

    if normalize_im is None:
        # No normalization
        frames_meta["zscore_median"] = 0
        frames_meta["zscore_iqr"] = 1
        return frames_meta
    elif normalize_im == "dataset":
        agg_cols = ["time_idx", "channel_idx", "dir_name"]
    elif normalize_im == "volume":
        agg_cols = ["time_idx", "channel_idx", "dir_name", "pos_idx"]
    else:
        agg_cols = ["time_idx", "channel_idx", "dir_name", "pos_idx", "slice_idx"]
    # median and inter-quartile range are more robust than mean and std
    ints_meta_sub = ints_meta[ints_meta["fg_frac"] >= min_fraction]
    ints_agg_median = ints_meta_sub[agg_cols + ["intensity"]].groupby(agg_cols).median()
    ints_agg_hq = (
        ints_meta_sub[agg_cols + ["intensity"]].groupby(agg_cols).quantile(0.75)
    )
    ints_agg_lq = (
        ints_meta_sub[agg_cols + ["intensity"]].groupby(agg_cols).quantile(0.25)
    )
    ints_agg = ints_agg_median
    ints_agg.columns = ["zscore_median"]
    ints_agg["zscore_iqr"] = ints_agg_hq["intensity"] - ints_agg_lq["intensity"]
    ints_agg.reset_index(inplace=True)

    cols_to_merge = frames_meta.columns[
        [col not in ["zscore_median", "zscore_iqr"] for col in frames_meta.columns]
    ]
    frames_meta = pd.merge(
        frames_meta[cols_to_merge],
        ints_agg,
        how="left",
        on=agg_cols,
    )
    if frames_meta["zscore_median"].isnull().values.any():
        raise ValueError(
            "Found NaN in normalization parameters. \
        min_fraction might be too low or images might be corrupted."
        )
    frames_meta_filename = os.path.join(input_dir, "frames_meta.csv")
    frames_meta.to_csv(frames_meta_filename, sep=",")

    cols_to_merge = ints_meta.columns[
        [col not in ["zscore_median", "zscore_iqr"] for col in ints_meta.columns]
    ]
    ints_meta = pd.merge(
        ints_meta[cols_to_merge],
        ints_agg,
        how="left",
        on=agg_cols,
    )
    ints_meta["intensity_norm"] = (
        ints_meta["intensity"] - ints_meta["zscore_median"]
    ) / (ints_meta["zscore_iqr"] + sys.float_info.epsilon)

    return frames_meta, ints_meta
