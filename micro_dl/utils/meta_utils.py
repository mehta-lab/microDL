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


def frames_meta_generator(
    input_dir,
    file_format="zarr",
    order="cztp",
    name_parser="parse_sms_name",
):
    """
    Generate metadata from file names, or metadata in the case of zarr files,
    for preprocessing.
    Will write found data in frames_metadata.csv in input directory.
    Assumed default file naming convention is for 'parse_idx_from_name':
    dir_name
    |
    |- im_c***_z***_t***_p***.png
    |- im_c***_z***_t***_p***.png

    c is channel
    z is slice in stack (z)
    t is time
    p is position (FOV)

    Naming convention for 'parse_sms_name':
    img_channelname_t***_p***_z***.tif for parse_sms_name

    The file structure for ome-zarr files is described here:
    https://ngff.openmicroscopy.org/0.1/

    :param str input_dir:   path to input directory containing image data
    :param str file_format: Image file format ('zarr' or 'tiff' or 'png')
    :param str order: Order in which file name encodes cztp (for tiff/png)
    :param str name_parser: Function in aux_utils for parsing indices from tiff/png file name
    :return pd.DataFrame frames_meta: Metadata for all frames in dataset
    """
    if "zarr" in file_format:
        zarr_files = glob.glob(os.path.join(input_dir, "*.zarr"))
        assert (
            len(zarr_files) > 0
        ), "file_format specified as zarr, but no zarr files found"
        # Generate frames_meta from zarr metadata
        frames_meta = frames_meta_from_zarr(input_dir, zarr_files)
    elif "tif" in file_format or "png" in file_format:
        frames_meta = frames_meta_from_filenames(
            input_dir,
            name_parser,
            order,
        )
    else:
        raise FileNotFoundError("Check that file_format matches image files")

    # Write metadata
    frames_meta_filename = os.path.join(input_dir, "frames_meta.csv")
    frames_meta.to_csv(frames_meta_filename, sep=",")
    return frames_meta


def frames_meta_from_filenames(input_dir, name_parser, order):
    """
    :param str input_dir:   path to input directory containing images
    :param str name_parser: Function in aux_utils for parsing indices from file name
    :return pd.DataFrame frames_meta: Metadata for all frames in dataset
    :param str order: Order in which file name encodes cztp (for tiff/png)
    """
    parse_func = aux_utils.import_object("utils.aux_utils", name_parser, "function")
    im_names = aux_utils.get_sorted_names(input_dir)
    frames_meta = aux_utils.make_dataframe(nbr_rows=len(im_names))
    channel_names = []
    # Fill dataframe with rows from image names
    for i in range(len(im_names)):
        kwargs = {"im_name": im_names[i], "dir_name": input_dir}
        if name_parser == "parse_idx_from_name":
            kwargs["order"] = order
        elif name_parser == "parse_sms_name":
            kwargs["channel_names"] = channel_names
        meta_row = parse_func(**kwargs)
        frames_meta.loc[i] = meta_row
    return frames_meta


def frames_meta_from_zarr(input_dir, file_names):
    """
    Reads ome-zarr file and creates frames_meta based on metadata and
    array information.
    Assumes one zarr store per position according to OME guidelines.

    :param str input_dir: Input directory
    :param list Zarr_files: List of full paths to all zarr files in dir
    :return pd.DataFrame frames_meta: Metadata for all frames in zarr
    """
    zarr_reader = io_utils.ZarrReader(file_names[0])
    nbr_channels = zarr_reader.channels
    nbr_slices = zarr_reader.slices
    nbr_times = zarr_reader.frames
    channel_names = zarr_reader.channel_names

    nbr_rows = len(file_names) * nbr_channels * nbr_slices * nbr_times
    frames_meta = aux_utils.make_dataframe(nbr_rows=nbr_rows)
    meta_row = dict.fromkeys(list(frames_meta))
    meta_row["dir_name"] = input_dir
    idx = 0
    for pos_idx in range(len(file_names)):
        zarr_reader = io_utils.ZarrReader(file_names[pos_idx])
        meta_row["file_name"] = os.path.basename(file_names[pos_idx])
        # Get position index from name
        meta_row["pos_idx"] = int(zarr_reader.columns[0].split("_")[-1])
        for channel_idx in range(nbr_channels):
            for slice_idx in range(nbr_slices):
                for time_idx in range(nbr_times):
                    meta_row["channel_idx"] = channel_idx
                    meta_row["slice_idx"] = slice_idx
                    meta_row["time_idx"] = time_idx
                    meta_row["channel_name"] = channel_names[channel_idx]
                    frames_meta.loc[idx] = meta_row
                    idx += 1
    return frames_meta


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

    mp_grid_sampler_args = []
    for position in modifier.position_map:
        ff_name = None

        position_metadata = modifier.get_position_meta(position)
        if "flat_field" in position_metadata:
            ff_name = position_metadata["flat_field"]["array_name"]

        mp_grid_sampler_args.append((position, ff_name, zarr_dir, grid_spacing))

    for channel in channel_ids:
        # NOTE: Doing sequential mp with pool execution creates synchronization
        #      points between each step. This could be detrimental to performance
        fov_sample_values = mp_utils.mp_sample_im_pixels(
            mp_grid_sampler_args, num_workers
        )
        dataset_sample_values = np.stack(fov_sample_values, 0)

        # get statistics
        fov_level_statistics = mp_utils.mp_get_val_stats(fov_sample_values, num_workers)
        dataset_level_statistiscs = mp_utils.get_val_stats(dataset_sample_values)

        for position in modifier.position_map:
            position_statistics = {
                "fov_statistics": fov_level_statistics[position],
                "dataset_statistics": dataset_level_statistiscs[position],
            }
            channel_position_statistics = {
                channel: position_statistics,
            }

            modifier.write_meta_field(
                position=position,
                metadata=channel_position_statistics,
                field_name="normalization",
            )


def mask_meta_generator(
    input_dir,
    num_workers=4,
):
    """
    Generate pixel intensity metadata for estimating image normalization
    parameters during preprocessing step. Pixels are sub-sampled from the image
    following a grid pattern defined by block_size to for efficient estimation of
    median and interquatile range. Grid sampling is preferred over random sampling
    in the case due to the spatial correlation in images.
    Will write found data in intensity_meta.csv in input directory.
    Assumed default file naming convention is:
    dir_name
    |
    |- im_c***_z***_t***_p***.png
    |- im_c***_z***_t***_p***.png

    c is channel
    z is slice in stack (z)
    t is time
    p is position (FOV)

    Other naming convention is:
    img_channelname_t***_p***_z***.tif for parse_sms_name

    :param str input_dir: path to input directory containing images
    :param str order: Order in which file name encodes cztp
    :param str name_parser: Function in aux_utils for parsing indices from file name
    :param int num_workers: number of workers for multiprocessing
    :return pd.DataFrame mask_meta: Metadata with mask info
    """
    frames_metadata = aux_utils.read_meta(input_dir)
    mp_fn_args = []
    # Fill dataframe with rows from image names
    for i, meta_row in frames_metadata.iterrows():
        meta_row["dir_name"] = input_dir
        im_path = os.path.join(input_dir, meta_row["file_name"])
        mp_fn_args.append((im_path, meta_row))

    meta_row_list = mp_utils.mp_wrapper(
        mp_utils.get_mask_meta_row,
        mp_fn_args,
        num_workers,
    )
    mask_meta = pd.DataFrame.from_dict(meta_row_list)

    mask_meta_filename = os.path.join(input_dir, "mask_meta.csv")
    mask_meta.to_csv(mask_meta_filename, sep=",")
    return mask_meta


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
