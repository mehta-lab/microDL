import cv2
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import sys
import scipy.stats

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
import micro_dl.utils.io_utils as io_utils
import micro_dl.utils.masks as mask_utils
import micro_dl.utils.tile_utils as tile_utils
from micro_dl.utils.normalize import hist_clipping
from micro_dl.torch_unet.utils.io import show_progress_bar


def mp_wrapper(fn, fn_args, workers):
    """Create and save masks with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned dicts from create_save_mask
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(fn, *zip(*fn_args))
    return list(res)


def mp_create_and_write_mask(fn_args, workers):
    """Create and save masks with multiprocessing. For argument parameters
    see mp_utils.create_and_write_mask.

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned dicts from create_save_mask
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(create_and_write_mask, *zip(*fn_args))
    return list(res)


def create_and_write_mask(
    zarr_dir,
    position,
    time_indices,
    channel_indices,
    structure_elem_radius,
    mask_type,
    mask_name,
    output_channel_index=None,
    verbose=False,
):
    # TODO: rewrite docstring
    """
    Create mask *for all depth slices* at each time and channel index specified
    in this position, and save them both as an additional channel in the data array
    of the given zarr store and a separate 'untracked' array with specified name.
    If output_channel_index is specified as an existing channel index, will overwrite
    this channel instead.

    Saves custom metadata related to the mask creation in the well-level
    .zattrs in the 'mask' field.

    When >1 channel are used to generate the mask, mask of each channel is
    generated then added together. Foreground fraction is calculated on
    a timepoint-position basis. That is, it will be recorded as an average
    foreground fraction over all slices in any given timepoint.


    :param str zarr_dir: directory to HCS compatible zarr store for usage
    :param int position: position to generate masks for
    :param list time_indices: list of time indices for mask generation,
                            if an index is skipped over, will populate with
                            zeros
    :param list channel_indices: list of channel indices for mask generation,
                            if more than 1 channel specified, masks from all
                            channels are aggregated
    :param int structure_elem_radius: size of structuring element used for binary
                            opening. str_elem: disk or ball
    :param str mask_type: thresholding type used for masking or str to map to
                            masking function
    :param str mask_name: name under which to save untracked copy of mask in
                            position
    :param int/None output_channel_index: if specified will overwrite data in this
                            channel with computed masks. By default does not
                            overwrite
    :param bool verbose: whether this process should send updates to stdout
    """

    # read in stack
    modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir)
    position_zarr = modifier.get_zarr(position=position)
    position_masks_shape = tuple(
        [modifier.shape[0], len(channel_indices), *modifier.shape[2:]]
    )

    # calculate masks over every time index and channel slice
    position_masks = np.zeros(position_masks_shape)
    position_foreground_fractions = {}

    for time_index in range(modifier.frames):
        timepoint_foreground_fraction = {}

        for channel_index in channel_indices:
            channel_name = modifier.channel_names[channel_index]
            mask_array_chan_idx = channel_indices.index(channel_index)

            if "mask" in channel_name:
                print("\n")
                if mask_type in channel_name:
                    print(f"Found existing channel: '{channel_name}'.")
                    print("You are likely creating duplicates, which is bad practice.")
                print(f"Skipping mask channel '{channel_name}' for thresholding")
            else:
                for slice_index in range(modifier.slices):
                    # print pyrimidal progress bar
                    if verbose:
                        time_progress = f"time {time_index+1}/{modifier.frames}"
                        channel_progress = f"chan {channel_index}/{channel_indices}"
                        position_progress = f"pos {position}/{modifier.positions}"
                        slice_progress = f"slice {slice_index}/{modifier.slices}"
                        p = (
                            f"Computing masks slice [{position_progress}, {time_progress},"
                            f" {channel_progress}, {slice_progress}]"
                        )
                        print(p)

                    # get mask for image slice or populate with zeros
                    if time_index in time_indices:
                        try:
                            flatfield_slice = modifier.get_untracked_array_slice(
                                position=position,
                                meta_field_name="flatfield",
                                time_index=time_index,
                                channel_index=channel_index,
                                z_index=slice_index,
                            )
                        except Exception as e:
                            flatfield_slice = None

                        mask = get_mask_slice(
                            position_zarr=position_zarr,
                            time_index=time_index,
                            channel_index=channel_index,
                            slice_index=slice_index,
                            mask_type=mask_type,
                            structure_elem_radius=structure_elem_radius,
                            flatfield_array=flatfield_slice,
                        )
                    else:
                        mask = np.zeros(modifier.shape[-2:])

                    position_masks[time_index, mask_array_chan_idx, slice_index] = mask

                # compute & record channel-wise foreground fractions
                frame_foreground_fraction = float(
                    np.mean(position_masks[time_index, mask_array_chan_idx]).item()
                )
                timepoint_foreground_fraction[channel_name] = frame_foreground_fraction
        position_foreground_fractions[time_index] = timepoint_foreground_fraction

    # combine masks along channels and compute & record combined foreground fraction
    position_masks = np.expand_dims(np.sum(position_masks, axis=1), 1)
    position_masks = np.where(position_masks > 0.5, 1, 0)
    for time_index in time_indices:
        frame_foreground_fraction = float(np.mean(position_masks[time_index]).item())
        timepoint_foreground_fraction["combined_fraction"] = frame_foreground_fraction

    # save masks as additional channel
    position_masks = position_masks.astype(position_zarr.dtype)
    contrast_limits = [
        0,
        float(np.shape(position_masks)[-1]),
        float(np.min(position_masks)),
        float(np.max(position_masks)),
    ]
    modifier.add_channel(
        new_channel_array=position_masks,
        position=position,
        omero_metadata=modifier.generate_omero_channel_meta(
            channel_name=mask_name,
            contrast_limits=contrast_limits,
        ),
        channel_index=output_channel_index,
    )

    # save masks as an 'untracked' array
    if mask_type in {"otsu", "unimodal","edge_detection"}:
        position_masks = position_masks.astype("bool")

    modifier.init_untracked_array(
        data_array=position_masks,
        position=position,
        name=mask_name,
    )

    # save custom tracking metadata
    metadata = {
        "array_name": mask_name,
        "masking_type": mask_type,
        "channel_ids": channel_indices,
        "time_idx": time_indices,
        "foreground_fractions_by_timepoint": position_foreground_fractions,
    }
    modifier.write_meta_field(
        position=position,
        metadata=metadata,
        field_name="mask",
    )


def get_mask_slice(
    position_zarr,
    time_index,
    channel_index,
    slice_index,
    mask_type,
    structure_elem_radius,
    flatfield_array=None,
):
    """
    Given a set of indices, mask type, and structuring element,
    pulls an image slice from the given zarr array, computes the
    requested mask and returns.

    If given a flatfield array, will flatfield correct the slice before
    computing the mask.

    :param zarr.Array position_zarr: zarr array of the desired position
    :param time_index: see name
    :param channel_index: see name
    :param slice_index: see name
    :param mask_type: see name,
                    options are {otsu, unimodal, edge_detection, borders_weight_loss_map}
    :param int structure_elem_radius: creation radius for the structuring
                    element
    :param np.ndarray flatfield_array: flatfield to correct image
    :return np.ndarray mask: 2d mask for this slice
    """
    # read and correct/preprocess slice
    im = position_zarr[time_index, channel_index, slice_index]
    if isinstance(flatfield_array, np.ndarray):
        im = image_utils.apply_flat_field_correction(
            input_image=im,
            flatfield_image=flatfield_array,
        )
    im = image_utils.preprocess_image(im, hist_clip_limits=(1, 99))
    # generate mask for slice
    if mask_type == "otsu":
        mask = mask_utils.create_otsu_mask(im.astype("float32"), structure_elem_radius)
    elif mask_type == "unimodal":
        mask = mask_utils.create_unimodal_mask(
            im.astype("float32"), structure_elem_radius
        )
    elif mask_type == "edge_detection":
        mask = mask_utils.create_edge_detection_mask(
            im.astype("float32"), structure_elem_radius
        )
    elif mask_type == "borders_weight_loss_map":
        mask = mask_utils.get_unet_border_weight_map(im)
        mask = image_utils.im_adjust(mask).astype(position_zarr.dtype)

    return mask


def mp_get_i_stats(fn_args, workers):
    """Read and computes statistics of images with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from get_im_stats
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(get_im_stats, fn_args)
        for r in res:
            print(r)
    return list(res)


def get_im_stats(im_path):
    """
    Read and computes statistics of images

    :param str im_path: Full path to image
    :return dict meta_row: Dict with intensity data for image
    """
    im = image_utils.read_image(im_path)
    meta_row = {"mean": np.nanmean(im), "std": np.nanstd(im)}
    return meta_row


def mp_get_val_stats(fn_args, workers):
    """
    Computes statistics of numpy arrays with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from get_im_stats
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(get_val_stats, fn_args)
    return list(res)


def get_val_stats(sample_values):
    """
    Computes the statistics of a numpy array and returns a dictionary
    of metadata corresponding to input sample values.

    :param list(float) sample_values: List of sample values at respective
                                        indices
    :return dict meta_row: Dict with intensity data for image
    """

    meta_row = {
        "mean": float(np.nanmean(sample_values)),
        "std": float(np.nanstd(sample_values)),
        "median": float(np.nanmedian(sample_values)),
        "iqr": float(scipy.stats.iqr(sample_values)),
    }
    return meta_row


def mp_sample_im_pixels(fn_args, workers):
    """Read and computes statistics of images with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from get_im_stats
    """

    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(sample_im_pixels, *zip(*fn_args))
    return list(res)


def sample_im_pixels(
    position,
    flatfield,
    zarr_dir,
    grid_spacing,
    channel,
):
    # TODO move out of mp utils into normalization utils
    """
    Read and computes statistics of images for each point in a grid.
    Grid spacing determines distance in pixels between grid points
    for rows and cols.
    Applies flatfield correction prior to intensity sampling if specified.
    By default, samples from every time position and every z-depth, and
    assumes that the data in the zarr store is stored in [T,C,Z,Y,X] format,
    for time, channel, z, y, x.

    :param int position: position currently being processed
    :param bool flatfield: whether to flatfield correct before sampling or not
    :param str zarr_dir: path to HCS-compatible zarr directory
    :param int grid_spacing: spacing of sampling grid in x and y
    :param int channel: channel to sample from

    :return list meta_rows: Dicts with intensity data for each grid point
    """
    modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir)
    image_zarr = modifier.get_zarr(position=position)

    all_sample_values = []
    all_time_indices = list(range(modifier.shape[0]))
    all_z_indices = list(range(modifier.shape[2]))

    # flatfield slice same for all time & z indices
    if flatfield:
        try:
            flatfield_slice = modifier.get_untracked_array_slice(
                position=position,
                meta_field_name="flatfield",
                time_index=0,
                channel_index=channel,
                z_index=0,
            )
        except:
            print(f"\nNo flatfield found: channel {channel}, position {position}")
            flatfield = False

    for time_index in all_time_indices:
        for z_index in all_z_indices:
            image_slice = image_zarr[time_index, channel, z_index, :, :]
            if flatfield:
                image_slice = image_utils.apply_flat_field_correction(
                    input_image=image_slice,
                    flatfield_image=flatfield_slice,
                )
            _, _, sample_values = image_utils.grid_sample_pixel_values(
                image_slice, grid_spacing
            )
            all_sample_values.append(sample_values)
    sample_values = np.stack(all_sample_values, 0).flatten()

    return sample_values
