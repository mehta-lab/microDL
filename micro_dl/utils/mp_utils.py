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


def mp_create_save_mask(fn_args, workers):
    """Create and save masks with multiprocessing

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
    str_elem_radius,
    int2str_len,
    mask_type,
    mask_name,
):
    # TODO: rewrite docstring
    """
    Create and save mask./home/christian.foley/virtual_staining/workspaces/microDL/micro_dl/utils
    When >1 channel are used to generate the mask, mask of each channel is
    generated then added together.

    :param pd.DataFrame channels_meta_sub: Metadata for given PTCZ
    :param list/None flat_field_fnames: Paths to corresponding flat field images
    :param int str_elem_radius: size of structuring element used for binary
     opening. str_elem: disk or ball
    :param str mask_dir: dir to save masks
    :param int mask_channel_idx: channel number of mask
    :param int time_idx: time points to use for generating mask
    :param int pos_idx: generate masks for given position / sample ids
    :param int slice_idx: generate masks for given slice ids
    :param int int2str_len: Length of str when converting ints
    :param str mask_type: thresholding type used for masking or str to map to
     masking function
    :param str mask_ext: '.npy' or '.png'. Save the mask as uint8 PNG or
     NPY files for otsu, unimodal masks, recommended to save as npy
     float64 for borders_weight_loss_map masks to avoid loss due to scaling it
     to uint8.
    :param str/None dir_name: Image directory (none if using frames_meta dir_name)
    :param list channel_thrs: list of threshold for each channel to generate
    binary masks. Only used when mask_type is 'dataset_otsu'
    :return dict cur_meta: One for each mask. fg_frac is added to metadata
            - how is it used?
    """
    # read in stack
    modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir)
    position_zarr = modifier.get_zarr(position=position)
    position_masks_shape = tuple([modifier.shape[0], 1, *modifier.shape[2:]])

    # calculate masks over every time index and channel slice
    position_masks = np.zeros(position_masks_shape).astype("float32")
    position_foreground_fractions = {}

    for time_index in time_indices:
        timepoint_foreground_fraction = {}

        for channel_index in channel_indices:
            im_stack = position_zarr[time_index, channel_index, ...]

            # compute mask for each slice in stack
            for slice_idx in range(im_stack.shape[0]):
                im = im_stack[slice_idx]
                if mask_type == "otsu":
                    mask = mask_utils.create_otsu_mask(
                        im.astype("float32"), str_elem_radius
                    )
                elif mask_type == "unimodal":
                    mask = mask_utils.create_unimodal_mask(
                        im.astype("float32"), str_elem_radius
                    )
                elif mask_type == "borders_weight_loss_map":
                    mask = mask_utils.get_unet_border_weight_map(im)

                mask = image_utils.im_adjust(mask).astype(position_zarr.dtype)
                position_masks[time_index, channel_index, slice_idx] = mask

            # compute & record the foreground fraction for this channel
            channel_foreground_fraction = np.mean(
                position_masks[time_index, channel_index]
            ).item()
            channel_name = modifier.channel_names[channel_index]
            timepoint_foreground_fraction[channel_name] = channel_foreground_fraction

        # aggregate all channel-wise foreground fractions
        position_foreground_fractions[time_index] = timepoint_foreground_fraction

    # save masks as additional channel
    position_masks = position_masks.astype(position_zarr.dtype)
    contrast_limits = [
        0,
        np.shape(position_masks)[-1],
        np.min(position_masks),
        np.max(position_masks),
    ]
    modifier.add_channel(
        new_channel_array=position_masks,
        position=position,
        metadata=modifier.generate_omero_channel_meta(
            channel_name=channel_name,
            contrast_limits=contrast_limits,
        ),
    )

    # save masks as an 'untracked' array
    if mask_type in {"otsu", "unimodal"}:
        position_masks = position_masks.astype("bool")

    modifier.init_untracked_array(
        array=position_masks,
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


def get_mask_meta_row(file_path, meta_row):
    mask = image_utils.read_image(file_path)
    fg_frac = np.sum(mask > 0) / mask.size
    meta_row = {**meta_row, "fg_frac": fg_frac}
    return meta_row


def mp_tile_save(fn_args, workers):
    """Tile and save with multiprocessing
    https://stackoverflow.com/questions/42074501/python-concurrent-futures-processpoolexecutor-performance-of-submit-vs-map
    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from tile_and_save
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(tile_and_save, *zip(*fn_args))
    return list(res)


def tile_and_save(
    meta_sub,
    flat_field_fname,
    hist_clip_limits,
    slice_idx,
    tile_size,
    step_size,
    min_fraction,
    image_format,
    save_dir,
    dir_name=None,
    int2str_len=3,
    is_mask=False,
    normalize_im=None,
    zscore_mean=None,
    zscore_std=None,
):
    """
    Crop image into tiles at given indices and save

    :param pd.DataFrame meta_sub: Subset of metadata for images to be tiled
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param int slice_idx: slice idx of input image
    :param list tile_size: size of tile along row, col (& slices)
    :param list step_size: step size along row, col (& slices)
    :param float min_fraction: min foreground volume fraction for keep tile
    :param str image_format: zyx / xyz
    :param str save_dir: output dir to save tiles
    :param str/None dir_name: Image directory
    :param int int2str_len: len of indices for creating file names
    :param bool is_mask: Indicates if files are masks
    :param str/None normalize_im: Normalization method
    :param float/None zscore_mean: Mean for normalization
    :param float/None zscore_std: Std for normalization
    :return: pd.DataFrame from a list of dicts with metadata
    """
    time_idx = meta_sub.loc[0, "time_idx"]
    channel_idx = meta_sub.loc[0, "channel_idx"]
    pos_idx = meta_sub.loc[0, "pos_idx"]
    try:
        input_image = image_utils.read_imstack_from_meta(
            frames_meta_sub=meta_sub,
            dir_name=dir_name,
            flat_field_fnames=flat_field_fname,
            hist_clip_limits=hist_clip_limits,
            is_mask=is_mask,
            normalize_im=normalize_im,
            zscore_mean=zscore_mean,
            zscore_std=zscore_std,
        )
        save_dict = {
            "time_idx": time_idx,
            "channel_idx": channel_idx,
            "pos_idx": pos_idx,
            "slice_idx": slice_idx,
            "save_dir": save_dir,
            "image_format": image_format,
            "int2str_len": int2str_len,
        }

        tile_meta_df = tile_utils.tile_image(
            input_image=input_image,
            tile_size=tile_size,
            step_size=step_size,
            min_fraction=min_fraction,
            save_dict=save_dict,
        )
    except Exception as e:
        err_msg = "error in t_{}, c_{}, pos_{}, sl_{}".format(
            time_idx, channel_idx, pos_idx, slice_idx
        )
        err_msg = err_msg + str(e)
        # TODO(Anitha) write to log instead
        print(err_msg)
        raise e
    return tile_meta_df


def mp_crop_save(fn_args, workers):
    """Crop and save images with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from crop_at_indices_save
    """

    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(crop_at_indices_save, *zip(*fn_args))
    return list(res)


def crop_at_indices_save(
    meta_sub,
    flat_field_fname,
    hist_clip_limits,
    slice_idx,
    crop_indices,
    image_format,
    save_dir,
    dir_name=None,
    int2str_len=3,
    is_mask=False,
    tile_3d=False,
    normalize_im=True,
    zscore_mean=None,
    zscore_std=None,
):
    """Crop image into tiles at given indices and save

    :param pd.DataFrame meta_sub: Subset of metadata for images to be cropped
    :param str flat_field_fname: File nname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param int time_idx: time point of input image
    :param int channel_idx: channel idx of input image
    :param int slice_idx: slice idx of input image
    :param int pos_idx: sample idx of input image
    :param tuple crop_indices: tuple of indices for cropping
    :param str image_format: zyx or xyz
    :param str save_dir: output dir to save tiles
    :param str/None dir_name: Input directory
    :param int int2str_len: len of indices for creating file names
    :param bool is_mask: Indicates if files are masks
    :param bool tile_3d: indicator for tiling in 3D
    :return: pd.DataFrame from a list of dicts with metadata
    """
    time_idx = meta_sub.loc[0, "time_idx"]
    channel_idx = meta_sub.loc[0, "channel_idx"]
    pos_idx = meta_sub.loc[0, "pos_idx"]
    try:
        input_image = image_utils.read_imstack_from_meta(
            frames_meta_sub=meta_sub,
            dir_name=dir_name,
            flat_field_fnames=flat_field_fname,
            hist_clip_limits=hist_clip_limits,
            is_mask=is_mask,
            normalize_im=normalize_im,
            zscore_mean=zscore_mean,
            zscore_std=zscore_std,
        )
        save_dict = {
            "time_idx": time_idx,
            "channel_idx": channel_idx,
            "pos_idx": pos_idx,
            "slice_idx": slice_idx,
            "save_dir": save_dir,
            "image_format": image_format,
            "int2str_len": int2str_len,
        }

        tile_meta_df = tile_utils.crop_at_indices(
            input_image=input_image,
            crop_indices=crop_indices,
            save_dict=save_dict,
            tile_3d=tile_3d,
        )
    except Exception as e:
        err_msg = "error in t_{}, c_{}, pos_{}, sl_{}".format(
            time_idx, channel_idx, pos_idx, slice_idx
        )
        err_msg = err_msg + str(e)
        # TODO(Anitha) write to log instead
        print(err_msg)
        raise e

    return tile_meta_df


def mp_resize_save(mp_args, workers):
    """
    Resize and save images with multiprocessing

    :param list mp_args: Function keyword arguments
    :param int workers: max number of workers
    """
    with ProcessPoolExecutor(workers) as ex:
        {ex.submit(resize_and_save, **kwargs): kwargs for kwargs in mp_args}


def resize_and_save(**kwargs):
    """
    Resizing images and saving them.

    Keyword arguments:
    :param pd.DataFrame meta_row: Row of metadata
    :param str/None dir_name: Image directory (none if using dir_name from frames_meta)
    :param str output_dir: Path to output directory
    :param float scale_factor: Scale factor for resizing
    :param str ff_path: Path to flatfield image
    """
    meta_row = kwargs["meta_row"]
    im = image_utils.read_image_from_row(meta_row, kwargs["dir_name"])

    if kwargs["ff_path"] is not None:
        im = image_utils.apply_flat_field_correction(
            im,
            flat_field_path=kwargs["ff_path"],
        )
    im_resized = image_utils.rescale_image(
        im=im,
        scale_factor=kwargs["scale_factor"],
    )
    # Write image
    # TODO: will we keep this functionality and thus write to zarr?
    # If so, should I create a zarr roots before mp?
    # Where to do init_array for each position?
    if "zarr" in meta_row["file_name"][-5:]:
        im_name = aux_utils.get_im_name(
            channel_idx=meta_row["channel_idx"],
            slice_idx=meta_row["slice_idx"],
            time_idx=meta_row["time_idx"],
            pos_idx=meta_row["pos_idx"],
        )
        write_path = os.path.join(kwargs["output_dir"], im_name)
    else:
        write_path = os.path.join(kwargs["output_dir"], meta_row["file_name"])
    cv2.imwrite(write_path, im_resized)


def mp_rescale_vol(fn_args, workers):
    """Rescale and save image stacks with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(rescale_vol_and_save, *zip(*fn_args))
    return list(res)


def rescale_vol_and_save(
    time_idx,
    pos_idx,
    channel_idx,
    slice_start_idx,
    slice_end_idx,
    frames_metadata,
    dir_name,
    output_fname,
    scale_factor,
    ff_path,
):
    """Rescale volumes and save

    :param int time_idx: time point of input image
    :param int pos_idx: sample idx of input image
    :param int channel_idx: channel idx of input image
    :param int slice_start_idx: start slice idx for the vol to be saved
    :param int slice_end_idx: end slice idx for the vol to be saved
    :param pd.Dataframe frames_metadata: metadata for the input slices
    :param str/None dir_name: Image directory (none if using dir_name from frames_meta)
    :param str output_fname: output_fname
    :param float/list scale_factor: scale factor for resizing
    :param str/None ff_path: path to flat field image
    """
    input_stack = []
    for slice_idx in range(slice_start_idx, slice_end_idx):
        meta_idx = aux_utils.get_meta_idx(
            frames_metadata,
            time_idx,
            channel_idx,
            slice_idx,
            pos_idx,
        )
        meta_row = frames_metadata.loc[meta_idx]
        if dir_name is None:
            dir_name = meta_row["dir_name"]
        cur_img = image_utils.read_image_from_row(meta_row, dir_name)
        if ff_path is not None:
            cur_img = image_utils.apply_flat_field_correction(
                cur_img,
                flat_field_path=ff_path,
            )
        input_stack.append(cur_img)
    input_stack = np.stack(input_stack, axis=2)
    resc_vol = image_utils.rescale_nd_image(input_stack, scale_factor)
    np.save(output_fname, resc_vol, allow_pickle=True, fix_imports=True)
    cur_metadata = {
        "time_idx": time_idx,
        "pos_idx": pos_idx,
        "channel_idx": channel_idx,
        "slice_idx": slice_start_idx,
        "file_name": os.path.basename(output_fname),
        "mean": np.mean(resc_vol),
        "std": np.std(resc_vol),
    }
    return cur_metadata


def mp_get_im_stats(fn_args, workers):
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
    ff_name,
    ff_channels,
    zarr_dir,
    grid_spacing,
    channel,
):
    # TODO move out of mp utils into normalization utils
    """
    Read and computes statistics of images for each point in a grid.
    Grid spacing determines distance in pixels between grid points
    for rows and cols.
    Applies flatfield correction prior to intensity sampling if flatfield
    path is specified.
    By default, samples from every time position and every z-depth, and
    assumes that the data in the zarr store is stored in [T,C,Z,Y,X] format,
    for time, channel, z, y, x.

    :param int position: position currently being processed
    :param str ff_name: name of 'untracked' flatfield array
    :param list ff_channels: list of channels with flatfield channel IDs,
                                only used if a flatfield name is provided
    :param str zarr_dir: path to HCS-compatible zarr directory
    :param int grid_spacing: spacing of sampling grid in x and y
    :param int channel: channel to sample from

    :return list meta_rows: Dicts with intensity data for each grid point
    """
    modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir)
    image_zarr = modifier.get_zarr(position=position)

    flatfield = None
    if ff_name is not None and channel in ff_channels:
        flatfield = modifier.get_untracked_array(position=position, name=ff_name)

    all_sample_values = []
    all_time_indices = list(range(modifier.shape[0]))
    all_z_indices = list(range(modifier.shape[2]))

    for time_index in all_time_indices:
        if flatfield is not None:
            # flatfield array might have collapsed indices
            ff_channel_pos = ff_channels.index(channel)
            flatfield_slice = flatfield[time_index, ff_channel_pos, 0, :, :]

        for z_index in all_z_indices:
            image_slice = image_zarr[time_index, channel, z_index, :, :]
            if flatfield is not None:
                image_slice = image_utils.apply_flat_field_correction(
                    input_image=image_slice,
                    flat_field_image=flatfield_slice,
                )
            _, _, sample_values = image_utils.grid_sample_pixel_values(
                image_slice, grid_spacing
            )
            all_sample_values.append(sample_values)
    sample_values = np.stack(all_sample_values, 0).flatten()

    return sample_values
