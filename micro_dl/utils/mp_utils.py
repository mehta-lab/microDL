import cv2
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import pickle
import sys

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
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
        res = ex.map(create_save_mask, *zip(*fn_args))
    return list(res)


def create_save_mask(channels_meta_sub,
                     flat_field_fnames,
                     str_elem_radius,
                     mask_dir,
                     mask_channel_idx,
                     int2str_len,
                     mask_type,
                     mask_ext,
                     zarr_bytes,
                     channel_thrs=None):

    """
    Create and save mask.
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
    :param list channel_thrs: list of threshold for each channel to generate
    binary masks. Only used when mask_type is 'dataset_otsu'
    :return dict cur_meta: One for each mask. fg_frac is added to metadata
            - how is it used?
    """
    im_stack = image_utils.read_imstack_from_meta(
        frames_meta_sub=channels_meta_sub,
        flat_field_fnames=flat_field_fnames,
        normalize_im=None,
    )
    if mask_type == 'dataset otsu':
        assert channel_thrs is not None, \
            'channel threshold is required for mask_type="dataset otsu"'
        assert len(channel_thrs) == range(im_stack.shape[-1]), \
            "Mismatch between channel thrs {} and im_stack {}".format(
                len(channel_thrs), im_stack.shape[-1])

    masks = []
    for idx in range(im_stack.shape[-1]):
        im = im_stack[..., idx]
        if mask_type == 'otsu':
            mask = mask_utils.create_otsu_mask(im.astype('float32'), str_elem_radius)
        elif mask_type == 'unimodal':
            mask = mask_utils.create_unimodal_mask(im.astype('float32'),  str_elem_radius)
        elif mask_type == 'dataset otsu':
            mask = mask_utils.create_otsu_mask(im.astype('float32'), str_elem_radius, channel_thrs[idx])
        elif mask_type == 'borders_weight_loss_map':
            mask = mask_utils.get_unet_border_weight_map(im)
        masks += [mask]
    # Border weight map mask is a float mask not binary like otsu or unimodal,
    # so keep it as is (assumes only one image in stack)
    fg_frac = None
    if mask_type == 'borders_weight_loss_map':
        mask = masks[0]
    else:
        masks = np.stack(masks, axis=-1)
        # mask = np.any(masks, axis=-1)
        mask = np.mean(masks, axis=-1)
        fg_frac = np.mean(mask)

    # Create mask name for given slice, time and position
    time_idx = int(channels_meta_sub['time_idx'].iloc[0])
    slice_idx = int(channels_meta_sub['slice_idx'].iloc[0])
    pos_idx = int(channels_meta_sub['pos_idx'].iloc[0])
    file_name = aux_utils.get_im_name(
        time_idx=channels_meta_sub['time_idx'],
        channel_idx=channels_meta_sub['mask_channel_idx'],
        slice_idx=channels_meta_sub['slice_idx'],
        pos_idx=channels_meta_sub['pos_idx'],
        int2str_len=int2str_len,
        ext=mask_ext,
    )
    overlay_name = aux_utils.get_im_name(
        time_idx=channels_meta_sub['time_idx'],
        channel_idx=mask_channel_idx,
        slice_idx=channels_meta_sub['slice_idx'],
        pos_idx=channels_meta_sub['pos_idx'],
        int2str_len=int2str_len,
        extra_field='overlay',
        ext=mask_ext,
    )
    if mask_ext == '.npy':
        # Save mask for given channels, mask is 2D
        np.save(os.path.join(mask_dir, file_name),
                mask,
                allow_pickle=True,
                fix_imports=True)
    elif mask_ext == '.png':
        # Covert mask to uint8
        # Border weight map mask is a float mask not binary like otsu or unimodal,
        # so keep it as is
        if mask_type == 'borders_weight_loss_map':
            assert im_stack.shape[-1] == 1
            # Note: Border weight map mask should only be generated from one binary image
        else:
            mask = image_utils.im_bit_convert(mask, bit=8, norm=True)
            mask = image_utils.im_adjust(mask)
            im_mean = np.mean(im_stack, axis=-1)
            im_mean = hist_clipping(im_mean, 1, 99)
            im_alpha = 255 / (np.max(im_mean) - np.min(im_mean) + sys.float_info.epsilon)
            im_mean = cv2.convertScaleAbs(
                im_mean - np.min(im_mean),
                alpha=im_alpha,
                )
            im_mask_overlay = np.stack([mask, im_mean, mask], axis=2)
            cv2.imwrite(os.path.join(mask_dir, overlay_name), im_mask_overlay)

        cv2.imwrite(os.path.join(mask_dir, file_name), mask)
    else:
        raise ValueError("mask_ext can be '.npy' or '.png', not {}".format(mask_ext))
    cur_meta = {'channel_idx': mask_channel_idx,
                'slice_idx': channels_meta_sub['slice_idx'],
                'time_idx': channels_meta_sub['time_idx'],
                'pos_idx': channels_meta_sub['pos_idx'],
                'file_name': file_name,
                'fg_frac': fg_frac,
                }
    return cur_meta


def get_mask_meta_row(file_path, meta_row):
    mask = image_utils.read_image(file_path)
    fg_frac = np.sum(mask > 0) / mask.size
    meta_row = {**meta_row, 'fg_frac': fg_frac}
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


def tile_and_save(meta_sub,
                  flat_field_fname,
                  hist_clip_limits,
                  slice_idx,
                  tile_size,
                  step_size,
                  min_fraction,
                  image_format,
                  save_dir,
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
    :param int int2str_len: len of indices for creating file names
    :param bool is_mask: Indicates if files are masks
    :param str/None normalize_im: Normalization method
    :param float/None zscore_mean: Mean for normalization
    :param float/None zscore_std: Std for normalization
    :return: pd.DataFrame from a list of dicts with metadata
    """
    time_idx = meta_sub.loc[0, 'time_idx']
    channel_idx = meta_sub.loc[0, 'channel_idx']
    pos_idx = meta_sub.loc[0, 'pos_idx']
    try:
        input_image = image_utils.read_imstack_from_meta(
            frames_meta_sub=meta_sub,
            flat_field_fnames=flat_field_fname,
            hist_clip_limits=hist_clip_limits,
            is_mask=is_mask,
            normalize_im=normalize_im,
            zscore_mean=zscore_mean,
            zscore_std=zscore_std
        )
        save_dict = {'time_idx': time_idx,
                     'channel_idx': channel_idx,
                     'pos_idx': pos_idx,
                     'slice_idx': slice_idx,
                     'save_dir': save_dir,
                     'image_format': image_format,
                     'int2str_len': int2str_len}

        tile_meta_df = tile_utils.tile_image(
            input_image=input_image,
            tile_size=tile_size,
            step_size=step_size,
            min_fraction=min_fraction,
            save_dict=save_dict,
        )
    except Exception as e:
        err_msg = 'error in t_{}, c_{}, pos_{}, sl_{}'.format(
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


def crop_at_indices_save(meta_sub,
                         flat_field_fname,
                         hist_clip_limits,
                         slice_idx,
                         crop_indices,
                         image_format,
                         save_dir,
                         int2str_len=3,
                         is_mask=False,
                         tile_3d=False,
                         normalize_im=True,
                         zscore_mean=None,
                         zscore_std=None
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
    :param int int2str_len: len of indices for creating file names
    :param bool is_mask: Indicates if files are masks
    :param bool tile_3d: indicator for tiling in 3D
    :return: pd.DataFrame from a list of dicts with metadata
    """
    time_idx = meta_sub.loc[0, 'time_idx']
    channel_idx = meta_sub.loc[0, 'channel_idx']
    pos_idx = meta_sub.loc[0, 'pos_idx']
    try:
        input_image = image_utils.read_imstack_from_meta(
            frames_meta_sub=meta_sub,
            flat_field_fnames=flat_field_fname,
            hist_clip_limits=hist_clip_limits,
            is_mask=is_mask,
            normalize_im=normalize_im,
            zscore_mean=zscore_mean,
            zscore_std=zscore_std
        )
        save_dict = {'time_idx': time_idx,
                     'channel_idx': channel_idx,
                     'pos_idx': pos_idx,
                     'slice_idx': slice_idx,
                     'save_dir': save_dir,
                     'image_format': image_format,
                     'int2str_len': int2str_len}

        tile_meta_df = tile_utils.crop_at_indices(
            input_image=input_image,
            crop_indices=crop_indices,
            save_dict=save_dict,
            tile_3d=tile_3d,
        )
    except Exception as e:
        err_msg = 'error in t_{}, c_{}, pos_{}, sl_{}'.format(
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
    :param str output_dir: Path to output directory
    :param float scale_factor: Scale factor for resizing
    :param str ff_path: Path to flatfield image
    """
    meta_row = kwargs['meta_row']
    im = image_utils.read_image_from_row(meta_row)

    if kwargs['ff_path'] is not None:
        im = image_utils.apply_flat_field_correction(
            im,
            flat_field_path=kwargs['ff_path'],
        )
    im_resized = image_utils.rescale_image(
        im=im,
        scale_factor=kwargs['scale_factor'],
    )
    # Write image
    # TODO: will we keep this functionality and thus write to zarr?
    # If so, should I create a zarr roots before mp?
    # Where to do init_array for each position?
    if 'zarr' in meta_row['file_name'][-5:]:
        im_name = aux_utils.get_im_name(
                        channel_idx=meta_row['channel_idx'],
                        slice_idx=meta_row['slice_idx'],
                        time_idx=meta_row['time_idx'],
                        pos_idx=meta_row['pos_idx'],
                    )
        write_path = os.path.join(kwargs['output_dir'], im_name)
    else:
        write_path = os.path.join(kwargs['output_dir'], meta_row['file_name'])
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


def rescale_vol_and_save(time_idx,
                         pos_idx,
                         channel_idx,
                         slice_start_idx,
                         slice_end_idx,
                         frames_metadata,
                         output_fname,
                         scale_factor,
                         ff_path):
    """Rescale volumes and save

    :param int time_idx: time point of input image
    :param int pos_idx: sample idx of input image
    :param int channel_idx: channel idx of input image
    :param int slice_start_idx: start slice idx for the vol to be saved
    :param int slice_end_idx: end slice idx for the vol to be saved
    :param pd.Dataframe frames_metadata: metadata for the input slices
    :param str output_fname: output_fname
    :param float/list scale_factor: scale factor for resizing
    :param str/None ff_path: path to flat field image
    :param bytes zarr_bytes: Serialized zarr object
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
        cur_img = image_utils.read_image_from_row(meta_row)
        if ff_path is not None:
            cur_img = image_utils.apply_flat_field_correction(
                cur_img,
                flat_field_path=ff_path,
            )
        input_stack.append(cur_img)
    input_stack = np.stack(input_stack, axis=2)
    resc_vol = image_utils.rescale_nd_image(input_stack, scale_factor)
    np.save(output_fname, resc_vol, allow_pickle=True, fix_imports=True)
    cur_metadata = {'time_idx': time_idx,
                    'pos_idx': pos_idx,
                    'channel_idx': channel_idx,
                    'slice_idx': slice_start_idx,
                    'file_name': os.path.basename(output_fname),
                    'mean': np.mean(resc_vol),
                    'std': np.std(resc_vol)}
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
    meta_row = {
        'mean': np.nanmean(im),
        'std': np.nanstd(im)
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


def sample_im_pixels(meta_row, ff_path, grid_spacing):
    """
    Read and computes statistics of images for each point in a grid.
    Grid spacing determines distance in pixels between grid points
    for rows and cols.
    Applies flatfield correction prior to intensity sampling if flatfield
    path is specified.

    :param dict meta_row: Metadata row for image
    :param str ff_path: Full path to flatfield image corresponding to image
    :param int grid_spacing: Distance in pixels between sampling points
    :return list meta_rows: Dicts with intensity data for each grid point
    """
    im = image_utils.read_image_from_row(meta_row)
    if ff_path is not None:
        im = image_utils.apply_flat_field_correction(
            input_image=im,
            flat_field_path=ff_path,
        )
    row_ids, col_ids, sample_values = \
        image_utils.grid_sample_pixel_values(im, grid_spacing)

    meta_rows = \
        [{**meta_row,
          'row_idx': row_idx,
          'col_idx': col_idx,
          'intensity': sample_value}
         for row_idx, col_idx, sample_value
         in zip(row_ids, col_ids, sample_values)]
    return meta_rows
