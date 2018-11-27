from concurrent.futures import ProcessPoolExecutor

from micro_dl.utils import tile_utils as tile_utils


def mp_tile_save(fn_args, workers):
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(tile_and_save, *zip(*fn_args))
    return list(res)


def tile_and_save(input_fnames,
                  flat_field_fname,
                  hist_clip_limits,
                  time_idx,
                  channel_idx,
                  pos_idx,
                  slice_idx,
                  tile_size,
                  step_size,
                  min_fraction,
                  data_format,
                  isotropic,
                  save_dir,
                  int2str_len=3):
    """Crop image into tiles at given indices and save

    :param tuple input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param int time_idx: time point of input image
    :param int channel_idx: channel idx of input image
    :param int slice_idx: slice idx of input image
    :param int pos_idx: sample idx of input image
    :param list tile_size: size of tile along row, col (& slices)
    :param list step_size: step size along row, col (& slices)
    :param float min_fraction: min foreground volume fraction for keep tile
    :param str data_format: channels_first / channels_last
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :param str save_dir: output dir to save tiles
    :param int int2str_len: len of indices for creating file names
    :return: pd.DataFrame from a list of dicts with metadata
    """

    try:
        input_image = tile_utils.read_imstack(
            input_fnames=input_fnames,
            flat_field_fname=flat_field_fname,
            hist_clip_limits=hist_clip_limits
        )

        save_dict = {'time_idx': time_idx,
                     'channel_idx': channel_idx,
                     'pos_idx': pos_idx,
                     'slice_idx': slice_idx,
                     'save_dir': save_dir,
                     'data_format': data_format,
                     'int2str_len': int2str_len}

        tile_meta_df = tile_utils.tile_image(input_image=input_image,
                                             tile_size=tile_size,
                                             step_size=step_size,
                                             isotropic=isotropic,
                                             min_fraction=min_fraction,
                                             save_dict=save_dict)
    except Exception as e:
        err_msg = 'error in t_{}, c_{}, pos_{}, sl_{}'.format(
            time_idx, channel_idx, pos_idx, slice_idx
        )
        err_msg = err_msg + str(e)
        # TODO(Anitha) write to log instead
        print(err_msg)
        raise e
    return tile_meta_df


def mp_crop_at_indices_save(fn_args, workers):
    # TODO modify to take funtion as input
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(crop_at_indices_save, *zip(*fn_args))
    return list(res)


def crop_at_indices_save(input_fnames,
                         flat_field_fname,
                         hist_clip_limits,
                         time_idx,
                         channel_idx,
                         pos_idx,
                         slice_idx,
                         crop_indices,
                         data_format,
                         isotropic,
                         save_dir,
                         int2str_len=3):
    """Crop image into tiles at given indices and save

    :param tuple input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param int time_idx: time point of input image
    :param int channel_idx: channel idx of input image
    :param int slice_idx: slice idx of input image
    :param int pos_idx: sample idx of input image
    :param tuple crop_indices: tuple of indices for cropping
    :param str data_format: channels_first / channels_last
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :param str save_dir: output dir to save tiles
    :param int int2str_len: len of indices for creating file names
    :return: pd.DataFrame from a list of dicts with metadata
    """

    try:
        input_image = tile_utils.read_imstack(
            input_fnames=input_fnames,
            flat_field_fname=flat_field_fname,
            hist_clip_limits=hist_clip_limits
        )

        save_dict = {'time_idx': time_idx,
                     'channel_idx': channel_idx,
                     'pos_idx': pos_idx,
                     'slice_idx': slice_idx,
                     'save_dir': save_dir,
                     'data_format': data_format,
                     'int2str_len': int2str_len}

        tile_meta_df = tile_utils.crop_at_indices(input_image=input_image,
                                                  crop_indices=crop_indices,
                                                  isotropic=isotropic,
                                                  save_dict=save_dict)
    except Exception as e:
        err_msg = 'error in t_{}, c_{}, pos_{}, sl_{}'.format(
            time_idx, channel_idx, pos_idx, slice_idx
        )
        err_msg = err_msg + str(e)
        # TODO(Anitha) write to log instead
        print(err_msg)
        raise e

    return tile_meta_df
