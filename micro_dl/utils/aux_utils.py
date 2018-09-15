"""Auxiliary utility functions"""
import inspect
import importlib
import logging
import numpy as np
import os
import pandas as pd


def import_class(module_name, cls_name):
    """Imports a class specified in yaml dynamically

    REFACTOR THIS!!

    :param str module_name: modules such as input, utils, train etc
    :param str cls_name: class to find
    """

    full_module_name = ".".join(('micro_dl', module_name))
    try:
        module = importlib.import_module(full_module_name)
        obj = getattr(module, cls_name)

        if inspect.isclass(obj):
            return obj
    except ImportError:
        raise


def get_row_idx(frames_metadata, time_idx,
                channel_idx, focal_plane_idx=None):
    """Get the indices for images with timepoint_idx and channel_idx

    :param pd.DataFrame frames_metadata: DF with columns time_idx,
     channel_idx, slice_idx, file_name]
    :param int time_idx: get info for this timepoint
    :param int channel_idx: get info for this channel
    :param int focal_plane_idx: get info for this focal plane (2D)
    """
    if focal_plane_idx is not None:
        row_idx = ((frames_metadata['time_idx'] == time_idx) &
                   (frames_metadata['channel_idx'] == channel_idx) &
                   (frames_metadata['slice_idx'] == focal_plane_idx))
    else:
        row_idx = ((frames_metadata['time_idx'] == time_idx) &
                   (frames_metadata['channel_idx'] == channel_idx))
    return row_idx


def get_meta_idx(metadata_df,
                 time_idx,
                 channel_idx,
                 slice_idx,
                 pos_idx):
    """
    Get row index in metadata dataframe given variable indices

    :param dataframe metadata_df: Dataframe with column names given below
    :param int time_idx: Timepoint index
    :param int channel_idx: Channel index
    :param int slice_idx: Slize (z) index
    :param int pos_idx: Position (FOV) index
    :return: int pos_idx: Row position matching indices above
    """
    frame_idx = metadata_df.index[
        (metadata_df['channel_idx'] == channel_idx) &
        (metadata_df['time_idx'] == time_idx) &
        (metadata_df["slice_idx"] == slice_idx) &
        (metadata_df["pos_idx"] == pos_idx)].tolist()
    return frame_idx[0]


def get_im_name(time_idx=None,
                channel_idx=None,
                slice_idx=None,
                pos_idx=None,
                extra_field=None,
                int2str_len=3):
    im_name = "im"
    if channel_idx is not None:
        im_name += "_c" + str(channel_idx).zfill(int2str_len)
    if slice_idx is not None:
        im_name += "_z" + str(slice_idx).zfill(int2str_len)
    if time_idx is not None:
        im_name += "_t" + str(time_idx).zfill(int2str_len)
    if pos_idx is not None:
        im_name += "_p" + str(pos_idx).zfill(int2str_len)
    if extra_field is not None:
        im_name += "_" + extra_field
    im_name += ".npy"
    return im_name


def validate_metadata_indices(frames_metadata,
                              time_ids=None,
                              channel_ids=None,
                              pos_ids=None):
    """
    Check the availability of provided timepoints and channels

    :param pd.DataFrame frames_metadata: DF with columns time_idx,
     channel_idx, slice_idx, pos_idx, file_name]
    :param int/list time_ids: check availability of these timepoints in
     frames_metadata
    :param int/list channel_ids: check availability of these channels in
     frames_metadata
    :param int/list pos_ids: Check availability of positions in metadata
    :param dict metadata_ids: All time and channel indices
    :raise AssertionError: If not all channels, timepoints or positions
        are present
    """

    metadata_ids = {}
    if time_ids is not None:
        if np.issubdtype(type(time_ids), np.integer):
            if time_ids == -1:
                time_ids = frames_metadata['time_idx'].unique()
            else:
                time_ids = [time_ids]
        all_tps = frames_metadata['time_idx'].unique()
        tp_indicator = [tp in all_tps for tp in time_ids]
        assert np.all(tp_indicator), 'time not available'
        metadata_ids['timepoints'] = time_ids

    if channel_ids is not None:
        if np.issubdtype(type(channel_ids), np.integer):
            if channel_ids == -1:
                channel_ids = frames_metadata['channel_idx'].unique()
            else:
                channel_ids = [channel_ids]
        all_channels = frames_metadata['channel_idx'].unique()
        channel_indicator = [c in all_channels for c in channel_ids]
        assert np.all(channel_indicator), 'channel not available'
        metadata_ids['channels'] = channel_ids

    if pos_ids is not None:
        if np.issubdtype(type(pos_ids), np.integer):
            if pos_ids == -1:
                pos_ids = frames_metadata['pos_idx'].unique()
            else:
                pos_ids = [pos_ids]
        all_pos = frames_metadata['pos_idx'].unique()
        pos_indicator = [c in all_pos for c in pos_ids]
        assert np.all(pos_indicator), 'position not available'
        metadata_ids['positions'] = pos_ids

    return metadata_ids


def init_logger(logger_name, log_fname, log_level):
    """Creates a logger instance

    :param str logger_name: name of the logger instance
    :param str log_fname: fname with full path of the log file
    :param int log_level: specifies the logging level: NOTSET:0, DEBUG:10,
    INFO:20, WARNING:30, ERROR:40, CRITICAL:50
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_fname)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    return logger


def save_tile_meta(tiles_meta,
                   cur_channel,
                   tiled_dir):
    """
    Save meta data for tiled images

    :param list tiles_meta: List of tuples holding meta info for tiled
        images
    :param int cur_channel: Channel being tiled
    :param str tiled_dir: Directory to save meta data in
    """

    fname_header = 'fname_{}'.format(cur_channel)
    cur_df = pd.DataFrame.from_records(
        tiles_meta,
        columns=['time_idx', 'channel_idx', 'pos_idx',
                 'slice_idx', fname_header]
    )
    metadata_fname = os.path.join(tiled_dir, 'tiles_meta.csv')
    if cur_channel == 0:
        df = cur_df
    else:
        df = pd.read_csv(metadata_fname, sep=',', index_col=0)
        df[fname_header] = cur_df[fname_header]
    df.to_csv(metadata_fname, sep=',')


def validate_config(config_dict, params):
    """Check if the required params are present in config

    :param dict config_dict: dictionary with params as keys
    :param list params: list of strings with expected params
    :return: list with bool values indicating if param is present or not
    """

    params = np.array(params)
    param_indicator = np.zeros(len(params), dtype='bool')
    for idx, exp_param in enumerate(params):
        cur_indicator = (exp_param in config_dict) and \
                        (config_dict[exp_param] is not None)
        param_indicator[idx] = cur_indicator
    check = np.all(param_indicator)
    msg = 'Params absent in network_config: {}'.\
        format(params[param_indicator == 0])
    return check, msg


def get_channel_axis(data_format):
    """Get the channel axis given the data format

    :param str data_format: as named. [channels_last, channel_first]
    :return int channel_axis
    """

    assert data_format in ['channels_first', 'channels_last'], \
        'Invalid data format %s' % data_format
    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    return channel_axis
