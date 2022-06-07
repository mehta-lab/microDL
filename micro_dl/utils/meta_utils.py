import glob
import itertools
<<<<<<< HEAD
=======
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.mp_utils as mp_utils
import numpy as np
>>>>>>> b0eb02d... sketching zarr support
import os
import pandas as pd
import pickle
import sys

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as im_utils
<<<<<<< HEAD
import micro_dl.utils.io_utils as io_utils
=======
>>>>>>> 2a79c8c... updated flatfield for zarr
import micro_dl.utils.mp_utils as mp_utils


def frames_meta_generator(
        input_dir,
        zarr_file=None,
        order='cztp',
        name_parser='parse_sms_name',
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
<<<<<<< HEAD
=======
    :param str/None zarr_file: Zarr file name in case of zarr input data format.
        None if using tiff/png/etc image files.
>>>>>>> b0eb02d... sketching zarr support
    :param str order: Order in which file name encodes cztp (for tiff/png)
    :param str name_parser: Function in aux_utils for parsing indices from tiff/png file name
    :return pd.DataFrame frames_meta: Metadata for all frames in dataset
    """
<<<<<<< HEAD
    zarr_files = glob.glob(os.path.join(input_dir, '*.zarr'))
    if len(zarr_files) == 0:
=======
    if zarr_file is None:
>>>>>>> b0eb02d... sketching zarr support
        frames_meta = frames_meta_from_filenames(
            input_dir,
            name_parser,
            order,
        )
    else:
        # Generate frames_meta from zarr metadata
<<<<<<< HEAD
        frames_meta = frames_meta_from_zarr(input_dir, zarr_files)
=======
        frames_meta = frames_meta_from_zarr(input_dir, zarr_file)
>>>>>>> b0eb02d... sketching zarr support

    # Write metadata
    frames_meta_filename = os.path.join(input_dir, 'frames_meta.csv')
    frames_meta.to_csv(frames_meta_filename, sep=",")
    return frames_meta


def frames_meta_from_filenames(input_dir, name_parser, order):
    """
    :param str input_dir:   path to input directory containing images
    :param str name_parser: Function in aux_utils for parsing indices from file name
    :return pd.DataFrame frames_meta: Metadata for all frames in dataset
    :param str order: Order in which file name encodes cztp (for tiff/png)
    """
    parse_func = aux_utils.import_object('utils.aux_utils', name_parser, 'function')
    im_names = aux_utils.get_sorted_names(input_dir)
    frames_meta = aux_utils.make_dataframe(nbr_rows=len(im_names))
    channel_names = []
    # Fill dataframe with rows from image names
    for i in range(len(im_names)):
        kwargs = {"im_name": im_names[i], "dir_name": input_dir}
        if name_parser == 'parse_idx_from_name':
            kwargs["order"] = order
        elif name_parser == 'parse_sms_name':
            kwargs["channel_names"] = channel_names
        meta_row = parse_func(**kwargs)
        frames_meta.loc[i] = meta_row
    return frames_meta


<<<<<<< HEAD
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
    meta_row['dir_name'] = input_dir
    idx = 0
    for pos_idx in range(len(file_names)):
        zarr_reader = io_utils.ZarrReader(file_names[pos_idx])
        meta_row['file_name'] = os.path.basename(file_names[pos_idx])
        # Get position index from name
        meta_row['pos_idx'] = int(zarr_reader.columns[0].split('_')[-1])
        for channel_idx in range(nbr_channels):
            for slice_idx in range(nbr_slices):
                for time_idx in range(nbr_times):
                    meta_row['channel_idx'] = channel_idx
                    meta_row['slice_idx'] = slice_idx
                    meta_row['time_idx'] = time_idx
                    meta_row['channel_name'] = zarr_object.get_channel_names[channel_idx]
                    frames_meta.loc[idx] = meta_row
                    idx += 1
=======
def frames_meta_from_zarr(input_dir, zarr_name):
    """
    Reads ome-zarr file and creates frames_meta based on metadata and
    array information.

    :param str input_dir: Input directory containing zarr file
    :param str zarr_name: Name of zarr file including extension
    :return pd.DataFrame frames_meta: Metadata for all frames in zarr
    """
    zarr_data = zarr.open(os.path.join(input_dir, zarr_name), mode='r')
    plate_info = zarr_data.attrs.get('plate')
    # TODO: Create a zarr object that can be passed around preprocessing?

    well_pos = []
    # Assumes that the positions are indexed in the order of Row-->Well-->FOV
    for well in plate_info['wells']:
        for pos in zarr_data[well['path']].attrs.get('well').get('images'):
            well_pos.append(
                {'well': well['path'], 'pos': pos['path']}
            )

    # Get channel names
    omero_meta = zarr_data[well_pos[0]['well']][well_pos[0]['pos']].attrs.asdict()['omero']
    channel_names = []
    for chan in omero_meta['channels']:
        channel_names.append(chan['label'])

    array_name = list(zarr_data[well_pos[0]['well']][well_pos[0]['pos']].array_keys())[0]
    array_shape = zarr_data[well_pos[0]['well']][well_pos[0]['pos']][array_name].shape

    nbr_pos = len(well_pos)
    nbr_times = array_shape[0]
    nbr_channels = array_shape[1]
    nbr_slices = array_shape[2]

    # If there isn't a channel name for each channel, set to nan
    if len(channel_names) != nbr_channels:
        channel_names = nbr_channels * [np.nan]

    nbr_rows = nbr_channels * nbr_pos * nbr_slices * nbr_times
    frames_meta = aux_utils.make_dataframe(nbr_rows=nbr_rows)
    meta_row = dict.fromkeys(list(frames_meta))
    meta_row['dir_name'] = input_dir
    meta_row['file_name'] = zarr_name
    idx = 0
    for channel_idx in range(nbr_channels):
        for pos_idx in range(nbr_pos):
            for slice_idx in range(nbr_slices):
                for time_idx in range(nbr_times):
                    meta_row['channel_idx'] = channel_idx
                    meta_row['pos_idx'] = pos_idx
                    meta_row['slice_idx'] = slice_idx
                    meta_row['time_idx'] = time_idx
                    meta_row['channel_name'] = channel_names[channel_idx]
                    frames_meta.loc[idx] = meta_row
                    idx += 1

>>>>>>> b0eb02d... sketching zarr support
    return frames_meta


def ints_meta_generator(
        input_dir,
        num_workers=4,
        block_size=256,
        flat_field_dir=None,
<<<<<<< HEAD
        channel_ids=-1):
=======
        channel_ids=-1,
        zarr_object=None,
        ):
>>>>>>> 2a79c8c... updated flatfield for zarr
    """
    Generate pixel intensity metadata for estimating image normalization
    parameters during preprocessing step. Pixels are sub-sampled from the image
    following a grid pattern defined by block_size to for efficient estimation of
    median and interquatile range. Grid sampling is preferred over random sampling
    in the case due to the spatial correlation in images.
    Will write found data in ints_meta.csv in input directory.
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
    :param int num_workers: number of workers for multiprocessing
    :param int block_size: block size for the grid sampling pattern. Default value works
        well for 2048 X 2048 images.
    :param str flat_field_dir: Directory containing flatfield images
    :param list/int channel_ids: Channel indices to process
    """
    if block_size is None:
        block_size = 256
    frames_metadata = aux_utils.read_meta(input_dir)
    if not isinstance(channel_ids, list):
        # Use all channels
        channel_ids = frames_metadata['channel_idx'].unique()
    # Pickle zarr object if passing it to multiprocessing
    zarr_pickle = None
    if zarr_object is not None:
        zarr_pickle = pickle.dumps(zarr_object)
    mp_fn_args = []
    # Fill dataframe with rows from image names
    for i, meta_row in frames_metadata.iterrows():
        channel_idx = meta_row['channel_idx']
        ff_path = im_utils.get_flat_field_path(
            flat_field_dir,
            channel_idx,
            channel_ids,
        )
<<<<<<< HEAD
        mp_fn_args.append((meta_row, ff_path, block_size))
=======
        mp_fn_args.append((meta_row, ff_path, block_size, zarr_pickle))
>>>>>>> 2a79c8c... updated flatfield for zarr

    im_ints_list = mp_utils.mp_sample_im_pixels(mp_fn_args, num_workers)
    im_ints_list = list(itertools.chain.from_iterable(im_ints_list))
    ints_meta = pd.DataFrame.from_dict(im_ints_list)

    ints_meta_filename = os.path.join(input_dir, 'intensity_meta.csv')
    ints_meta.to_csv(ints_meta_filename, sep=",")


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
        meta_row['dir_name'] = input_dir
        im_path = os.path.join(input_dir, meta_row['file_name'])
        mp_fn_args.append((im_path, meta_row))

    meta_row_list = mp_utils.mp_wrapper(
        mp_utils.get_mask_meta_row,
        mp_fn_args,
        num_workers,
    )
    mask_meta = pd.DataFrame.from_dict(meta_row_list)

    mask_meta_filename = os.path.join(input_dir, 'mask_meta.csv')
    mask_meta.to_csv(mask_meta_filename, sep=",")
    return mask_meta


def compute_zscore_params(frames_meta,
                          ints_meta,
                          input_dir,
                          normalize_im,
                          min_fraction=0.99):
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

    assert normalize_im in [None, 'slice', 'volume', 'dataset'], \
        'normalize_im must be None or "slice" or "volume" or "dataset"'

    if normalize_im is None:
        # No normalization
        frames_meta['zscore_median'] = 0
        frames_meta['zscore_iqr'] = 1
        return frames_meta
    elif normalize_im == 'dataset':
        agg_cols = ['time_idx', 'channel_idx', 'dir_name']
    elif normalize_im == 'volume':
        agg_cols = ['time_idx', 'channel_idx', 'dir_name', 'pos_idx']
    else:
        agg_cols = ['time_idx', 'channel_idx', 'dir_name', 'pos_idx', 'slice_idx']
    # median and inter-quartile range are more robust than mean and std
    ints_meta_sub = ints_meta[ints_meta['fg_frac'] >= min_fraction]
    ints_agg_median = \
        ints_meta_sub[agg_cols + ['intensity']].groupby(agg_cols).median()
    ints_agg_hq = \
        ints_meta_sub[agg_cols + ['intensity']].groupby(agg_cols).quantile(0.75)
    ints_agg_lq = \
        ints_meta_sub[agg_cols + ['intensity']].groupby(agg_cols).quantile(0.25)
    ints_agg = ints_agg_median
    ints_agg.columns = ['zscore_median']
    ints_agg['zscore_iqr'] = ints_agg_hq['intensity'] - ints_agg_lq['intensity']
    ints_agg.reset_index(inplace=True)

    cols_to_merge = frames_meta.columns[[
            col not in ['zscore_median', 'zscore_iqr']
            for col in frames_meta.columns]]
    frames_meta = pd.merge(
        frames_meta[cols_to_merge],
        ints_agg,
        how='left',
        on=agg_cols,
    )
    if frames_meta['zscore_median'].isnull().values.any():
        raise ValueError('Found NaN in normalization parameters. \
        min_fraction might be too low or images might be corrupted.')
    frames_meta_filename = os.path.join(input_dir, 'frames_meta.csv')
    frames_meta.to_csv(frames_meta_filename, sep=",")

    cols_to_merge = ints_meta.columns[[
            col not in ['zscore_median', 'zscore_iqr']
            for col in ints_meta.columns]]
    ints_meta = pd.merge(
        ints_meta[cols_to_merge],
        ints_agg,
        how='left',
        on=agg_cols,
    )
    ints_meta['intensity_norm'] = \
        (ints_meta['intensity'] - ints_meta['zscore_median']) / \
        (ints_meta['zscore_iqr'] + sys.float_info.epsilon)

    return frames_meta, ints_meta



