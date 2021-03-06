"""Script for preprocessing stack"""
import argparse
import numpy as np
import os
import time
import warnings

from micro_dl.preprocessing.estimate_flat_field import FlatFieldEstimator2D
from micro_dl.preprocessing.generate_masks import MaskProcessor
from micro_dl.preprocessing.resize_images import ImageResizer
from micro_dl.preprocessing.tile_3d import ImageTilerUniform3D
from micro_dl.preprocessing.tile_uniform_images import ImageTilerUniform
from micro_dl.preprocessing.tile_nonuniform_images import \
    ImageTilerNonUniform
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.preprocess_utils as preprocess_utils


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='path to yaml configuration file',
    )
    args = parser.parse_args()
    return args


def flat_field_correct(params_dict, block_size):
    """Estimate flat_field_images

    :param dict params_dict: dict with keys: input_dir, output_dir, time_ids,
     channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :return str flat_field_dir: full path of dir with flat field correction
     images
    """

    flat_field_inst = FlatFieldEstimator2D(
        input_dir=params_dict['input_dir'],
        output_dir=params_dict['output_dir'],
        channel_ids=params_dict['channel_ids'],
        slice_ids=params_dict['slice_ids'],
        block_size=block_size,
    )
    flat_field_inst.estimate_flat_field()
    flat_field_dir = flat_field_inst.get_flat_field_dir()
    return flat_field_dir


def resize_images(params_dict,
                  scale_factor,
                  num_slices_subvolume,
                  resize_3d,
                  flat_field_dir):
    """Resample images first

    :param dict params_dict: dict with keys: input_dir, output_dir, time_ids,
     channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param int/list scale_factor: scale factor for each dimension
    :param int num_slices_subvolume: num of slices to be included in each
     volume. If -1, include all slices in slice_ids
    :param bool resize_3d: indicator for resize 2d or 3d
    :param str flat_field_dir: dir with flat field correction images. if None,
     does not correct for illumination changes
    :return:
     str resize_dir: dir with resized images
     int, list: slice_ids corrected for gaps due to 3d. For ex.
      slice_ids=[0,1,...8] and num_slices_subvolume=3, returned
      slice_ids=[0, 2, 4, 6]
    """

    if isinstance(scale_factor, list):
        scale_factor = np.array(scale_factor)

    if np.all(scale_factor == 1):
        return params_dict['input_dir'], params_dict['slice_ids']

    resize_inst = ImageResizer(
        input_dir=params_dict['input_dir'],
        output_dir=params_dict['output_dir'],
        scale_factor=scale_factor,
        channel_ids=params_dict['channel_ids'],
        time_ids=params_dict['time_ids'],
        slice_ids=params_dict['slice_ids'],
        pos_ids=params_dict['pos_ids'],
        int2str_len=params_dict['int2strlen'],
        num_workers=params_dict['num_workers'],
        flat_field_dir=flat_field_dir
    )

    if resize_3d:
        # return slice_ids from resize_volumes to deal with slice_ids=-1
        slice_ids = resize_inst.resize_volumes(num_slices_subvolume)
    else:
        resize_inst.resize_frames()
        slice_ids = params_dict['slice_ids']
    resize_dir = resize_inst.get_resize_dir()
    return resize_dir, slice_ids


def generate_masks(params_dict,
                   mask_from_channel,
                   flat_field_dir,
                   str_elem_radius,
                   mask_type,
                   mask_channel,
                   mask_ext,
                   mask_dir=None,
                   normalize_im=False):
    """Generate masks per image or volume

    :param dict params_dict: dict with keys: input_dir, output_dir, time_ids,
        channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param int/list mask_from_channel: generate masks from sum of these
        channels
    :param str/None flat_field_dir: dir with flat field correction images
    :param int str_elem_radius: structuring element size for morphological
        opening
    :param str mask_type: string to map to masking function. otsu or unimodal
        or borders_weight_loss_map
    :param int/None mask_channel: channel num assigned to mask channel. I
    :param str mask_ext: 'npy' or 'png'. Save the mask as uint8 PNG or
         NPY files
    :param str/None mask_dir: If creating weight maps from mask directory,
        specify mask dir
    :param bool normalize_im: indicator to normalize image based on z-score or not
    :return str mask_dir: Directory with created masks
    :return int mask_channel: Channel number assigned to masks
    """
    assert mask_type in {'otsu', 'unimodal', 'borders_weight_loss_map'},\
        "Supported mask types: 'otsu', 'unimodal', 'borders_weight_loss_map'" +\
        ", not {}".format(mask_type)

    # If generating weights map, input dir is the mask dir
    input_dir = params_dict['input_dir']
    if mask_dir is not None:
        input_dir = mask_dir
    # Instantiate channel to mask processor
    mask_processor_inst = MaskProcessor(
        input_dir=input_dir,
        output_dir=params_dict['output_dir'],
        channel_ids=mask_from_channel,
        flat_field_dir=flat_field_dir,
        time_ids=params_dict['time_ids'],
        slice_ids=params_dict['slice_ids'],
        pos_ids=params_dict['pos_ids'],
        int2str_len=params_dict['int2strlen'],
        uniform_struct=params_dict['uniform_struct'],
        num_workers=params_dict['num_workers'],
        mask_type=mask_type,
        mask_channel=mask_channel,
        mask_ext=mask_ext,
        normalize_im=normalize_im,
    )

    correct_flat_field = False
    if flat_field_dir is not None:
        correct_flat_field = True
    mask_processor_inst.generate_masks(
        correct_flat_field=correct_flat_field,
        str_elem_radius=str_elem_radius,
    )
    mask_dir = mask_processor_inst.get_mask_dir()
    mask_channel = mask_processor_inst.get_mask_channel()
    return mask_dir, mask_channel


def tile_images(params_dict,
                resize_flag,
                flat_field_dir,
                tile_dict):
    """
    Tile images.

    :param dict params_dict: dict with keys: input_dir, output_dir, time_ids,
     channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param dict tile_dict: dict with tiling related keys: tile_size, step_size,
     image_format, depths. optional: min_fraction, mask_channel, mask_dir,
     mask_depth, tile_3d
    :param bool resize_flag: indicator if resize related params in pp_config
     passed to pre_process()
    :param str/None flat_field_dir: dir with flat field correction images
    :return str tile_dir: dir with tiled images
    """
    # Check tile args
    tile_3d = False
    if 'tile_3d' in tile_dict:
        tile_3d = tile_dict['tile_3d']
    tile_dict['tile_3d'] = tile_3d
    hist_clip_limits = None
    if 'hist_clip_limits' in tile_dict:
        hist_clip_limits = tile_dict['hist_clip_limits']
    # setup tiling keyword arguments
    kwargs = {'input_dir': params_dict['input_dir'],
              'output_dir': params_dict['output_dir'],
              'normalize_channels': params_dict["normalize_channels"],
              'tile_size': tile_dict['tile_size'],
              'step_size': tile_dict['step_size'],
              'depths': tile_dict['depths'],
              'time_ids': params_dict['time_ids'],
              'channel_ids': params_dict['channel_ids'],
              'slice_ids': params_dict['slice_ids'],
              'pos_ids': params_dict['pos_ids'],
              'hist_clip_limits': hist_clip_limits,
              'flat_field_dir': flat_field_dir,
              'num_workers': params_dict['num_workers'],
              'int2str_len': params_dict['int2strlen'],
              'tile_3d': tile_3d}

    if params_dict['uniform_struct']:
        if tile_3d:
            if resize_flag:
                warnings.warn(
                    'If resize_3d was used, slice_idx corresponds to start'
                    'slice of each volume.If slice_ids=-1, the slice_ids'
                    'will be read from frames_meta.csv. Assuming slice_ids'
                    'provided here is fixed for these gaps.', Warning)
            tile_inst = ImageTilerUniform3D(**kwargs)
        else:
            tile_inst = ImageTilerUniform(**kwargs)
    else:
        # currently not supported but should be easy to extend
        tile_inst = ImageTilerNonUniform(**kwargs)

    tile_dir = tile_inst.get_tile_dir()

    # retain tiles with a minimum amount of foreground
    if 'mask_dir' in tile_dict:
        min_fraction = 0.
        if 'min_fraction' in tile_dict:
            min_fraction = tile_dict['min_fraction']
        mask_channel = tile_dict['mask_channel']
        mask_dir = tile_dict['mask_dir']

        mask_depth = 1
        if 'mask_depth' in tile_dict:
            mask_depth = tile_dict['mask_depth']

        tile_inst.tile_mask_stack(
            mask_dir=mask_dir,
            mask_channel=mask_channel,
            min_fraction=min_fraction,
            mask_depth=mask_depth,
        )
    else:
        # retain all tiles
        tile_inst.tile_stack()

    return tile_dir


def pre_process(pp_config, req_params_dict):
    """
    Preprocess data. Possible options are:

    correct_flat_field: Perform flatfield correction (2D only currently)
    resample: Resize 2D images (xy-plane) according to a scale factor,
        e.g. to match resolution in z. Resize 3d images
    create_masks: Generate binary masks from given input channels
    do_tiling: Split frames (stacked frames if generating 3D tiles) into
    smaller tiles with tile_size and step_size.

    This script will preprocess your dataset, save tiles and associated
    metadata. Then in the train_script, a dataframe for training data
    will be assembled based on the inputs and target you specify.

    :param dict pp_config: dict with key options:
    [input_dir, output_dir, slice_ids, time_ids, pos_ids
    correct_flat_field, use_masks, masks, tile_stack, tile]
    :param dict req_params_dict: dict with commom params for all tasks
    :raises AssertionError: If 'masks' in pp_config contains both channels
     and mask_dir (the former is for generating masks from a channel)
    """
    time_start = time.time()

    # -----------------Estimate flat field images--------------------
    flat_field_dir = None
    if 'flat_field' in pp_config:
        if 'estimate' in pp_config['flat_field'] and \
                pp_config['flat_field']['estimate']:
            assert 'flat_field_dir' not in pp_config['flat_field'], \
                'estimate_flat_field or use images in flat_field_dir.'
            block_size = None
            if 'block_size' in pp_config['flat_field']:
                block_size = pp_config['flat_field']['block_size']
            flat_field_dir = flat_field_correct(req_params_dict, block_size)
            pp_config['flat_field']['flat_field_dir'] = flat_field_dir

        elif 'correct' in pp_config['flat_field'] and \
                pp_config['flat_field']['correct']:
            flat_field_dir = pp_config['flat_field']['flat_field_dir']

    # -------------------------Resize images--------------------------
    if 'resize' in pp_config:
        scale_factor = pp_config['resize']['scale_factor']
        num_slices_subvolume = -1
        if 'num_slices_subvolume' in pp_config['resize']:
            num_slices_subvolume = \
                pp_config['resize']['num_slices_subvolume']

        resize_dir, slice_ids = resize_images(
            req_params_dict,
            scale_factor,
            num_slices_subvolume,
            pp_config['resize']['resize_3d'],
            flat_field_dir,
        )
        # the images are resized after flat field correction
        flat_field_dir = None
        pp_config['resize']['resize_dir'] = resize_dir
        req_params_dict['input_dir'] = resize_dir
        req_params_dict['slice_ids'] = slice_ids

    # ------------------------Generate masks-------------------------
    mask_dir = None
    mask_channel = None
    if 'masks' in pp_config:
        if 'channels' in pp_config['masks']:
            # Generate masks from channel
            assert 'mask_dir' not in pp_config['masks'], \
                "Don't specify a mask_dir if generating masks from channel"
            mask_from_channel = pp_config['masks']['channels']
            str_elem_radius = 5
            if 'str_elem_radius' in pp_config['masks']:
                str_elem_radius = pp_config['masks']['str_elem_radius']
            mask_type = 'otsu'
            if 'mask_type' in pp_config['masks']:
                mask_type = pp_config['masks']['mask_type']
            mask_ext = '.png'
            if 'mask_ext' in pp_config['masks']:
                mask_ext = pp_config['masks']['mask_ext']
            normalize_im = False
            if 'normalize_im' in pp_config['masks']:
                normalize_im = pp_config['masks']['normalize_im']
            mask_dir, mask_channel = generate_masks(
                params_dict=req_params_dict,
                mask_from_channel=mask_from_channel,
                flat_field_dir=flat_field_dir,
                str_elem_radius=str_elem_radius,
                mask_type=mask_type,
                mask_channel=mask_channel,
                mask_ext=mask_ext,
                normalize_im=normalize_im,
            )
        elif 'mask_dir' in pp_config['masks']:
            assert 'channels' not in pp_config['masks'], \
                "Don't specify channels to mask if using pre-generated masks"
            mask_dir = pp_config['masks']['mask_dir']
            # Get preexisting masks from directory and match to input dir
            mask_meta_fname = None
            if 'csv_name' in pp_config['masks']:
                mask_meta_fname = pp_config['masks']['csv_name']
            mask_channel = preprocess_utils.validate_mask_meta(
                mask_dir=mask_dir,
                input_dir=req_params_dict['input_dir'],
                csv_name=mask_meta_fname,
                mask_channel=mask_channel,
            )
        else:
            raise ValueError("If using masks, specify either mask_channel",
                             "or mask_dir.")
        pp_config['masks']['mask_dir'] = mask_dir
        pp_config['masks']['mask_channel'] = mask_channel

    # ----------------------Generate weight map-----------------------
    if 'make_weight_map' in pp_config and pp_config['make_weight_map']:
        # Must have mask dir and mask channel defined to generate weight map
        assert mask_dir is not None,\
            "Must have mask dir to generate weights"
        assert mask_channel is not None,\
            "Must have mask channel to generate weights"
        mask_type = 'borders_weight_loss_map'
        # Mask channel should be highest channel value in dataset at this point
        weights_channel = mask_channel + 1
        # Generate weights
        weights_dir, _ = generate_masks(
            params_dict=req_params_dict,
            mask_from_channel=mask_channel,
            flat_field_dir=None,
            str_elem_radius=5,
            mask_type=mask_type,
            mask_channel=weights_channel,
            mask_ext='.npy',
            mask_dir=mask_dir,
            normalize_im=False,
        )
        pp_config['weights'] = {
            'weights_dir': weights_dir,
            'weights_channel': weights_channel,
        }

    # ------------Tile images, targets, masks, weight maps------------
    if 'tile' in pp_config:
        resize_flag = False
        if 'resize' not in pp_config:
            resize_flag = True
        # Always tile masks if they exist
        if mask_dir is not None:
            if 'mask_dir' not in pp_config['tile']:
                pp_config['tile']['mask_dir'] = mask_dir
            if 'mask_channel' not in pp_config['tile']:
                pp_config['tile']['mask_channel'] = mask_channel
        tile_dir = tile_images(
            params_dict=req_params_dict,
            tile_dict=pp_config['tile'],
            resize_flag=resize_flag,
            flat_field_dir=flat_field_dir,
        )
        # Tile weight maps as well if they exist
        if 'weights' in pp_config:
            weight_params_dict = req_params_dict.copy()
            weight_params_dict["input_dir"] = weights_dir
            weight_params_dict["channel_ids"] = [weights_channel]
            weight_tile_config = pp_config['tile'].copy()
            weight_params_dict['normalize_channels'] = [False]
            # Weights depth should be the same as mask depth
            weight_tile_config['depths'] = 1
            weight_tile_config.pop('mask_dir')
            if 'mask_depth' in pp_config['tile']:
                weight_tile_config['depths'] = [pp_config['tile']['mask_depth']]
            tile_dir = tile_images(
                params_dict=weight_params_dict,
                tile_dict=weight_tile_config,
                resize_flag=resize_flag,
                flat_field_dir=None,
            )
        pp_config['tile']['tile_dir'] = tile_dir

    # Write in/out/mask/tile paths and config to json in output directory
    time_el = time.time() - time_start
    return pp_config, time_el


def save_config(cur_config, runtime):
    """Save the cur_config or append to existing config"""

    # Read preprocessing.json if exists in input dir
    parent_dir = cur_config['input_dir'].split(os.sep)[:-1]
    parent_dir = os.sep.join(parent_dir)

    prior_config_fname = os.path.join(parent_dir, 'preprocessing_info.json')
    prior_pp_config = None
    if os.path.exists(prior_config_fname):
        prior_pp_config = aux_utils.read_json(prior_config_fname)

    meta_path = os.path.join(cur_config['output_dir'],
                             'preprocessing_info.json')

    processing_info = [{'processing_time': runtime,
                        'config': cur_config}]
    if prior_pp_config is not None:
        prior_pp_config.append(processing_info[0])
        processing_info = prior_pp_config
    aux_utils.write_json(processing_info, meta_path)


if __name__ == '__main__':
    args = parse_args()

    pp_config = aux_utils.read_config(args.config)
    input_dir = pp_config['input_dir']
    output_dir = pp_config['output_dir']
    slice_ids = -1
    if 'slice_ids' in pp_config:
        slice_ids = pp_config['slice_ids']

    time_ids = -1
    if 'time_ids' in pp_config:
        time_ids = pp_config['time_ids']

    pos_ids = -1
    if 'pos_ids' in pp_config:
        pos_ids = pp_config['pos_ids']

    channel_ids = -1
    if 'channel_ids' in pp_config:
        channel_ids = pp_config['channel_ids']

    normalize_channels = -1
    if 'normalize_channels' in pp_config['tile']:
        normalize_channels = pp_config['tile']['normalize_channels']
        if isinstance(channel_ids, list):
            assert len(channel_ids) == len(normalize_channels),\
                "Nbr channels {} and normalization {} mismatch".format(
                    channel_ids,
                    normalize_channels,
                )

    uniform_struct = False
    if 'uniform_struct' in pp_config:
        uniform_struct = pp_config['uniform_struct']

    int2str_len = 3
    if 'int2str_len' in pp_config:
        int2str_len = pp_config['int2str_len']

    num_workers = 4
    if 'num_workers' in pp_config:
        num_workers = pp_config['num_workers']

    base_config = {'input_dir': input_dir,
                   'output_dir': output_dir,
                   'slice_ids': slice_ids,
                   'time_ids': time_ids,
                   'pos_ids': pos_ids,
                   'channel_ids': channel_ids,
                   'uniform_struct': uniform_struct,
                   'int2strlen': int2str_len,
                   'num_workers': num_workers,
                   'normalize_channels': normalize_channels}

    pp_config, runtime = pre_process(pp_config, base_config)
    save_config(pp_config, runtime)
