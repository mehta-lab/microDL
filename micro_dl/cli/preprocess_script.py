"""Script for preprocessing stack"""

import argparse
import json
import os
import yaml

from micro_dl.input.generate_masks import MaskProcessor
from micro_dl.input.tile_stack import ImageStackTiler
from micro_dl.utils.aux_utils import import_class


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


def read_config(config_fname):
    """Read the config file in yml format

    TODO: validate config!

    :param str config_fname: fname of config yaml with its full path
    :return: dict config: Configuration parameters
    """

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    return config


def pre_process(pp_config):
    """
    Preprocess data. Possible options are:
    split_volumes: Split .lif file into individual 2D frames
    correct_flat_field: Perform flatfield correction (2D only currently)
    use_masks: Generate binary masks from given input channels
    tile_stack: Split frames into smaller tiles with tile_size and step_size
    This script will preprocess your dataset, save tiles and associated
    metadata. Then in the train_script, a dataframe for training data
    will be assembled based on the inputs and target you specify.

    :param dict pp_config: dict with keys [input_fname, base_output_dir,
     split_volumes, crop_volumes]
    """
    input_dir = pp_config['input_dir']
    output_dir = pp_config['output_dir']

    # split images (if input is a lif file)
    if pp_config['split_volumes']:
        stack_splitter_cls = pp_config['splitter_class']
        stack_splitter_cls = import_class(
            'input.split_lif_stack',
            stack_splitter_cls,
        )
        stack_splitter = stack_splitter_cls(
            lif_fname=pp_config['input_dir'],
            base_output_dir=pp_config['output_dir'],
            verbose=pp_config['verbose']
        )
        stack_splitter.save_images()
        input_dir = os.path.join(
            pp_config['output_dir'],
            'split_images',
        )

    focal_plane_idx = -1
    if 'focal_plane_idx' in pp_config:
        focal_plane_idx = pp_config['focal_plane_idx']
    timepoint_ids = -1
    if 'timepoints' in pp_config:
        timepoint_ids = pp_config['timepoints']

    # estimate flat_field images
    correct_flat_field = True if pp_config['correct_flat_field'] else False
    flat_field_dir = None
    if correct_flat_field:
        # Create flat_field_dir as a subdirectory of output_dir
        flat_field_dir = os.path.join(output_dir, 'flat_field_images')
        os.makedirs(flat_field_dir, exist_ok=True)

        flat_field_estimator_cls = pp_config['flat_field_class']
        flat_field_estimator_cls = import_class(
            'input.estimate_flat_field',
            flat_field_estimator_cls,
        )
        flat_field_estimator = flat_field_estimator_cls(
            input_dir=input_dir,
            flat_field_dir=flat_field_dir,
        )
        flat_field_estimator.estimate_flat_field(focal_plane_idx)

    # generate masks
    mask_dir = None
    if pp_config['use_masks']:
        # Create mask_dir as a subdirectory of output_dir
        channel_ids = pp_config['masks']['mask_channels']
        mask_dir = os.path.join(
            output_dir,
            'mask_channels_' + '-'.join(map(str, channel_ids)),
        )
        os.makedirs(mask_dir, exist_ok=True)

        mask_processor_inst = MaskProcessor(
            input_dir=input_dir,
            output_dir=mask_dir,
            channel_ids=channel_ids,
            flat_field_dir=flat_field_dir,
            timepoint_ids=timepoint_ids,
            focal_plane_idx=focal_plane_idx,
        )
        str_elem_radius = 5
        if 'str_elem_radius' in pp_config['masks']:
            str_elem_radius = pp_config['masks']['str_elem_radius']

        mask_processor_inst.generate_masks(
            correct_flat_field=correct_flat_field,
            str_elem_radius=str_elem_radius,
        )

    # tile all frames, after flatfield correction if flatfields are generated
    tile_dir = None
    tile_mask_dir = None
    if pp_config['tile_stack']:
        tile_size = pp_config['tile']['tile_size']
        step_size = pp_config['tile']['step_size']
        str_tile_size = '-'.join([str(val) for val in tile_size])
        str_step_size = '-'.join([str(val) for val in step_size])
        tile_dir = 'tiles_{}_step_{}'.format(str_tile_size,
                                             str_step_size)
        tile_dir = os.path.join(output_dir, tile_dir)
        os.makedirs(tile_dir, exist_ok=True)
        isotropic = False
        if 'isotropic' in pp_config['tile']:
            isotropic = pp_config['tile']['isotropic']
        channel_ids = -1
        if "channels" in pp_config['tile']:
            channel_ids = pp_config['tile']['channels']
        hist_clip_limits = None
        if 'hist_clip_limits' in pp_config['tile']:
            hist_clip_limits = pp_config['tile']['hist_clip_limits']

        tile_inst = ImageStackTiler(
            input_dir=input_dir,
            output_dir=tile_dir,
            tile_size=tile_size,
            step_size=step_size,
            timepoint_ids=timepoint_ids,
            channel_ids=channel_ids,
            focal_plane_idx=focal_plane_idx,
            hist_clip_limits=hist_clip_limits,
            flat_field_dir=flat_field_dir,
            isotropic=isotropic,
        )
        # If you're using min fraction, it assumes you've generated masks
        # and want to tile only the ones with a minimum amount of foreground
        if 'min_fraction' in pp_config['tile'] and pp_config['use_masks']:
            if pp_config['tile']['save_cropped_masks']:
                tile_mask_dir = 'mask_tiles_{}_step_{}'.format(str_tile_size,
                                                               str_step_size)
                tile_mask_dir = os.path.join(output_dir, tile_mask_dir)
                os.makedirs(tile_mask_dir, exist_ok=True)
            tile_inst.tile_mask_stack(
                min_fraction=pp_config['tile']['min_fraction'],
                mask_dir=mask_dir,
                tile_mask_dir=tile_mask_dir,
            )
        else:
            tile_inst.tile_stack()

    # Write in/out/mask/tile paths and config to json in output directory
    processing_info = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "flat_field_dir": flat_field_dir,
        "mask_dir": mask_dir,
        "tile_dir": tile_dir,
        "tile_mask_dir": tile_mask_dir,
        "config": pp_config,
    }
    json_dump = json.dumps(processing_info)
    meta_path = os.path.join(output_dir, "preprocessing_info.json")
    with open(meta_path, "w") as write_file:
        write_file.write(json_dump)


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    pre_process(config)
