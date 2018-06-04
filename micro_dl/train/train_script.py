#!/usr/bin/env python
"""Train neural network models in keras"""
import argparse
import os
import pandas as pd
import yaml

from micro_dl.input.crop_stack import ImageStackCropper
from micro_dl.input.dataset import BaseDataSet, DataSetWithMask
from micro_dl.input.gen_crop_masks import MaskProcessor
from micro_dl.input.training_table import (
    BaseTrainingTable, TrainingTableWithMask
)
from micro_dl.train.trainer import BaseKerasTrainer
from micro_dl.utils.aux_utils import import_class
from micro_dl.utils.train_utils import check_gpu_availability


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify the gpu to use: 0,1,...')
    parser.add_argument('--gpu_mem_frac', type=float, default=1,
                        help='specify the gpu memory fraction to use')
    parser.add_argument('--action', type=str, default='train',
                        choices=('train', 'tune_hyperparam'),
                        help=('action to take on the model: train,'
                              'tune_hyperparam'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str,
                       help='path to yaml configuration file')
    group.add_argument('--model', type=str, default=None,
                       help='path to checkpoint file/directory')
    parser.add_argument('--port', type=int, default=-1,
                        help='port to use for the tensorboard callback')
    args = parser.parse_args()
    return args


def read_config(config_fname):
    """Read the config file in yml format

    TODO: validate config!

    :param str config_fname: fname of config yaml with its full path
    :return:
    """

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    return config


def pre_process(config):
    """Split and crop volumes from lif data

    :param dict config: dict with keys [input_fname, base_output_dir,
     split_volumes, crop_volumes]
    """

    if config['dataset']['split_volumes']:
        stack_splitter_cls = config['dataset']['splitter_class']
        stack_splitter_cls = import_class('input', stack_splitter_cls)
        stack_splitter = stack_splitter_cls(
            lif_fname=config['dataset']['input_fname'],
            base_output_dir=config['dataset']['base_output_dir'],
            verbose=config['verbose']
        )
        stack_splitter.save_images()

    split_dir = os.path.join(config['dataset']['base_output_dir'],
                             'split_images')

    if config['dataset']['correct_flat_field']:
        correct_flat_field = True
    else:
        correct_flat_field = False

    if 'focal_plane_idx' in config['dataset']:
        focal_plane_idx = config['dataset']['focal_plane_idx']
    else:
        focal_plane_idx = None

    if correct_flat_field:
        flat_field_estimator_cls = config['dataset']['flat_field_class']
        flat_field_estimator_cls = import_class('input',
                                                flat_field_estimator_cls)
        flat_field_estimator = flat_field_estimator_cls(split_dir)
        flat_field_estimator.estimate_flat_field(focal_plane_idx)

    if 'mask_channels' in config['dataset']:
        if 'timepoints' in config['dataset']:
            mask_processor_inst = MaskProcessor(
                split_dir, config['dataset']['mask_channels'],
                config['dataset']['timepoints']
            )
        else:
            mask_processor_inst = MaskProcessor(
                split_dir, config['dataset']['mask_channels']
            )

        mask_processor_inst.generate_masks(
            focal_plane_idx=focal_plane_idx,
            correct_flat_field=correct_flat_field
        )

    if config['dataset']['crop_volumes']:
        if 'isotropic' in config['dataset']['cropper']:
            isotropic = ['dataset']['cropper']['isotropic']
        else:
            isotropic = False

        cropper_inst = ImageStackCropper(
            config['dataset']['base_output_dir'],
            config['dataset']['cropper']['tile_size'],
            config['dataset']['cropper']['step_size'],
            correct_flat_field=correct_flat_field, isotropic=isotropic
        )
        if 'min_fraction' in config['dataset']:
            cropper_inst.crop_stack_by_indices(
                mask_channels=config['dataset']['mask_channels'],
                min_fraction=config['dataset']['min_fraction'],
                save_cropped_masks=config['dataset']['save_cropped_masks'],
                isotropic=isotropic, focal_plane_idx=focal_plane_idx
            )
        else:
            cropper_inst.crop_stack(focal_plane_idx=focal_plane_idx)


def train_xy(df_meta, config):
    """Train using fit_generator"""

    tt = BaseTrainingTable(df_meta, config['dataset']['input_channels'],
                           config['dataset']['target_channels'],
                           config['dataset']['split_by_column'],
                           config['dataset']['split_ratio'])
    if 'val' in config['dataset']['split_ratio']:
        df_train, df_val, df_test = tt.train_test_split()
        val_ds_params = {}
        if 'augmentations' in config['trainer']:
            val_ds_params['augmentations'] = (
                config['trainer']['augmentations']
            )

        ds_val = BaseDataSet(input_fnames=df_val['fpaths_input'],
                             target_fnames=df_val['fpaths_target'],
                             batch_size=config['trainer']['batch_size'],
                             **val_ds_params)
        train_ds_params = val_ds_params.copy()
    else:
        df_train, df_test = tt.train_test_split()
        ds_val = None
        if 'augmentations' in config['trainer']:
            train_ds_params['augmentations'] = (
                config['trainer']['augmentations']
            )
    ds_train = BaseDataSet(input_fnames=df_train['fpaths_input'],
                           target_fnames=df_train['fpaths_target'],
                           batch_size=config['trainer']['batch_size'],
                           **train_ds_params)
    ds_test = BaseDataSet(input_fnames=df_train['fpaths_input'],
                          target_fnames=df_train['fpaths_target'],
                          batch_size=config['trainer']['batch_size'])
    return ds_train, ds_val, ds_test


def train_xyweights(df_meta, config):
    """Train using fit_generator"""

    tt = TrainingTableWithMask(df_meta, config['dataset']['input_channels'],
                               config['dataset']['target_channels'],
                               config['dataset']['mask_channels'],
                               config['dataset']['split_by_column'],
                               config['dataset']['split_ratio'])
    if 'val' in config['dataset']['split_ratio']:
        df_train, df_val, df_test = tt.train_test_split()
        val_ds_params = {}
        if 'augmentations' in config['trainer']:
            val_ds_params['augmentations'] = (
                config['trainer']['augmentations']
            )

        ds_val = DataSetWithMask(input_fnames=df_val['fpaths_input'],
                                 target_fnames=df_val['fpaths_target'],
                                 mask_fnames=df_val['fpaths_mask'],
                                 batch_size=config['trainer']['batch_size'],
                                 **val_ds_params)
        train_ds_params = val_ds_params.copy()
    else:
        df_train, df_test = tt.train_test_split()
        ds_val = None
        if 'augmentations' in config['trainer']:
            train_ds_params['augmentations'] = (
                config['trainer']['augmentations']
            )
    ds_train = DataSetWithMask(input_fnames=df_train['fpaths_input'],
                               target_fnames=df_train['fpaths_target'],
                               mask_fnames=df_train['fpaths_mask'],
                               batch_size=config['trainer']['batch_size'],
                               **train_ds_params)
    ds_test = DataSetWithMask(input_fnames=df_train['fpaths_input'],
                              target_fnames=df_train['fpaths_target'],
                              mask_fnames=df_train['fpaths_mask'],
                              batch_size=config['trainer']['batch_size'])
    return ds_train, ds_val, ds_test



def run_action(args):
    """Performs training or tune hyper parameters

    :param Namespace args: namespace containing the arguments passed
    """

    action = args.action
    config = read_config(args.config)
    if action == 'train':
        if config['dataset']['preprocess']:
            pre_process(config)

        df_meta_fname = os.path.join(config['dataset']['data_dir'],
                                     'cropped_images_info.csv')
        df_meta = pd.read_csv(df_meta_fname)
        if 'weighted_loss' in config['trainer']:
            ds_train, ds_val, ds_test = train_xyweights(df_meta, config)
        else:
            ds_train, ds_val, ds_test = train_xy(df_meta, config)

        if 'model_name' in config['trainer']:
            model_name = config['trainer']['model_name']
        else:
            model_name = None

        trainer = BaseKerasTrainer(config=config,
                                   model_dir=config['trainer']['model_dir'],
                                   train_dataset=ds_train, val_dataset=ds_val,
                                   model_name=model_name, gpu_ids=args.gpu,
                                   gpu_mem_frac=args.gpu_mem_frac)
        trainer.train()
    elif action == 'tune_hyperparam':
        raise NotImplementedError
    else:
        raise TypeError(('action {} not permitted. options: train or '
                        'tune_hyperparam').format(action))


if __name__ == '__main__':
    args = parse_args()
    gpu_availability = check_gpu_availability(args.gpu,
                                              args.gpu_mem_frac)
    if not isinstance(args.gpu, int):
        raise NotImplementedError
    run_action(args)




