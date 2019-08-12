#!/usr/bin/env/python
"""Model inference on larger images with and w/o stitching"""
import argparse
import glob
import natsort
import os
import yaml

from micro_dl.inference import image_inference as image_inf
import micro_dl.utils.train_utils as train_utils


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=None,
                        help=('Optional: specify the gpu to use: 0,1,...',
                              ', -1 for debugging. Default: pick best GPU'))
    parser.add_argument('--gpu_mem_frac', type=float, default=None,
                        help='Optional: specify gpu memory fraction to use')
    parser.add_argument(
        '--config',
        type=str,
        help='path to inference yaml configuration file',
    )

    args = parser.parse_args()
    return args


def run_inference(config_fname,
                   gpu_ids,
                   gpu_mem_frac):
    """

    :param str config_fname: Full path to config yaml file
    :param int gpu_ids: gpu id to use
    :param float gpu_mem_frac: gpu memory fraction to use
    """

    with open(config_fname, 'r') as f:
        inf_config = yaml.safe_load(f)
    # Load train config from model dir
    train_config_fname = glob.glob(
        os.path.join(inf_config['model_dir'], '*.yml')
    )
    assert len(train_config_fname) == 1, \
        'more than one train config yaml found in model dir'
    with open(train_config_fname[0], 'r') as f:
        train_config = yaml.safe_load(f)
    # Use model_dir from inference config if present, otherwise use train
    if 'model_dir' in inf_config:
        model_dir = inf_config['model_dir']
    else:
        model_dir = train_config['trainer']['model_dir']
    if 'model_fname' in inf_config:
        model_fname = inf_config['model_fname']
    else:
        # If model filename not listed, grab latest one
        fnames = [f for f in os.listdir(inf_config['model_dir'])
                  if f.endswith('.hdf5')]
        assert len(fnames) > 0, 'No weight files found in model dir'
        fnames = natsort.natsorted(fnames)
        model_fname = fnames[-1]

    # Set defaults
    vol_inf_dict = None
    if 'vol_inf_dict' in inf_config:
        vol_inf_dict = inf_config['vol_inf_dict']
    mask_params_dict = None
    if 'mask_params_dict' in inf_config:
        mask_params_dict = inf_config['mask_params_dict']
    data_split = 'test'
    if 'data_split' in inf_config:
        data_split = inf_config['data_split']

    inference_inst = image_inf.ImagePredictor(
        train_config=train_config,
        model_dir=model_dir,
        model_fname=model_fname,
        image_dir=inf_config['image_dir'],
        image_param_dict=inf_config['image_params_dict'],
        gpu_id=gpu_ids,
        gpu_mem_frac=gpu_mem_frac,
        data_split=data_split,
        metrics_list=inf_config['metrics'],
        metrics_orientations=inf_config['metrics_orientations'],
        mask_param_dict=mask_params_dict,
        vol_inf_dict=vol_inf_dict,
    )
    inference_inst.run_prediction()


if __name__ == '__main__':
    args = parse_args()
    # Get GPU ID and memory fraction
    gpu_id, gpu_mem_frac = train_utils.select_gpu(
        args.gpu,
        args.gpu_mem_frac,
    )
    run_inference(
        config_fname=args.config,
        gpu_ids=gpu_id,
        gpu_mem_frac=gpu_mem_frac,
    )
