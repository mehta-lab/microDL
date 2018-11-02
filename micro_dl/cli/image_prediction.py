#!/usr/bin/env/python
"""Model inference"""
import argparse
import natsort
import numpy as np
import os
import pandas as pd
import yaml

import micro_dl.train.model_inference as inference
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.train_utils import check_gpu_availability

def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='specify the gpu to use: 0,1,... (-1 for debugging)',
    )
    parser.add_argument(
        '--gpu_mem_frac',
        type=float,
        default=1.,
        help='specify the gpu memory fraction to use',
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Directory containing model weights, config and csv files',
    )
    parser.add_argument(
        '--model_fname',
        type=str,
        default=None,
        help='File name of weights in model dir (.hdf5). If None grab newest.',
    )
    parser.add_argument(
        '--test_data',
        type=bool,
        default=True,
        help='True if use test data indices in split_samples.',
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing images',
    )
    args = parser.parse_args()
    return args


def run_prediction(args):
    """
    Predict images given model + weights.
    If the test_data flag is set to True, the test indices in
    split_samples.json file in model directory will be predicted
    Otherwise, all images in image directory will be predicted.
    """
    # Load config file
    config_name = os.path.join(args.model_dir, 'config.yml')
    with open(config_name, 'r') as f:
        config = yaml.load(f)
    # Load frames metadata and determine indices
    frames_meta = pd.read_csv(os.path.join(args.image_dir, 'frames_meta.csv'))
    idx_name = config['dataset']['split_by_column']
    if args.test_data:
        idx_fname = os.path.join(args.model_dir, 'split_samples.json')
        try:
            split_samples = aux_utils.read_json(idx_fname)
            test_indices = split_samples['test']
        except FileNotFoundError as e:
            print("No split_samples file. Will predict all images in dir.")
            test_indices = np.unique(frames_meta[idx_name])

    # Get model weight file name
    model_fname = args.model_fname
    if model_fname is None:
        fnames = [f for f in os.listdir(args.model_dir) if f.endswith('.hdf5')]
        assert len(fnames) > 0, 'No weight files found in model dir'
        fnames = natsort.natsorted(fnames)
        model_fname = fnames[-1]
    weights_path = os.path.join(args.model_dir, model_fname)

    # Create image subdirectory to write predicted images
    pred_dir = os.path.join(args.image_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    # Get images, assemble frames if input is 3D

    im_pred = inference.predict_on_larger_image(
        network_config=config['network'],
        model_fname=weights_path,
        input_image=im,
    )




if __name__ == '__main__':
    args = parse_args()
    gpu_available = False
    assert isinstance(args.gpu, int)
    if args.gpu == -1:
        run_prediction(args)
    if args.gpu >= 0:
        gpu_available = check_gpu_availability(args.gpu, args.gpu_mem_frac)
    if gpu_available:
        run_prediction(args)
