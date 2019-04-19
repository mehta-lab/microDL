#!/usr/bin/env/python

import argparse
import numpy as np
import os
import pandas as pd
import yaml

import micro_dl.inference.evaluation_metrics as metrics
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.tile_utils as tile_utils


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
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
        dest='test_data',
        action='store_true',
        help="Use test indices in split_samples.json",
    )
    parser.add_argument(
        '--all_data',
        dest='test_data',
        action='store_false',
    )
    parser.set_defaults(test_data=True)
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help="Directory containing images",
    )
    parser.add_argument(
        '--ext',
        type=str,
        default='.tif',
        help="Image extension. If .png rescales to uint16, otherwise save as is",
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        nargs='*',
        help='Metrics for model evaluation'
    )
    parser.add_argument(
        '--orientations',
        type=str,
        default=None,
        nargs='*',
        help='Evaluate metrics along these orientations (xy, xz, yz, xyz)'
    )
    parser.add_argument(
        '--start_z',
        type=int,
        default=None,
        help="If none, start at first slice",
    )
    parser.add_argument(
        '--end_z',
        type=int,
        default=None,
        help="If none, end at last slice",
    )
    return parser.parse_args()


def compute_metrics(args):

    # Load config file
    config_name = os.path.join(args.model_dir, 'config.yml')
    with open(config_name, 'r') as f:
        config = yaml.load(f)
    # Load frames metadata and determine indices
    frames_meta = pd.read_csv(os.path.join(args.image_dir, 'frames_meta.csv'))

    metrics_list = args.metrics
    if isinstance(metrics_list, str):
        metrics_list = [metrics_list]
    metrics_inst = metrics.MetricsEstimator(metrics_list=metrics_list)

    split_idx_name = config['dataset']['split_by_column']
    if args.test_data:
        idx_fname = os.path.join(args.model_dir, 'split_samples.json')
        try:
            split_samples = aux_utils.read_json(idx_fname)
            test_ids = split_samples['test']
        except FileNotFoundError as e:
            print("No split_samples file. Will predict all images in dir.")
    else:
        test_ids = np.unique(frames_meta[split_idx_name])

    # Find other indices to iterate over than split index name
    # E.g. if split is position, we also need to iterate over time and slice
    metadata_ids = {split_idx_name: test_ids}
    iter_ids = ['slice_idx', 'pos_idx', 'time_idx']

    for id in iter_ids:
        if id != split_idx_name:
            metadata_ids[id] = np.unique(frames_meta[id])

    # Create image subdirectory to write predicted images
    pred_dir = os.path.join(args.model_dir, 'predictions')

    target_channel = config['dataset']['target_channels'][0]

    # If network depth is > 3 determine depth margins for +-z
    depth = 1
    if 'depth' in config['network']:
        depth = config['network']['depth']
        if depth > 1:
            metadata_ids['slice_idx'] = aux_utils.adjust_slice_margins(
                slice_ids=metadata_ids['slice_idx'],
                depth=depth,
            )
    print(metadata_ids['slice_idx'])
    # Get input channel(s)
    input_channels = config['dataset']['input_channels']
    pred_channel = input_channels[0]

    orientations_list = args.orientations
    if isinstance(orientations_list, str):
        orientations_list = [orientations_list]
    available_orientations = {'xy', 'xz', 'yz', 'xyz'}
    assert set(orientations_list).issubset(available_orientations), \
        "Orientations must be subset of {}".format(available_orientations)
    metrics_xy = pd.DataFrame()
    metrics_xz = pd.DataFrame()
    metrics_yz = pd.DataFrame()
    metrics_xyz = pd.DataFrame()

    # Iterate over all indices for test data
    for time_idx in metadata_ids['time_idx']:
        for pos_idx in metadata_ids['pos_idx']:
            target_fnames = []
            pred_fnames = []
            for slice_idx in metadata_ids['slice_idx']:
                im_idx = aux_utils.get_meta_idx(
                    metadata_df=frames_meta,
                    time_idx=time_idx,
                    channel_idx=target_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                )
                target_fname = os.path.join(
                    args.image_dir,
                    frames_meta.loc[im_idx, 'file_name'],
                )
                target_fnames.append(target_fname)
                pred_fname = aux_utils.get_im_name(
                    time_idx=time_idx,
                    channel_idx=pred_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                    ext=args.ext,
                )
                pred_fname = os.path.join(pred_dir, pred_fname)
                pred_fnames.append(pred_fname)
            print(pred_fnames)
            print('----------------------')
            print(target_fnames)
            target_stack = tile_utils.read_imstack(
                input_fnames=tuple(target_fnames),
            )
            pred_stack = tile_utils.read_imstack(
                input_fnames=tuple(pred_fnames),
            )
            if depth == 1:
                # Remove singular z dimension for 2D image
                target_stack = np.squeeze(target_stack)
                pred_stack = np.squeeze(pred_stack)
            print(target_stack.shape, pred_stack.shape)

            pred_name = "t{}_p{}".format(time_idx, pos_idx)
            if 'xy' in orientations_list:
                metrics_inst.estimate_xy_metrics(
                    target=target_stack,
                    prediction=pred_stack,
                    pred_name = pred_name,
                )
                metrics_xy = metrics_xy.append(
                    metrics_inst.get_metrics_xy(),
                    ignore_index=True,
                )
            if 'xz' in orientations_list:
                metrics_inst.estimate_xz_metrics(
                    target=target_stack,
                    prediction=pred_stack,
                    pred_name = pred_name,
                )
                metrics_xz = metrics_xz.append(
                    metrics_inst.get_metrics_xz(),
                    ignore_index=True,
                )
            if 'yz' in orientations_list:
                metrics_inst.estimate_yz_metrics(
                    target=target_stack,
                    prediction=pred_stack,
                    pred_name=pred_name,
                )
                metrics_yz = metrics_yz.append(
                    metrics_inst.get_metrics_yz(),
                    ignore_index=True,
                )
            if 'xyz' in orientations_list:
                metrics_inst.estimate_xyz_metrics(
                    target=target_stack,
                    prediction=pred_stack,
                    pred_name = pred_name,
                )
                metrics_xyz = metrics_xyz.append(
                    metrics_inst.get_metrics_xyz(),
                    ignore_index=True,
                )

    if not metrics_xy.empty:
        metrics_name = os.path.join(pred_dir, 'metrics_xy.csv')
        metrics_xy.to_csv(metrics_name, sep=",")
    if not metrics_xz.empty:
        metrics_name = os.path.join(pred_dir, 'metrics_xz.csv')
        metrics_xz.to_csv(metrics_name, sep=",")
    if not metrics_yz.empty:
        metrics_name = os.path.join(pred_dir, 'metrics_yz.csv')
        metrics_yz.to_csv(metrics_name, sep=",")
    if not metrics_xyz.empty:
        metrics_name = os.path.join(pred_dir, 'metrics_xyz.csv')
        metrics_xyz.to_csv(metrics_name, sep=",")


if __name__ == '__main__':
    args = parse_args()
    compute_metrics(args)
