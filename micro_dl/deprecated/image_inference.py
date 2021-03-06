#!/usr/bin/env/python
"""Model inference on large images"""
import argparse
import cv2
import natsort
import numpy as np
import os
import pandas as pd
import time
import yaml
import keras.backend as K

import micro_dl.plotting.plot_utils as plot_utils
import micro_dl.inference.model_inference as inference
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
from micro_dl.utils.tile_utils import preprocess_imstack
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
                        help='Optional: specify the gpu memory fraction to use')

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
        '--save_figs',
        dest='save_figs',
        action='store_true',
        help="Saves input, target, prediction plots. Assumes you have target channel",
    )
    parser.add_argument(
        '--no_figs',
        dest='save_figs',
        action='store_false',
        help="Don't save plots"
    )
    parser.set_defaults(save_figs=False)

    parser.add_argument(
        '--normalize_im',
        dest='normalize_im',
        action='store_true',
        help="normalizes input image",
    )
    parser.add_argument(
        '--dont_normalize_im',
        dest='normalize_im',
        action='store_false',
        help="Don't normalize input image"
    )
    parser.set_defaults(normalize_im=False)

    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        nargs='*',
        help='Metrics for model evaluation'
    )
    args = parser.parse_args()
    return args


def run_prediction(model_dir,
                   image_dir,
                   gpu_ids,
                   gpu_mem_frac,
                   model_fname=None,
                   metrics=None,
                   test_data=True,
                   ext='.tif',
                   save_figs=False,
                   normalize_im=False):
    """
    Predict images given model + weights.
    If the test_data flag is set to True, the test indices in
    split_samples.json file in model directory will be predicted
    Otherwise, all images in image directory will be predicted.
    It will load the config.yml file save in model_dir to reconstruct the model.
    Predictions are converted to uint16 and saved as png as default, but can
    also be saved as is in .npy format.
    If saving figures, it assumes that input as well as target channels are
    present in image_dir.

    :param str model_dir: Model directory
    :param str image_dir: Directory containing images for inference
    :param int gpu_ids: GPU ID to use for session
    :param float gpu_mem_frac: What fraction of GPU memory to use
    :param str model_fname: Model weights file name (in model dir)
    :param str metrics: String or list thereof of train/metrics.py functions
        to be computed during inference
    :param bool test_data: Use test indices from metadata, else use all
    :param str ext: File extension for inference output
    :param bool save_figs: Save plots of input/target/prediction
    """
    if gpu_ids >= 0:
        sess = train_utils.set_keras_session(
            gpu_ids=gpu_ids, gpu_mem_frac=gpu_mem_frac
        )
    # Load config file
    config_name = os.path.join(model_dir, 'config.yml')
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)
    # Load frames metadata and determine indices
    network_config = config['network']
    dataset_config = config['dataset']
    trainer_config = config['trainer']
    frames_meta = pd.read_csv(
        os.path.join(image_dir, 'frames_meta.csv'),
        index_col=0,
    )
    test_tile_meta = pd.read_csv(
        os.path.join(model_dir, 'test_metadata.csv'),
        index_col=0,
    )
    # TODO: generate test_frames_meta.csv together with tile csv during training
    test_frames_meta_filename = os.path.join(
        model_dir,
        'test_frames_meta.csv',
    )
    if metrics is not None:
        if isinstance(metrics, str):
            metrics = [metrics]
        metrics_cls = train_utils.get_metrics(metrics)
    else:
        metrics_cls = metrics
    loss = trainer_config['loss']
    loss_cls = train_utils.get_loss(loss)
    split_idx_name = dataset_config['split_by_column']
    K.set_image_data_format(network_config['data_format'])
    if test_data:
        idx_fname = os.path.join(model_dir, 'split_samples.json')
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
            metadata_ids[id] = np.unique(test_tile_meta[id])

    # create empty dataframe for test image metadata
    if metrics is not None:
        test_frames_meta = pd.DataFrame(
            columns=frames_meta.columns.values.tolist() + metrics,
        )
    else:
        test_frames_meta = pd.DataFrame(
            columns=frames_meta.columns.values.tolist()
        )
    # Get model weight file name, if none, load latest saved weights
    if model_fname is None:
        fnames = [f for f in os.listdir(model_dir) if f.endswith('.hdf5')]
        assert len(fnames) > 0, 'No weight files found in model dir'
        fnames = natsort.natsorted(fnames)
        model_fname = fnames[-1]
    weights_path = os.path.join(model_dir, model_fname)

    # Create image subdirectory to write predicted images
    pred_dir = os.path.join(model_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    target_channel = dataset_config['target_channels'][0]
    # If saving figures, create another subdirectory to predictions
    if save_figs:
        fig_dir = os.path.join(pred_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

    # If network depth is > 3 determine depth margins for +-z
    depth = 1
    if 'depth' in network_config:
        depth = network_config['depth']

    # Get input channel
    # TODO: Add multi channel support once such models are tested
    input_channel = dataset_config['input_channels'][0]
    assert isinstance(input_channel, int),\
        "Only supporting single input channel for now"
    # Get data format
    data_format = 'channels_first'
    if 'data_format' in network_config:
        data_format = network_config['data_format']
    # Load model with predict = True
    model = inference.load_model(
        network_config=network_config,
        model_fname=weights_path,
        predict=True,
    )
    print(model.summary())
    optimizer = trainer_config['optimizer']['name']
    model.compile(loss=loss_cls, optimizer=optimizer, metrics=metrics_cls)
    # Iterate over all indices for test data
    for time_idx in metadata_ids['time_idx']:
        for pos_idx in metadata_ids['pos_idx']:
            for slice_idx in metadata_ids['slice_idx']:
                # TODO: Add flatfield support
                im_stack = preprocess_imstack(
                    frames_metadata=frames_meta,
                    input_dir=image_dir,
                    depth=depth,
                    time_idx=time_idx,
                    channel_idx=input_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                    normalize_im=normalize_im
                )
                # Crop image shape to nearest factor of two
                im_stack = image_utils.crop2base(im_stack)
                # Change image stack format to zyx
                im_stack = np.transpose(im_stack, [2, 0, 1])
                if depth == 1:
                    # Remove singular z dimension for 2D image
                    im_stack = np.squeeze(im_stack)
                # Add channel dimension
                if data_format == 'channels_first':
                    im_stack = im_stack[np.newaxis, ...]
                else:
                    im_stack = im_stack[..., np.newaxis]
                # add batch dimensions
                im_stack = im_stack[np.newaxis, ...]
                # Predict on large image
                start = time.time()
                im_pred = inference.predict_large_image(
                    model=model,
                    input_image=im_stack,
                )
                print("Inference time:", time.time() - start)
                # Write prediction image
                im_name = aux_utils.get_im_name(
                    time_idx=time_idx,
                    channel_idx=input_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                    ext=ext,
                )
                file_name = os.path.join(pred_dir, im_name)
                if ext == '.png':
                    # Convert to uint16 for now
                    im_pred = 2 ** 16 * (im_pred - im_pred.min()) / \
                              (im_pred.max() - im_pred.min())
                    im_pred = im_pred.astype(np.uint16)
                    cv2.imwrite(file_name, np.squeeze(im_pred))
                if ext == '.tif':
                    # Convert to float32 and remove batch dimension
                    im_pred = im_pred.astype(np.float32)
                    cv2.imwrite(file_name, np.squeeze(im_pred))
                elif ext == '.npy':
                    np.save(file_name, im_pred, allow_pickle=True)
                else:
                    raise ValueError('Unsupported file extension')

                # assuming target and predicted images are always 2D for now
                # Load target
                meta_idx = aux_utils.get_meta_idx(
                    frames_meta,
                    time_idx,
                    target_channel,
                    slice_idx,
                    pos_idx,
                )
                # get a single row of frame meta data
                test_frames_meta_row = frames_meta.loc[meta_idx].copy()
                im_target = preprocess_imstack(
                    frames_metadata=frames_meta,
                    input_dir=image_dir,
                    depth=1,
                    time_idx=time_idx,
                    channel_idx=target_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                )
                im_target = image_utils.crop2base(im_target)
                # TODO: Add image_format option to network config
                # Change image stack format to zyx
                im_target = np.transpose(im_target, [2, 0, 1])
                if depth == 1:
                    # Remove singular z dimension for 2D image
                    im_target = np.squeeze(im_target)
                # Add channel dimension
                if data_format == 'channels_first':
                    im_target = im_target[np.newaxis, ...]
                else:
                    im_target = im_target[..., np.newaxis]
                # add batch dimensions
                im_target = im_target[np.newaxis, ...]

                metric_vals = model.evaluate(x=im_pred, y=im_target)
                for metric, metric_val in zip([loss] + metrics, metric_vals):
                    test_frames_meta_row[metric] = metric_val

                test_frames_meta = test_frames_meta.append(
                    test_frames_meta_row,
                    ignore_index=True,
                )
                # Save figures if specified
                if save_figs:
                    # save predicted images assumes 2D
                    if depth > 1:
                        im_stack = im_stack[..., depth // 2, :, :]
                        im_target = im_target[0, ...]
                    plot_utils.save_predicted_images(
                        input_batch=im_stack,
                        target_batch=im_target,
                        pred_batch=im_pred,
                        output_dir=fig_dir,
                        output_fname=im_name[:-4],
                        ext='jpg',
                        clip_limits=1,
                        font_size=15
                    )

    # Save metrics as csv
    test_frames_meta.to_csv(test_frames_meta_filename, sep=",")


if __name__ == '__main__':
    args = parse_args()
    # Get GPU ID and memory fraction
    gpu_id, gpu_mem_frac = train_utils.select_gpu(
        args.gpu,
        args.gpu_mem_frac,
    )
    run_prediction(
        model_dir=args.model_dir,
        image_dir=args.image_dir,
        gpu_ids=gpu_id,
        gpu_mem_frac=gpu_mem_frac,
        model_fname=args.model_fname,
        metrics=args.metrics,
        test_data=args.test_data,
        ext=args.ext,
        save_figs=args.save_figs,
        normalize_im=args.normalize_im
    )

