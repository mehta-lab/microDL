"""Model inference at the image/volume level"""
import cv2
import natsort
import numpy as np
import os
import pandas as pd

from micro_dl.input.inference_dataset import InferenceDataset
import micro_dl.train.model_inference as inference
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.train_utils import set_keras_session


def create_model(training_config, model_fname):
    """Load model given the model_fname or the saved model in model_dir

    :param dict training_config:
    :param str model_fname:
    :return keras.Model instance
    """

    model_dir = training_config['trainer']['model_dir']

    # Get model weight file name, if none, load latest saved weights
    if model_fname is None:
        fnames = [f for f in os.listdir(model_dir) if f.endswith('.hdf5')]
        assert len(fnames) > 0, 'No weight files found in model dir'
        fnames = natsort.natsorted(fnames)
        model_fname = fnames[-1]
    weights_path = os.path.join(model_dir, model_fname)

    # Load model with predict = True
    model = inference.load_model(
        network_config=training_config['network'],
        model_fname=weights_path,
        predict=True,
    )
    return model


def get_split_meta(training_config, image_dir, data_split='test'):
    """Get the cztp indices for the data split: train, test, val or all

    :param dict training_config:
    :param str image_dir: dir containing images AND NOT TILES!
    :param str data_split:
    :return pd.Dataframe df_split_meta: dataframe with slice, pos, time
     indices and file names
    """

    # Load frames metadata and determine indices
    frames_meta = pd.read_csv(os.path.join(image_dir,
                                           'frames_meta.csv'))
    split_idx_name = training_config['dataset']['split_by_column']

    model_dir = training_config['trainer']['model_dir']
    if data_split == 'test':
        idx_fname = os.path.join(model_dir, 'split_samples.json')
        try:
            split_samples = aux_utils.read_json(idx_fname)
            test_ids = split_samples['test']
        except FileNotFoundError as e:
            print("No split_samples file. Will predict all images in dir.")
    else:
        test_ids = np.unique(frames_meta[split_idx_name])

    df_split_meta_idx = frames_meta[split_idx_name].isin(test_ids)
    df_split_meta = frames_meta[df_split_meta_idx]
    return df_split_meta


def _get_sub_block(input_image,
                   start_z_idx,
                   end_z_idx,
                   image_format,
                   data_format):
    """Get the sub block along z given start and end slice indices"""

    if image_format == 'xyz' and data_format == 'channels_first':
        cur_block = input_image[:, :, :, :, start_z_idx: end_z_idx]
    elif image_format == 'xyz' and data_format == 'channels_last':
        cur_block = input_image[:, :, :, start_z_idx: end_z_idx, :]
    elif image_format == 'zyx' and data_format == 'channels_first':
        cur_block = input_image[:, :, start_z_idx: end_z_idx, :, :]
    elif image_format == 'zyx' and data_format == 'channels_last':
        cur_block = input_image[:, start_z_idx: end_z_idx, :, :, :]
    return cur_block


def place_block(pred_block, pred_image, start_idx, end_idx):
    """Place the current block prediction in the larger vol

    zyx and channels_first for now
    """
    pred_image[:, :, start_idx + 1: end_idx, :, :] = pred_block[:, :, 1:, :, :]
    if start_idx > 0:
        pred_image[:, :, start_idx, :, :] = (
            pred_image[:, :, start_idx, :, :] + start_idx[:, :, 0, :, :]
        ) / 2
    else:
        pred_image[:, :, start_idx, :, :] = start_idx[:, :, 0, :, :]
    return pred_image


def save_pred_image(predicted_image,
                    save_dir,
                    time_idx, tar_ch_idx, pos_idx, slice_idx,
                    im_ext):
    """Save predicted images"""

    # Write prediction image
    im_name = aux_utils.get_im_name(
        time_idx=time_idx,
        channel_idx=tar_ch_idx,
        slice_idx=slice_idx,
        pos_idx=pos_idx,
        ext=im_ext,
    )
    file_name = os.path.join(save_dir, im_name)
    # save 3D image as npy.
    if len(predicted_image.shape) == 3:
        np.save(file_name, predicted_image, allow_pickle=True)
    else:
        if im_ext == '.png':
            # Convert to uint16 for now
            im_pred = 2 ** 16 * (predicted_image - predicted_image.min()) / \
                      (predicted_image.max() - predicted_image.min())
            im_pred = im_pred.astype(np.uint16)
            cv2.imwrite(file_name, np.squeeze(im_pred))
        elif im_ext == '.tif':
            # Convert to float32 and remove batch dimension
            im_pred = predicted_image.astype(np.float32)
            cv2.imwrite(file_name, np.squeeze(im_pred))
        elif im_ext == '.npy':
            np.save(file_name, predicted_image, allow_pickle=True)
        else:
            raise ValueError('Unsupported file extension')


def run_prediction(training_config,
                   model_fname,
                   image_dir,
                   data_split,
                   gpu_id,
                   gpu_mem_frac,
                   image_format='zyx',
                   flat_field_dir=None,
                   im_ext='.npy',
                   num_slices=1):
    """Run prediction for entire 2D image or a 3D stack

    :param dict training_config:
    :param str model_fname:
    :param str image_dir:
    :param str data_split:
    :param int gpu_id:
    :param float gpu_mem_frac:
    :param str image_format:
    :param str flat_field_dir:
    :param int num_slices: in case of 3D, the full volume will not fit in
     GPU memory, specify the number of slices to use and this will depend on
     the network depth, for ex 8 for a network of depth 4
    """

    if num_slices > 1:
        assert training_config['network']['class'] == 'UNet3D', \
            'num_slices is used for splitting a volume into block along z.' \
            'Preferable if it is in powers of 2'
        network_depth = len(training_config['network']['num_filters_per_block'])
        min_num_slices = 2 ** (network_depth - 1)
        assert num_slices >= min_num_slices, \
            'Insufficient number of slices {} for the network depth {}'.format(
                num_slices, network_depth
            )

    if gpu_id >= 0:
        sess = set_keras_session(gpu_ids=gpu_id,
                                 gpu_mem_frac=gpu_mem_frac)

    # create network instance and load weights
    model_inst = create_model(training_config, model_fname)

    # create a inference dataset instance
    df_split_meta = get_split_meta(training_config, image_dir, data_split)

    dataset_inst = InferenceDataset(image_dir=image_dir,
                                    dataset_config=training_config['dataset'],
                                    network_config=training_config['network'],
                                    df_meta=df_split_meta,
                                    image_format=image_format,
                                    flat_field_dir=flat_field_dir)

    # Create image subdirectory to write predicted images
    model_dir = training_config['trainer']['model_dir']
    pred_dir = os.path.join(model_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    # assign zdim if not Unet2D
    data_format = training_config['network']['data_format']
    if image_format == 'zyx':
        z_dim = 2 if data_format == 'channels_first' else 1
    elif image_format == 'xyz':
        z_dim = 4 if data_format == 'channels_first' else 3

    df_iteration_meta = dataset_inst.get_df_iteration_meta()
    for ds_idx in range(len(dataset_inst)):
        cur_input, cur_target = dataset_inst.__getitem__(ds_idx)
        # get blocks with an overlap of one slice
        if num_slices > 1:
            pred_image = np.zeros_like(cur_input)
            num_z = cur_input.shape[z_dim]
            num_blocks = np.floor(num_z / (num_slices - 1)).astype('int')
            for block_idx in range(num_blocks):
                start_idx = block_idx * (num_slices - 1)
                end_idx = start_idx + num_slices
                cur_block = _get_sub_block(cur_input,
                                           start_idx,
                                           end_idx,
                                           image_format,
                                           data_format)
                pred_block = inference.predict_on_larger_image(
                    model=model_inst,
                    input_image=cur_block
                )
                # overlap of 1 slice is hard-coded for now!
                pred_image = place_block(pred_block,
                                         pred_image,
                                         start_idx,
                                         end_idx)
        else:
            pred_image = inference.predict_on_larger_image(
                model=model_inst, input_image=cur_input
            )
        pred_image = np.squeeze(pred_image)
        cur_row = df_iteration_meta.iloc[ds_idx]
        save_pred_image(predicted_image=pred_image,
                        save_dir=pred_dir,
                        time_idx=cur_row['time_idx'],
                        tar_ch_idx=cur_row['channel_idx'],
                        pos_idx=cur_row['pos_idx'],
                        slice_idx=cur_row['slice_idx'],
                        im_ext=im_ext)
