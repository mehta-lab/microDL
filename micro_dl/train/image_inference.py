"""Model inference at the image/volume level"""
import cv2
import natsort
import numpy as np
import os
import pandas as pd

from micro_dl.input.inference_dataset import InferenceDataset
import micro_dl.train.model_inference as inference
from micro_dl.train.evaluation_metrics import MetricsEstimator
from micro_dl.train.stitch_predictions import ImageStitcher
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.image_utils import center_crop_to_shape
from micro_dl.utils.train_utils import set_keras_session
import micro_dl.utils.tile_utils as tile_utils


def create_model(training_config, model_fname):
    """Load model given the model_fname or the saved model in model_dir

    :param dict training_config: config dict with params related to dataset,
     trainer and network
    :param str model_fname: fname of the hdf5 file with model weights
    :return keras.Model instance with trained weights loaded
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

    :param dict training_config: config dict with params related to dataset,
     trainer and network
    :param str image_dir: dir containing images AND NOT TILES!
    :param str data_split: in [train, val, test]
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
            print("No split_samples file. Will predict all images in dir." + e)
    else:
        test_ids = np.unique(frames_meta[split_idx_name])

    df_split_meta_idx = frames_meta[split_idx_name].isin(test_ids)
    df_split_meta = frames_meta[df_split_meta_idx]
    return df_split_meta


def _get_sub_block_z(input_image,
                     start_z_idx,
                     end_z_idx,
                     image_format,
                     data_format):
    """Get the sub block along z given start and end slice indices

    :param np.array input_image: 3D image
    :param int start_z_idx: start slice for the current block
    :param int end_z_idx: end slice for the current block
    :param str image_format: xyz or zxy
        :param str data_format: channels_first or channels last
    :return np.array cur_block: sub block / volume
    """

    if image_format == 'xyz' and data_format == 'channels_first':
        cur_block = input_image[:, :, :, :, start_z_idx: end_z_idx]
    elif image_format == 'xyz' and data_format == 'channels_last':
        cur_block = input_image[:, :, :, start_z_idx: end_z_idx, :]
    elif image_format == 'zxy' and data_format == 'channels_first':
        cur_block = input_image[:, :, start_z_idx: end_z_idx, :, :]
    elif image_format == 'zxy' and data_format == 'channels_last':
        cur_block = input_image[:, start_z_idx: end_z_idx, :, :, :]
    return cur_block


def save_pred_image(predicted_image,
                    save_dir,
                    time_idx, tar_ch_idx, pos_idx, slice_idx,
                    im_ext):
    """Save predicted images

    :param np.array predicted_image: 2D / 3D predicted image
    :param str save_dir: directory to save predicted images
    :param int time_idx: time index
    :param int tar_ch_idx: target / predicted channel index
    :param int pos_idx: FOV / position index
    :param int slice_idx: slice index
    :param str im_ext: npy or png or tiff. FOR 3D IMAGES USE NPY AS PNG AND
     TIFF ARE CURRENTLY NOT SUPPORTED
    """

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


def _predict_sub_block_z(input_image,
                         z_dim,
                         num_slices,
                         num_overlap,
                         image_format,
                         data_format,
                         model_inst):
    """Predict sub blocks along z

    :param np.array input_image:
    :param int z_dim:
    :param int num_slices:
    :param int num_overlap:
    :param str image_format:
    :param str data_format:
    :param keras.Model model_inst:
    """

    pred_imgs_list = []
    start_end_idx = []
    num_z = input_image.shape[z_dim]
    num_blocks = np.ceil(num_z / (num_slices - num_overlap)).astype(
        'int')
    print('block pred:', num_blocks)
    for block_idx in range(num_blocks):
        start_idx = block_idx * (num_slices - num_overlap)
        end_idx = start_idx + num_slices
        if end_idx >= num_z:
            end_idx = num_z
            start_idx = end_idx - num_slices
        cur_block = _get_sub_block_z(input_image,
                                     start_idx,
                                     end_idx,
                                     image_format,
                                     data_format)
        pred_block = inference.predict_on_larger_image(
            model=model_inst,
            input_image=cur_block
        )
        pred_imgs_list.append(pred_block)
        start_end_idx.append(start_idx, end_idx)
    return pred_imgs_list, start_end_idx


def _predict_sub_block_xyz(input_image,
                           crop_indices,
                           data_format,
                           model_inst):
    """Predict sub blocks along xyz

    :param np.array input_image:
    :param list crop_indices:
    :param str data_format,
    :param keras.Model model_inst:
    """

    pred_imgs_list = []
    start_end_idx = []
    for crop_idx in crop_indices:
        if data_format == 'channels_first':
            cur_block = input_image[:, :, crop_idx[0]: crop_indices[1],
                                    crop_idx[2]: crop_indices[3],
                                    crop_idx[4]: crop_indices[5]]
        elif data_format == 'channels_last':
            cur_block = input_image[:, crop_idx[0]: crop_indices[1],
                                    crop_idx[2]: crop_indices[3],
                                    crop_idx[4]: crop_indices[5], :]
        pred_block = inference.predict_on_larger_image(
            model=model_inst,
            input_image=cur_block
        )
        pred_imgs_list.append(pred_block)
        start_end_idx.append(crop_idx)
    return pred_imgs_list, start_end_idx


def estimate_metrics(metrics_est_inst,
                     cur_target,
                     cur_prediction,
                     cur_pred_fname,
                     mask_param_dict,
                     cur_row):
    """Estimate evaluation metrics

    :param MetricsEstimator metrics_est_inst:
    :param np.array cur_target:
    :param np.array cur_prediction:
    :param str cur_pred_fname:
    :param dict mask_param_dict:
    :param pd.Series cur_row:
    """

    kw_args = {'target': cur_target,
               'prediction': cur_prediction,
               'pred_fname': cur_pred_fname}

    if mask_param_dict is not None:
        # mask_channel = 9
        mask_fname = aux_utils.get_im_name(
            time_idx=cur_row['time_idx'],
            channel_idx=mask_param_dict['mask_channel'],
            slice_idx=cur_row['slice_idx'],
            pos_idx=cur_row['pos_idx']
        )
        # mask_dir = '/data/kidney_tiles/mask_channels_2',
        mask_fname = os.path.join(mask_param_dict['mask_dir'],
                                  mask_fname)
        cur_mask = np.load(mask_fname)
        cur_mask = np.transpose(cur_mask, [2, 0, 1])
        kw_args['cur_mask'] = cur_mask
    metrics_est_inst.estimate_metrics(kw_args)


def run_prediction(training_config,
                   model_fname,
                   image_dir,
                   data_split,
                   gpu_id,
                   gpu_mem_frac,
                   image_param_dict,
                   mask_param_dict=None,
                   vol_inf_dict=None,
                   metrics_list=None):
    """Run prediction for entire 2D image or a 3D stack

    :param dict training_config:
    :param str model_fname:
    :param str image_dir:
    :param str data_split:
    :param int gpu_id:
    :param float gpu_mem_frac:
    :param dict image_param_dict: dict with keys image_format, flat_field_dir,
     im_ext
    :param dict mask_param_dict: dict with keys mask_dir and mask_channel
    :param dict vol_inf_dict: dict with params for 3D inference with keys:
     num_slices, inf_shape, tile_shape, num_overlap, overlap_operation.
     num_slices - in case of 3D, the full volume will not fit in GPU memory,
     specify the number of slices to use and this will depend on the network
     depth, for ex 8 for a network of depth 4.
     inf_shape: inference on a center sub volume
    :param list metrics_list:
    """

    if mask_param_dict is not None:
        assert ('mask_channel' in mask_param_dict and
                'mask_dir' in mask_param_dict), \
            'Both mask_channel and mask_dir are needed'

    tile_option = None
    if 'num_slices' in vol_inf_dict and vol_inf_dict['num_slices'] > 1:
        tile_option = 'tile_z'
        num_slices = vol_inf_dict['num_slices']
        assert training_config['network']['class'] == 'UNet3D', \
            'num_slices is used for splitting a volume into block along z.' \
            'Preferable if it is in powers of 2'
        network_depth = len(
            training_config['network']['num_filters_per_block']
        )
        min_num_slices = 2 ** (network_depth - 1)
        assert num_slices >= min_num_slices, \
            'Insufficient number of slices {} for the network depth {}'.format(
                num_slices, network_depth
            )
        msg = ('num_overlap: The sub blocks should have an overlap of at least'
               ' 1 slice')
        assert 'num_overlap' in vol_inf_dict and \
               vol_inf_dict['num_overlap'] >= 1, msg
        num_overlap = vol_inf_dict['num_overlap']

    if 'inf_shape' in vol_inf_dict:
        tile_option = 'infer_on_center'

    if 'tile_shape' in vol_inf_dict:
        tile_option = 'tile_xyz'
        assert len(vol_inf_dict['tile_shape']) == len(num_overlap), \
            'len of tile_shape is not equal to len of num_overlap'

    if 'vol_inf_dict' is not None:
        # assign zdim if not Unet2D
        data_format = training_config['network']['data_format']
        if image_param_dict['image_format'] == 'zxy':
            z_dim = 2 if data_format == 'channels_first' else 1
        elif image_param_dict['image_format'] == 'xyz':
            z_dim = 4 if data_format == 'channels_first' else 3

        # create an instance of ImageStitcher
        if tile_option in ['tile_z', 'tile_xyz']:
            overlap_dict = {
                'overlap_shape': vol_inf_dict['num_overlap'],
                'overlap_operation': vol_inf_dict['overlap_operation'],
                'z_dim': z_dim
            }
            snitch_inst = ImageStitcher(
                tile_option=tile_option,
                overlap_dict=overlap_dict,
                image_format=image_param_dict['image_format'],
                data_format=data_format
            )
    if gpu_id >= 0:
        sess = set_keras_session(gpu_ids=gpu_id,
                                 gpu_mem_frac=gpu_mem_frac)

    # create network instance and load weights
    model_inst = create_model(training_config, model_fname)

    # create a inference dataset instance
    df_split_meta = get_split_meta(training_config, image_dir, data_split)

    dataset_inst = InferenceDataset(
        image_dir=image_dir,
        dataset_config=training_config['dataset'],
        network_config=training_config['network'],
        df_meta=df_split_meta,
        image_format=image_param_dict['image_format'],
        flat_field_dir=image_param_dict['flat_field_dir']
    )

    # Create image subdirectory to write predicted images
    model_dir = training_config['trainer']['model_dir']
    pred_dir = os.path.join(model_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    # create an instance of MetricsEstimator
    df_iteration_meta = dataset_inst.get_df_iteration_meta()
    if metrics_list is not None:
        masked_metrics = False
        if mask_param_dict is not None:
            masked_metrics = True
        metrics_est_inst = MetricsEstimator(metrics_list=metrics_list,
                                            masked_metrics=masked_metrics,
                                            len_data_split=len(dataset_inst))

    crop_indices = None
    for ds_idx in range(len(dataset_inst)):
        cur_input, cur_target = dataset_inst.__getitem__(ds_idx)
        # get blocks with an overlap of one slice
        if tile_option == 'infer_on_center':
            inf_shape = vol_inf_dict['inf_shape']
            center_block = center_crop_to_shape(cur_input,
                                                inf_shape)
            pred_image = inference.predict_on_larger_image(
                model=model_inst, input_image=center_block
            )
        elif tile_option == 'tile_z':
            pred_block_list, start_end_idx = _predict_sub_block_z(
                cur_input,
                z_dim,
                num_slices,
                num_overlap,
                image_param_dict['image_format'],
                training_config['network']['data_format'],
                model_inst
            )
            pred_image = snitch_inst.stitch_predictions(cur_input.shape,
                                                        pred_block_list,
                                                        start_end_idx)
        elif tile_option == 'tile_xyz':
            step_size = (vol_inf_dict['tile_shape'] -
                         vol_inf_dict['num_overlap'])
            if crop_indices is None:
                crop_img_list, crop_idx = tile_utils.tile_image(
                    input_image=cur_input,
                    tile_size=vol_inf_dict['tile_shape'],
                    step_size=step_size,
                    return_index=True
                )
            pred_block_list, crop_indices = _predict_sub_block_xyz(
                cur_input,
                crop_idx,
                training_config['network']['data_format'],
                model_inst
            )
            pred_image = snitch_inst.stitch_predictions(cur_input.shape,
                                                        pred_block_list,
                                                        start_end_idx)
        else:
            pred_image = inference.predict_on_larger_image(
                model=model_inst, input_image=cur_input
            )

        pred_image = np.squeeze(pred_image)
        cur_target = np.squeeze(cur_target)
        cur_row = df_iteration_meta.iloc[ds_idx]
        pred_fname = save_pred_image(predicted_image=pred_image,
                                     save_dir=pred_dir,
                                     time_idx=cur_row['time_idx'],
                                     tar_ch_idx=cur_row['channel_idx'],
                                     pos_idx=cur_row['pos_idx'],
                                     slice_idx=cur_row['slice_idx'],
                                     im_ext=image_param_dict['im_ext'])
        if metrics_list is not None:
            estimate_metrics(metrics_est_inst,
                             cur_target,
                             pred_image,
                             pred_fname,
                             mask_param_dict,
                             cur_row)
    if metrics_list is not None:
        df_metrics = metrics_est_inst.get_metrics_df()
        df_metrics.to_csv(os.path.join(model_dir, 'test_metrics.csv'),
                          sep=',')
