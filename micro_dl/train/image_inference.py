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


class ImagePredictor:
    """Infer on larger images"""

    def __init__(self, config,
                 model_fname,
                 image_dir,
                 data_split,
                 image_param_dict,
                 gpu_id,
                 gpu_mem_frac,
                 metrics_list=None,
                 mask_param_dict=None,
                 vol_inf_dict=None):
        """Init

        :param dict config: config dict with params related to dataset,
         trainer and network
        :param str model_fname: fname of the hdf5 file with model weights
        :param str image_dir: dir containing images AND NOT TILES!
        :param dict image_param_dict: dict with keys image_format,
         flat_field_dir, im_ext. im_ext: npy or png or tiff. FOR 3D IMAGES USE
         NPY AS PNG AND TIFF ARE CURRENTLY NOT SUPPORTED
        :params int gpu_id: gpu to use
        :params float gpu_mem_frac: Memory fractions to use corresponding
         to gpu_ids
        :param list metrics_list: list of metrics to estimate
        :param dict mask_param_dict: dict with keys mask_dir and mask_channel
        :param dict vol_inf_dict: dict with params for 3D inference with keys:
         num_slices, inf_shape, tile_shape, num_overlap, overlap_operation.
         num_slices - in case of 3D, the full volume will not fit in GPU
         memory, specify the number of slices to use and this will depend on
         the network depth, for ex 8 for a network of depth 4. inf_shape -
         inference on a center sub volume.
        """

        self.config = config
        self.data_format = self.config['network']['data_format']
        if gpu_id >= 0:
            sess = set_keras_session(gpu_ids=gpu_id,
                                     gpu_mem_frac=gpu_mem_frac)
        # create network instance and load weights
        model_inst = self._create_model(model_fname)
        self.model_inst = model_inst

        assert data_split in ['train', 'val', 'test'], \
            'data_split not in [train, val, test]'
        df_split_meta = self._get_split_meta(image_dir, data_split)
        self.df_split_meta = df_split_meta
        assert ('image_format' in image_param_dict and
                'im_ext' in image_param_dict), \
            'image_format and/or im_ext not in image_param_dict'
        dataset_inst = InferenceDataset(
            image_dir=image_dir,
            dataset_config=config['dataset'],
            network_config=config['network'],
            df_meta=df_split_meta,
            image_format=image_param_dict['image_format'],
            flat_field_dir=image_param_dict['flat_field_dir']
        )
        self.dataset_inst = dataset_inst
        self.image_format = image_param_dict['image_format']
        self.image_ext = image_param_dict['image_ext']

        # Create image subdirectory to write predicted images
        model_dir = config['trainer']['model_dir']
        pred_dir = os.path.join(model_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        self.pred_dir = pred_dir

        # create an instance of MetricsEstimator
        self.df_iteration_meta = dataset_inst.get_df_iteration_meta()
        metrics_est_inst = None
        if mask_param_dict is not None:
            assert ('mask_channel' in mask_param_dict and
                    'mask_dir' in mask_param_dict), \
                'Both mask_channel and mask_dir are needed'
            metrics_est_inst = MetricsEstimator(
                metrics_list=metrics_list,
                masked_metrics=True,
                len_data_split=len(dataset_inst)
            )
        self.mask_param_dict = mask_param_dict
        self.metrics_est_inst = metrics_est_inst

        num_overlap = 0
        snitch_inst = None
        tile_option = None
        z_dim = 2
        if vol_inf_dict is not None:
            tile_option, num_overlap, snitch_inst, z_dim = \
                self._assign_vol_inf_options(image_param_dict,
                                             vol_inf_dict)
        self.tile_option = tile_option
        self.num_overlap = num_overlap
        self.snitch_inst = snitch_inst
        self.z_dim = z_dim
        self.vol_inf_dict = vol_inf_dict

    def _create_model(self, model_fname):
        """Load model given the model_fname or the saved model in model_dir

        :param str model_fname: fname of the hdf5 file with model weights
        :return keras.Model instance with trained weights loaded
        """

        model_dir = self.config['trainer']['model_dir']

        # Get model weight file name, if none, load latest saved weights
        if model_fname is None:
            fnames = [f for f in os.listdir(model_dir) if f.endswith('.hdf5')]
            assert len(fnames) > 0, 'No weight files found in model dir'
            fnames = natsort.natsorted(fnames)
            model_fname = fnames[-1]
        weights_path = os.path.join(model_dir, model_fname)

        # Load model with predict = True
        model = inference.load_model(
            network_config=self.config['network'],
            model_fname=weights_path,
            predict=True,
        )
        return model

    def _get_split_meta(self, image_dir, data_split='test'):
        """Get the cztp indices for the data split: train, test, val or all

        :param str image_dir: dir containing images AND NOT TILES!
        :param str data_split: in [train, val, test]
        :return pd.Dataframe df_split_meta: dataframe with slice, pos, time
         indices and file names
        """

        # Load frames metadata and determine indices
        frames_meta = pd.read_csv(os.path.join(image_dir,
                                               'frames_meta.csv'))
        split_idx_name = self.config['dataset']['split_by_column']

        model_dir = self.config['trainer']['model_dir']
        if data_split == 'test':
            idx_fname = os.path.join(model_dir, 'split_samples.json')
            try:
                split_samples = aux_utils.read_json(idx_fname)
                test_ids = split_samples['test']
            except FileNotFoundError as e:
                print("No split_samples file. "
                      "Will predict all images in dir." + e)
        else:
            test_ids = np.unique(frames_meta[split_idx_name])

        df_split_meta_idx = frames_meta[split_idx_name].isin(test_ids)
        df_split_meta = frames_meta[df_split_meta_idx]
        return df_split_meta

    def _assign_vol_inf_options(self, image_param_dict, vol_inf_dict):
        """Assign vol inf options

        :param dict image_param_dict:
        :param dict vol_inf_dict:
        """

        # assign zdim if not Unet2D
        if image_param_dict['image_format'] == 'zxy':
            z_dim = 2 if self.data_format == 'channels_first' else 1
        elif image_param_dict['image_format'] == 'xyz':
            z_dim = 4 if self.data_format == 'channels_first' else 3

        if 'num_slices' in vol_inf_dict and vol_inf_dict['num_slices'] > 1:
            tile_option = 'tile_z'
            num_slices = vol_inf_dict['num_slices']
            assert self.config['network']['class'] == 'UNet3D', \
                'num_slices is used for splitting a volume into block ' \
                'along z. Preferable if it is in powers of 2'
            network_depth = len(
                self.config['network']['num_filters_per_block']
            )
            min_num_slices = 2 ** (network_depth - 1)
            assert num_slices >= min_num_slices, \
                'Insufficient number of slices {} for the network ' \
                'depth {}'.format(num_slices, network_depth)
            num_overlap = vol_inf_dict['num_overlap'] \
                if 'num_overlap' in vol_inf_dict else 0
        elif 'tile_shape' in vol_inf_dict:
            tile_option = 'tile_xyz'
            num_overlap = vol_inf_dict['num_overlap'] \
                if 'num_overlap' in vol_inf_dict else [0, 0, 0]
        elif 'inf_shape' in vol_inf_dict:
            tile_option = 'infer_on_center'

        snitch_inst = None
        # create an instance of ImageStitcher
        if tile_option in ['tile_z', 'tile_xyz']:
            overlap_dict = {
                'overlap_shape': num_overlap,
                'overlap_operation': vol_inf_dict['overlap_operation'],
                'z_dim': z_dim
            }
            snitch_inst = ImageStitcher(
                tile_option=tile_option,
                overlap_dict=overlap_dict,
                image_format=image_param_dict['image_format'],
                data_format=self.data_format
            )
        return tile_option, num_overlap, snitch_inst, z_dim

    def _get_sub_block_z(self,
                         input_image,
                         start_z_idx,
                         end_z_idx):
        """Get the sub block along z given start and end slice indices

        :param np.array input_image: 3D image
        :param int start_z_idx: start slice for the current block
        :param int end_z_idx: end slice for the current block
        :return np.array cur_block: sub block / volume
        """

        if self.image_format == 'xyz' and \
                self.data_format == 'channels_first':
            cur_block = input_image[:, :, :, :, start_z_idx: end_z_idx]
        elif self.image_format == 'xyz' and \
                self.data_format == 'channels_last':
            cur_block = input_image[:, :, :, start_z_idx: end_z_idx, :]
        elif self.image_format == 'zxy' and \
                self.data_format == 'channels_first':
            cur_block = input_image[:, :, start_z_idx: end_z_idx, :, :]
        elif self.image_format == 'zxy' and \
                self.data_format == 'channels_last':
            cur_block = input_image[:, start_z_idx: end_z_idx, :, :, :]
        return cur_block

    def _predict_sub_block_z(self, input_image):
        """Predict sub blocks along z

        :param np.array input_image:
        """

        pred_imgs_list = []
        start_end_idx = []
        num_z = input_image.shape[self.z_dim]
        num_slices = self.vol_inf_dict['num_slices']
        num_blocks = np.ceil(
            num_z / (num_slices - self.num_overlap)
        ).astype('int')
        print('block pred:', num_blocks)
        for block_idx in range(num_blocks):
            start_idx = block_idx * (num_slices - self.num_overlap)
            end_idx = start_idx + num_slices
            if end_idx >= num_z:
                end_idx = num_z
                start_idx = end_idx - num_slices
            cur_block = self._get_sub_block_z(input_image,
                                              start_idx,
                                              end_idx)
            pred_block = inference.predict_on_larger_image(
                model=self.model_inst,
                input_image=cur_block
            )
            pred_imgs_list.append(pred_block)
            start_end_idx.append(start_idx, end_idx)
        return pred_imgs_list, start_end_idx

    def _predict_sub_block_xyz(self,
                               input_image,
                               crop_indices):
        """Predict sub blocks along xyz

        :param np.array input_image:
        :param list crop_indices:
        """

        pred_imgs_list = []
        start_end_idx = []
        for crop_idx in crop_indices:
            if self.data_format == 'channels_first':
                cur_block = input_image[:, :, crop_idx[0]: crop_indices[1],
                                        crop_idx[2]: crop_indices[3],
                                        crop_idx[4]: crop_indices[5]]
            elif self.data_format == 'channels_last':
                cur_block = input_image[:, crop_idx[0]: crop_indices[1],
                                        crop_idx[2]: crop_indices[3],
                                        crop_idx[4]: crop_indices[5], :]
            pred_block = inference.predict_on_larger_image(
                model=self.model_inst,
                input_image=cur_block
            )
            pred_imgs_list.append(pred_block)
            start_end_idx.append(crop_idx)
        return pred_imgs_list, start_end_idx

    def save_pred_image(self,
                        predicted_image,
                        time_idx, tar_ch_idx, pos_idx, slice_idx):
        """Save predicted images

        :param np.array predicted_image: 2D / 3D predicted image
        :param int time_idx: time index
        :param int tar_ch_idx: target / predicted channel index
        :param int pos_idx: FOV / position index
        :param int slice_idx: slice index
        """

        # Write prediction image
        im_name = aux_utils.get_im_name(
            time_idx=time_idx,
            channel_idx=tar_ch_idx,
            slice_idx=slice_idx,
            pos_idx=pos_idx,
            ext=self.image_ext,
        )
        file_name = os.path.join(self.pred_dir, im_name)
        # save 3D image as npy.
        if len(predicted_image.shape) == 3:
            np.save(file_name, predicted_image, allow_pickle=True)
        else:
            if self.image_ext == '.png':
                # Convert to uint16 for now
                im_pred = 2 ** 16 * \
                          (predicted_image - predicted_image.min()) / \
                          (predicted_image.max() - predicted_image.min())
                im_pred = im_pred.astype(np.uint16)
                cv2.imwrite(file_name, np.squeeze(im_pred))
            elif self.image_ext == '.tif':
                # Convert to float32 and remove batch dimension
                im_pred = predicted_image.astype(np.float32)
                cv2.imwrite(file_name, np.squeeze(im_pred))
            elif self.image_ext == '.npy':
                np.save(file_name, predicted_image, allow_pickle=True)
            else:
                raise ValueError('Unsupported file extension')

    def estimate_metrics(self,
                         cur_target,
                         cur_prediction,
                         cur_pred_fname,
                         cur_row):
        """Estimate evaluation metrics

        :param np.array cur_target:
        :param np.array cur_prediction:
        :param str cur_pred_fname:
        :param pd.Series cur_row:
        """

        kw_args = {'target': cur_target,
                   'prediction': cur_prediction,
                   'pred_fname': cur_pred_fname}

        if self.mask_param_dict is not None:
            # mask_channel = 9
            mask_fname = aux_utils.get_im_name(
                time_idx=cur_row['time_idx'],
                channel_idx=self.mask_param_dict['mask_channel'],
                slice_idx=cur_row['slice_idx'],
                pos_idx=cur_row['pos_idx']
            )
            # mask_dir = '/data/kidney_tiles/mask_channels_2',
            mask_fname = os.path.join(self.mask_param_dict['mask_dir'],
                                      mask_fname)
            cur_mask = np.load(mask_fname)
            cur_mask = np.transpose(cur_mask, [2, 0, 1])
            kw_args['cur_mask'] = cur_mask
        self.metrics_est_inst.estimate_metrics(kw_args)

    def run_prediction(self):
        """Run prediction for entire 2D image or a 3D stack"""

        crop_indices = None
        for ds_idx in range(len(self.dataset_inst)):
            cur_input, cur_target = self.dataset_inst.__getitem__(ds_idx)
            # get blocks with an overlap of one slice
            if self.tile_option == 'infer_on_center':
                inf_shape = self.vol_inf_dict['inf_shape']
                center_block = center_crop_to_shape(cur_input, inf_shape)
                pred_image = inference.predict_on_larger_image(
                    model=self.model_inst, input_image=center_block
                )
            elif self.tile_option == 'tile_z':
                pred_block_list, start_end_idx = \
                    self._predict_sub_block_z(cur_input)
                pred_image = self.snitch_inst.stitch_predictions(
                    cur_input.shape,
                    pred_block_list,
                    start_end_idx
                )
            elif self.tile_option == 'tile_xyz':
                step_size = (self.vol_inf_dict['tile_shape'] -
                             self.num_overlap)
                if crop_indices is None:
                    _, crop_indices = tile_utils.tile_image(
                        input_image=cur_input,
                        tile_size=self.vol_inf_dict['tile_shape'],
                        step_size=step_size,
                        return_index=True
                    )
                pred_block_list, crop_indices = \
                    self._predict_sub_block_xyz(cur_input, crop_indices)
                pred_image = \
                    self.snitch_inst.stitch_predictions(cur_input.shape,
                                                        pred_block_list,
                                                        crop_indices)
            else:
                pred_image = inference.predict_on_larger_image(
                    model=self.model_inst, input_image=cur_input
                )

            pred_image = np.squeeze(pred_image)
            cur_target = np.squeeze(cur_target)
            cur_row = self.df_iteration_meta.iloc[ds_idx]
            pred_fname = self.save_pred_image(
                predicted_image=pred_image,
                time_idx=cur_row['time_idx'],
                tar_ch_idx=cur_row['channel_idx'],
                pos_idx=cur_row['pos_idx'],
                slice_idx=cur_row['slice_idx']
            )
            if self.metrics_est_inst is not None:
                self.estimate_metrics(cur_target,
                                      pred_image,
                                      pred_fname,
                                      cur_row)
        if self.metrics_est_inst is not None:
            df_metrics = self.metrics_est_inst.get_metrics_df()
            df_metrics.to_csv(
                os.path.join(self.config['trainer']['model_dir'],
                             'test_metrics.csv'),
                sep=','
            )
