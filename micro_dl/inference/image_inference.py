"""Model inference at the image/volume level"""
import cv2
import natsort
import numpy as np
import os
import pandas as pd
import pdb
from micro_dl.input.inference_dataset import InferenceDataSet
import micro_dl.inference.model_inference as inference
from micro_dl.inference.evaluation_metrics import MetricsEstimator
from micro_dl.inference.stitch_predictions import ImageStitcher
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.image_utils import center_crop_to_shape
from micro_dl.utils.train_utils import set_keras_session
import micro_dl.utils.tile_utils as tile_utils


class ImagePredictor:
    """Infer on larger images"""

    def __init__(self,
                 train_config,
                 model_dir,
                 model_fname,
                 image_dir,
                 inference_config,
                 gpu_id,
                 gpu_mem_frac,
                 data_split='test'):
        """Init

        :param dict train_config: Training config dict with params related
            to dataset, trainer and network
        :param str model_dir: Path to model directory
        :param str model_fname: File name of weights in model dir (.hdf5).
        :param str image_dir: dir containing input images AND NOT TILES!
        :param dict inference_config: dict of dicts:
            dict images:
             str image_format: 'zyx' or 'xyz'
             str/None flat_field_dir: flatfield directory
             str im_ext: e.g. '.png' or '.npy' or '.tiff'
             FOR 3D IMAGES USE NPY AS PNG AND TIFF ARE CURRENTLY NOT SUPPORTED.
             list crop_shape: center crop the image to a specified shape before
             tiling for inference
            dict metrics:
             list metrics_list: list of metrics to estimate. available
             metrics: [ssim, corr, r2, mse, mae}]
             list metrics_orientations: xy, xyz, xz or yz
            dict masks: dict with keys
             str mask_dir: path to masks
             str mask_type: 'target' for segmentation, 'metrics' for weighted
             int mask_channel: mask channel as in training
            dict inference_3d: dict with params for 3D inference with keys:
             num_slices, inf_shape, tile_shape, num_overlap, overlap_operation.
             int num_slices: in case of 3D, the full volume will not fit in GPU
              memory, specify the number of slices to use and this will depend on
              the network depth, for ex 8 for a network of depth 4. inf_shape -
              inference on a center sub volume.
             list tile_shape: shape of tile for tiling along xyz.
             int/list num_overlap: int for tile_z, list for tile_xyz
             str overlap_operation: e.g. 'mean'
        :param int gpu_id: gpu to use
        :param float gpu_mem_frac: Memory fractions to use corresponding
         to gpu_ids
        :param str data_split: Which data (train/test/val) to run inference on.
         (default = test)
         TODO: add accuracy and dice coeff to metrics list
        """
        self.config = train_config
        self.model_dir = model_dir
        self.data_format = self.config['network']['data_format']
        if gpu_id >= 0:
            self.sess = set_keras_session(
                gpu_ids=gpu_id,
                gpu_mem_frac=gpu_mem_frac,
            )

        # create network instance and load weights
        self.model_inst = self._create_model(
            os.path.join(self.model_dir, model_fname),
        )

        assert data_split in ['train', 'val', 'test'], \
            'data_split not in [train, val, test]'
        self.df_split_meta = self._get_split_meta(image_dir, data_split)

        flat_field_dir = None
        images_dict = inference_config['images']
        if 'flat_field_dir' in images_dict:
            flat_field_dir = images_dict['flat_field_dir']
        self.dataset_inst = InferenceDataSet(
            image_dir=image_dir,
            dataset_config=self.config['dataset'],
            network_config=self.config['network'],
            df_meta=self.df_split_meta,
            image_format=images_dict['image_format'],
            flat_field_dir=flat_field_dir,
        )
        # Set defaults
        self.image_format = 'zyx'
        if 'image_format' in images_dict:
            self.image_format = images_dict['image_format']
        self.image_ext = '.png'
        if 'image_ext' in images_dict:
            self.image_ext = images_dict['im_ext']
        crop_shape = None
        if 'crop_shape' in images_dict:
            crop_shape = images_dict['crop_shape']
        self.crop_shape = crop_shape
        self.nrows = self.config['network']['width']
        self.ncols = self.config['network']['height']

        # Create image subdirectory to write predicted images
        self.pred_dir = os.path.join(self.model_dir, 'predictions')
        os.makedirs(self.pred_dir, exist_ok=True)
        # create an instance of MetricsEstimator ??
        self.df_iteration_meta = self.dataset_inst.get_df_iteration_meta()

        # Handle masks as either targets or for masked metrics
        self.masks_dict = None
        self.mask_metrics = False
        self.mask_target_dir = None
        if 'masks' in inference_config:
            self.masks_dict = inference_config['masks']
        if self.masks_dict is not None:
            assert 'mask_channel' in self.masks_dict , 'mask_channel is needed'
            assert 'mask_dir' in self.masks_dict, 'mask_dir is needed'
            assert 'mask_type' in self.masks_dict, \
                'mask_type (target/metrics) is needed'
            if self.masks_dict['mask_type'] == 'metrics':
                # Compute weighted metrics
                self.mask_metrics = True
            else:
                # Use masks as targets for metrics computations
                self.mask_target_dir = self.metrics_dict['mask_dir']

        # Handle metrics config settings
        self.metrics_inst = None
        self.metrics_dict = None
        if 'metrics' in inference_config:
            self.metrics_dict = inference_config['metrics']
        if self.metrics_dict is not None:
            assert 'metrics' in self.metrics_dict,\
                'Must specify with metrics to use'
            self.metrics_inst = MetricsEstimator(
                metrics_list=self.metrics_dict['metrics'],
                masked_metrics=self.mask_metrics,
            )
            self.metrics_orientations = ['xy']
            available_orientations = ['xy', 'xyz', 'xz', 'yz']
            if 'metrics_orientations' in self.metrics_dict:
                self.metrics_orientations = \
                    self.metrics_dict['metrics_orientations']
                assert set(self.metrics_orientations).\
                    issubset(available_orientations,),\
                    'orientation not in [xy, xyz, xz, yz]'
            self.df_xy = pd.DataFrame()
            self.df_xyz = pd.DataFrame()
            self.df_xz = pd.DataFrame()
            self.df_yz = pd.DataFrame()

        # Handle 3D volume inference settings
        self.num_overlap = 0
        self.stitch_inst = None
        self.tile_option = None
        self.z_dim = 2
        self.inference_3d_dict = None
        if 'inference_3d' in inference_config:
            self.inference_3d_dict = inference_config['inference_3d']
        if self.inference_3d_dict is not None:
            self._assign_vol_inf_options(
                self.inference_3d_dict,
            )

    def _create_model(self, model_fname):
        """Load model given the model_fname or the saved model in model_dir

        :param str model_fname: fname of the hdf5 file with model weights
        :return keras.Model instance with trained weights loaded
        """
        weights_path = os.path.join(self.model_dir, model_fname)
        # Load model with predict = True
        model = inference.load_model(
            network_config=self.config['network'],
            model_fname=weights_path,
            predict=True,
        )
        return model

    def _get_split_meta(self, image_dir, data_split='test'):
        """Get the meta dataframe for data_split

        :param str image_dir: dir containing images AND NOT TILES!
        :param str data_split: in [train, val, test]
        :return pd.Dataframe df_split_meta: dataframe with slice, pos, time
         indices and file names
        """
        # Load frames metadata and determine indices
        frames_meta = pd.read_csv(
            os.path.join(image_dir, 'frames_meta.csv'),
        )
        split_idx_name = self.config['dataset']['split_by_column']

        if data_split == 'test':
            idx_fname = os.path.join(self.model_dir, 'split_samples.json')
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

    def _assign_vol_inf_options(self, vol_inf_dict):
        """
        Assign inference options for 3D volumes

        tile_z - 2d/3d predictions on full xy extent, stitch predictions along
         z axis
        tile_xyz - 2d/3d prediction on sub-blocks, stitch predictions along xyz
        infer_on_center - infer on center block

        :param dict vol_inf_dict: same as in __init__
        """
        # assign zdim if not Unet2D
        if self.image_format == 'zyx':
            self.z_dim = 2 if self.data_format == 'channels_first' else 1
        elif self.image_format == 'xyz':
            self.z_dim = 4 if self.data_format == 'channels_first' else 3

        if 'num_slices' in vol_inf_dict and vol_inf_dict['num_slices'] > 1:
            self.tile_option = 'tile_z'
            train_num_slices = self.config['network']['depth']
            assert vol_inf_dict['num_slices'] >= train_num_slices, \
                'inference num of slies < num of slices used for training. ' \
                'Inference on reduced num of slices gives sub optimal results'
            num_slices = vol_inf_dict['num_slices']

            assert self.config['network']['class'] == 'UNet3D', \
                'currently stitching predictions available for 3D models only'
            network_depth = len(
                self.config['network']['num_filters_per_block']
            )
            min_num_slices = 2 ** (network_depth - 1)
            assert num_slices >= min_num_slices, \
                'Insufficient number of slices {} for the network ' \
                'depth {}'.format(num_slices, network_depth)
            self.num_overlap = vol_inf_dict['num_overlap'] \
                if 'num_overlap' in vol_inf_dict else 0
        elif 'tile_shape' in vol_inf_dict:
            self.tile_option = 'tile_xyz'
            self.num_overlap = vol_inf_dict['num_overlap'] \
                if 'num_overlap' in vol_inf_dict else [0, 0, 0]
        elif 'inf_shape' in vol_inf_dict:
            self.tile_option = 'infer_on_center'
            self.num_overlap = 0

        # create an instance of ImageStitcher
        if self.tile_option in ['tile_z', 'tile_xyz']:
            overlap_dict = {
                'overlap_shape': self.num_overlap,
                'overlap_operation': vol_inf_dict['overlap_operation']
            }
            self.stitch_inst = ImageStitcher(
                tile_option=self.tile_option,
                overlap_dict=overlap_dict,
                image_format=self.image_format,
                data_format=self.data_format
            )

    def _get_sub_block_z(self,
                         input_image,
                         start_z_idx,
                         end_z_idx):
        """Get the sub block along z given start and end slice indices

        :param np.array input_image: 5D tensor with the entire 3D volume
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
        elif self.image_format == 'zyx' and \
                self.data_format == 'channels_first':
            cur_block = input_image[:, :, start_z_idx: end_z_idx, :, :]
        elif self.image_format == 'zyx' and \
                self.data_format == 'channels_last':
            cur_block = input_image[:, start_z_idx: end_z_idx, :, :, :]
        return cur_block

    def _predict_sub_block_z(self, input_image):
        """Predict sub blocks along z

        :param np.array input_image: 5D tensor with the entire 3D volume
        :return list pred_imgs_list - list of predicted sub blocks
         list start_end_idx - list of tuples with start and end z indices
        """

        pred_imgs_list = []
        start_end_idx = []
        num_z = input_image.shape[self.z_dim]
        num_slices = self.inference_3d_dict['num_slices']
        num_blocks = np.ceil(
            num_z / (num_slices - self.num_overlap)
        ).astype('int')
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
            # reduce predictions from 5D to 3D for simplicity
            pred_imgs_list.append(np.squeeze(pred_block))
            start_end_idx.append((start_idx, end_idx))
        return pred_imgs_list, start_end_idx

    def _predict_sub_block_xyz(self,
                               input_image,
                               crop_indices):
        """Predict sub blocks along xyz

        :param np.array input_image: 5D tensor with the entire 3D volume
        :param list crop_indices: list of crop indices
        :return list pred_imgs_list - list of predicted sub blocks
        """

        pred_imgs_list = []
        for crop_idx in crop_indices:
            if self.data_format == 'channels_first':
                cur_block = input_image[:, :, crop_idx[0]: crop_idx[1],
                                        crop_idx[2]: crop_idx[3],
                                        crop_idx[4]: crop_idx[5]]
            elif self.data_format == 'channels_last':
                cur_block = input_image[:, crop_idx[0]: crop_idx[1],
                                        crop_idx[2]: crop_idx[3],
                                        crop_idx[4]: crop_idx[5], :]
            pred_block = inference.predict_on_larger_image(
                model=self.model_inst,
                input_image=cur_block
            )
            # retain the full 5D tensor to experiment for multichannel case
            pred_imgs_list.append(pred_block)
        return pred_imgs_list

    def save_pred_image(self,
                        predicted_image,
                        time_idx,
                        target_channel_idx,
                        pos_idx,
                        slice_idx):
        """
        Save predicted images

        :param np.array predicted_image: 2D / 3D predicted image
        :param int time_idx: time index
        :param int target_channel_idx: target / predicted channel index
        :param int pos_idx: FOV / position index
        :param int slice_idx: slice index
        """

        # Write prediction image
        im_name = aux_utils.get_im_name(
            time_idx=time_idx,
            channel_idx=target_channel_idx,
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
        return

    def estimate_metrics(self,
                         cur_target,
                         cur_prediction,
                         cur_pred_fname,
                         cur_mask):
        """
        Estimate evaluation metrics
        The row of metrics gets added to metrics_est.df_metrics

        :param np.array cur_target: ground truth
        :param np.array cur_prediction: model prediction
        :param str cur_pred_fname: fname for saving model predictions
        :param np.array cur_mask: foreground/ background mask
        """

        kw_args = {'target': cur_target,
                   'prediction': cur_prediction,
                   'pred_name': cur_pred_fname}

        if cur_mask is not None:
            kw_args['mask'] = cur_mask

        if 'xy' in self.metrics_orientations:
            self.metrics_inst.estimate_xy_metrics(**kw_args)
            self.df_xy = self.df_xy.append(
                self.metrics_inst.get_metrics_xy()
            )
        if 'xyz' in self.metrics_orientations:
            self.metrics_inst.estimate_xyz_metrics(**kw_args)
            self.df_xyz = self.df_xyz.append(
                self.metrics_inst.get_metrics_xyz()
            )
        if 'xz' in self.metrics_orientations:
            self.metrics_inst.estimate_xz_metrics(**kw_args)
            self.df_xz = self.df_xz.append(
                self.metrics_inst.get_metrics_xz()
            )
        if 'yz' in self.metrics_orientations:
            self.metrics_inst.estimate_yz_metrics(**kw_args)
            self.df_yz = self.df_yz.append(
                self.metrics_inst.get_metrics_yz()
            )

    def get_mask(self, cur_row, transpose=False):
        """Get mask"""

        mask_fname = aux_utils.get_im_name(
            time_idx=cur_row['time_idx'],
            channel_idx=self.masks_dict['mask_channel'],
            slice_idx=cur_row['slice_idx'],
            pos_idx=cur_row['pos_idx']
        )
        mask_fname = os.path.join(
            self.masks_dict['mask_dir'],
            mask_fname)
        cur_mask = np.load(mask_fname)
        # moves z from last axis to first axis
        if transpose:
            cur_mask = np.transpose(cur_mask, [2, 0, 1])
        if self.crop_shape is not None:
            cur_mask = center_crop_to_shape(cur_mask,
                                            self.crop_shape)
        return cur_mask

    def run_prediction(self):
        """Run prediction for entire 2D image or a 3D stack"""

        crop_indices = None
        df_iteration_meta = self.dataset_inst.get_df_iteration_meta()
        pos_idx = df_iteration_meta['pos_idx'].unique()
        for idx, cur_pos_idx in enumerate(pos_idx):
            print(cur_pos_idx, ',{}/{}'.format(idx, len(pos_idx)))
            df_iter_meta_row_idx = df_iteration_meta.index[
                df_iteration_meta['pos_idx'] == cur_pos_idx
            ].values
            if self.tile_option is None:
                # 2D, 2.5D
                max_sl = \
                    df_iteration_meta[df_iter_meta_row_idx]['slice_idx'].max()
                min_sl = \
                    df_iteration_meta[df_iter_meta_row_idx]['slice_idx'].min()
                pred_vol = np.zeros([self.nrows, self.ncols, max_sl],
                                    dtype='float32')
                tar_vol = np.zeros_like(pred_vol)
                mask_vol = np.zeros([self.nrows, self.ncols, max_sl],
                                    dtype='bool')

                for row_idx in df_iter_meta_row_idx:
                    cur_input, cur_target = \
                        self.dataset_inst.__getitem__(row_idx)
                    if self.crop_shape is not None:
                        cur_input = center_crop_to_shape(
                            cur_input,
                            self.crop_shape,
                        )
                        cur_target = center_crop_to_shape(
                            cur_target,
                            self.crop_shape,
                        )
                    pred_image = inference.predict_on_larger_image(
                        model=self.model_inst, input_image=cur_input
                    )
                    pred_image = np.squeeze(pred_image)

                    # save prediction
                    cur_row = self.df_iteration_meta.iloc[row_idx]
                    self.save_pred_image(
                        predicted_image=pred_image,
                        time_idx=cur_row['time_idx'],
                        target_channel_idx=cur_row['channel_idx'],
                        pos_idx=cur_row['pos_idx'],
                        slice_idx=cur_row['slice_idx']
                    )

                    cur_sl = df_iteration_meta[row_idx]['slice_idx']
                    # get mask
                    if self.masks_dict is not None:
                        cur_mask = self.get_mask(cur_row)
                        mask_vol[:, :, cur_sl] = cur_mask

                    # add to vol
                    pred_vol[:, :, cur_sl] = pred_image
                    tar_vol[:, :, cur_sl] = np.squeeze(cur_target)
                pred_image = pred_vol[:, :, min_sl:]
                target_image = tar_vol[:, :, min_sl:]
                mask_vol = mask_vol[:, :, min_sl:]

            else:  # 3D
                assert len(df_iter_meta_row_idx) == 1, \
                    'more than one matching row found for position ' \
                    '{}'.format(cur_pos_idx)
                cur_input, cur_target = \
                    self.dataset_inst.__getitem__(df_iter_meta_row_idx[0])
                if self.crop_shape is not None:
                    cur_input = center_crop_to_shape(cur_input,
                                                     self.crop_shape)
                    cur_target = center_crop_to_shape(cur_target,
                                                      self.crop_shape)
                if self.tile_option == 'infer_on_center':
                    inf_shape = self.inference_3d_dict['inf_shape']
                    center_block = center_crop_to_shape(cur_input, inf_shape)
                    cur_target = center_crop_to_shape(cur_target, inf_shape)
                    pred_image = inference.predict_on_larger_image(
                        model=self.model_inst, input_image=center_block
                    )
                elif self.tile_option == 'tile_z':
                    pred_block_list, start_end_idx = \
                        self._predict_sub_block_z(cur_input)
                    pred_image = self.stitch_inst.stitch_predictions(
                        np.squeeze(cur_input).shape,
                        pred_block_list,
                        start_end_idx
                    )
                elif self.tile_option == 'tile_xyz':
                    step_size = (np.array(self.inference_3d_dict['tile_shape']) -
                                 np.array(self.num_overlap))
                    if crop_indices is None:
                        # TODO tile_image works for 2D/3D imgs, modify for multichannel
                        _, crop_indices = tile_utils.tile_image(
                            input_image=np.squeeze(cur_input),
                            tile_size=self.inference_3d_dict['tile_shape'],
                            step_size=step_size,
                            return_index=True
                        )
                    pred_block_list = self._predict_sub_block_xyz(cur_input,
                                                                  crop_indices)
                    pred_image = self.stitch_inst.stitch_predictions(
                        np.squeeze(cur_input).shape,
                        pred_block_list,
                        crop_indices
                    )
                pred_image = np.squeeze(pred_image)
                target_image = np.squeeze(cur_target)
                # save prediction
                cur_row = self.df_iteration_meta.iloc[df_iter_meta_row_idx[0]]
                self.save_pred_image(predicted_image=pred_image,
                                     time_idx=cur_row['time_idx'],
                                     target_channel_idx=cur_row['channel_idx'],
                                     pos_idx=cur_row['pos_idx'],
                                     slice_idx=cur_row['slice_idx'])
                # get mask
                if self.masks_dict is not None:
                    mask_vol = self.get_mask(cur_row, transpose=True)
                # 3D uses zyx, estimate metrics expects xyz
                pred_image = np.transpose(pred_image, [1, 2, 0])
                target_image = np.transpose(target_image, [1, 2, 0])
                mask_vol = np.transpose(mask_vol, [1, 2, 0])

            pred_fname = 'im_t{}_c{}_p{}'.format(cur_row['time_idx'],
                                                 cur_row['channel_idx'],
                                                 cur_row['pos_idx'])
            if self.metrics_inst is not None:
                if self.masks_dict is None:
                    mask_vol = None
                self.estimate_metrics(cur_target=target_image,
                                      cur_prediction=pred_image,
                                      cur_pred_fname=pred_fname,
                                      cur_mask=mask_vol)
            del pred_image, target_image
            if self.metrics_inst is not None:
                metrics_mapping = {
                    'xy': self.df_xy,
                    'xz': self.df_xz,
                    'yz': self.df_yz,
                    'xyz': self.df_xyz,
                }
                for orientation in self.metrics_orientations:
                    metrics_df = metrics_mapping[orientation]
                    df_name = 'metrics_{}.csv'.format(orientation)
                    metrics_df.to_csv(
                        os.path.join(self.model_dir, df_name),
                        sep=','
                    )
