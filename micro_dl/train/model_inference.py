"""Model inference related functions"""
from keras import Model
import numpy as np
import os
from skimage.io import imsave

import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.image_utils import tile_image
from micro_dl.plotting.plot_utils import save_predicted_images
import micro_dl.utils.train_utils as train_utils


def load_model(config, model_fname):
    """Load the model from model_dir

    Due to the lambda layer only model weights are saved and not the model
    config. Hence load_model wouldn't work here!
    :param yaml config: a yaml file with all the required parameters
    :param str model_fname: fname with full path of the .hdf5 file with saved
     weights
    :return: Keras.Model instance
    """

    network_cls = config['network']['class']
    # not ideal as more networks get added
    network_cls = aux_utils.import_class('networks', network_cls)
    network = network_cls(config)
    inputs, outputs = network.build_net()
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights(model_fname)
    return model


class ModelEvaluator:
    """Evaluates model performance on test set"""

    def __init__(self, config, model=None, model_fname=None,
                 gpu_ids=0, gpu_mem_frac=0.95):
        """Init

        :param dict config: dict read from a config yaml. Need network related
         parameters for creating the model
        :param Keras.Model model: trained model
        :param str model_fname: fname with full path of the .hdf5 file
         containing the trained model weights
        :param int/list gpu_ids: gpu to use
        :param float/list gpu_mem_frac: Memory fractions to use corresponding
         to gpu_ids
        """

        msg = 'either model or model_fname has to be provided but not both'
        assert np.logical_xor(model is not None,
                              model_fname is not None), msg
        self.config = config
        if model is not None:
            self.model = model
        else:
            self.model = load_model(config, model_fname)
        if gpu_ids >= 0:
            self.sess = train_utils.set_keras_session(
                gpu_ids=gpu_ids, gpu_mem_frac=gpu_mem_frac
            )

    def evaluate_model(self, ds_test):
        """Evaluate model performance on the test set

        https://github.com/keras-team/keras/issues/2621

        :param BaseDataSet/DataSetWithMask ds_test: generator used for
         batching test images
        """

        loss = train_utils.get_loss(self.config['trainer']['loss'])
        metrics = train_utils.get_metrics(self.config['trainer']['metrics'])
        # the optimizer here is a dummy and is not used!
        self.model.compile(loss=loss, optimizer='Adam', metric=metrics)
        test_performance = self.model.evaluate_generator(generator=ds_test)
        return test_performance

    @staticmethod
    def _read_one(tp_dir, channel_ids, fname):
        """Read one image set

        :param str tp_dir: timepoint dir
        :param list channel_ids: list of channels to read from
        :param str fname: fname of the image. Expects the fname to be the same
         in all channels
        :returns
        """

        cur_images = []
        for ch in channel_ids:
            cur_fname = os.path.join(tp_dir,
                                     'channel_{}'.format(ch),
                                     fname)
            cur_images.append(np.load(cur_fname))
        cur_images = np.stack(cur_images)
        return cur_images

    @staticmethod
    def _get_crop_indices(ip_image_shape, n_dim, cur_index):
        """Assemble slice indices from the crop_indices

        :param np.array ip_image_shape: shape of input image
        :param int n_dim: dimensionality of the image (in each channel)
        :param tuple cur_index: tuple of indices
        :return: slice array
        """

        cur_index_slice = []
        # for multi-channel input, include all channels
        if len(ip_image_shape) > n_dim:
            cur_index_slice.append(np.s_[:])
        cur_index_slice.append(np.s_[cur_index[0]:cur_index[1]])
        cur_index_slice.append(np.s_[cur_index[2]:cur_index[3]])
        if n_dim == 3:
            cur_index_slice.append(np.s_[cur_index[4]:cur_index[5]])
        return cur_index_slice

    def _pred_image(self, ip_image, crop_indices, batch_size):
        """Batch images

        :param np.array ip_image: input image to be tiled
        :param list crop_indices: list of tuples with crop indices
        :param int batch_size: number of tiles to batch
        :return: list of np.array with shape (batch_size, ip_image.shape)
        """

        tile_dim = int(len(crop_indices[0]) / 2)
        num_batches = np.ceil(len(crop_indices) / batch_size).astype('int')
        pred_tiles = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = np.min([(batch_idx + 1) * batch_size,
                             len(crop_indices)])
            ip_batch_list = []
            for cur_index in crop_indices[start_idx:end_idx]:
                cur_index_slice = self._get_crop_indices(ip_image.shape,
                                                         tile_dim,
                                                         cur_index)
                cropped_image = ip_image[cur_index_slice]
                ip_batch_list.append(cropped_image)

            ip_batch = np.stack(ip_batch_list)
            pred_batch = self.model.predict(ip_batch)
            pred_tiles.append(pred_batch)
        return pred_tiles

    def _place_patch(self, full_image, tile_image, input_image_dim, full_idx,
                     tile_idx, operation='mean'):
        """Place individual patches on the full image

        Allowed operations are insert, mean and max.

        :param np.array full_image: image with the same shape as input image
         and initialized with zeros
        :param np.array tile_image: predicted tile or model inference on the
         tile
        :param int input_image_dim: dimensionality of input image
        :param list full_idx: indices in the full image
        :param list tile_idx: indices in the tile image
        :param str operation: operation on the patch [insert, mean, max]
        :return: np.array modified full_image
        """

        n_dim = int(len(full_idx) / 2)
        tile_slice = self._get_crop_indices(input_image_dim, n_dim, tile_idx)
        full_slice = self._get_crop_indices(input_image_dim, n_dim, full_idx)
        if operation == 'mean':
            full_image[full_slice] = (full_image[full_slice] +
                                      tile_image[tile_slice]) / 2
        elif operation == 'max':
            #  check if np.max does pixelwise comparison
            full_image[full_slice] = np.maximum(full_image[full_slice],
                                                tile_image[tile_slice])
        elif operation == 'insert':
            full_image[full_slice] = tile_image[tile_slice]
        return full_image

    def _stich_image(self, pred_tiles, crop_indices, input_image_shape,
                     batch_size, tile_size, overlap_size):
        """Stiches the full image from predicted tiles

        :param list pred_tiles: list of predicted np.arrays
        :param list crop_indices: list of tuples of crop indices
        :param np.array input_image_shape: shape of the input image
        :param int batch_size:
        :param list tile_size:
        :param list overlap_size: tile_size - step_size
        :return: np.array, stiched predicted image
        """

        predicted_image = np.zeros(input_image_shape)
        for b_idx, pred_batch in enumerate(pred_tiles):
            for i_idx, pred_tile in enumerate(pred_batch):
                c_idx = b_idx * batch_size + i_idx
                cur_index = crop_indices[c_idx]
                n_dim = int(len(cur_index) / 2)
                tile_top_index = [0, overlap_size[0], 0, tile_size[1]]
                full_top_index = [cur_index[0], cur_index[0] + overlap_size[0],
                                  cur_index[2], cur_index[3]]
                tile_left_index = [overlap_size[0], tile_size[0],
                                   0, overlap_size[1]]
                full_left_index = [cur_index[0] + overlap_size[0],
                                   cur_index[1], cur_index[2],
                                   cur_index[2] + overlap_size[1]]
                tile_non_index = [overlap_size[0], tile_size[0],
                                  overlap_size[1], tile_size[1]]
                full_non_index = [cur_index[0] + overlap_size[0], cur_index[1],
                                  cur_index[2] + overlap_size[1], cur_index[3]]
                if n_dim == 3:
                    tile_top_index.extend([0, tile_size[2]])
                    tile_left_index.extend([0, tile_size[2]])
                    tile_front_index = tile_non_index.copy()
                    tile_front_index.extend([0, overlap_size[2]])
                    tile_non_index.extend([overlap_size[2], tile_size[2]])

                    full_top_index.extend([cur_index[4], cur_index[5]])
                    full_left_index.extend([cur_index[4], cur_index[5]])
                    full_front_index = full_non_index.copy()
                    full_front_index.extend([cur_index[4],
                                             cur_index[4] + overlap_size[2]])
                    full_non_index.extend([cur_index[4] + overlap_size[2],
                                           cur_index[5]])
                chk_index = np.array(cur_index)[[0, 2]]
                place_operation = np.empty_like(chk_index, dtype='U8')
                place_operation[chk_index == 0] = 'insert'
                place_operation[chk_index != 0] = 'mean'
                print(chk_index, cur_index, place_operation)
                self._place_patch(predicted_image, pred_tile,
                                  input_image_shape, full_top_index,
                                  tile_top_index,
                                  operation=place_operation[0])

                self._place_patch(predicted_image, pred_tile,
                                  input_image_shape, full_left_index,
                                  tile_left_index,
                                  operation=place_operation[1])

                self._place_patch(predicted_image, pred_tile,
                                  input_image_shape, full_non_index,
                                  tile_non_index, operation='insert')
                if n_dim == 3:
                    if cur_index[4] == 0:
                        place_operation = 'insert'
                    else:
                        place_operation = 'mean'
                    self._place_patch(predicted_image, pred_tile,
                                      input_image_shape, full_front_index,
                                      tile_front_index,
                                      operation=place_operation)
        return predicted_image

    def predict_on_full_image(self, image_meta, test_sample_idx,
                              focal_plane_idx=None, depth=None,
                              per_tile_overlap=1/8):
        """Tile and run inference on tiles and assemble the full image

        If 3D and isotropic, it is not possible to find the original
        tile_size i.e. depth from config used for training

        :param pd.DataFrame image_meta: Df with individual image info,
         timepoint', 'channel_num', 'sample_num', 'slice_num', 'fname',
         'size_x_microns', 'size_y_microns', 'size_z_microns'
        :param list test_sample_idx: list of sample numbers to be used in the
         test set
        :param int focal_plane_idx: focal plane to be used
        :param int depth: if 3D - num of slices used for tiling
        :param float per_tile_overlap: percent overlap between successive tiles
        """
        if 'timepoints' not in self.config['dataset']:
            timepoint_ids = -1
        else:
            timepoint_ids = self.config['dataset']['timepoints']

        ip_channel_ids = self.config['dataset']['input_channels']
        op_channel_ids = self.config['dataset']['target_channels']
        tp_channel_ids = aux_utils.validate_tp_channel(
            image_meta, timepoint_ids=timepoint_ids
        )
        tp_idx = tp_channel_ids['timepoints']
        tile_size = [self.config['network']['height'],
                     self.config['network']['width']]

        isotropic = False
        if depth is not None:
            assert 'depth' in self.config['network']
            tile_size.insert(0, depth)
            if depth == self.config['network']['depth']:
                isotropic = False  # no need to resample
            else:
                isotropic = True

        step_size = (1 - per_tile_overlap) * np.array(tile_size)
        step_size = step_size.astype('int')
        step_size[step_size < 1] = 1

        overlap_size = tile_size - step_size
        batch_size = self.config['trainer']['batch_size']

        for tp in tp_idx:
            # get the meta for all images in tp_dir and channel_dir
            row_idx_ip0 = aux_utils.get_row_idx(
                image_meta, tp, ip_channel_ids[0],
                focal_plane_idx=focal_plane_idx
            )
            ip0_meta = image_meta[row_idx_ip0]

            # get rows corr. to test_sample_idx from this DF
            test_row_ip0 = ip0_meta.loc[
                ip0_meta['sample_num'].isin(test_sample_idx)
            ]
            test_ip0_fnames = test_row_ip0['fname'].tolist()
            test_image_fnames = (
                [fname.split(os.sep)[-1] for fname in test_ip0_fnames]
            )
            tp_dir = str(os.sep).join(test_ip0_fnames[0].split(os.sep)[:-2])
            test_image = np.load(test_ip0_fnames[0])
            _, crop_indices = tile_image(test_image, tile_size, step_size,
                                           isotropic, return_index=True)
            pred_dir = os.path.join(self.config['trainer']['model_dir'],
                                    'predicted_images', 'tp_{}'.format(tp))
            for fname in test_image_fnames:
                target_image = self._read_one(tp_dir, op_channel_ids, fname)
                input_image = self._read_one(tp_dir, ip_channel_ids, fname)
                pred_tiles = self._pred_image(input_image,
                                              crop_indices,
                                              batch_size)
                pred_image = self._stich_image(pred_tiles, crop_indices,
                                               input_image.shape, batch_size,
                                               tile_size, overlap_size)
                pred_fname = '{}.tiff'.format(fname.split('.')[0])
                # in which format to write 3D images - nifti perhaps?
                for idx, op_ch in enumerate(op_channel_ids):
                    op_dir = os.path.join(pred_dir, 'channel_{}'.format(op_ch))
                    imsave(os.path.join(op_dir, pred_fname),
                           np.squeeze(pred_image[idx]))
                    save_predicted_images(input_image, target_image,
                                          pred_image, op_dir, pred_fname)


def predict_and_save_images(config, model_fname, ds_test, model_dir):
    """Run inference on images

    :param yaml config: config used to train the model
    :param str model_fname: fname with full path for the saved model
     (.hdf5)
    :param dataset ds_test: generator for the test set
    :param str model_dir: dir where model results are to be saved
    """

    model = load_model(config, model_fname)
    output_dir = os.path.join(model_dir, 'test_predictions')
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx in range(ds_test.__len__()):
        if 'weighted_loss' in config['trainer']:
            cur_input, cur_target, cur_mask = ds_test.__getitem__(batch_idx)
        else:
            cur_input, cur_target = ds_test.__getitem__(batch_idx)
        pred_batch = model.predict(cur_input)
        save_predicted_images(cur_input, cur_target, pred_batch,
                              output_dir, batch_idx)