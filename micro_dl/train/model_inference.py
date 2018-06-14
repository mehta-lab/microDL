"""Model inference related functions"""
import numpy as np
import os
from keras import Model

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

    def __init__(self, config, model_fname, gpu_ids=0, gpu_mem_frac=0.95):
        """Init

        :param dict config: dict read from a config yaml. Need network related
         parameters for creating the model
        :param str model_fname: fname with full path of the .hdf5 file
         containing the trained model weights
        :param int/list gpu_ids: gpu to use
        :param float/list gpu_mem_frac: Memory fractions to use corresponding
         to gpu_ids
        """

        self.config = config
        self.model_fname = model_fname
        self.model = load_model(config, model_fname)
        self.sess = train_utils.set_keras_session(gpu_ids=gpu_ids,
                                                  gpu_mem_frac=gpu_mem_frac)

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
        """

        cur_images = []
        for ch in channel_ids:
            cur_fname = os.path.join(tp_dir,
                                     'channel_{}'.format(ch),
                                     fname)
            cur_images.append(np.load(cur_fname))
        cur_images = np.stack(cur_images)
        return cur_images

    def _pred_batch(self, ip_image, crop_indices, overlap_size,
                    predicted_image):
        """Batch images

        :param np.arrray ip_image: input image to be tiled
        ;param list crop_indices: list of tuples with crop indices
        """

        ip_batch_list = [ip_image[cur_index] for cur_index in crop_indices]
        ip_batch = np.stack(ip_batch_list)
        pred_batch = self.model.predict(ip_batch)

        return pred_batch





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
            num_batches = np.ceil(len(crop_indices) / batch_size)

            for fname in test_image_fnames:
                target_image = self._read_one(tp_dir, op_channel_ids, fname)
                input_image = self._read_one(tp_dir, ip_channel_ids, fname)
                predicted_image = np.zeros( )
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = np.min(batch_idx * batch_size,
                                     len(test_image_fnames))
                    pred_batch = self._pred_batch(
                        input_image, crop_indices[start_idx:end_idx],
                        tile_size - step_size,

                    )












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