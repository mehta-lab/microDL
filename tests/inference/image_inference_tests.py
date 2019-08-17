import cv2
import numpy as np
import os
from testfixtures import TempDirectory
import unittest
from unittest.mock import patch

import micro_dl.inference.image_inference as image_inference
import micro_dl.utils.aux_utils as aux_utils


class TestImageInference(unittest.TestCase):

    @patch('micro_dl.inference.model_inference.load_model')
    def setUp(self, mock_model):
        """
        Set up a directory with images
        """
        mock_model.return_value = 'dummy_model'

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.tempdir.makedir('image_dir')
        self.tempdir.makedir('mask_dir')
        self.tempdir.makedir('model_dir')
        self.image_dir = os.path.join(self.temp_path, 'image_dir')
        self.mask_dir = os.path.join(self.temp_path, 'mask_dir')
        self.model_dir = os.path.join(self.temp_path, 'model_dir')
        # Create a temp image dir
        im = np.zeros((10, 16), dtype=np.uint8)
        self.frames_meta = aux_utils.make_dataframe()
        self.time_idx = 2
        self.slice_idx = 3
        for p in range(5):
            for c in range(3):
                im_name = aux_utils.get_im_name(
                    time_idx=self.time_idx,
                    channel_idx=c,
                    slice_idx=self.slice_idx,
                    pos_idx=p,
                    ext='.png',
                )
                cv2.imwrite(os.path.join(self.image_dir, im_name), im + c * 10)
                self.frames_meta = self.frames_meta.append(
                    aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                    ignore_index=True,
                )
        # Write frames meta to image dir too
        self.frames_meta.to_csv(os.path.join(self.image_dir, 'frames_meta.csv'))
        # Save masks and mask meta
        self.mask_meta = aux_utils.make_dataframe()
        self.mask_channel = 50
        for p in range(5):
            im_name = aux_utils.get_im_name(
                time_idx=2,
                channel_idx=self.mask_channel,
                slice_idx=3,
                pos_idx=p,
                ext='.png',
            )
            cv2.imwrite(os.path.join(self.mask_dir, im_name), im + 1)
            self.mask_meta = self.mask_meta.append(
                aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                ignore_index=True,
            )
        # Write frames meta to mask dir too
        self.mask_meta.to_csv(os.path.join(self.mask_dir, 'frames_meta.csv'))
        # Setup model dir
        split_samples = {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        }
        aux_utils.write_json(
            split_samples,
            os.path.join(self.model_dir, 'split_samples.json'),
        )
        # Select inference split of dataset
        self.split_col_ids = ('pos_idx', [1, 3])
        # Make configs with fields necessary for 2D segmentation inference
        self.train_config = {
            'network': {
                'class': 'UNet2D',
                'data_format': 'channels_first',
                'depth': 1,
                'width': 10,
                'height': 10},
            'dataset': {
                'split_by_column': 'pos_idx',
                'input_channels': [1],
                'target_channels': [self.mask_channel],
                'model_task': 'segmentation',
            },
        }
        self.inference_config = {
            'model_dir': self.model_dir,
            'model_fname': 'dummy_weights.hdf5',
            'image_dir': self.image_dir,
            'data_split': 'test',
            'images': {
                'image_format': 'zyx',
                'image_ext': '.png',
            },
            'metrics': {
                'metrics': ['mae'],
                'metrics_orientations': ['xy'],
            },
            'masks': {
                'mask_dir': self.mask_dir,
                'mask_type': 'target',
                'mask_channel': 50,
            }
        }
        # Instantiate class
        self.infer_inst = image_inference.ImagePredictor(
            train_config=self.train_config,
            inference_config=self.inference_config,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        self.assertEqual(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test init of inference dataset
        """
        # Check proper init
        self.assertEqual(self.infer_inst.model_dir, self.model_dir)
        self.assertEqual(self.infer_inst.image_dir, self.image_dir)
        self.assertEqual(self.infer_inst.data_format, 'channels_first')
        self.assertEqual(self.infer_inst.model, 'dummy_model')
        self.assertEqual(self.infer_inst.image_format, 'zyx')
        self.assertEqual(self.infer_inst.image_ext, '.png')
        self.assertFalse(self.infer_inst.mask_metrics)
        self.assertEqual(self.infer_inst.mask_dir, self.mask_dir)
        self.assertListEqual(self.infer_inst.metrics_orientations, ['xy'])
        self.assertEqual(self.infer_inst.num_overlap, 0)
        self.assertIsNone(self.infer_inst.stitch_inst)
        self.assertIsNone(self.infer_inst.tile_option)
        self.assertIsNone(self.infer_inst.crop_shape)

    def test_get_split_ids(self):
        split_col, infer_ids = self.infer_inst._get_split_ids()
        self.assertEqual(split_col, 'pos_idx')
        self.assertListEqual(infer_ids, [3, 4])

    def test_get_split_ids_no_json(self):
        self.infer_inst.model_dir = self.infer_inst.image_dir
        split_col, infer_ids = self.infer_inst._get_split_ids()
        self.assertEqual(split_col, 'pos_idx')
        self.assertListEqual(infer_ids, [0, 1, 2, 3, 4])

    def test_save_pred_image(self):
        im = np.zeros((1, 10, 15), dtype=np.uint8)
        im[:, 5, :] = 128
        self.infer_inst.save_pred_image(
            predicted_image=im,
            time_idx=10,
            target_channel_idx=20,
            pos_idx=30,
            slice_idx=40,
        )
        pred_name = os.path.join(
            self.model_dir,
            'predictions/im_c020_z040_t010_p030.png',
        )
        im_pred = cv2.imread(pred_name, cv2.IMREAD_ANYDEPTH)
        self.assertEqual(im_pred.dtype, np.uint16)
        self.assertTupleEqual(im_pred.shape, (10, 15))
        # Prediction intensities are maximized to range
        self.assertEqual(im_pred.max(), 65535)
        self.assertEqual(im_pred.min(), 0)

    def test_estimate_metrics_xy(self):
        target = np.ones((10, 15, 5), dtype=np.float32)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        self.infer_inst.estimate_metrics(target, prediction, 'test_name', None)
        metrics = self.infer_inst.df_xy
        self.assertTupleEqual(metrics.shape, (5, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae.mean(), 0.5)

    def test_estimate_metrics_xyz(self):
        target = np.ones((10, 15, 5), dtype=np.float32)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        self.infer_inst.metrics_orientations = ['xyz']
        self.infer_inst.estimate_metrics(target, prediction, 'test_name', None)
        metrics = self.infer_inst.df_xyz
        self.assertTupleEqual(metrics.shape, (1, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae[0], 0.5)
        self.assertEqual(metrics.pred_name[0], 'test_name')

    def test_estimate_metrics_xz(self):
        target = np.ones((10, 15, 5), dtype=np.float32)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        self.infer_inst.metrics_orientations = ['xz']
        self.infer_inst.estimate_metrics(target, prediction, 'test_name', None)
        metrics = self.infer_inst.df_xz
        self.assertTupleEqual(metrics.shape, (10, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae[0], 0.0)
        self.assertEqual(metrics.mae[5], 1.0)
        self.assertEqual(metrics.pred_name[9], 'test_name_xz9')

    def test_estimate_metrics_yz(self):
        target = np.ones((10, 15, 5), dtype=np.float32)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        self.infer_inst.metrics_orientations = ['yz']
        self.infer_inst.estimate_metrics(target, prediction, 'test_name', None)
        metrics = self.infer_inst.df_yz
        self.assertTupleEqual(metrics.shape, (15, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae[0], 0.5)
        self.assertEqual(metrics.pred_name[14], 'test_name_yz14')

