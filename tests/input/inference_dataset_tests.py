import cv2
import numpy as np
import os
from testfixtures import TempDirectory
import unittest

import micro_dl.input.inference_dataset as inference_dataset
import micro_dl.utils.aux_utils as aux_utils


class TestInferenceDataSet(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with images
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        # Create a temp image dir
        im = np.zeros((10, 15), dtype=np.uint8)
        self.frames_meta = aux_utils.make_dataframe()
        for p in range(5):
            im_name = aux_utils.get_im_name(
                time_idx=2,
                channel_idx=1,
                slice_idx=3,
                pos_idx=p,
                ext='.png',
            )
            cv2.imwrite(os.path.join(self.temp_path, im_name), im)
            self.frames_meta = self.frames_meta.append(
                aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                ignore_index=True,
            )

        # Fix these
        dataset_config = {
            'input_channels': [1],
            'target_channels': [999],
            'model_task': 'segmentation',
        }
        network_config = {
            'class': 'UNet2D',
            'network': {'depth': 1},
            'data_format': 'channels_first',
        }
        # Instantiate class
        self.data_inst = inference_dataset.InferenceDataSet(
            image_dir=self.temp_path,
            dataset_config=dataset_config,
            network_config=network_config,
            df_meta=self.frames_meta,
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
        self.assertEqual(self.data_inst.image_dir, self.temp_path)
        self.assertIsNone(self.data_inst.flat_field_dir)
        self.assertEqual(self.data_inst.image_format, 'zyx')
        self.assertEqual(self.data_inst.model_task, 'segmentation')
        self.assertEqual(self.data_inst.depth, 1)
        self.assertTupleEqual(self.data_inst.df_meta.shape, self.frames_meta.shape)
        self.assertTrue(self.data_inst.squeeze)
        self.assertFalse(self.data_inst.im_3d)
        self.assertEqual(self.data_inst.data_format, 'channels_first')
        self.assertListEqual(self.data_inst.input_channels, [1])
        self.assertListEqual(self.data_inst.target_channels, [999])
