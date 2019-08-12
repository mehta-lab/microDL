import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.input.inference_dataset as inference_dataset


class TestInferenceDataSet(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with images
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.input_fnames = pd.Series(['in1.npy',
                                  'in2.npy',
                                  'in3.npy',
                                  'in4.npy'])
        self.target_fnames = pd.Series(['out1.npy',
                                   'out2.npy',
                                   'out3.npy',
                                   'out4.npy'])
        self.batch_size = 2
        # Normally, tiles would have the same shape in x, y but this helps
        # us test augmentations
        self.im = np.zeros((5, 7, 3))
        self.im[:, :5, 0] = np.diag([1, 2, 3, 4, 5])
        self.im[:4, :4, 1] = 1

        self.im_target = np.zeros((5, 7, 3))
        self.im_target[:4, :4, 0] = 1
        # Batch size is 2, input images of shape (5, 7, 3)
        # stack adds singleton dimension
        self.batch_shape = (2, 1, 5, 7, 3)
        for i, (in_name, out_name) in enumerate(zip(self.input_fnames,
                                                    self.target_fnames)):
            np.save(os.path.join(self.temp_path, in_name), self.im + i)
            np.save(os.path.join(self.temp_path, out_name), self.im_target + i)
        # Fix these
        dataset_config = {
            'input_channels': 5,
            'target_channels': 3,
        }
        network_config = {
            'class': 'UNet2D',
            'network': {'depth': 1},
            'data_format': 'channels_first',
        }
        df_meta = None
        # Instantiate class
        self.data_inst = inference_dataset.BaseDataSet(
            image_dir=self.temp_path,
            dataset_config=dataset_config,
            network_config=network_config,
            df_meta=df_meta,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test image tiler on frames temporary dir
        """
        nose.tools.assert_equal(self.data_inst.image_dir, self.temp_path)
        self.assertListEqual(
            self.input_fnames.tolist(),
            self.data_inst.input_fnames.tolist(),
        )
        self.assertListEqual(
            self.target_fnames.tolist(),
            self.data_inst.target_fnames.tolist(),
        )
        nose.tools.assert_equal(
            self.data_inst.num_samples,
            len(self.input_fnames),
        )
        nose.tools.assert_equal(self.data_inst.model_task, 'regression')
