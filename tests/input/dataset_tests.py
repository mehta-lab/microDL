import cv2
import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.input.dataset as dataset
import micro_dl.utils.aux_utils as aux_utils


class TestBaseDataSet(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory for tiling with flatfield, no mask
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        input_fnames = pd.Series(['in1.npy',
                                  'in2.npy',
                                  'in3.npy',
                                  'in4.npy'])
        target_fnames = pd.Series(['out1.npy',
                                   'out2.npy',
                                   'out3.npy',
                                   'out4.npy'])
        self.batch_size = 2
        # Make artificial image
        im = np.zeros((3, 5, 2))
        im[:, :3, 0] = np.diag([1, 2, 3])
        im[:2, :2, 1] = 1
        for i, (in_name, out_name) in enumerate(zip(input_fnames, target_fnames)):
            np.save(os.path.join(self.temp_path, in_name), im + i)
            np.save(os.path.join(self.temp_path, out_name), im + i)
        # Instantiate class
        self.data_inst = dataset.BaseDataSet(
            tile_dir=self.temp_path,
            input_fnames=input_fnames,
            target_fnames=target_fnames,
            batch_size=self.batch_size,
            augmentations=True,
            normalize=False,
            data_format='channels_last',
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
        nose.tools.assert_equal(self.data_inst.tile_dir, self.temp_path)


