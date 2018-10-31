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
        self.im = np.zeros((3, 5, 2))
        self.im[:, :3, 0] = np.diag([1, 2, 3])
        self.im[:2, :2, 1] = 1
        for i, (in_name, out_name) in enumerate(zip(self.input_fnames,
                                                    self.target_fnames)):
            np.save(os.path.join(self.temp_path, in_name), self.im + i)
            np.save(os.path.join(self.temp_path, out_name), self.im + i)
        # Instantiate class
        self.data_inst = dataset.BaseDataSet(
            tile_dir=self.temp_path,
            input_fnames=self.input_fnames,
            target_fnames=self.target_fnames,
            batch_size=self.batch_size,
            augmentations=True,
            random_seed=42,
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
        self.assertListEqual(
            self.input_fnames.tolist(),
            self.data_inst.input_fnames.tolist(),
        )
        self.assertListEqual(
            self.target_fnames.tolist(),
            self.data_inst.target_fnames.tolist(),
        )
        nose.tools.assert_equal(self.data_inst.batch_size, self.batch_size)
        nose.tools.assert_true(self.data_inst.shuffle)
        nose.tools.assert_equal(
            self.data_inst.num_samples,
            len(self.input_fnames),
        )
        nose.tools.assert_true(self.data_inst.augmentations)
        nose.tools.assert_equal(self.data_inst.model_task, 'regression')
        nose.tools.assert_equal(self.data_inst.random_seed, 42)
        nose.tools.assert_false(self.data_inst.normalize)
        nose.tools.assert_equal(self.data_inst.data_format, 'channels_last')

    def test__len__(self):
        nbr_batches = self.data_inst.__len__()
        expected_batches = len(self.input_fnames) / self.batch_size
        nose.tools.assert_equal(nbr_batches, expected_batches)

    def test_augment_image_asis(self):
        trans_im = self.data_inst._augment_image(self.im, 0)
        np.testing.assert_array_equal(trans_im, self.im)

    def test_augment_image_lr(self):
        trans_im = self.data_inst._augment_image(self.im, 1)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.fliplr(self.im[..., i]),
            )

    def test_augment_image_ud(self):
        trans_im = self.data_inst._augment_image(self.im, 2)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.flipud(self.im[..., i]),
            )

    def test_augment_image_rot90(self):
        trans_im = self.data_inst._augment_image(self.im, 3)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.rot90(self.im[..., i], k=1),
            )

    def test_augment_image_rot180(self):
        trans_im = self.data_inst._augment_image(self.im, 4)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.rot90(self.im[..., i], k=2),
            )

    def test_augment_image_rot270(self):
        trans_im = self.data_inst._augment_image(self.im, 5)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.rot90(self.im[..., i], k=3),
            )

    @nose.tools.raises(ValueError)
    def test_augment_image_6(self):
        self.data_inst._augment_image(self.im, 6)

    @nose.tools.raises(ValueError)
    def test_augment_image_m1(self):
        self.data_inst._augment_image(self.im, -1)

    def test_get_volume(self):
        image_volume = self.data_inst._get_volume(self.input_fnames[0:2])
        print(image_volume.shape)
