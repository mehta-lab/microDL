import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.meta_utils as meta_utils


class TestMetaUtils(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with some images to resample
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.input_dir = os.path.join(self.temp_path, 'input_dir')
        self.tempdir.makedir('input_dir')
        self.ff_dir = os.path.join(self.temp_path, 'ff_dir')
        self.tempdir.makedir('ff_dir')
        self.slice_idx = 1
        self.time_idx = 2
        self.im = np.zeros((10, 20), np.uint8) + 5
        ff_im = np.ones((10, 20), np.float) * 2
        # Mask meta file
        self.csv_name = 'mask_image_matchup.csv'
        self.input_meta = aux_utils.make_dataframe()
        # Make input meta
        for c in range(3):
            ff_path = os.path.join(
                self.ff_dir,
                'flat-field_channel-{}.npy'.format(c)
            )
            np.save(ff_path, ff_im, allow_pickle=True, fix_imports=True)
            for p in range(5):
                im_name = aux_utils.get_im_name(
                    channel_idx=c,
                    slice_idx=self.slice_idx,
                    time_idx=self.time_idx,
                    pos_idx=p,
                )
                cv2.imwrite(
                    os.path.join(self.input_dir, im_name),
                    self.im,
                )
                self.input_meta = self.input_meta.append(
                    aux_utils.parse_idx_from_name(im_name),
                    ignore_index=True,
                )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_frames_meta_generator(self):
        frames_meta = meta_utils.frames_meta_generator(
            input_dir=self.input_dir,
            name_parser='parse_idx_from_name',
        )
        for idx, row in frames_meta.iterrows():
            input_row = self.input_meta.iloc[idx]
            nose.tools.assert_equal(input_row['file_name'], row['file_name'])
            nose.tools.assert_equal(input_row['slice_idx'], row['slice_idx'])
            nose.tools.assert_equal(input_row['time_idx'], row['time_idx'])
            nose.tools.assert_equal(input_row['channel_idx'], row['channel_idx'])
            nose.tools.assert_equal(input_row['pos_idx'], row['pos_idx'])
