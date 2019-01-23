import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.preprocess_utils as preprocess_utils


class TestPreprocessUtils(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with some images to resample
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.mask_dir = os.path.join(self.temp_path, 'mask_dir')
        self.tempdir.makedir('mask_dir')
        self.input_dir = os.path.join(self.temp_path, 'input_dir')
        self.tempdir.makedir('input_dir')
        self.mask_channel = 1
        self.slice_idx = 7
        self.time_idx = 8
        # Mask meta file
        self.csv_name = 'mask_image_matchup.csv'
        input_meta = aux_utils.make_dataframe()
        # Make input meta
        for c in range(4):
            for p in range(10):
                im_name = aux_utils.get_im_name(
                    channel_idx=c,
                    slice_idx=self.slice_idx,
                    time_idx=self.time_idx,
                    pos_idx=p,
                    ext='.png',
                )
                input_meta = input_meta.append(
                    aux_utils.get_ids_from_imname(im_name),
                    ignore_index=True,
                )
        input_meta.to_csv(
            os.path.join(self.input_dir, 'frames_meta.csv'),
            sep=',',
        )
        # Make mask meta
        mask_meta = pd.DataFrame()
        for p in range(10):
            im_name = aux_utils.get_im_name(
                channel_idx=self.mask_channel,
                slice_idx=self.slice_idx,
                time_idx=self.time_idx,
                pos_idx=p,
                ext='.png',
            )
            # Indexing can be different
            mask_name = 'mask_{}.png'.format(p + 1)
            mask_meta = mask_meta.append(
                {'mask_name': mask_name, 'file_name': im_name},
                ignore_index=True,
            )
        mask_meta.to_csv(
            os.path.join(self.mask_dir, self.csv_name),
            sep=',',
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_validate_mask_meta(self):
        pp_config = {
            'input_dir': self.input_dir,
            'channel_ids': [1, 2, 4],
            'masks': {'mask_dir': self.mask_dir, 'csv_name': self.csv_name},
        }
        mask_out_channel = preprocess_utils.validate_mask_meta(pp_config)
        self.assertEqual(mask_out_channel, 5)

        out_meta = aux_utils.read_meta(self.mask_dir)
        print(out_meta)
        for i, row in out_meta.iterrows():
            self.assertEqual(row.slice_idx, self.slice_idx)
            self.assertEqual(row.time_idx, self.time_idx)
            self.assertEqual(row.channel_idx, self.mask_channel)
            self.assertEqual(row.pos_idx, i)
            self.assertEqual(row.file_name, "mask_{}.png".format(i + 1))

