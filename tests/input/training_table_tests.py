import nose.tools
import numpy as np
import os
import pandas as pd
import unittest

import micro_dl.input.training_table as training_table
import micro_dl.utils.aux_utils as aux_utils


class TestTrainingTable(unittest.TestCase):

    def setUp(self):
        """
        Set up a dataframe for training table
        """
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        self.frames_meta = aux_utils.make_dataframe()
        self.time_ids = [3, 4, 5]
        self.pos_ids = [7, 8, 10, 12, 15]
        self.channel_ids = [0, 1, 2, 3]
        self.slice_ids = [0, 1, 2, 3, 4, 5]
        # Tiles will typically be split into image subsections
        # but it doesn't matter for testing
        for c in self.channel_ids:
            for p in self.pos_ids:
                for z in self.slice_ids:
                    for t in self.time_ids:
                        im_name = aux_utils.get_im_name(
                            channel_idx=c,
                            slice_idx=z,
                            time_idx=t,
                            pos_idx=p,
                        )
                        self.frames_meta = self.frames_meta.append(
                            aux_utils.parse_idx_from_name(im_name),
                            ignore_index=True,
                        )
        self.tiles_meta = aux_utils.sort_meta_by_channel(self.frames_meta)
        print(self.tiles_meta.head())

        self.input_channels = [0, 2]
        self.target_channels = [3]
        self.mask_channels = [1]
        self.split_ratio = {
            'train': 0.6,
            'val': 0.2,
            'test': 0.2,
        }
        # Instantiate class
        self.table_inst = training_table.BaseTrainingTable(
            df_metadata=self.tiles_meta,
            input_channels=self.input_channels,
            target_channels=self.target_channels,
            split_by_column='pos_idx',
            split_ratio=self.split_ratio,
            mask_channels=[1],
            random_seed=42,
        )

    def test__init__(self):
        col_names = ['index', 'channel_idx', 'slice_idx', 'time_idx',
                      'channel_name', 'file_name_0', 'pos_idx',
                      'file_name_1', 'file_name_2', 'file_name_3']

        self.assertListEqual(list(self.table_inst.df_metadata), col_names)
        self.assertListEqual(
            self.table_inst.input_channels,
            self.input_channels,
        )
        self.assertListEqual(
            self.table_inst.target_channels,
            self.target_channels,
        )
        self.assertListEqual(
            self.table_inst.mask_channels,
            self.mask_channels,
        )
        self.assertEqual(self.table_inst.split_by_column, 'pos_idx')
        self.assertDictEqual(self.table_inst.split_ratio, self.split_ratio)
        self.assertEqual(self.table_inst.random_seed, 42)

    def test_get_col_name(self):
        col_names = self.table_inst._get_col_name([1, 3])
        self.assertListEqual(col_names, ['file_name_1', 'file_name_3'])
