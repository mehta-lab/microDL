import cv2
import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.utils.tile_utils as tile_utils
import micro_dl.utils.aux_utils as aux_utils


class TestTileUtils(unittest.TestCase):

    def setUp(self):
        """Set up a dictionary with images"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        meta_fname = 'frames_meta.csv'
        self.df_columns = ['channel_idx',
                           'slice_idx',
                           'time_idx',
                           'channel_name',
                           'file_name',
                           'pos_idx']
        frames_meta = pd.DataFrame(columns=self.df_columns)

        x = np.linspace(-4, 4, 32)
        y = x.copy()
        z = np.linspace(-3, 3, 8)
        xx, yy, zz = np.meshgrid(x, y, z)
        sph = (xx ** 2 + yy ** 2 + zz ** 2)
        sph = (sph <= 8) * (8 - sph)
        sph = (sph / sph.max()) * 255
        sph = sph.astype('uint8')
        self.sph = sph

        self.channel_idx = 1
        self.time_idx = 0
        self.pos_idx = 1
        self.int2str_len = 3

        for z in range(sph.shape[2]):
            im_name = aux_utils.get_im_name(
                channel_idx=1,
                slice_idx=z,
                time_idx=self.time_idx,
                pos_idx=self.pos_idx,
            )
            cv2.imwrite(os.path.join(self.temp_path, im_name), sph[:, :, z])
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name, self.df_columns),
                ignore_index=True
            )

        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, meta_fname), sep=',')
        self.frames_meta = frames_meta

        self.sph_fname = os.path.join(
            self.temp_path,
            'im_c001_z000_t000_p001_3d.npy',
        )
        np.save(self.sph_fname, self.sph, allow_pickle=True, fix_imports=True)
        meta_3d = pd.DataFrame.from_dict([{
            'channel_idx': 1,
            'slice_idx': 0,
            'time_idx': 0,
            'channel_name': '3d_test',
            'file_name': 'im_c001_z000_t000_p001_3d.npy',
            'pos_idx': 1,
        }])
        self.meta_3d = meta_3d

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_tile_image(self):
        """Test tile_image"""

        input_image = self.sph[:, :, 3:6]
        tile_size = [16, 16]
        step_size = [8, 8]
        # returns at tuple of (img_id, tile)
        tiled_image_list = tile_utils.tile_image(
            input_image,
            tile_size=tile_size,
            step_size=step_size,
        )
        nose.tools.assert_equal(len(tiled_image_list), 9)
        c = 0
        for row in range(0, 17, 8):
            for col in range(0, 17, 8):
                id_str = 'r{}-{}_c{}-{}_sl{}-{}'.format(
                    row, row + tile_size[0], col, col + tile_size[1], 0, 3
                )
                nose.tools.assert_equal(id_str, tiled_image_list[c][0])
                tile = input_image[row:row + tile_size[0],
                                   col: col + tile_size[1], ...]
                numpy.testing.assert_array_equal(tile, tiled_image_list[c][1])
                c += 1

        # returns tuple_list, cropping_index
        _, tile_index = tile_utils.tile_image(
            input_image,
            tile_size=tile_size,
            step_size=step_size,
            return_index=True,
        )
        exp_tile_index = [(0, 16, 0, 16), (0, 16, 8, 24),
                          (0, 16, 16, 32), (8, 24, 0, 16),
                          (8, 24, 8, 24), (8, 24, 16, 32),
                          (16, 32, 0, 16), (16, 32, 8, 24),
                          (16, 32, 16, 32)]

        numpy.testing.assert_equal(exp_tile_index, tile_index)

        # save tiles in place and return meta_df
        tile_dir = os.path.join(self.temp_path, 'tile_dir')
        os.makedirs(tile_dir, exist_ok=True)
        meta_dir = os.path.join(tile_dir, 'meta_dir')
        os.makedirs(meta_dir, exist_ok=True)
        save_dict = {'time_idx': self.time_idx,
                     'channel_idx': self.channel_idx,
                     'slice_idx': 4,
                     'pos_idx': self.pos_idx,
                     'image_format': 'zyx',
                     'int2str_len': 3,
                     'save_dir': tile_dir}
        tile_meta_df = tile_utils.tile_image(
            input_image,
            tile_size=tile_size,
            step_size=step_size,
            save_dict=save_dict,
        )
        tile_meta = []
        for row in range(0, 17, 8):
            for col in range(0, 17, 8):
                id_str = 'r{}-{}_c{}-{}_sl{}-{}'.format(
                    row, row + tile_size[0], col, col + tile_size[1], 0, 3
                )
                cur_fname = aux_utils.get_im_name(
                    time_idx=self.time_idx,
                    channel_idx=self.channel_idx,
                    slice_idx=4,
                    pos_idx=self.pos_idx,
                    int2str_len=3,
                    extra_field=id_str,
                    ext='.npy',
                )
                cur_path = os.path.join(tile_dir, cur_fname)
                nose.tools.assert_equal(os.path.exists(cur_path), True)
                cur_meta = {'channel_idx': self.channel_idx,
                            'slice_idx': 4,
                            'time_idx': self.time_idx,
                            'file_name': cur_fname,
                            'pos_idx': self.pos_idx,
                            'row_start': row,
                            'col_start': col}
                tile_meta.append(cur_meta)
        exp_tile_meta_df = pd.DataFrame.from_dict(tile_meta)
        exp_tile_meta_df = exp_tile_meta_df.sort_values(by=['file_name'])
        pd.testing.assert_frame_equal(tile_meta_df, exp_tile_meta_df)

        # use mask and min_fraction to select tiles to retain
        input_image_bool = input_image > 128
        _, tile_index = tile_utils.tile_image(
            input_image_bool,
            tile_size=tile_size,
            step_size=step_size,
            min_fraction=0.3,
            return_index=True,
        )
        exp_tile_index = [(0, 16, 8, 24),
                          (8, 24, 0, 16), (8, 24, 8, 24),
                          (8, 24, 16, 32),
                          (16, 32, 8, 24)]
        numpy.testing.assert_array_equal(tile_index, exp_tile_index)

        # tile_3d
        input_image = self.sph
        tile_size = [16, 16, 6]
        step_size = [8, 8, 4]
        # returns at tuple of (img_id, tile)
        tiled_image_list = tile_utils.tile_image(
            input_image,
            tile_size=tile_size,
            step_size=step_size,
        )
        nose.tools.assert_equal(len(tiled_image_list), 18)
        c = 0
        for row in range(0, 17, 8):
            for col in range(0, 17, 8):
                for sl in range(0, 8, 6):
                    if sl == 0:
                        sl_start_end = [0, 6]
                    else:
                        sl_start_end = [2, 8]

                    id_str = 'r{}-{}_c{}-{}_sl{}-{}'.format(
                        row, row + tile_size[0], col, col + tile_size[1],
                        sl_start_end[0], sl_start_end[1]
                    )
                    nose.tools.assert_equal(id_str, tiled_image_list[c][0])
                    tile = input_image[row:row + tile_size[0],
                                       col: col + tile_size[1],
                                       sl_start_end[0]: sl_start_end[1]]
                    numpy.testing.assert_array_equal(tile,
                                                     tiled_image_list[c][1])
                    c += 1

    def test_crop_at_indices(self):
        """Test crop_at_indices"""

        crop_indices = [(0, 16, 8, 24, 0, 3),
                        (8, 24, 0, 16, 0, 3), (8, 24, 8, 24, 0, 3),
                        (8, 24, 16, 32, 0, 3),
                        (16, 32, 8, 24, 0, 3)]
        input_image = self.sph[:, :, 3:6]

        # return tuple_list
        tiles_list = tile_utils.crop_at_indices(input_image, crop_indices)
        for idx, cur_idx in enumerate(crop_indices):
            tile = input_image[cur_idx[0]: cur_idx[1],
                               cur_idx[2]: cur_idx[3],
                               cur_idx[4]: cur_idx[5]]
            id_str = 'r{}-{}_c{}-{}_sl{}-{}'.format(cur_idx[0], cur_idx[1],
                                                    cur_idx[2], cur_idx[3],
                                                    cur_idx[4], cur_idx[5])
            nose.tools.assert_equal(id_str, tiles_list[idx][0])
            numpy.testing.assert_array_equal(tiles_list[idx][1], tile)

        # save tiles in place and return meta_df
        tile_dir = os.path.join(self.temp_path, 'tile_dir')
        os.makedirs(tile_dir, exist_ok=True)
        meta_dir = os.path.join(tile_dir, 'meta_dir')
        os.makedirs(meta_dir, exist_ok=True)
        save_dict = {'time_idx': self.time_idx,
                     'channel_idx': self.channel_idx,
                     'slice_idx': 4,
                     'pos_idx': self.pos_idx,
                     'image_format': 'zyx',
                     'int2str_len': 3,
                     'save_dir': tile_dir}

        tile_meta_df = tile_utils.crop_at_indices(
            input_image,
            crop_indices,
            save_dict=save_dict,
        )
        exp_tile_meta = []

        for idx, cur_idx in enumerate(crop_indices):
            id_str = 'r{}-{}_c{}-{}_sl{}-{}'.format(cur_idx[0], cur_idx[1],
                                                    cur_idx[2], cur_idx[3],
                                                    cur_idx[4], cur_idx[5])
            cur_fname = aux_utils.get_im_name(
                time_idx=self.time_idx,
                channel_idx=self.channel_idx,
                slice_idx=4,
                pos_idx=self.pos_idx,
                int2str_len=3,
                extra_field=id_str,
                ext='.npy',
            )
            cur_path = os.path.join(tile_dir, cur_fname)
            nose.tools.assert_equal(os.path.exists(cur_path), True)
            cur_meta = {'channel_idx': self.channel_idx,
                        'slice_idx': 4,
                        'time_idx': self.time_idx,
                        'file_name': cur_fname,
                        'pos_idx': self.pos_idx,
                        'row_start': cur_idx[0],
                        'col_start': cur_idx[2]}
            exp_tile_meta.append(cur_meta)
        exp_tile_meta_df = pd.DataFrame.from_dict(exp_tile_meta)
        exp_tile_meta_df = exp_tile_meta_df.sort_values(by=['file_name'])
        pd.testing.assert_frame_equal(tile_meta_df, exp_tile_meta_df)

    def test_write_tile(self):
        """Test write_tile"""

        tile_dir = os.path.join(self.temp_path, 'tile_dir')
        os.makedirs(tile_dir, exist_ok=True)
        save_dict = {'time_idx': self.time_idx,
                     'channel_idx': self.channel_idx,
                     'slice_idx': 4,
                     'pos_idx': self.pos_idx,
                     'image_format': 'zyx',
                     'int2str_len': 3,
                     'save_dir': tile_dir}

        input_image = self.sph[:, :, 3:6]
        cur_tile = input_image[8: 24, 8: 24, 0: 3]
        img_id = 'r8-24_c8-24_sl0-3'
        fname = tile_utils.write_tile(cur_tile, save_dict, img_id)

        exp_fname = '{}_{}.npy'.format('im_c001_z004_t000_p001', img_id)
        nose.tools.assert_equal(fname, exp_fname)
        fpath = os.path.join(tile_dir, fname)
        nose.tools.assert_equal(os.path.exists(fpath), True)

    def test_write_meta(self):
        """Test write_meta"""

        # save tiles in place and return meta_df
        tile_dir = os.path.join(self.temp_path, 'tile_dir')
        os.makedirs(tile_dir, exist_ok=True)
        meta_dir = os.path.join(tile_dir, 'meta_dir')
        os.makedirs(meta_dir, exist_ok=True)
        save_dict = {'time_idx': self.time_idx,
                     'channel_idx': self.channel_idx,
                     'slice_idx': 4,
                     'pos_idx': self.pos_idx,
                     'image_format': 'zyx',
                     'int2str_len': 3,
                     'save_dir': tile_dir}

        tile_meta = [{'channel_idx': self.channel_idx,
                      'slice_idx': 4,
                      'time_idx': self.time_idx,
                      'file_name': 'im_c001_z004_t000_p001_r8-24_c8-24_sl0-3',
                      'pos_idx': self.pos_idx,
                      'row_start': 8,
                      'col_start': 8}]

        tile_meta_df = tile_utils.write_meta(tile_meta, save_dict)

        exp_tile_meta_df = pd.DataFrame.from_dict(tile_meta)
        exp_tile_meta_df = exp_tile_meta_df.sort_values(by=['file_name'])
        pd.testing.assert_frame_equal(tile_meta_df, exp_tile_meta_df)

        # when tile_meta is an empty list
        tile_meta = []
        tile_meta_df = tile_utils.write_meta(tile_meta, save_dict)
        nose.tools.assert_equal(tile_meta_df, None)
