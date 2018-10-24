import cv2
import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.input.tile_images as tile_images
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as norm_util


class TestImageTiler(unittest.TestCase):

    def setUp(self):
        """
        Set up a folder structure containing one timepoint (0)
        one channel (1) and two images in channel subfolder
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        df_names = ["channel_idx",
                    "slice_idx",
                    "time_idx",
                    "channel_name",
                    "file_name",
                    "pos_idx"]
        frames_meta = pd.DataFrame(
            columns=df_names,
        )
        # Write images as bytes
        self.im = 350 * np.ones((15, 12), dtype=np.uint16)
        self.im2 = 8000 * np.ones((15, 12), dtype=np.uint16)
        res, im_encoded = cv2.imencode('.png', self.im)
        im_encoded = im_encoded.tostring()
        res, im2_encoded = cv2.imencode('.png', self.im2)
        im2_encoded = im2_encoded.tostring()
        self.channel_idx = 1
        self.time_idx = 5
        self.pos_idx1 = 7
        self.pos_idx2 = 8
        int2str_len = 3
        # Write test images with 4 z and 2 pos idx
        for z in range(15, 20):
            im_name = "im_c" + str(self.channel_idx).zfill(int2str_len) + \
            "_z" + str(z).zfill(int2str_len) + \
            "_t" + str(self.time_idx).zfill(int2str_len) + \
            "_p" + str(self.pos_idx1).zfill(int2str_len) + ".png"
            self.tempdir.write(im_name, im_encoded)
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_names),
                ignore_index=True,
            )
        for z in range(15, 20):
            im_name = "im_c" + str(self.channel_idx).zfill(int2str_len) + \
            "_z" + str(z).zfill(int2str_len) + \
            "_t" + str(self.time_idx).zfill(int2str_len) + \
            "_p" + str(self.pos_idx2).zfill(int2str_len) + ".png"
            self.tempdir.write(im_name, im2_encoded)
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_names),
                ignore_index=True,
            )
        frames_meta.to_csv(
            os.path.join(self.temp_path, self.meta_name),
            sep=',',
        )
        # Instantiate tiler class
        self.output_dir = os.path.join(self.temp_path, "tile_dir")
        self.tile_dict = {
            'channels': [1],
            'tile_size': [5, 5],
            'step_size': [3, 3],
            'depths': 3,
            'data_format': 'channels_last',
        }
        self.tile_inst = tile_images.ImageTiler(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            tile_dict=self.tile_dict,
            time_ids=-1,
            slice_ids=-1,
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
        nose.tools.assert_equal(self.tile_inst.depths, 3)
        nose.tools.assert_equal(self.tile_inst.mask_depth, 1)
        nose.tools.assert_equal(self.tile_inst.tile_size, [5, 5])
        nose.tools.assert_equal(self.tile_inst.step_size, [3, 3])
        nose.tools.assert_false(self.tile_inst.isotropic)
        nose.tools.assert_equal(self.tile_inst.hist_clip_limits, None)
        nose.tools.assert_equal(self.tile_inst.data_format, 'channels_last')
        nose.tools.assert_equal(
            self.tile_inst.str_tile_step,
            'tiles_5-5_step_3-3',
        )
        nose.tools.assert_equal(self.tile_inst.channel_ids, [self.channel_idx])
        nose.tools.assert_equal(self.tile_inst.time_ids, [self.time_idx])
        # Depth is 3 so first and last frame will not be used
        numpy.testing.assert_array_equal(
            self.tile_inst.slice_ids,
            np.asarray([16, 17, 18]),
        )
        numpy.testing.assert_array_equal(
            self.tile_inst.pos_ids,
            np.asarray([7, 8]),
        )

        # channel_depth should be a dict containing depths for each channel
        print(self.tile_inst.channel_depth)
        self.assertListEqual(
            list(self.tile_inst.channel_depth),
            [self.channel_idx],
        )
        nose.tools.assert_equal(
            self.tile_inst.channel_depth[self.channel_idx],
            3,
        )

    def test_tile_dir(self):
        nose.tools.assert_equal(self.tile_inst.get_tile_dir(),
                                os.path.join(self.output_dir,
                                             "tiles_5-5_step_3-3"))

    def test_tile_mask_dir(self):
        nose.tools.assert_equal(self.tile_inst.get_tile_mask_dir(), None)

    def test_preprocess_im(self):
        im_stack, channel_name = self.tile_inst._preprocess_im(
            time_idx=self.time_idx,
            channel_idx=self.channel_idx,
            slice_idx=16,
            pos_idx=self.pos_idx1,
        )
        self.assertTupleEqual(im_stack.shape, (15, 12, 3))
        im_norm = norm_util.zscore(self.im)
        for z in range(0, 3):
            numpy.testing.assert_array_equal(im_stack[..., z], im_norm)

    def test_write_tiled_data(self):
        tiled_data = [('r0-5_c0-5_sl0-3', np.zeros((5, 5, 3), dtype=np.float)),
                      ('r3-8_c0-5_sl0-3', np.ones((5, 5, 3), dtype=np.float))]
        tiled_metadata = self.tile_inst._get_dataframe()
        tile_indices = [(0, 5, 0, 5), (3, 8, 0, 5)]
        tile_dir = self.tile_inst.get_tile_dir()

        out_metadata = self.tile_inst._write_tiled_data(
            tiled_data=tiled_data,
            save_dir=tile_dir,
            time_idx=self.time_idx,
            channel_idx=self.channel_idx,
            slice_idx=17,
            pos_idx=self.pos_idx2,
            tile_indices=tile_indices,
            tiled_metadata=tiled_metadata,
        )

        self.assertListEqual(
            out_metadata.channel_idx.tolist(),
            [self.channel_idx] * 2,
        )
        self.assertListEqual(
            out_metadata.slice_idx.tolist(),
            [17] * 2,
        )
        self.assertListEqual(
            out_metadata.time_idx.tolist(),
            [self.time_idx] * 2,
        )
        self.assertListEqual(
            out_metadata.pos_idx.tolist(),
            [self.pos_idx2] * 2,
        )
        self.assertListEqual(
            out_metadata.row_start.tolist(),
            [0, 3],
        )
        self.assertListEqual(
            out_metadata.col_start.tolist(),
            [0, 0],
        )
        self.assertListEqual(
            out_metadata.file_name.tolist(),
            ['im_c001_z017_t005_p008_r0-5_c0-5_sl0-3.npy',
             'im_c001_z017_t005_p008_r3-8_c0-5_sl0-3.npy'],
        )
        # Load and assert tiles
        tile = np.load(
            os.path.join(tile_dir,
            'im_c001_z017_t005_p008_r0-5_c0-5_sl0-3.npy'),
        )
        numpy.testing.assert_array_equal(tile, tiled_data[0][1])
        tile = np.load(
            os.path.join(tile_dir,
            'im_c001_z017_t005_p008_r3-8_c0-5_sl0-3.npy'),
        )
        numpy.testing.assert_array_equal(tile, tiled_data[1][1])









