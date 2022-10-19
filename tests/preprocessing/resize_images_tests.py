import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.preprocessing.resize_images as resize_images
import micro_dl.utils.aux_utils as aux_utils


class TestResizeImages(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with some images to resample
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.output_dir = os.path.join(self.temp_path, 'out_dir')
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        self.frames_meta = aux_utils.make_dataframe()
        # Write images
        self.time_ids = [5]
        self.slice_ids = [6]
        self.pos_idx = 7
        self.pos_ids = [7, 8, 9]
        self.channel_ids = list(range(4))
        self.im = 1500 * np.ones((30, 20), dtype=np.uint16)

        for c in self.channel_ids:
            for p in self.pos_ids:
                im_name = aux_utils.get_im_name(
                    channel_idx=c,
                    slice_idx=self.slice_ids[0],
                    time_idx=self.time_ids[0],
                    pos_idx=p,
                )
                cv2.imwrite(os.path.join(self.temp_path, im_name),
                            self.im + c * 100)
                self.frames_meta = self.frames_meta.append(
                    aux_utils.parse_idx_from_name(im_name=im_name, dir_name=self.temp_path),
                    ignore_index=True,
                )
        # Write metadata
        self.frames_meta.to_csv(
            os.path.join(self.temp_path, self.meta_name),
            sep=',',
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_downsample(self):
        # Half the image size
        scale_factor = 0.5
        resize_inst = resize_images.ImageResizer(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            channel_ids=self.channel_ids,
            time_ids=self.time_ids,
            slice_ids=self.slice_ids,
            pos_ids=self.pos_ids,
            scale_factor=scale_factor,
        )
        self.assertEqual(resize_inst.time_ids, self.time_ids)
        self.assertListEqual(resize_inst.channel_ids, self.channel_ids)
        self.assertListEqual(resize_inst.slice_ids, self.slice_ids)
        self.assertListEqual(resize_inst.pos_ids, self.pos_ids)
        resize_dir = resize_inst.get_resize_dir()
        self.assertEqual(os.path.join(self.output_dir, 'resized_images'),
                         resize_dir)
        # Resize
        resize_inst.resize_frames()
        # Validate
        new_shape = tuple([int(scale_factor * x) for x in self.im.shape])
        for i, row in self.frames_meta.iterrows():
            file_name = os.path.join(resize_dir, row['file_name'])
            im = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            self.assertTupleEqual(new_shape, im.shape)
            self.assertEqual(im.dtype, self.im.dtype)
            im_expected = self.im + row['channel_idx'] * 100
            im_expected = cv2.resize(im_expected, (new_shape[1], new_shape[0]))
            np.testing.assert_array_equal(im, im_expected)

    def test_upsample(self):
        # Half the image size
        scale_factor = 2.0
        resize_inst = resize_images.ImageResizer(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            channel_ids=self.channel_ids,
            time_ids=self.time_ids,
            slice_ids=self.slice_ids,
            pos_ids=self.pos_ids,
            scale_factor=scale_factor,
        )
        self.assertEqual(resize_inst.time_ids, self.time_ids)
        self.assertListEqual(resize_inst.channel_ids, self.channel_ids)
        self.assertListEqual(resize_inst.slice_ids, self.slice_ids)
        self.assertListEqual(resize_inst.pos_ids, self.pos_ids)
        resize_dir = resize_inst.get_resize_dir()
        self.assertEqual(os.path.join(self.output_dir, 'resized_images'),
                         resize_dir)
        # Resize
        resize_inst.resize_frames()
        # Validate
        new_shape = tuple([int(scale_factor * x) for x in self.im.shape])
        for i, row in self.frames_meta.iterrows():
            file_name = os.path.join(resize_dir, row['file_name'])
            im = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            self.assertTupleEqual(new_shape, im.shape)
            self.assertEqual(im.dtype, self.im.dtype)
            im_expected = self.im + row['channel_idx'] * 100
            im_expected = cv2.resize(im_expected, (new_shape[1], new_shape[0]))
            np.testing.assert_array_equal(im, im_expected)

    def test_resize_volumes(self):
        """Test resizing volumes"""

        # set up a volume with 5 slices, 2 channels
        slice_ids = [0, 1, 2, 3, 4]
        channel_ids = [2, 3]
        resize_dir = os.path.join(self.output_dir, 'resized_images')
        frames_meta = aux_utils.make_dataframe()
        exp_meta_dict = []
        for c in channel_ids:
            for s in slice_ids:
                im_name = aux_utils.get_im_name(
                    channel_idx=c,
                    slice_idx=s,
                    time_idx=self.time_ids[0],
                    pos_idx=self.pos_idx,
                )
                cv2.imwrite(os.path.join(self.temp_path, im_name),
                            self.im + c * 100)
                frames_meta = frames_meta.append(
                    aux_utils.parse_idx_from_name(im_name=im_name, dir_name=self.temp_path),
                    ignore_index=True,
                )
            op_fname = 'im_c00{}_z000_t005_p007_3.3-0.8-1.0.npy'.format(c)
            exp_meta_dict.append({'time_idx': self.time_ids[0],
                                  'pos_idx': self.pos_idx,
                                  'channel_idx': c,
                                  'slice_idx': 0,
                                  'file_name': op_fname,
                                  'mean': np.mean(self.im) + c * 100,
                                  'std': float(0),
                                  'dir_name': resize_dir})
        exp_meta_df = pd.DataFrame.from_dict(exp_meta_dict)
        # Write metadata
        frames_meta.to_csv(
            os.path.join(self.temp_path, self.meta_name),
            sep=',',
        )

        scale_factor = [3.3, 0.8, 1.0]
        resize_inst = resize_images.ImageResizer(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            channel_ids=channel_ids,
            time_ids=self.time_ids,
            slice_ids=slice_ids,
            pos_ids=[self.pos_idx],
            scale_factor=scale_factor,
        )

        # save all slices in one volume
        resize_inst.resize_volumes()
        saved_meta = aux_utils.read_meta(resize_dir)
        pd.testing.assert_frame_equal(saved_meta, exp_meta_df)

        # num_slices_subvolume = 3, save vol chunks
        exp_meta_dict = []
        for c in channel_ids:
            for s in [0, 2]:
                op_fname = 'im_c00{}_z00{}_t005_p007_3.3-0.8-1.0.npy'.format(c,
                                                                             s)
                exp_meta_dict.append({'time_idx': self.time_ids[0],
                                      'pos_idx': self.pos_idx,
                                      'channel_idx': c,
                                      'slice_idx': s,
                                      'file_name': op_fname,
                                      'mean': np.mean(self.im) + c * 100,
                                      'std': float(0),
                                      'dir_name': resize_dir})
        exp_meta_df = pd.DataFrame.from_dict(exp_meta_dict)
        resize_inst.resize_volumes(num_slices_subvolume=3)
        saved_meta = aux_utils.read_meta(resize_dir)
        pd.testing.assert_frame_equal(saved_meta, exp_meta_df)
