import glob
import nose.tools
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import tempfile
import unittest

from micro_dl.input.gen_crop_masks import MaskProcessor


class TestMaskProcessor(unittest.TestCase):

    def setUp(self):
        """
        Set up a folder structure with one timepoint (0),
        two channels: one with a spherical object and the other with a square
        object in 2D
        """

        tempfile.tempdir = '/tmp'
        self.tempdir = tempfile.mkdtemp()
        tp_dir = os.path.join(self.tempdir, 'timepoint_0')
        os.mkdir(tp_dir)

        ch0_dir = os.path.join(tp_dir, 'channel_0')
        os.mkdir(ch0_dir)
        ch1_dir = os.path.join(tp_dir, 'channel_1')
        os.mkdir(ch1_dir)

        # create a sphere
        x = np.linspace(-4, 4, 32)
        y = x.copy()
        z = np.linspace(-3, 3, 8)
        xx, yy, zz = np.meshgrid(x, y, z)
        sph = (xx ** 2 + yy ** 2 + zz ** 2)
        sph = (sph <= 8) * (8 - sph)

        # create an image with a rect
        rec = np.zeros(sph.shape)
        rec[3:30, 14:18, 3:6] = 2
        rec[14:18, 3:30, 3:6] = 2

        meta_info = []
        for sl_idx in range(sph.shape[-1]):
            fname = os.path.join(ch0_dir, 'im_n{}_z0.npy'.format(sl_idx))
            np.save(fname, np.squeeze(sph[:, :, sl_idx]))
            meta_info.append((0, 0, sl_idx, 0, fname, 1, 1, 3))
            fname = os.path.join(ch1_dir, 'im_n{}_z0.npy'.format(sl_idx))
            np.save(fname, np.squeeze(rec[:, :, sl_idx]))
            meta_info.append((0, 1, sl_idx, 0, fname, 1, 1, 3))
        df = pd.DataFrame.from_records(
            meta_info,
            columns=['timepoint', 'channel_num', 'sample_num', 'slice_num',
                     'fname', 'size_x_microns', 'size_y_microns',
                     'size_z_microns']
        )
        metadata_fname = os.path.join(self.tempdir,
                                      'split_images_info.csv')
        df.to_csv(metadata_fname, sep=',')

    def tearDown(self):
        """Tear down temporary folder and file structure"""

        shutil.rmtree(self.tempdir)
        nose.tools.assert_equal(os.path.isdir(self.tempdir), False)

    def test_mask_processor(self):
        """Test the methods in MaskProcessor"""

        mask_proc_inst = MaskProcessor(self.tempdir, mask_channels=[0, 1])
        # test generate_masks()
        mask_proc_inst.generate_masks(focal_plane_idx=0,
                                      str_elem_radius=3)
        # should create a folder mask_0-1 with 8 masks
        mask_dir = os.path.join(self.tempdir, 'timepoint_0', 'mask_0-1')
        nose.tools.assert_equal(os.path.isdir(mask_dir), True)
        mask_fnames = glob.glob(mask_dir + os.sep + '*.npy')
        nose.tools.assert_equal(len(mask_fnames), 8)
        # test get_crop_indices()
        op_dir = os.path.join(self.tempdir, 'cropped_masks')
        os.mkdir(op_dir)
        mask_proc_inst.get_crop_indices(min_fraction=0.4, tile_size=[8, 8],
                                        step_size=[8, 8],
                                        cropped_mask_dir=op_dir,
                                        save_cropped_masks=True)
        # saves a pickle file containing a dict with fnames as keys and
        # indices as values
        dict_fname = os.path.join(self.tempdir, 'timepoint_0',
                                  'mask_0-1_vf-0.4.pkl')
        nose.tools.assert_equal(os.path.exists(dict_fname), True)
        with open(dict_fname, 'rb') as f:
            crop_indices_dict = pickle.load(f)
        nose.tools.assert_equal(len(crop_indices_dict['im_n0_z0.npy']), 0)
        nose.tools.assert_equal(len(crop_indices_dict['im_n7_z0.npy']), 0)

        nose.tools.assert_equal(len(crop_indices_dict['im_n1_z0.npy']), 4)
        nose.tools.assert_equal(len(crop_indices_dict['im_n6_z0.npy']), 4)

        nose.tools.assert_equal(len(crop_indices_dict['im_n2_z0.npy']), 4)
        nose.tools.assert_equal(len(crop_indices_dict['im_n5_z0.npy']), 4)

        nose.tools.assert_equal(len(crop_indices_dict['im_n3_z0.npy']), 4)
        nose.tools.assert_equal(len(crop_indices_dict['im_n4_z0.npy']), 4)
        # test _process_cropped_masks()
        cropped_mask_fnames = glob.glob(op_dir + os.sep + '*.npy')
        nose.tools.assert_equal(len(cropped_mask_fnames), 24)

        exp_fnames = ['n1_r8-16_c8-16.npy', 'n1_r8-16_c16-24.npy',
                      'n1_r16-24_c8-16.npy', 'n1_r16-24_c16-24.npy',
                      'n2_r8-16_c8-16.npy', 'n2_r8-16_c16-24.npy',
                      'n2_r16-24_c8-16.npy', 'n2_r16-24_c16-24.npy',
                      'n3_r8-16_c8-16.npy', 'n3_r8-16_c16-24.npy',
                      'n3_r16-24_c8-16.npy', 'n3_r16-24_c16-24.npy',
                      'n4_r8-16_c8-16.npy', 'n4_r8-16_c16-24.npy',
                      'n4_r16-24_c8-16.npy', 'n4_r16-24_c16-24.npy',
                      'n5_r8-16_c8-16.npy', 'n5_r8-16_c16-24.npy',
                      'n5_r16-24_c8-16.npy', 'n5_r16-24_c16-24.npy',
                      'n6_r8-16_c8-16.npy', 'n6_r8-16_c16-24.npy',
                      'n6_r16-24_c8-16.npy', 'n6_r16-24_c16-24.npy']
        for fname in exp_fnames:
            cur_fname = os.path.join(op_dir, fname)
            nose.tools.assert_equal(cur_fname in cropped_mask_fnames, True)
