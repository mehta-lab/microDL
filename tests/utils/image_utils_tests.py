import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest
import zarr

# Create a test image and its corresponding coordinates and values
# Create a test image with a bright block to the right
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
import micro_dl.utils.normalize as normalize
from tests.utils.masks_utils_tests import uni_thr_tst_image

test_im = np.zeros((10, 15), np.uint16) + 100
test_im[:, 9:] = 200
x, y = np.meshgrid(np.linspace(1, 7, 3), np.linspace(1, 13, 5))
test_coords = np.vstack((x.flatten(), y.flatten())).T
test_values = np.zeros((15,), dtype=np.float64) + 100.
test_values[9:] = 200.


def test_upscale_image():
    im_out = image_utils.rescale_image(test_im, 2)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], test_shape[0] * 2)
    nose.tools.assert_equal(im_shape[1], test_shape[1] * 2)
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_downscale_image():
    im_out = image_utils.rescale_image(test_im, 0.5)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], round(test_shape[0] * .5))
    nose.tools.assert_equal(im_shape[1], round(test_shape[1] * .5))
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_samescale_image():
    im_out = image_utils.rescale_image(test_im, 1)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], test_shape[0])
    nose.tools.assert_equal(im_shape[1], test_shape[1])
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_fit_polynomial_surface():
    flatfield = image_utils.fit_polynomial_surface_2D(
        test_coords,
        test_values,
        im_shape=(10, 15),
    )
    # Since there's a bright block to the right, the left col should be
    # < right col
    nose.tools.assert_true(np.mean(flatfield[:, 0]) <
                           np.mean(flatfield[:, -1]))
    # Since flatfield is normalized, the mean should be close to one
    nose.tools.assert_almost_equal(np.mean(flatfield), 1., places=3)


def test_rescale_volume():
    # shape (5, 31, 31)
    nd_image = np.repeat(uni_thr_tst_image[np.newaxis], 5, axis=0)
    # upsample isotropically, 0.5 upsampling
    res_volume = image_utils.rescale_nd_image(nd_image, 1.3)
    nose.tools.assert_tuple_equal(res_volume.shape, (6, 40, 40))
    # upsample anisotropically
    res_volume = image_utils.rescale_nd_image(nd_image, [2.1, 1.1, 1.7])
    nose.tools.assert_tuple_equal(res_volume.shape, (10, 34, 53))
    # downsample isotropically, 0.5 downsampling
    res_volume = image_utils.rescale_nd_image(nd_image, 0.7)
    nose.tools.assert_tuple_equal(res_volume.shape, (4, 22, 22))
    # assertion error


@nose.tools.raises(AssertionError)
def test_rescale_volume_vrong_dims():
    nd_image = np.repeat(uni_thr_tst_image[np.newaxis], 5, axis=0)
    image_utils.rescale_nd_image(nd_image, [1.2, 1.8])


def test_center_crop_to_shape():
    im = np.zeros((5, 10, 15))
    output_shape = [5, 6, 9]
    im_center = image_utils.center_crop_to_shape(im, output_shape)
    nose.tools.assert_tuple_equal(im_center.shape, (5, 6, 9))


def test_center_crop_to_shape_2d():
    im = np.zeros((2, 5, 10))
    output_shape = [3, 7]
    im_center = image_utils.center_crop_to_shape(im, output_shape)
    nose.tools.assert_tuple_equal(im_center.shape, (2, 3, 7))


def test_center_crop_to_shape_2d_xyx():
    im = np.zeros((5, 10, 2))
    output_shape = [3, 7]
    im_center = image_utils.center_crop_to_shape(im, output_shape, 'xyz')
    nose.tools.assert_tuple_equal(im_center.shape, (3, 7, 2))


@nose.tools.raises(AssertionError)
def test_center_crop_to_shape_2d_too_big():
    im = np.zeros((2, 5, 10))
    output_shape = [7, 7]
    image_utils.center_crop_to_shape(im, output_shape)


# class TestImageUtils(unittest.TestCase):
#
#     def setUp(self):
#         """Set up a dictionary with images"""
#
#         self.tempdir = TempDirectory()
#         self.temp_path = self.tempdir.path
#         meta_fname = 'frames_meta.csv'
#         self.frames_meta = aux_utils.make_dataframe()
#
#         x = np.linspace(-4, 4, 32)
#         y = x.copy()
#         z = np.linspace(-3, 3, 8)
#         xx, yy, zz = np.meshgrid(x, y, z)
#         sph = (xx ** 2 + yy ** 2 + zz ** 2)
#         sph = (sph <= 8) * (8 - sph)
#         sph = (sph / sph.max()) * 255
#         sph = sph.astype('uint8')
#         self.sph = sph
#
#         self.channel_idx = 1
#         self.time_idx = 0
#         self.pos_idx = 1
#         self.int2str_len = 3
#
#         for z in range(sph.shape[2]):
#             im_name = aux_utils.get_im_name(
#                 channel_idx=1,
#                 slice_idx=z,
#                 time_idx=self.time_idx,
#                 pos_idx=self.pos_idx,
#             )
#             cv2.imwrite(os.path.join(self.temp_path, im_name), sph[:, :, z])
#             meta_row = aux_utils.parse_idx_from_name(im_name)
#             meta_row['zscore_median'] = np.nanmean(sph[:, :, z])
#             meta_row['zscore_iqr'] = np.nanstd(sph[:, :, z])
#             self.frames_meta = self.frames_meta.append(
#                 meta_row,
#                 ignore_index=True
#             )
#         self.frames_meta['dir_name'] = self.temp_path
#         self.dataset_mean = self.frames_meta['zscore_median'].mean()
#         self.dataset_std = self.frames_meta['zscore_iqr'].mean()
#         # Write metadata
#         self.frames_meta.to_csv(os.path.join(self.temp_path, meta_fname), sep=',')
#         # Write 3D sphere data
#         self.sph_fname = os.path.join(
#             self.temp_path,
#             'im_c001_z000_t000_p001_3d.npy',
#         )
#         np.save(self.sph_fname, self.sph, allow_pickle=True, fix_imports=True)
#         meta_3d = pd.DataFrame.from_dict([{
#             'channel_idx': 1,
#             'slice_idx': 0,
#             'time_idx': 0,
#             'channel_name': '3d_test',
#             'file_name': 'im_c001_z000_t000_p001_3d.npy',
#             'pos_idx': 1,
#             'zscore_median': np.nanmean(sph),
#             'zscore_iqr': np.nanstd(sph)
#         }])
#         self.meta_3d = meta_3d
#
#     def tearDown(self):
#         """
#         Tear down temporary folder and file structure
#         """
#         TempDirectory.cleanup_all()
#         nose.tools.assert_equal(os.path.isdir(self.temp_path), False)
#
#     def test_read_image(self):
#         file_path = os.path.join(
#             self.temp_path,
#             self.frames_meta['file_name'][0],
#         )
#         im = image_utils.read_image(file_path)
#         np.testing.assert_array_equal(im, self.sph[..., 0])
#
#     def test_read_image_npy(self):
#         im = image_utils.read_image(self.sph_fname)
#         np.testing.assert_array_equal(im, self.sph)
#
#     def test_read_imstack(self):
#         """Test read_imstack"""
#
#         fnames = self.frames_meta['file_name'][:3]
#         fnames = [os.path.join(self.temp_path, fname) for fname in fnames]
#         # non-boolean
#         im_stack = image_utils.read_imstack(
#             input_fnames=fnames,
#             normalize_im=True,
#             zscore_mean=self.dataset_mean,
#             zscore_std=self.dataset_std,
#         )
#         exp_stack = normalize.zscore(
#             self.sph[:, :, :3],
#             im_mean=self.dataset_mean,
#             im_std=self.dataset_std,
#         )
#         np.testing.assert_equal(im_stack.shape, (32, 32, 3))
#         np.testing.assert_array_equal(
#             exp_stack[:, :, :3],
#             im_stack,
#         )
#         # read a 3D image
#         im_stack = image_utils.read_imstack([self.sph_fname])
#         np.testing.assert_equal(im_stack.shape, (32, 32, 8))
#
#         # read multiple 3D images
#         im_stack = image_utils.read_imstack((self.sph_fname, self.sph_fname))
#         np.testing.assert_equal(im_stack.shape, (32, 32, 8, 2))
#
#     def test_preprocess_imstack(self):
#         """Test preprocess_imstack"""
#         im_stack = image_utils.preprocess_imstack(
#             frames_metadata=self.frames_meta,
#             depth=3,
#             time_idx=self.time_idx,
#             channel_idx=self.channel_idx,
#             slice_idx=2,
#             pos_idx=self.pos_idx,
#             normalize_im='dataset',
#         )
#         np.testing.assert_equal(im_stack.shape, (32, 32, 3))
#         exp_stack = np.zeros((32, 32, 3))
#         # Right now the code normalizes on a z slice basis for all
#         # normalization schemes
#         for z in range(exp_stack.shape[2]):
#             exp_stack[..., z] = normalize.zscore(self.sph[..., z + 1])
#         np.testing.assert_array_equal(im_stack, exp_stack)
#
#     def test_preprocess_imstack_3d(self):
#         # preprocess a 3D image
#         im_stack = image_utils.preprocess_imstack(
#             frames_metadata=self.meta_3d,
#             depth=1,
#             time_idx=0,
#             channel_idx=1,
#             slice_idx=0,
#             pos_idx=1,
#             normalize_im='dataset',
#         )
#         np.testing.assert_equal(im_stack.shape, (32, 32, 8))
#         # Normalization for 3D image is done on the entire volume
#         exp_stack = normalize.zscore(
#             self.sph,
#             im_mean=np.nanmean(self.sph),
#             im_std=np.nanstd(self.sph),
#         )
#         np.testing.assert_array_equal(im_stack, exp_stack)


class TestZarrWriter(unittest.TestCase):

    def setUp(self):
        """Create data set"""

        self.tempdir = TempDirectory()
        self.write_dir = self.tempdir.path
        self.zarr_name = 'test_data.zarr'
        self.channel_names = ['test_ch1', 'test_ch2']

        self.P = 10
        self.T = 3
        self.C = 2
        self.Z = 15
        self.Y = 25
        self.X = 35
        self.data_set = np.random.rand(self.P, self.T, self.C, self.Z, self.Y, self.X)
        self.zarr_writer = image_utils.ZarrWriter(
            write_dir=self.write_dir,
            zarr_name=self.zarr_name,
            channel_names=self.channel_names,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        self.assertFalse(os.path.isdir(self.write_dir))

    def test_init(self):
        self.assertEqual(self.zarr_writer.write_dir, self.write_dir)
        self.assertEqual(self.zarr_writer.zarr_name, self.zarr_name)
        zarr_path = os.path.join(self.write_dir, self.zarr_name)
        self.assertEqual(self.zarr_writer.zarr_path, zarr_path)
        zarr_data = zarr.open(zarr_path, mode='r')
        plate_info = zarr_data.attrs.get('plate')
        well = plate_info['wells'][0]
        pos = zarr_data[well['path']].attrs.get('well').get('images')[0]
        first_pos = zarr_data[well['path']][pos['path']]
        omero_meta = first_pos.attrs.asdict()['omero']
        for i, chan in enumerate(omero_meta['channels']):
            self.assertEqual(chan['label'], self.channel_names[i])

    def test_get_col_name(self):
        col_name = self.zarr_writer.get_col_name(15)
        self.assertEqual(col_name, 'Col_15')

    def test_get_pos_name(self):
        pos_name = self.zarr_writer.get_pos_name(666)
        self.assertEqual(pos_name, 'Pos_666')

    def test_create_row(self):
        self.zarr_writer.create_row(10)
        self.assertEqual(self.zarr_writer.row_name, 'Row_10')

    @nose.tools.raises(FileExistsError)
    def test_create_existing_row(self):
        self.zarr_writer.create_row(0)

    def test_create_column(self):
        self.zarr_writer.create_column(55)
        self.assertTrue('Col_55' in self.zarr_writer.store[self.zarr_writer.row_name])

    @nose.tools.raises(FileExistsError)
    def test_create_existing_column(self):
        self.zarr_writer.create_column(0)

    def test_create_meta(self):
        self.zarr_writer.create_meta()
        meta = self.zarr_writer.store['Row_0']['Col_0']['Pos_000'].attrs.asdict()
        for i, chan in enumerate(meta['omero']['channels']):
            self.assertEqual(chan['label'], self.channel_names[i])

    def test_create_position(self):
        self.zarr_writer.create_position(1)
        self.assertTrue('Pos_001' in self.zarr_writer.store['Row_0']['Col_1'])

    @nose.tools.raises(AssertionError)
    def test_create_not_consecutive_position(self):
        self.zarr_writer.create_position(9)

    @nose.tools.raises(FileExistsError)
    def test_create_existing_position(self):
        self.zarr_writer.create_position(0)

    def test_update_meta(self):
        self.zarr_writer.update_meta(0)
        self.assertEqual(
            self.zarr_writer.plate_meta['plate']['columns'][0]['name'],
            'Col_0',
        )
        plate_meta = self.zarr_writer.plate_meta['plate']['wells'][0]
        self.assertEqual(
            plate_meta['path'],
            'Row_0/Col_0',
        )
        self.assertEqual(
            self.zarr_writer.well_meta['well']['images'][0]['path'],
            'Pos_000',
        )

    def test_write_data_set(self):
        # Data has to have format (P (optional), T, C, Z, Y, X)
        P = 5
        data_set = np.zeros((P, 3, 2, 10, 15, 20))
        for pos_idx in range(P):
            data_set[pos_idx, ...] = pos_idx + 1
        self.zarr_writer.write_data_set(data_set)

        zarr_data = zarr.open(os.path.join(self.write_dir, self.zarr_name), mode='r')
        for pos_idx in range(P):
            col_name = 'Col_{}'.format(pos_idx)
            pos_name = 'Pos_{:03d}'.format(pos_idx)
            array = zarr_data['Row_0'][col_name][pos_name]['array']
            self.assertTupleEqual(array.shape, (3, 2, 10, 15, 20))
            self.assertEqual(np.mean(array), pos_idx + 1)

    def test_write_data_set_position(self):
        # Data has to have format (P (optional), T, C, Z, Y, X)
        data_set = np.ones((3, 2, 10, 15, 20))
        self.zarr_writer.write_data_set(data_set)

        zarr_data = zarr.open(os.path.join(self.write_dir, self.zarr_name), mode='r')
        array = zarr_data['Row_0']['Col_0']['Pos_000']['array']
        self.assertTupleEqual(array.shape, (3, 2, 10, 15, 20))
        self.assertEqual(np.mean(array), 1)

    @nose.tools.raises(AssertionError)
    def test_write_wrong_data_set(self):
        data_set = np.zeros((10, 15, 20))
        self.zarr_writer.write_data_set(data_set)

    def test_write_position(self):
        # Data has to have format (P (optional), T, C, Z, Y, X)
        data_set = np.ones((3, 2, 10, 15, 20))
        self.zarr_writer.write_position(data_set, 1)

        zarr_data = zarr.open(os.path.join(self.write_dir, self.zarr_name), mode='r')
        array = zarr_data['Row_0']['Col_1']['Pos_001']['array']
        self.assertTupleEqual(array.shape, (3, 2, 10, 15, 20))
        self.assertEqual(np.mean(array), 1)

    @nose.tools.raises(AssertionError)
    def test_write_wrong_position(self):
        data_set = np.ones((10, 15, 20))
        self.zarr_writer.write_position(data_set, 1)

    @nose.tools.raises(AssertionError)
    def test_write_position_wrong_channels(self):
        data_set = np.ones((3, 7, 10, 15, 20))
        self.zarr_writer.write_position(data_set, 1)

    def test_write_frame(self):
        frame = np.ones((10, 15, 20))
        data_shape = (5, 3, 2, 10, 15, 20)
        self.zarr_writer.write_frame(
            frame=frame,
            data_shape=data_shape,
            pos_idx=0,
            time_idx=2,
            channel_idx=1,
        )
        zarr_data = zarr.open(os.path.join(self.write_dir, self.zarr_name), mode='r')
        array = zarr_data['Row_0']['Col_0']['Pos_000']['array']
        self.assertTupleEqual(array.shape, (3, 2, 10, 15, 20))
        # Get subsection containing ones
        self.assertEqual(np.mean(array[2, 1, ...]), 1)
        