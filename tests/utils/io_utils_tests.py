import nose
from testfixtures import TempDirectory
import unittest
import zarr

import micro_dl.utils.io_utils as io_utils


class TestZarrReader(unittest.TestCase):

    def setUp(self):
        """Create data set"""

        self.tempdir = TempDirectory()
        self.input_dir = self.tempdir.path
        self.zarr_name = 'test_data.zarr'
        self.channel_names = ['test_channel1', 'test_channel2']
        # Write test dataset
        zarr_writer = image_utils.ZarrWriter(
            write_dir=self.input_dir,
            zarr_name=self.zarr_name,
            channel_names=self.channel_names,
        )
        self.nbr_pos = 5
        self.nbr_times = 3
        self.nbr_channels = 2
        self.nbr_slices = 10
        self.size_y = 15
        self.size_x = 20
        data_set = np.zeros((self.nbr_pos,
                             self.nbr_times,
                             self.nbr_channels,
                             self.nbr_slices,
                             self.size_y,
                             self.size_x))
        for pos_idx in range(self.nbr_pos):
            data_set[pos_idx, ...] = pos_idx + 1
        zarr_writer.write_data_set(data_set)
        # Instantiate zarr reader
        self.zarr_reader = image_utils.ZarrReader(self.input_dir, self.zarr_name)

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        self.assertFalse(os.path.isdir(self.input_dir))

    def test_init(self):
        self.assertEqual(len(self.zarr_reader.well_pos), self.nbr_pos)
        self.assertListEqual(self.zarr_reader.channel_names, self.channel_names)
        self.assertEqual(self.zarr_reader.array_name, 'array')
        self.assertEqual(self.zarr_reader.nbr_pos, self.nbr_pos)
        self.assertEqual(self.zarr_reader.nbr_times, self.nbr_times)
        self.assertEqual(self.zarr_reader.nbr_channels, self.nbr_channels)
        self.assertEqual(self.zarr_reader.nbr_slices, self.nbr_slices)

    def test_get_positions(self):
        self.assertEqual(self.zarr_reader.get_positions(), self.nbr_pos)

    def test_get_times(self):
        self.assertEqual(self.zarr_reader.get_times(), self.nbr_times)

    def test_get_channel_names(self):
        self.assertEqual(self.zarr_reader.get_channel_names(), self.channel_names)

    def test_image_from_row(self):
        meta_row = {
            "channel_idx": 1,
            "pos_idx": 3,
            "slice_idx": 7,
            "time_idx": 1,
            "channel_name": 'test_channel2',
            "dir_name": self.input_dir,
            "file_name": self.zarr_name,
        }
        meta_row = pd.DataFrame(meta_row, index=[0]).loc[0]
        im = self.zarr_reader.image_from_row(meta_row)
        self.assertTupleEqual(im.shape, (self.size_y, self.size_x))
        self.assertEqual(np.mean(im), 4)


class TestZarrWriter(unittest.TestCase):

    def setUp(self):
        """Create data set"""

        self.tempdir = TempDirectory()
        self.write_dir = self.tempdir.path
        self.zarr_name = 'test_data.zarr'
        self.channel_names = ['test_ch1', 'test_ch2']

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
        nbr_pos = 5
        data_set = np.zeros((nbr_pos, 3, 2, 10, 15, 20))
        for pos_idx in range(nbr_pos):
            data_set[pos_idx, ...] = pos_idx + 1
        self.zarr_writer.write_data_set(data_set)

        zarr_data = zarr.open(os.path.join(self.write_dir, self.zarr_name), mode='r')
        for pos_idx in range(nbr_pos):
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

    def test_write_image(self):
        frame = np.ones((10, 15, 20))
        data_shape = (5, 3, 2, 10, 15, 20)
        self.zarr_writer.write_image(
            im=frame,
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

    def test_write_2d_image(self):
        frame = np.ones((15, 20))
        data_shape = (5, 3, 2, 10, 15, 20)
        self.zarr_writer.write_image(
            im=frame,
            data_shape=data_shape,
            pos_idx=0,
            time_idx=2,
            channel_idx=1,
            slice_idx=5,
        )
        zarr_data = zarr.open(os.path.join(self.write_dir, self.zarr_name), mode='r')
        array = zarr_data['Row_0']['Col_0']['Pos_000']['array']
        self.assertTupleEqual(array.shape, (3, 2, 10, 15, 20))
        # Get subsection containing ones
        self.assertEqual(np.mean(array[2, 1, 5, ...]), 1)

    @nose.tools.raises(AssertionError)
    def test_write_image_wrong_y(self):
        self.zarr_writer.write_image(
            im=np.ones((10, 15, 20)),
            data_shape=(5, 3, 2, 10, 25, 20),
            pos_idx=0,
            time_idx=2,
            channel_idx=1,
        )

    @nose.tools.raises(AssertionError)
    def test_write_image_wrong_x(self):
        self.zarr_writer.write_image(
            im=np.ones((10, 15, 25)),
            data_shape=(5, 3, 2, 10, 25, 20),
            pos_idx=0,
            time_idx=2,
            channel_idx=1,
        )

    @nose.tools.raises(AssertionError)
    def test_write_image_wrong_shape(self):
        self.zarr_writer.data_shape = (5, 3, 2, 10, 25, 20)
        self.zarr_writer.write_image(
            im=np.ones((10, 15, 20)),
            data_shape=(5, 3, 2, 10, 25, 25),
            pos_idx=0,
            time_idx=2,
            channel_idx=1,
        )
