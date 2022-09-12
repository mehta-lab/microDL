"""Utility functions for processing images"""

import cv2
import itertools
import math
import numpy as np
import os
import pandas as pd
import sys
from scipy.ndimage.interpolation import zoom
from skimage.transform import resize
import zarr

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as normalize


def im_bit_convert(im, bit=16, norm=False, limit=[]):
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm:
        if not limit:
            # scale each image individually based on its min and max
            limit = [np.nanmin(im[:]), np.nanmax(im[:])]
        im = (im-limit[0]) / \
            (limit[1]-limit[0] + sys.float_info.epsilon) * (2**bit-1)
    im = np.clip(im, 0, 2**bit-1) # clip the values to avoid wrap-around by np.astype
    if bit == 8:
        im = im.astype(np.uint8, copy=False)  # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False)  # convert to 16 bit
    return im


def im_adjust(img, tol=1, bit=8):
    """
    Adjust contrast of the image

    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=bit, norm=True, limit=limit.tolist())
    return im_adjusted


def resize_image(input_image, output_shape):
    """Resize image to a specified shape

    :param np.ndarray input_image: image to be resized
    :param tuple/np.array output_shape: desired shape of the output image
    :return: np.array, resized image
    """

    msg = 'the output shape does not match the image dimension'
    assert len(output_shape) == len(input_image.shape), msg
    assert input_image.dtype is not 'bool'

    resized_image = resize(input_image, output_shape)
    return resized_image


def rescale_image(im, scale_factor):
    """
    Rescales a 2D image equally in x and y given a scale factor.
    Uses bilinear interpolation (the OpenCV default).

    :param np.array im: 2D image
    :param float scale_factor:
    :return np.array: 2D image resized by scale factor
    """

    assert scale_factor > 0,\
        'Scale factor must be > 0, not {}'.format(scale_factor)

    im_shape = im.shape
    assert len(im_shape) == 2, "OpenCV only works with 2D images"
    dsize = (int(round(im_shape[1] * scale_factor)),
             int(round(im_shape[0] * scale_factor)))

    return cv2.resize(im, dsize=dsize)


def rescale_nd_image(input_volume, scale_factor):
    """Rescale a nd array, mainly used for 3D volume

    For non-int dims, the values are rounded off to closest int. 0.5 is iffy,
    when downsampling the value gets floored and upsampling it gets rounded to
    next int

    :param np.array input_volume: 3D stack
    :param float/list scale_factor: if scale_factor is a float, scale all
     dimensions by this. Else scale_factor has to be specified for each
     dimension in a list or tuple
    :return np.array res_volume: rescaled volume
    """

    assert not input_volume.dtype == 'bool', \
        'input image is binary, not ideal for spline interpolation'

    if not isinstance(scale_factor, float):
        assert len(input_volume.shape) == len(scale_factor), \
            'Missing scale factor:' \
            'scale_factor:{} != input_volume:{}'.format(
                len(scale_factor), len(input_volume.shape)
            )

    res_image = zoom(input_volume, scale_factor)
    return res_image


def crop2base(im, base=2):
    """
    Crop image to nearest smaller factor of the base (usually 2), assumes xyz
    format, will work for zyx too but the x_shape, y_shape and z_shape will be
    z_shape, y_shape and x_shape respectively

    :param nd.array im: Image
    :param int base: Base to use, typically 2
    :param bool crop_z: crop along z dim, only for UNet3D
    :return nd.array im: Cropped image
    :raises AssertionError: if base is less than zero
    """
    assert base > 0, "Base needs to be greater than zero, not {}".format(base)
    im_shape = im.shape

    x_shape = base ** int(math.log(im_shape[0], base))
    y_shape = base ** int(math.log(im_shape[1], base))
    if x_shape < im_shape[0]:
        # Approximate center crop
        start_idx = (im_shape[0] - x_shape) // 2
        im = im[start_idx:start_idx + x_shape, ...]
    if y_shape < im_shape[1]:
        # Approximate center crop
        start_idx = (im_shape[1] - y_shape) // 2
        im = im[:, start_idx:start_idx + y_shape, ...]
    return im


def resize_mask(input_image, target_size):
    """Resample label/bool images"""
    raise NotImplementedError


def apply_flat_field_correction(input_image, **kwargs):
    """Apply flat field correction.

    :param np.array input_image: image to be corrected
    Kwargs, either:
        flat_field_image (np.float): flat_field_image for correction
        flat_field_path (str): Full path to flatfield image
    :return: np.array (float) corrected image
    """
    corrected_image = input_image.astype('float')
    if 'flat_field_image' in kwargs:
        flat_field_im = kwargs['flat_field_image']
        if flat_field_im is not None:
            corrected_image = input_image.astype('float') / flat_field_im
    elif 'flat_field_path' in kwargs:
        flat_field_path = kwargs['flat_field_path']
        if flat_field_path is not None:
            flat_field_image = np.load(flat_field_path)
            corrected_image = input_image.astype('float') / flat_field_image
    else:
        print("Incorrect kwargs: {}, returning input image".format(kwargs))
    return corrected_image


def fit_polynomial_surface_2D(sample_coords,
                              sample_values,
                              im_shape,
                              order=2,
                              normalize=True):
    """
    Given coordinates and corresponding values, this function will fit a
    2D polynomial of given order, then create a surface of given shape.

    :param np.array sample_coords: 2D sample coords (nbr of points, 2)
    :param np.array sample_values: Corresponding intensity values (nbr points,)
    :param tuple im_shape:         Shape of desired output surface (height, width)
    :param int order:              Order of polynomial (default 2)
    :param bool normalize:         Normalize surface by dividing by its mean
                                   for flatfield correction (default True)

    :return np.array poly_surface: 2D surface of shape im_shape
    """
    assert (order + 1) ** 2 <= len(sample_values), \
        "Can't fit a higher degree polynomial than there are sampled values"
    # Number of coefficients in determined by order + 1 squared
    orders = np.arange(order + 1)
    variable_matrix = np.zeros((sample_coords.shape[0], (order + 1) ** 2))
    variable_iterator = itertools.product(orders, orders)
    for idx, (m, n) in enumerate(variable_iterator):
        variable_matrix[:, idx] = sample_coords[:, 0] ** n * sample_coords[:, 1] ** m
    # Least squares fit of the points to the polynomial
    coeffs, _, _, _ = np.linalg.lstsq(variable_matrix, sample_values, rcond=-1)
    # Create a grid of image (x, y) coordinates
    x_mesh, y_mesh = np.meshgrid(np.linspace(0, im_shape[1] - 1, im_shape[1]),
                                 np.linspace(0, im_shape[0] - 1, im_shape[0]))
    # Reconstruct the surface from the coefficients
    poly_surface = np.zeros(im_shape, np.float)
    variable_iterator = itertools.product(orders, orders)
    for coeff, (m, n) in zip(coeffs, variable_iterator):
        poly_surface += coeff * x_mesh ** m * y_mesh ** n

    if normalize:
        poly_surface /= np.mean(poly_surface)
    return poly_surface


def center_crop_to_shape(input_image, output_shape, image_format='zyx'):
    """Center crop the image to a given shape

    :param np.array input_image: input image to be cropped
    :param list output_shape: desired crop shape
    :param str image_format: Image format; zyx or xyz
    :return np.array center_block: Center of input image with output shape
    """

    input_shape = np.array(input_image.shape)
    singleton_dims = np.where(input_shape == 1)[0]
    input_image = np.squeeze(input_image)
    modified_shape = output_shape.copy()
    if len(input_image.shape) == len(output_shape) + 1:
        # This means we're dealing with multichannel 2D
        if image_format == 'zyx':
            modified_shape.insert(0, input_image.shape[0])
        else:
            modified_shape.append(input_image.shape[-1])
    assert np.all(np.array(modified_shape) <= np.array(input_image.shape)), \
        'output shape is larger than image shape, use resize or rescale'

    start_0 = (input_image.shape[0] - modified_shape[0]) // 2
    start_1 = (input_image.shape[1] - modified_shape[1]) // 2
    if len(input_image.shape) > 2:
        start_2 = (input_image.shape[2] - modified_shape[2]) // 2
        center_block = input_image[
                       start_0: start_0 + modified_shape[0],
                       start_1: start_1 + modified_shape[1],
                       start_2: start_2 + modified_shape[2]]
    else:
        center_block = input_image[
                       start_0: start_0 + modified_shape[0],
                       start_1: start_1 + modified_shape[1]]
    for idx in singleton_dims:
        center_block = np.expand_dims(center_block, axis=idx)
    return center_block


def grid_sample_pixel_values(im, grid_spacing):
    """Sample pixel values in the input image at the grid. Any incomplete
    grids (remainders of modulus operation) will be ignored.

    :param np.array im: 2D image
    :param int grid_spacing: spacing of the grid
    :return int row_ids: row indices of the grids
    :return int col_ids: column indices of the grids
    :return np.array sample_values: sampled pixel values
    """

    im_shape = im.shape
    assert grid_spacing < im_shape[0], "grid spacing larger than image height"
    assert grid_spacing < im_shape[1], "grid spacing larger than image width"
    # leave out the grid points on the edges
    sample_coords = np.array(list(itertools.product(
        np.arange(grid_spacing, im_shape[0], grid_spacing),
        np.arange(grid_spacing, im_shape[1], grid_spacing))))
    row_ids = sample_coords[:, 0]
    col_ids = sample_coords[:, 1]
    sample_values = im[row_ids, col_ids]
    return row_ids, col_ids, sample_values


def read_image(file_path):
    """
    Read 2D grayscale image from file.
    Checks file extension for npy and load array if true. Otherwise
    reads regular image using OpenCV (png, tif, jpg, see OpenCV for supported
    files) of any bit depth.

    :param str file_path: Full path to image
    :return array im: 2D image
    :raise IOError if image can't be opened
    """
    if file_path[-3:] == 'npy':
        im = np.load(file_path)
    else:
        im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        if im is None:
            raise IOError('Image "{}" cannot be found.'.format(file_path))
    return im


def read_image_from_row(meta_row, zarr_reader=None):
    """
    Read 2D grayscale image from file.
    Checks file extension for npy and load array if true. Otherwise
    reads regular image using OpenCV (png, tif, jpg, see OpenCV for supported
    files) of any bit depth.

    :param pd.DataFrame meta_row: Row in metadata
    :param None/class zarr_reader: ZarrReader class instance if zarr data
    :return array im: 2D image
    :raise IOError if image can't be opened
    """
    if isinstance(meta_row, (pd.DataFrame, pd.Series)):
        meta_row = meta_row.squeeze()
    file_path = os.path.join(meta_row['dir_name'], meta_row['file_name'])
    if file_path[-3:] == 'npy':
        im = np.load(file_path)
    elif 'zarr' in file_path[-5:]:
        assert zarr_reader is not None, "No zarr class instance present."
        im = zarr_reader.image_from_row(meta_row)
    else:
        # Assumes files are tiff or png
        im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        if im is None:
            raise IOError('Image "{}" cannot be found.'.format(file_path))
    return im


def get_flat_field_path(flat_field_dir, channel_idx, channel_ids):
    """
    Given channel and flatfield dir, check that corresponding flatfield
    is present and returns its path.

    :param str flat_field_dir: Flatfield directory
    :param int channel_idx: Channel index for flatfield
    :param list channel_ids: All channel indices being processed
    """
    ff_path = None
    if flat_field_dir is not None:
        if isinstance(channel_idx, (int, float)) and channel_idx in channel_ids:
            ff_name = 'flat-field_channel-{}.npy'.format(channel_idx)
            if ff_name in os.listdir(flat_field_dir):
                ff_path = os.path.join(
                    flat_field_dir,
                    ff_name,
                )
    return ff_path


def preprocess_image(im,
                     hist_clip_limits=None,
                     is_mask=False,
                     normalize_im=None,
                     zscore_mean=None,
                     zscore_std=None):
    """
    Do histogram clipping, z score normalization, and potentially binarization.

    :param np.array im: Image (stack)
    :param tuple hist_clip_limits: Percentile histogram clipping limits
    :param bool is_mask: True if mask
    :param str/None normalize_im: Normalization, if any
    :param float/None zscore_mean: Data mean
    :param float/None zscore_std: Data std
    """
    # remove singular dimension for 3D images
    if len(im.shape) > 3:
        im = np.squeeze(im)
    if not is_mask:
        if hist_clip_limits is not None:
            im = normalize.hist_clipping(
                im,
                hist_clip_limits[0],
                hist_clip_limits[1]
            )
        if normalize_im is not None:
            im = normalize.zscore(
                im,
                im_mean=zscore_mean,
                im_std=zscore_std,
            )
    else:
        if im.dtype != bool:
            im = im > 0
    return im


def read_imstack_from_meta(frames_meta_sub,
                           zarr_reader=None,
                           flat_field_fnames=None,
                           hist_clip_limits=None,
                           is_mask=False,
                           normalize_im=None,
                           zscore_mean=None,
                           zscore_std=None):
    """
    Read images (>1) from metadata rows and assembles a stack.
    If images are masks, make sure they're boolean by setting >0 to True

    :param pd.DataFrame frames_meta_sub: Selected subvolume to be read
    :param class/None zarr_reader: ZarrReader class instance
    :param str/list flat_field_fnames: Path(s) to flat field image(s)
    :param tuple hist_clip_limits: Percentile limits for histogram clipping
    :param bool is_mask: Indicator for if files contain masks
    :param bool/None normalize_im: Whether to zscore normalize im stack
    :param float zscore_mean: mean for z-scoring the image
    :param float zscore_std: std for z-scoring the image
    :return np.array: input stack flat_field correct and z-scored if regular
        images, booleans if they're masks
    """
    im_stack = []
    meta_shape = frames_meta_sub.shape
    nbr_images = meta_shape[0] if len(meta_shape) > 1 else 1
    if isinstance(flat_field_fnames, list):
        assert len(flat_field_fnames) == nbr_images, \
            "Number of flatfields don't match number of input images"
    else:
        flat_field_fnames = nbr_images * [flat_field_fnames]

    if nbr_images > 1:
        for idx in range(meta_shape[0]):
            meta_row = frames_meta_sub.iloc[idx]
            im = read_image_from_row(meta_row, zarr_reader)
            flat_field_fname = flat_field_fnames[idx]
            if flat_field_fname is not None:
                if not is_mask and not normalize_im:
                    im = apply_flat_field_correction(
                        im,
                        flat_field_path=flat_field_fname,
                    )
            im_stack.append(im)
    else:
        # In case of series
        im = read_image_from_row(frames_meta_sub, zarr_reader)
        flat_field_fname = flat_field_fnames[0]
        if flat_field_fname is not None:
            if not is_mask and not normalize_im:
                im = apply_flat_field_correction(
                    im,
                    flat_field_path=flat_field_fname,
                )
        im_stack = [im]

    input_image = np.stack(im_stack, axis=-1)
    # Norm, hist clip, binarize for mask
    input_image = preprocess_image(
        input_image,
        hist_clip_limits,
        is_mask,
        normalize_im,
        zscore_mean,
        zscore_std,
    )
    return input_image


def read_imstack(input_fnames,
                 flat_field_fnames=None,
                 hist_clip_limits=None,
                 is_mask=False,
                 normalize_im=None,
                 zscore_mean=None,
                 zscore_std=None):
    """
    Read the images in the fnames and assembles a stack.
    If images are masks, make sure they're boolean by setting >0 to True

    :param tuple/list input_fnames: Paths to input files
    :param str/list flat_field_fnames: Path(s) to flat field image(s)
    :param tuple hist_clip_limits: limits for histogram clipping
    :param bool is_mask: Indicator for if files contain masks
    :param bool/None normalize_im: Whether to zscore normalize im stack
    :param float zscore_mean: mean for z-scoring the image
    :param float zscore_std: std for z-scoring the image
    :return np.array: input stack flat_field correct and z-scored if regular
        images, booleans if they're masks
    """
    im_stack = []
    if isinstance(flat_field_fnames, list):
        assert len(flat_field_fnames) == len(input_fnames), \
            "Number of flatfields don't match number of input images"
    else:
        flat_field_fnames = len(input_fnames) * [flat_field_fnames]

    for idx, fname in enumerate(input_fnames):
        im = read_image(fname)
        flat_field_fname = flat_field_fnames[idx]
        if flat_field_fname is not None:
            if not is_mask and not normalize_im:
                im = apply_flat_field_correction(
                    im,
                    flat_field_path=flat_field_fname,
                )
        im_stack.append(im)

    input_image = np.stack(im_stack, axis=-1)
    # Norm, hist clip, binarize for mask
    input_image = preprocess_image(
        input_image,
        hist_clip_limits,
        is_mask,
        normalize_im,
        zscore_mean,
        zscore_std,
    )
    return input_image


def preprocess_imstack(frames_metadata,
                       depth,
                       time_idx,
                       channel_idx,
                       slice_idx,
                       pos_idx,
                       zarr_reader=None,
                       flat_field_path=None,
                       hist_clip_limits=None,
                       normalize_im='stack',
                       ):
    """
    Preprocess image given by indices: flatfield correction, histogram
    clipping and z-score normalization is performed.

    :param pd.DataFrame frames_metadata: DF with meta info for all images
    :param int depth: num of slices in stack if 2.5D or depth for 3D
    :param int time_idx: Time index
    :param int channel_idx: Channel index
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param class/None zarr_reader: ZarrReader class instance if zarr data
    :param np.array flat_field_path: Path to flat field image for channel
    :param list hist_clip_limits: Limits for histogram clipping (size 2)
    :param str or None normalize_im: options to z-score the image
    :return np.array im: 3D preprocessed image
    """
    assert normalize_im in ['stack', 'dataset', 'volume', 'slice', None], \
        "'normalize_im' can only be 'stack', 'dataset', 'volume', 'slice', or None"

    metadata_ids, _ = aux_utils.validate_metadata_indices(
        frames_metadata=frames_metadata,
        slice_ids=-1,
        uniform_structure=True,
    )
    margin = 0 if depth == 1 else depth // 2
    im_stack = []
    for z in range(slice_idx - margin, slice_idx + margin + 1):
        meta_idx = aux_utils.get_meta_idx(
            frames_metadata,
            time_idx,
            channel_idx,
            z,
            pos_idx,
        )
        meta_row = frames_metadata.loc[meta_idx]
        im = read_image_from_row(meta_row, zarr_reader)
        # Only flatfield correct images that won't be normalized
        if flat_field_path is not None:
            assert normalize_im in [None, 'stack'], \
                "flat field correction currently only supports " \
                "None or 'stack' option for 'normalize_im'"
            im = apply_flat_field_correction(
                im,
                flat_field_path=flat_field_path,
            )

        zscore_median = None
        zscore_iqr = None
        if normalize_im in ['dataset', 'volume', 'slice']:
            if 'zscore_median' in frames_metadata:
                zscore_median = frames_metadata.loc[meta_idx, 'zscore_median']
            if 'zscore_iqr' in frames_metadata:
                zscore_iqr = frames_metadata.loc[meta_idx, 'zscore_iqr']
        if normalize_im is not None:
            im = normalize.zscore(
                im,
                im_mean=zscore_median,
                im_std=zscore_iqr,
            )
        im_stack.append(im)

    if len(im.shape) == 3:
        # each channel is tiled independently and stacked later in dataset cls
        im_stack = im
        assert depth == 1, 'more than one 3D volume gets read'
    else:
        # Stack images in same channel
        im_stack = np.stack(im_stack, axis=2)
    # normalize
    if hist_clip_limits is not None:
        im_stack = normalize.hist_clipping(
            im_stack,
            hist_clip_limits[0],
            hist_clip_limits[1],
        )

    return im_stack


class ZarrReader:
    """
    Handles zarr data from input .zarr file in a class that is serializable,
    thus suitable for multiprocessing.
    Assumes ome-zarr data, next generation file format defined here:
    https://ngff.openmicroscopy.org/0.1/
    Reading and writing zarr data is based on WaveOrder:
    https://github.com/mehta-lab/waveorder/tree/master/waveorder/io

    Note: There is an ome-zarr-py package in PyPI but I can't currently
    figure out how to use it with the ome-zarr datasets I'm prototyping with.
    https://ome-zarr.readthedocs.io/en/stable/index.html
    """

    def __init__(self, input_dir, zarr_name, single_pos=True):
        """
        Finds metadata for zarr file

        :param str input_dir: Input directory
        :param str zarr_name: Name of zarr file in input_dir
        :param bool single_pos: If each zarr file contains a single position
        """
        zarr_store = zarr.open(os.path.join(input_dir, zarr_name), mode='r')
        self.zarr_store = None
        if not single_pos:
            self.zarr_store = zarr_store
        self.single_pos = single_pos
        plate_info = zarr_store.attrs.get('plate')

        self.well_pos = []
        # Assumes that the positions are indexed in the order of Row-->Well-->FOV
        for well in plate_info['wells']:
            for pos in zarr_store[well['path']].attrs.get('well').get('images'):
                self.well_pos.append(
                    {'well': well['path'], 'pos': pos['path']}
                )

        # Get channel names
        first_pos = zarr_store[self.well_pos[0]['well']][self.well_pos[0]['pos']]
        omero_meta = first_pos.attrs.asdict()['omero']
        self.channel_names = []
        for chan in omero_meta['channels']:
            self.channel_names.append(chan['label'])

        self.array_name = list(first_pos.array_keys())[0]
        array_shape = first_pos[self.array_name].shape

        self.nbr_pos = len(self.well_pos)
        self.nbr_times = array_shape[0]
        self.nbr_channels = array_shape[1]
        self.nbr_slices = array_shape[2]

        # If there isn't a channel name for each channel, set to nan
        if len(self.channel_names) != self.nbr_channels:
            self.channel_names = self.nbr_channels * [np.nan]

    def get_pos(self):
        return self.nbr_pos

    def get_times(self):
        return self.nbr_times

    def get_channels(self):
        return self.nbr_channels

    def get_slices(self):
        return self.nbr_slices

    def get_channel_names(self):
        return self.channel_names

    def image_from_row(self, meta_row):
        """
        Fetches an image given indices of position, time, channel and slice.

        :param tuple meta_row: Row of metadata
        """
        pos_idx = 0
        if not self.single_pos:
            pos_idx = meta_row['pos_idx']
        if self.zarr_store is None:
            zarr_store = zarr.open(
                os.path.join(meta_row['dir_name'], meta_row['file_name']),
                mode='r',
            )
        else:
            zarr_store = self.zarr_store
        well_pos_idx = self.well_pos[pos_idx]
        array = zarr_store[well_pos_idx['well']][well_pos_idx['pos']][self.array_name]
        im = array[meta_row['time_idx'], meta_row['channel_idx'], meta_row['slice_idx']]
        return im

    def image_from_indices(self, pos_idx, time_idx, channel_idx, slice_idx):
        """
        Fetches an image given indices of position, time, channel and slice.

        :param int pos_idx: Position index
        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int slice_idx: Slice index
        """
        well_pos_idx = self.well_pos[pos_idx]
        array = self.zarr_store[well_pos_idx['well']][well_pos_idx['pos']][self.array_name]
        im = array[time_idx, channel_idx, slice_idx]
        return im


class ZarrWriter:
    """
    Writes data into a .zarr file.
    Assumes ome-zarr data, next generation file format defined here:
    https://ngff.openmicroscopy.org/0.1/
    Reading and writing zarr data is based on WaveOrder:
    https://github.com/mehta-lab/waveorder/tree/master/waveorder/io
    """

    def __init__(self, write_dir, zarr_name, channel_names, pos_idx):
        """
        Sets up zarr store

        :param str write_dir: Input directory
        :param str zarr_name: Name of zarr file, ending with .zarr
        :param list channel_names: List of channel names (strs)
            Assuming they're the same throughout acquisition.
        """
        self.write_dir = write_dir
        self.zarr_path = os.path.join(self.write_dir, zarr_name)
        if os.path.exists(self.zarr_path):
            raise FileExistsError('Zarr data path: {} already exists'.format(self.zarr_path))

        store = zarr.open(self.zarr_path)
        # Initialize hierarchy and metadata
        self.plate_meta = dict()
        self.well_meta = dict()
        self.row_name = None
        self.channel_names = channel_names
        self.data_shape = None
        self.chunk_size = None
        self.data_dtype = np.uint16
        self.positions = []
        self.create_position(store, pos_idx=pos_idx)
        self.create_meta()
        self.update_meta(pos_idx=pos_idx)

    def get_col_name(self, pos_idx):
        """
        :param int pos_idx: Positional index
        """
        return 'Col_{}'.format(pos_idx)

    def get_pos_name(self, pos_idx):
        """
        :param int pos_idx: Positional index
        """
        return 'Pos_{:03d}'.format(pos_idx)

    def create_row(self, store, row_idx):
        """
        Creates a row in the hierarchy (first level below zarr store).
        Keeps track of the row name (for now only created once per dataset).

        :param int row_idx: Row index
        """
        self.row_name = 'Row_{}'.format(row_idx)
        # check if row that already exists
        if self.row_name in store:
            raise FileExistsError('A row named {} already exists'.format(self.row_name))
        else:
            store.create_group(self.row_name)

    def create_column(self, store, pos_idx):
        """
        Creates a column in the hierarchy with same index as position.

        :param int pos_idx: Position index (order in which it is placed)
        """
        col_name = self.get_col_name(pos_idx)
        # check to see if col already exists
        if col_name in store[self.row_name]:
            raise FileExistsError(
                'A column subgroup named {} already exists'.format(col_name),
            )
        else:
            store[self.row_name].create_group(col_name)

    def create_meta(self, store):
        """
        Create metadata according to OME standards. Version 0.1?
        """
        ome_version = '0.1'  # Don't know specifics of different versions
        self.plate_meta['plate'] = {'acquisitions': [{'id': 1,
                                                      'maximumfieldcount': 1,
                                                      'name': 'Dataset',
                                                      'starttime': 0}],
                                    'columns': [],
                                    'field_count': 1,
                                    'name': self.zarr_name.strip('.zarr'),
                                    'rows': [],
                                    'version': ome_version,
                                    'wells': []}

        self.plate_meta['plate']['rows'].append({'name': self.row_name})
        self.well_meta['well'] = {'images': [], 'version': ome_version}

        multiscale_dict = [{'datasets': [{'path': "arr_0"}],
                            'version': '0.1'}]

        rdefs = {'defaultT': 0,
                 'model': 'color',
                 'projection': 'normal',
                 'defaultZ': 0}

        dict_list = []
        for i, channel_name in enumerate(self.channel_names):
            first_chan = True if i == 0 else False
            # Hardcoding contrast limits for now to uint16
            channel_dict = {
                'active': first_chan,
                'coefficient': 1.0,
                'color': 'FFFFFF',
                'family': 'linear',
                'inverted': False,
                'label': channel_name,
                'window': {'end': 65535, 'max': 65535, 'min': 0, 'start': 0}
            }
            dict_list.append(channel_dict)

        full_dict = {'multiscales': multiscale_dict,
                     'omero': {
                         'channels': dict_list,
                         'rdefs': rdefs,
                         'version': 0.1}
                     }

        pos = self.positions[0]
        pos_group = store[pos['row']][pos['col']][pos['name']]
        pos_group.attrs.put(full_dict)

    def create_position(self, zarr_name, pos_idx, single_pos=True):
        """
        Creates a column and position subgroup given the index.

        :param int pos_idx: Index of the position to create
        """
        self.store
        if single_pos:
            self.create_row(row_idx=0)
        else:
            assert pos_idx == len(self.positions), \
                "There are {} existing positions, but index is {}".format(
                    len(self.positions), pos_idx,
                )
            # Positions are appended to list. Could add index and check for
            # it to remove assertion above if positions are not created
            # consecutively.
        self.create_column(pos_idx=pos_idx)
        pos_name = self.get_pos_name(pos_idx=pos_idx)
        col_name = self.get_col_name(pos_idx=pos_idx)

        # create position subgroup
        if pos_name in self.store[self.row_name][col_name]:
            raise FileExistsError(
                'A pos named {} already exists'.format(pos_name),
            )
        else:
            self.store[self.row_name][col_name].create_group(pos_name)
            self.positions.append({'name': pos_name, 'row': self.row_name, 'col': col_name})

    def update_meta(self, pos_idx):
        """
        Update metadata for given position.

        :param int pos_idx: Position index
        """
        pos_name = self.get_pos_name(pos_idx=pos_idx)
        col_name = self.get_col_name(pos_idx=pos_idx)
        self.plate_meta['plate']['columns'].append({'name': col_name})
        self.plate_meta['plate']['wells'].append({'path': f'{self.row_name}/{col_name}'})
        self.store.attrs.put(self.plate_meta)
        # Update well meta
        self.well_meta['well']['images'] = [{'path': pos_name}]
        self.store[self.row_name][col_name].attrs.put(self.well_meta)

    def write_data_set(self, data_array):
        """
        This function will write an entire (small) 5D/6D data data set for
        given position, or if P is in data array, loop over positions.
        Data has to have format (P (optional), T, C, Z, Y, X)
        This function does not check if it's overwriting.

        Chunk size is currently set to one z-slice = (1,1,1,Y,X)
        Zarr will load one chunk at a time with this specified size.

        chan_names describe the names of the channels of your data in the
        order in which they will be written.

        :param np.array data_array: Image for given position
        """
        data_shape = data_array.shape
        # Check data shape and determine positions
        if len(data_shape) == 5:
            self.write_position(data_array, 0)
        else:
            assert len(data_shape) == 6, \
                "Data set must have format (P (optional), T, C, Z, Y, X)"
            for pos_idx in range(data_shape[0]):
                self.write_position(data_array[pos_idx], pos_idx)

    def write_position(self, zarr_name, data_array, pos_idx):
        """
        This function will write an entire (small) 5D data data set for
        given position.
        Data has to have format (T, C, Z, Y, X)
        This function does not check if it's overwriting.

        Chunk size is currently set to one z-slice = (1,1,1,Y,X)
        Zarr will load one chunk at a time with this specified size.

        chan_names describe the names of the channels of your data in the
        order in which they will be written.

        :param np.array data_array: Image for given position
        :param int pos_idx: Position index
        """
        data_shape = data_array.shape
        assert len(data_shape) == 5, \
            "Data shape has to be 5, not {}".format(len(data_shape))
        assert data_shape[1] == len(self.channel_names), \
            "Data has {} channels, but class was instantiated with {} channels".format(
                data_shape[1], len(self.channel_names),
            )

        chunk_size = (1, 1, 1, data_shape[3], data_shape[4])
        col_name = self.get_col_name(pos_idx)
        pos_name = self.get_pos_name(pos_idx)
        if pos_idx > 0:
            self.create_position(pos_idx)
            self.update_meta(pos_idx)

        current_pos_group = self.store[self.row_name][col_name][pos_name]

        if current_pos_group.__len__() == 0:
            current_pos_group.zeros(
                'array',
                shape=data_shape,
                chunks=chunk_size,
                dtype=data_array.dtype,
            )
        current_pos_group['array'] = data_array

    def write_image(self,
                    im,
                    data_shape,
                    pos_idx,
                    time_idx,
                    channel_idx,
                    slice_idx=None):
        """
        Writes image (2D/3D) frame for given position, time, channel and slice.
        Positions have to be written in consecutive order.

        Data has to have format (P, T, C, Z, Y, X)
        This function does not check if it's overwriting.

        clims corresponds to the the display contrast limits in the
        metadata for every channel, if none, default values will be used

        :param np.array im: Image data (Z (optional), Y, X)
        :param tuple data_shape: Shape of image data (P, T, C, Z, Y, X)
        :param int pos_idx: Position index
        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int/None slice_idx: Slice index
        """
        im_shape = im.shape
        assert im_shape[-2] == data_shape[-2], "Image Y must match data shape"
        assert im_shape[-1] == data_shape[-1], "Image X must match data shape"
        # Once data shape is established, it's fixed for the whole data set
        if self.data_shape is not None:
            assert self.data_shape == data_shape, \
                "Data shape doesn't match previously established shape {}".format(
                    self.data_shape,
                )
        else:
            # Fix data shape for whole data set
            self.data_shape = data_shape
            self.chunk_size = (1, 1, 1, data_shape[-2], data_shape[-1])
            self.data_dtype = im.dtype

        col_name = self.get_col_name(pos_idx)
        pos_name = self.get_pos_name(pos_idx)
        if pos_idx > len(self.positions):
            self.create_position(pos_idx)

        current_pos_group = self.store[self.row_name][col_name][pos_name]
        if current_pos_group.__len__() == 0:
            current_pos_group.zeros(
                'array',
                shape=self.data_shape[1:],
                chunks=self.chunk_size,
                dtype=self.data_dtype,
            )
        if slice_idx is None:
            assert len(im_shape) == 3, \
                "If slice isn't specified, shape needs to be 3D, it's {}D".format(len(im_shape))
            current_pos_group['array'][time_idx, channel_idx, ...] = im
        else:
            assert len(im_shape) == 2, \
                "If slice is specified, shape needs to be 2D, it's {}D".format(len(im_shape))
            current_pos_group['array'][time_idx, channel_idx, slice_idx, ...] = im
