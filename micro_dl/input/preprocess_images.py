"""Classes for handling microscopy data in lif format

Uses dir structure:
base_output_dir
 |-image_volume, image_volumes_info.csv
    |-tp0
        |-channel0
 |-img_512_512_8_.., cropped_images_info.csv
    |-tp-0
        |-channel0: contains all npy files for cropped images from channel0
        |-channel1: contains all npy files for cropped images from channel1..
        and so on
"""
from abc import ABCMeta, abstractmethod
import glob
import cv2
import logging
import natsort
import numpy as np
import os
import pandas as pd
import re
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_opening

import micro_dl.utils.image_utils as image_utils
import micro_dl.input.preprocessor_utils as preproc_utils


class ImagePreprocessor(metaclass=ABCMeta):
    """Base class for volume splitting, cropping and flatfield correction"""

    def __init__(self, input_dir, base_output_dir, verbose=0):
        """
        :param str input_dir: Input directory,
            containing all channels (inputs and target) you will use for training
        :param str base_output_dir: base folder for storing the individual
         image and cropped volumes
        :param int verbose: specifies the logging level: NOTSET:0, DEBUG:10,
         INFO:20, WARNING:30, ERROR:40, CRITICAL:50
        """

        self.input_dir = input_dir
        self.channel_dirs = glob.glob(self.input_dir + "/*/")
        self.base_output_dir = base_output_dir
        # Create output directory if it doesn't exist already
        os.makedirs(self.base_output_dir, exist_ok=True)
        # Create volume dir if it doesn't exist already
        self.volume_dir = os.path.join(self.base_output_dir, 'image_volumes')
        os.makedirs(self.volume_dir, exist_ok=True)
        # Validate and instantiate logging
        log_levels = [0, 10, 20, 30, 40, 50]
        if verbose in log_levels:
            self.verbose = verbose
        else:
            self.verbose = 10
        self.logger = self._init_logger()

    def _init_logger(self):
        """
        Initialize logger for pre-processing

        Logger outputs to console and log_file
        """

        logger = logging.getLogger('preprocessing')
        logger.setLevel(self.verbose)
        logger.propagate = False

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.verbose)
        logger.addHandler(stream_handler)

        logger_fname = os.path.join(self.base_output_dir, 'preprocessing.log')
        file_handler = logging.FileHandler(logger_fname)
        file_handler.setLevel(self.verbose)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _log_info(self, msg):
        """Log info"""

        if self.verbose > 0:
            self.logger.info(msg)

    def channel_validator(self):
        """
        Input directory should contain subdirectories consisting of channels
        numbered 0, ...
        This function makes sure images have matching shapes and unique indices
        in each channel.

        :return list of ints channel_nrbs: Channel numbers determined by searching
            input_dir subfolder names for ints
        :return list of ints im_indices: Unique image indices. Must be matching
            in all the subfolders of input_dir
        """
        assert len(self.channel_dirs) > 1, "Must be at least an input and a target channel"
        # Make sure all input directories contain images with the same indices and shape
        im_shapes = []
        im_indices = []
        channel_nbrs = []
        for i in self.channel_dirs:
            im_shape, im_ind = self.image_validator(i)
            im_shapes.append(im_shape)
            im_indices.append(im_ind)
            channel_nbrs.append(self.get_channel_from_dir(i))
        # Make sure image shapes and indices match across channels
        nbr_idxs = len(im_indices[0])
        for i in range(1, len(self.channel_dirs)):
            idx_overlap = set(im_indices[0]).intersection(im_indices[i])
            assert len(idx_overlap) == nbr_idxs, \
                "Index mismatch between channels {} and {}".format(
                    self.channel_dirs[0],
                    self.channel_dirs[i])
            assert im_shapes[0] == im_shapes[i], \
                "Image shape mismatch between channels {} and {}".format(
                    self.channel_dirs[0],
                    self.channel_dirs[i])
        self._log_info("found channels: {}".format(channel_nbrs))
        return channel_nbrs, im_indices[0]

    def _get_sorted_names(self, image_dir):
        """
        Get image names in directory and sort them by their indices
        :param image_dir:
        :return:
        """
        ims = [f for f in os.listdir(image_dir) if not f.startswith('.')]
        # Sort image names according to indices
        return natsort.natsorted(ims)

    def _read_or_catch(self, dir_name, im_name):
        try:
            im = cv2.imread(os.path.join(dir_name, im_name), cv2.IMREAD_ANYDEPTH)
        except IOError as e:
            print(e)
        return im

    def image_validator(self, image_dir):
        """
        Make sure all images in a directory have unique indexing and the same
        shape.

        :param str image_dir: Directory containing opencv readable images
        :return tuple im_shape: Image shape if all image have the same one

        :return list im_indices: Unique indices for the images

        :throws IOError: If images can't be read
        """
        im_names = self._get_sorted_names(image_dir)
        assert len(im_names) > 1, "Only one or less images in directory " + image_dir
        # Read first image to determine shape
        im = self._read_or_catch(image_dir, im_names[0])
        im_shape = im.shape
        # Determine indexing
        idx0 = re.findall("\d+", im_names[0])
        idx1 = re.findall("\d+", im_names[1])
        assert len(idx0) == len(idx1), "Different numbers of indices in file names"
        potential_idxs = np.zeros(len(idx0))
        for idx, (i, j) in enumerate(zip(idx0, idx1)):
            potential_idxs[idx] = abs(int(j) - int(i))
        idx_pos = np.where(potential_idxs > 0)[0]
        # There should only be one index (varying integer) in filenames
        assert len(idx_pos) == 1, ("Unclear indexing,"
                                   "more than one varying int in file names")
        # Loop through all images
        # check that shape is constant and collect indices
        im_indices = np.zeros(len(im_names), dtype=int)
        for i, im_name in enumerate(im_names):
            im = self._read_or_catch(image_dir, im_name)
            assert im.shape == im_shape, "Mismatching image shape in " + im_name
            im_indices[i] = int(re.findall("\d+", im_name)[idx_pos[0]])

        # Make sure there's a unique index for each image
        assert len(im_indices) == len(np.unique(im_indices)), \
            "Images don't have unique indexing"
        msg = '{} contains indices: {}'.format(image_dir, im_indices)
        self._log_info(msg)
        return im_shape, im_indices

    def get_channel_from_dir(self, image_dir):
        """
        Get channel number, assuming it's the last part of the
        image directory name.

        :param str image_dir: Image directory

        :return int channel_nbr: channel number
        """
        strs = image_dir.split("/")
        pos = -1
        if len(strs[pos]) == 0:
            pos = -2

        channel_nbr = re.findall("\d+", strs[pos])
        assert len(channel_nbr) == 1, ("Couldn't find channel number in ",
            image_dir)
        return int(channel_nbr[0])

    def save_images_as_npy(self,
                           channel_nbrs,
                           im_indices,
                           mask_channels=None,
                           num_timepoints=0):
        """
        Saves the individual images as a npy files.
        This only supports images from one timepoint for now, timepoints is
        just added for consistency with the lif preprocessor.

        :param str img_fname: fname with full path of the Lif file
        :param int/list mask_channels: channels from which masks have to be
         generated
        :param int focal_plane_idx: focal plane to consider
        """
        # Skipping this for now
        num_pix_z = 1
        size_x_um = 0
        size_y_um = 0
        size_z_um = 0

        # Loop through all timepoints, add image info to csv and write data as npy
        records = []
        for timepoint_idx in range(num_timepoints):
            # Create timepoint directory
            timepoint_dir = os.path.join(self.volume_dir,
                                         'timepoint_{}'.format(timepoint_idx))
            os.makedirs(timepoint_dir, exist_ok=True)

            for channel_i, channel_idx in enumerate(channel_nbrs):
                # Create output channel directory
                out_channel_dir = os.path.join(
                    timepoint_dir,
                    'channel_{}'.format(channel_idx))
                os.makedirs(out_channel_dir, exist_ok=True)
                # Get sorted image names
                in_channel_dir = self.channel_dirs[channel_i]
                im_names = self._get_sorted_names(in_channel_dir)
                # Save images for indices found in image_validator
                for sample_i, sample_idx in enumerate(im_indices):
                    cur_records = self.save_each_image(
                        in_channel_dir,
                        out_channel_dir,
                        im_names[sample_i],
                        channel_idx,
                        sample_idx,
                        num_pix_z,
                        timepoint_idx,
                        size_x_um,
                        size_y_um,
                        size_z_um)

                    records.extend(cur_records)
                msg = 'Wrote files for tp:{}, channel:{}'.format(
                    timepoint_idx, channel_idx
                )
                self._log_info(msg)
        # Create pandas dataframe
        df = pd.DataFrame.from_records(
            records,
            columns=['timepoint', 'channel_num', 'sample_num', 'slice_num',
                     'fname', 'size_x_microns', 'size_y_microns',
                     'size_z_microns']
        )
        metadata_fname = os.path.join(self.volume_dir,
                                      'image_volumes_info.csv')
        df.to_csv(metadata_fname, sep=',')

        # if mask_channels is not None:
        #     timepoints = [0]
        #     self.gen_mask(timepoints, mask_channels, 0)

    def save_each_image(self,
                        in_channel_dir,
                        out_channel_dir,
                        im_name,
                        channel_idx,
                        sample_idx,
                        num_pix_z,
                        timepoint_idx,
                        size_x_um,
                        size_y_um,
                        size_z_um):
        """
        Saves each individual image as a npy file

        Have to decide when to reprocess the file and when not to. Currently
        doesn't check if the file has already been processed.
        """
        records = []
        for z_idx in range(num_pix_z):
            cur_fname = os.path.join(
                out_channel_dir, 'image_n{}_z{}.npy'.format(sample_idx, z_idx)
            )
            # image voxels are 16 bits
            img = self._read_or_catch(in_channel_dir, im_name)
            np.save(cur_fname, img, allow_pickle=True, fix_imports=True)
            msg = 'Generated file:{}'.format(cur_fname)
            self._log_info(msg)
            # add wavelength info perhaps?
            records.append((timepoint_idx, channel_idx, sample_idx, z_idx,
                            cur_fname, size_x_um, size_y_um, size_z_um))
        return records

    def get_row_idx(self, volume_metadata, timepoint_idx,
                    channel_idx, focal_plane_idx=None):
        """Get the indices for images with timepoint_idx and channel_idx"""

        row_idx = ((volume_metadata['timepoint'] == timepoint_idx) &
                   (volume_metadata['channel_num'] == channel_idx) &
                   (volume_metadata['slice_num'] == focal_plane_idx))
        return row_idx

    def flat_field_corr(self, focal_plane_idx=None):
        """Estimates flat field correction image"""

        meta_fname = os.path.join(self.volume_dir, 'image_volumes_info.csv')
        try:
            volume_metadata = pd.read_csv(meta_fname)
        except IOError as e:
            self.logger.error('cannot read individual image info:' + str(e))
            raise

        all_channels = volume_metadata['channel_num'].unique()
        flat_field_dir = os.path.join(self.base_output_dir,
                                      'flat_field_images')
        os.makedirs(flat_field_dir, exist_ok=True)
        for tp_idx in volume_metadata['timepoint'].unique():
            for channel_idx in all_channels:
                row_idx = self.get_row_idx(volume_metadata, tp_idx,
                                           channel_idx, focal_plane_idx)
                channel_metadata = volume_metadata[row_idx]
                for idx, row in channel_metadata.iterrows():
                    sample_fname = row['fname']
                    cur_image = np.load(sample_fname)
                    n_dim = len(cur_image.shape)
                    if n_dim == 3:
                        cur_image = np.mean(cur_image, axis=2)
                    if idx == 0:
                        summed_image = cur_image.astype('float64')
                    else:
                        summed_image += cur_image
                mean_image = summed_image / len(row_idx)
                # Compute flatfield from from mean image of stack
                # TODO (Jenny): it currently samples median values
                # from a mean images, not very statistically meaningful
                # but easier than computing median of image stack
                flatfield = image_utils.get_flatfield(mean_image)
                fname = 'flat-field_tp-{}_channel-{}.npy'.format(tp_idx,
                                                                 channel_idx)
                cur_fname = os.path.join(flat_field_dir, fname)
                np.save(cur_fname, flatfield,
                        allow_pickle=True, fix_imports=True)

    def save_flat_field_corrected_images(self):
        """Saves flat field corrected images

        For the current set, no structure in flat field or averaged images,
        ignore for now
        """
        raise NotImplementedError

    def gen_mask(self, timepoint_ids, channel_ids, focal_plane_idx=None):
        """Generates a binary mask based on summation of channels"""

        meta_fname = os.path.join(self.volume_dir, 'image_volumes_info.csv')
        try:
            volume_metadata = pd.read_csv(meta_fname)
        except IOError as e:
            self.logger.error('cannot read individual image info:', str(e))
            raise

        if isinstance(channel_ids, int):
            channel_ids = [channel_ids]
        ch_str = '-'.join(map(str, channel_ids))

        for tp_idx in timepoint_ids:
            row_idx = self.get_row_idx(volume_metadata, tp_idx, channel_ids[0],
                                       focal_plane_idx)
            metadata = volume_metadata[row_idx]
            tp_dir = os.path.join(self.volume_dir,
                                  'timepoint_{}'.format(tp_idx))
            mask_dir = os.path.join(tp_dir, 'mask_{}'.format(ch_str))
            os.makedirs(mask_dir, exist_ok=True)

            fnames = [os.path.split(row['fname'])[1]
                      for _, row in metadata.iterrows()]
            for fname in fnames:
                for idx, channel in enumerate(channel_ids):
                    cur_fname = os.path.join(
                        tp_dir, 'channel_{}'.format(channel), fname
                    )
                    if idx == 0:
                        summed_image = np.load(cur_fname)
                    else:
                        summed_image += np.load(cur_fname)
                thr = threshold_otsu(summed_image, nbins=512)
                str_elem = disk(5)
                thr_image = binary_opening(summed_image >= thr, str_elem)
                mask = binary_fill_holes(thr_image)
                np.save(os.path.join(mask_dir, fname), mask,
                        allow_pickle=True, fix_imports=True)

    def crop_images(self, tile_size, step_size,
                    channel_ids=-1, focal_plane_idx=0,
                    mask_channel_ids=None, min_fraction=None,
                    isotropic=True):
        """Crop image volumes in the specified channels

        Isotropic here refers to the same dimension/shape along x,y,z and not
        really isotropic resolution in mm.

        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In
         case of no overlap, the step size is tile_size. If overlap,
         step_size < tile_size
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param list channel_ids: crop volumes in the given channels.
         default=-1, crop all channels
        :param int focal_plane_idx: focal plane to consider
        :param list/int mask_channel_ids: channels from which masks have to be
         generated
        :param float min_fraction: minimum volume fraction of the ROI to retain
         a tile
        """
        print(os.path.join(self.volume_dir, 'image_volumes_info.csv'))
        volume_metadata = pd.read_csv(os.path.join(self.volume_dir,
                                                   'image_volumes_info.csv'))
        available_channels = volume_metadata['channel_num'].unique()
        if isinstance(channel_ids, int) and channel_ids == -1:
            channel_ids = available_channels

        channel_indicator = [c in available_channels for c in channel_ids]
        assert np.all(channel_indicator)

        if mask_channel_ids is not None:
            assert min_fraction > 0.0
            if isinstance(channel_ids, int):
                channel_ids = [channel_ids]
            ch_str = '-'.join(map(str, mask_channel_ids))
            mask_dir_name = 'mask_{}'.format(ch_str)

        str_tile_size = '-'.join([str(val) for val in tile_size])
        str_step_size = '-'.join([str(val) for val in step_size])
        cropped_dir_name = 'image_tile_{}_step_{}'.format(str_tile_size,
                                                          str_step_size)
        cropped_dir = os.path.join(self.base_output_dir, cropped_dir_name)
        os.makedirs(cropped_dir, exist_ok=True)
        metadata_fname = os.path.join(cropped_dir,
                                      'cropped_images_info.csv')

        for timepoint_idx in volume_metadata['timepoint'].unique():
            timepoint_dir = os.path.join(cropped_dir,
                                         'timepoint_{}'.format(timepoint_idx))
            os.makedirs(timepoint_dir, exist_ok=True)
            if mask_channel_ids is not None:
                mask_dir = os.path.join(self.volume_dir,
                                        'timepoint_{}'.format(timepoint_idx),
                                        mask_dir_name)
                cropped_mask_dir = os.path.join(timepoint_dir, mask_dir_name)
                os.makedirs(cropped_mask_dir, exist_ok=True)
                crop_indices_dict = preproc_utils.get_crop_indices(
                    mask_dir, min_fraction, cropped_mask_dir, tile_size,
                    step_size, isotropic
                )
            for channel_idx in channel_ids:
                print(timepoint_idx, channel_idx, focal_plane_idx)
                row_idx = self.get_row_idx(volume_metadata, timepoint_idx,
                                           channel_idx, focal_plane_idx)
                channel_metadata = volume_metadata[row_idx]
                channel_dir = os.path.join(timepoint_dir,
                                           'channel_{}'.format(channel_idx))
                os.makedirs(channel_dir, exist_ok=True)
                metadata = []
                for _, row in channel_metadata.iterrows():
                    sample_fname = row['fname']
                    print(sample_fname)
                    cur_image = np.load(sample_fname)
                    if mask_channel_ids is not None:
                        _, fname = os.path.split(sample_fname)
                        cropped_image_data = preproc_utils.crop_at_indices(
                            cur_image, crop_indices_dict[fname], isotropic
                        )
                    else:
                        cropped_image_data = preproc_utils.crop_image(
                            input_image=cur_image, tile_size=tile_size,
                            step_size=step_size, isotropic=isotropic
                        )
                    for id_img_tuple in cropped_image_data:
                        xyz_idx = id_img_tuple[0]
                        img_fname = 'n{}_{}'.format(row['sample_num'], xyz_idx)
                        cropped_img = id_img_tuple[1]
                        cropped_img_fname = os.path.join(
                            channel_dir, '{}.npy'.format(img_fname)
                        )
                        np.save(cropped_img_fname, cropped_img,
                                allow_pickle=True, fix_imports=True)
                        metadata.append((row['timepoint'], row['channel_num'],
                                         row['sample_num'], row['slice_num'],
                                         cropped_img_fname))
                msg = 'Cropped images for channel:{}'.format(channel_idx)
                self._log_info(msg)
                fname_header = 'fname_{}'.format(channel_idx)
                cur_df = pd.DataFrame.from_records(
                    metadata,
                    columns=['timepoint', 'channel_num', 'sample_num',
                             'slice_num', fname_header]
                )
                if channel_idx == 0:
                    df = cur_df
                else:
                    df = pd.read_csv(metadata_fname, sep=',', index_col=0)
                    df[fname_header] = cur_df[fname_header]
                df.to_csv(metadata_fname, sep=',')
