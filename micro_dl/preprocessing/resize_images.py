"""Generate masks from sum of flurophore channels"""

import numpy as np
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as im_utils
import micro_dl.utils.mp_utils as mp_utils


class ImageResizer:
    """Resize images for given indices"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 scale_factor,
                 channel_ids,
                 time_ids,
                 slice_ids,
                 pos_ids,
                 int2str_len=3,
                 num_workers=4,
                 flat_field_dir=None,
                 flat_field_channels=[]):
        """
        :param str input_dir: Directory with image frames
        :param str output_dir: Base output directory
        :param float/list scale_factor: Scale factor for resizing frames.
        :param int/list channel_ids: Channel indices to resize
            (default -1 includes all slices)
        :param list time_ids: timepoints to use
        :param list slice_ids: Index of slice (z) indices to use
        :param list pos_ids: Position (FOV) indices to use
        :param int int2str_len: Length of str when converting ints
        :param int num_workers: number of workers for multiprocessing
        :param str flat_field_dir: dir with flat field images
        :param list flat_field_channels: Channels to apply flatfield correction
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        if isinstance(scale_factor, list):
            scale_factor = np.array(scale_factor)
        assert np.all(scale_factor > 0), \
            "Scale factor should be positive float, not {}".format(scale_factor)
        self.scale_factor = scale_factor

        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        self.time_ids = time_ids
        self.channel_ids = channel_ids
        self.slice_ids = slice_ids
        self.pos_ids = pos_ids
        self.flat_field_channels = flat_field_channels

        # Create resize_dir as a subdirectory of output_dir
        self.resize_dir = os.path.join(
            self.output_dir,
            'resized_images',
        )
        os.makedirs(self.resize_dir, exist_ok=True)

        self.int2str_len = int2str_len
        self.num_workers = num_workers
        self.flat_field_dir = flat_field_dir

    def get_resize_dir(self):
        """
        Return directory with resized images
        :return str resize_dir: Directory where resized images are stored
        """
        return self.resize_dir

    def resize_frames(self):
        """
        Resize frames for given indices.
        """
        assert isinstance(self.scale_factor, (float, int)), \
            'different scale factors provided for x and y'
        mp_args = []
        # Loop through all the given indices and resize images
        resized_metadata = aux_utils.get_sub_meta(
            self.frames_metadata,
            self.time_ids,
            self.channel_ids,
            self.slice_ids,
            self.pos_ids,
        )
        for i, meta_row in resized_metadata.iterrows():
            ff_path = im_utils.get_flat_field_path(
                self.flat_field_dir,
                meta_row['channel_idx'],
                self.flat_field_channels,
            )
            kwargs = {
                'meta_row': meta_row,
                'dir_name': self.input_dir,
                'output_dir': self.resize_dir,
                'scale_factor': self.scale_factor,
                'ff_path': ff_path,
            }
            mp_args.append(kwargs)
        # Multiprocessing of kwargs
        mp_utils.mp_resize_save(mp_args, self.num_workers)
        resized_metadata['dir_name'] = self.resize_dir
        resized_metadata = resized_metadata.sort_values(by=['file_name'])
        resized_metadata.to_csv(
            os.path.join(self.resize_dir, "frames_meta.csv"),
            sep=',',
        )

    def resize_volumes(self, num_slices_subvolume=-1):
        """Down or up sample volumes

        Overlap of one slice across subvolumes

        :param int num_slices_subvolume: num of 2D slices to include in each
         volume. if -1, include all slices
        """

        # assuming slice_ids will be continuous
        num_total_slices = len(self.slice_ids)
        if not isinstance(self.scale_factor, float):
            sc_str = '-'.join(self.scale_factor.astype('str'))
        else:
            sc_str = self.scale_factor

        mp_args = []
        if num_slices_subvolume == -1:
            num_slices_subvolume = len(self.slice_ids)
        num_blocks = np.floor(
            num_total_slices / (num_slices_subvolume - 1)
        ).astype('int')
        for time_idx in self.time_ids:
            for pos_idx in self.pos_ids:
                for channel_idx in self.channel_ids:
                    ff_path = im_utils.get_flat_field_path(
                        self.flat_field_dir,
                        channel_idx,
                        self.flat_field_channels,
                    )
                    for block_idx in range(num_blocks):
                        idx = self.slice_ids[0] + \
                              block_idx * (num_slices_subvolume - 1)
                        slice_start_idx = np.maximum(self.slice_ids[0], idx)
                        slice_end_idx = slice_start_idx + num_slices_subvolume
                        if slice_end_idx > self.slice_ids[-1]:
                            slice_end_idx = self.slice_ids[-1] + 1
                            slice_start_idx = slice_end_idx - num_slices_subvolume
                        op_fname = aux_utils.get_im_name(
                            time_idx,
                            channel_idx,
                            slice_start_idx,
                            pos_idx,
                            extra_field=sc_str,
                            ext='.npy',
                        )
                        write_fpath = os.path.join(self.resize_dir, op_fname)
                        mp_args.append((time_idx,
                                        pos_idx,
                                        channel_idx,
                                        slice_start_idx,
                                        slice_end_idx,
                                        self.frames_metadata,
                                        self.input_dir,
                                        write_fpath,
                                        self.scale_factor,
                                        ff_path))

        # Multiprocessing of kwargs
        resized_metadata_list = mp_utils.mp_rescale_vol(mp_args, self.num_workers)
        resized_metadata_df = pd.DataFrame.from_dict(resized_metadata_list)
        resized_metadata_df['dir_name'] = self.resize_dir
        resized_metadata_df.to_csv(
            os.path.join(self.resize_dir, 'frames_meta.csv'),
            sep=',',
        )

        if num_slices_subvolume == -1:
            slice_ids = self.slice_ids[0]
        else:
            slice_ids = self.slice_ids[0: -1: num_slices_subvolume - 1]

        return slice_ids
