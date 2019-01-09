"""Generate masks from sum of flurophore channels"""

import numpy as np
import os

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils


class ImageResizer:
    """Resize images for given indices"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 scale_factor,
                 channel_ids=-1,
                 time_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 int2str_len=3):
        """
        :param str input_dir: Directory with image frames
        :param str output_dir: Base output directory
        :param float scale_factor: Scale factor for resizing frames
        :param int/list channel_ids: Channel indices to resize
            (default -1 includes all slices)
        :param list/int time_ids: timepoints to use
        :param int slice_ids: Index of slize (z) indices to use
        :param int pos_ids: Position (FOV) indices to use
        :param int int2str_len: Length of str when converting ints
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        assert scale_factor > 0, \
            "Scale factor should be positive float, not {}".format(scale_factor)
        self.scale_factor = scale_factor

        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        metadata_ids = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
        )
        self.time_ids = metadata_ids['time_ids']
        self.channel_ids = metadata_ids['channel_ids']
        self.slice_ids = metadata_ids['slice_ids']
        self.pos_ids = metadata_ids['pos_ids']

        # Create resize_dir as a subdirectory of output_dir
        self.resize_dir = os.path.join(
            self.output_dir,
            'resized_frames',
        )
        os.makedirs(self.resize_dir, exist_ok=True)

        self.int2str_len = int2str_len

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
        # Loop through all the indices and resize images
        for slice_idx in self.slice_ids:
            for time_idx in self.time_ids:
                for pos_idx in self.pos_ids:
                    for channel_idx in self.channel_ids:
                        frame_idx = aux_utils.get_meta_idx(
                            self.frames_metadata,
                            time_idx,
                            channel_idx,
                            slice_idx,
                            pos_idx,
                        )
                        file_path = os.path.join(
                            self.input_dir,
                            self.frames_metadata.loc[frame_idx, "file_name"],
                        )
                        im = image_utils.read_image(file_path)
                        # Here's where the resizing happens
                        im_resized = image_utils.rescale_image(im, self.scale_factor)
                        # Get name for given indices
                        file_name = aux_utils.get_im_name(
                            time_idx=time_idx,
                            channel_idx=channel_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                        )
                        # Save mask for given channels
                        np.save(os.path.join(self.resize_dir, file_name),
                                im_resized,
                                allow_pickle=True,
                                fix_imports=True)