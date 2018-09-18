"""Generate masks from sum of flurophore channels"""

import glob
import numpy as np
import os
import pandas as pd
import pickle

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils


class MaskProcessor:
    """Generate masks from channels"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 channel_ids,
                 flat_field_dir=None,
                 timepoint_ids=-1,
                 focal_plane_idx=-1,
                 int2str_len=3):
        """
        :param str input_dir: Directory with image frames
        :param str output_dir: Directory where masks will be saved
        :param str flat_field_dir: Directory with flatfield images if
            flatfield correction is applied
        :param int/list channel_ids: generate mask from the sum of these
         (flurophore) channel indices
        :param list/int timepoint_ids: timepoints to consider
        :param int focal_plane_idx: Index of which focal plane (z)
            acquisition to use (default -1 includes all slices)
        :param int int2str_len: Length of str when converting ints
        """

        meta_fname = glob.glob(os.path.join(input_dir, 'frames_meta.csv'))
        assert len(meta_fname) == 1, \
            "Can't find info.csv file in {}".format(input_dir)

        try:
            frames_metadata = pd.read_csv(meta_fname[0])
        except IOError as e:
            e.args += 'cannot read split image info'
            raise

        self.frames_metadata = frames_metadata
        metadata_ids = aux_utils.validate_metadata_indices(
            frames_metadata=frames_metadata,
            time_ids=timepoint_ids,
            channel_ids=channel_ids,
            slice_ids=focal_plane_idx,
        )
        self.timepoint_ids = metadata_ids['timepoint_ids']
        self.channel_ids = metadata_ids['channel_ids']
        self.slice_ids = metadata_ids['slice_ids']

        self.input_dir = input_dir
        self.mask_dir_name = output_dir
        self.flat_field_dir = flat_field_dir
        self.int2str_len = int2str_len

    def generate_masks(self,
                       correct_flat_field=False,
                       str_elem_radius=5):
        """
        Generate masks from flat-field corrected flurophore images.
        The sum of flurophore channels is thresholded to generate a foreground
        mask.

        :param bool correct_flat_field: bool indicator to correct for flat
         field or not
        :param int str_elem_radius: Radius of structuring element for morphological
            operations
        """
        # Loop through all the indices and create masks
        for slice_idx in self.slice_ids:
            for time_idx in self.timepoint_ids:
                for pos_idx in np.unique(self.frames_metadata["pos_idx"]):
                    mask_images = []
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
                        if correct_flat_field:
                            im = image_utils.apply_flat_field_correction(
                                input_image=im,
                                flat_field_dir=self.flat_field_dir,
                                channel_idx=channel_idx,
                            )
                        mask_images.append(im)
                    # Combine channel images and generate mask
                    summed_image = np.sum(np.stack(mask_images), axis=0)
                    summed_image = summed_image.astype('float32')
                    mask = image_utils.create_mask(
                        summed_image,
                        str_elem_radius,
                    )
                    # Create mask name for given slice, time and position
                    file_name = aux_utils.get_im_name(
                        time_idx=time_idx,
                        channel_idx=None,
                        slice_idx=slice_idx,
                        pos_idx=pos_idx,
                    )
                    # Save mask for given channels
                    np.save(os.path.join(self.mask_dir_name, file_name),
                            mask,
                            allow_pickle=True,
                            fix_imports=True)
