"""Generate masks to be used as target images for segmentation"""

import numpy as np
import os
import pandas as pd
import pickle

from micro_dl.utils.aux_utils import validate_tp_channel, get_row_idx
from micro_dl.utils.image_utils import apply_flat_field_correction, create_mask


class MaskCreator:
    """Creates masks for segmentation"""

    def __init__(self, input_dir, input_channel_id, output_dir,
                 output_channel_id, timepoint_id=0, correct_flat_field=True,
                 tile_index_fname=None, study_meta_fname=None):
        """Init

        """

        assert os.path.exists(input_dir), 'input_dir does not exist'
        assert os.path.exists(output_dir), 'output_dir does not exist'
        self.input_dir = input_dir
        self.output_dir = output_dir

        if tile_index_fname:
            msg = 'tile index file does not exist'
            assert (os.path.exists(tile_index_fname) and
                    os.path.isfile(tile_index_fname)), msg
        self.tile_index_fname = tile_index_fname

        self.correct_flat_field = correct_flat_field

        if study_meta_fname:
            meta_fname = os.path.join(input_dir, 'split_images_info.csv')
            study_metadata = pd.read_csv(meta_fname)
        else:
            study_metadata = pd.read_csv(study_meta_fname)
        self.study_metadata = study_metadata

        avail_tp_channels = validate_tp_channel(study_metadata,
                                                timepoint_ids=timepoint_id,
                                                channel_ids=input_channel_id)

        msg = 'timepoint_id is not available'
        assert timepoint_id in avail_tp_channels['timepoints'], msg
        self.timepoint_id = timepoint_id

        msg = 'input_channel_id is not available'
        assert input_channel_id in avail_tp_channels['channels'], msg
        self.input_channel_id = input_channel_id

        msg = 'output_channel_id is already present'
        assert output_channel_id not in avail_tp_channels['channels'], msg
        self.output_channel_id = output_channel_id

    def create_masks_for_stack(self, str_elem_radius=3, focal_plane_idx=0):
        """Create masks

        :param...
        """

        for tp_idx in self.timepoint_id:
            for ch in self.input_channel_id:
                row_idx = get_row_idx(self.study_metadata, tp_idx,
                                      ch, focal_plane_idx)
                ch_meta = self.study_metadata[row_idx]
                cur_flat_field = np.load(os.path.join(
                    self.input_dir, 'flat_field_images',
                    'flat-field_channel-{}.npy'.format(ch)
                ))
                for meta_row in  ch_meta.iterrows():
                    cur_image = np.load(meta_row['fname'])
                    if self.correct_flat_field:
                        cur_image = apply_flat_field_correction(
                            cur_image, flat_field_image=cur_flat_field
                        )
                    mask = create_mask(cur_image,
                                       str_elem_size=str_elem_radius)
                    if tile_index_fname:
                        # read the pickle file and crop_at_indices
                    else:
                        # crop_image
                    # save the stack

    def _append_to_csv





