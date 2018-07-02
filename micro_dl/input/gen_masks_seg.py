"""Generate masks to be used as target images for segmentation"""
import cv2
import glob
import numpy as np
import os
import pandas as pd
import pickle

from micro_dl.utils.aux_utils import validate_tp_channel, get_row_idx
import micro_dl.utils.image_utils as image_utils


class MaskCreator:
    """Creates masks for segmentation"""

    def __init__(self, input_dir, input_channel_id, output_dir,
                 output_channel_id, timepoint_id=0, correct_flat_field=True,
                 study_meta_fname=None, focal_plane_idx=0):
        """Init

        If a pickle exists, it always uses crop_by_indices with a vf constraint
        """

        assert os.path.exists(input_dir), 'input_dir does not exist'
        assert os.path.exists(output_dir), 'output_dir does not exist'
        self.input_dir = input_dir
        self.output_dir = output_dir

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
        msg = 'input and output channel ids are not of same length'
        assert len(input_channel_id) == len(output_channel_id), msg
        self.output_channel_id = output_channel_id
        self.focal_place_idx = focal_plane_idx

    def create_masks_for_stack(self, str_elem_radius=3):
        """Create masks

        :param...
        """

        for tp_idx in self.timepoint_id:
            for ch_idx, ch in enumerate(self.input_channel_id):
                row_idx = get_row_idx(self.study_metadata, tp_idx,
                                      ch, self.focal_plane_idx)
                ch_meta = self.study_metadata[row_idx]
                cur_flat_field = np.load(os.path.join(
                    self.input_dir, 'flat_field_images',
                    'flat-field_channel-{}.npy'.format(ch)
                ))
                mask_dir = os.path.join(
                    self.input_dir, tp_idx,
                    'channel_{}'.format(self.output_channel_id[ch_idx])
                )
                os.makedirs(mask_dir, exist_ok=True)
                for _, meta_row in  ch_meta.iterrows():
                    sample_fname = meta_row['fname']
                    if sample_fname[-3:] == 'npy':
                        cur_image = np.load(meta_row['fname'])
                    else:
                        cur_image = cv2.imread(sample_fname,
                                               cv2.IMREAD_ANYDEPTH)

                    if self.correct_flat_field:
                        cur_image = image_utils.apply_flat_field_correction(
                            cur_image, flat_field_image=cur_flat_field
                        )
                    mask = image_utils.create_mask(
                        cur_image, str_elem_size=str_elem_radius
                    )
                    _, fname = os.path.split
                    mask_fname = os.path.join(mask_dir, fname)
                    np.save(mask, mask_fname,
                            allow_pickle=True, fix_imports=True)

    def tile_mask_stack(self, input_mask_dir, tile_index_fname=None,
                        tile_size=None, step_size=None, isotropic=False):
        """Tiles a stack of masks"""

        if tile_index_fname:
            msg = 'tile index file does not exist'
            assert (os.path.exists(tile_index_fname) and
                    os.path.isfile(tile_index_fname)), msg
            with open(tile_index_fname, 'rb') as f:
                crop_indices_dict = pickle.load(f)
        else:
            msg = 'tile_size and step_size are needed'
            assert tile_size is not None and step_size is not None, msg

        meta_dict = {}
        for cur_dir in input_mask_dir:
            cur_tp = int((cur_dir.split(os.sep)[-2]).split('_')[-1])
            cur_ch = int((cur_dir.split(os.sep)[-1]).split('_')[-1])
            #  read all mask npy files
            mask_fnames = glob.glob(os.path.join(cur_dir, '*.npy'))
            cropped_meta = []
            for cur_mask in mask_fnames:
                _, fname = os.path.split(cur_mask)
                sample_num = int(fname.split('_')[1][1:])
                if tile_index_fname:
                    cropped_image_data = image_utils.crop_at_indices(
                        input_image=cur_mask,
                        crop_indices=crop_indices_dict[fname],
                        isotropic=isotropic
                    )
                else:
                    cropped_image_data = image_utils.tile_image(
                        input_image=cur_mask,
                        tile_size=tile_size,
                        step_size=step_size,
                        isotropic=isotropic
                    )
                # save the stack
                for id_img_tuple in cropped_image_data:
                    rcsl_idx = id_img_tuple[0]
                    img_fname = 'n{}_{}.npy'.format(sample_num, rcsl_idx)
                    cropped_img = id_img_tuple[1]
                    cropped_img_fname = os.path.join(
                        self.output_dir,
                        'timepoint_{}'.format(cur_tp),
                        'channel_{}'.format(cur_ch),
                        img_fname
                    )
                    np.save(cropped_img_fname, cropped_img,
                            allow_pickle=True, fix_imports=True)
                    cropped_meta.append(
                            (cur_tp, cur_ch, sample_num, self.focal_place_idx,
                             cropped_img_fname)
                        )
            meta_dict[cur_dir] = cropped_meta
        return meta_dict
