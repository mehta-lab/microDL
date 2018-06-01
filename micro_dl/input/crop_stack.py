"""Crop images for training"""

import numpy as np
import os
import pandas as pd
import pickle

from micro_dl.input.gen_crop_masks import MaskProcessor
from micro_dl.utils.aux_utils import get_row_idx, validate_tp_channel
from micro_dl.utils.image_utils import (apply_flat_field_correction,
                                        crop_image, crop_at_indices)


def save_cropped_images(cropped_image_info, meta_row, channel_dir,
                        cropped_meta):
    """Save cropped images for individual/sample image

    :param list cropped_image_info: a list with tuples of cropped image id
     of the format xxmin-xmz_yymin-ymax_zzmin-zmax and cropped image
    :param pd.DataFrame(row) meta_row: row of metadata from split images
    :param str channel_dir: dir to save cropped images
    :param list cropped_meta: list of tuples holding meta info for cropped
     images
    """

    for id_img_tuple in cropped_image_info:
        xyz_idx = id_img_tuple[0]
        img_fname = 'n{}_{}'.format(meta_row['sample_num'], xyz_idx)
        cropped_img = id_img_tuple[1]
        cropped_img_fname = os.path.join(
            channel_dir, '{}.npy'.format(img_fname)
        )
        np.save(cropped_img_fname, cropped_img,
                allow_pickle=True, fix_imports=True)
        cropped_meta.append(
            (meta_row['timepoint'], meta_row['channel_num'],
             meta_row['sample_num'], meta_row['slice_num'],
             cropped_img_fname)
        )


def save_cropped_meta(cropped_meta, cur_channel, cropped_dir):
    """Save meta data for cropped images

    :param list cropped_meta: list of tuples holding meta info for cropped
     images
    :param int cur_channel: channel being cropped
    :param str cropped_dir: dir to save meta data
    """

    fname_header = 'fname_{}'.format(cur_channel)
    cur_df = pd.DataFrame.from_records(
        cropped_meta,
        columns=['timepoint', 'channel_num', 'sample_num',
                 'slice_num', fname_header]
    )
    metadata_fname = os.path.join(cropped_dir,
                                  'cropped_images_info.csv')
    if cur_channel == 0:
        df = cur_df
    else:
        df = pd.read_csv(metadata_fname, sep=',', index_col=0)
        df[fname_header] = cur_df[fname_header]
    df.to_csv(metadata_fname, sep=',')


class ImageStackCropper:
    """Crops all images images in a stack"""

    def __init__(self, base_output_dir, tile_size, step_size,
                 timepoint_ids=-1, crop_channels=-1, correct_flat_field=False,
                 isotropic=False):
        """Init

        Isotropic here refers to the same dimension/shape along x,y,z and not
        really isotropic resolution in mm.

        :param str base_output_dir: base folder for storing the individual
         and cropped images
        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In case
         of no overlap, the step size is tile_size. If overlap, step_size <
         tile_size
        :param list/int timepoint_ids: timepoints to consider
        :param list/int crop_channels: crop images in the given channels.
         default=-1, crop all channels
        :param bool correct_flat_field: bool indicator for correcting for flat
         field
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :return: a list with tuples of cropped image id of the format
         xxmin-xmax_yymin-ymax_zzmin-zmax and cropped image
        """

        self.base_output_dir = base_output_dir
        volume_metadata = pd.read_csv(os.path.join(
            self.base_output_dir, 'split_images', 'split_images_info.csv'
        ))
        self.volume_metadata = volume_metadata

        tp_channel_ids = validate_tp_channel(volume_metadata,
                                             timepoint_ids=timepoint_ids,
                                             channel_ids=crop_channels)

        self.crop_channels = tp_channel_ids['channels']
        self.timepoint_ids = tp_channel_ids['timepoints']

        msg = 'Incompatible tile and step size, not the same length'
        assert len(tile_size) == len(step_size), msg
        self.tile_size = tile_size
        self.step_size = step_size

        str_tile_size = '-'.join([str(val) for val in tile_size])
        str_step_size = '-'.join([str(val) for val in step_size])
        cropped_dir_name = 'image_tile_{}_step_{}'.format(str_tile_size,
                                                          str_step_size)
        cropped_dir = os.path.join(base_output_dir, cropped_dir_name)
        os.makedirs(cropped_dir, exist_ok=True)
        self.cropped_dir = cropped_dir
        if isotropic:
            isotropic_shape = [tile_size[0], ] * len(tile_size)
            msg = 'tile size is not isotropic'
            assert list(tile_size) == isotropic_shape, msg
        self.isotropic = isotropic
        self.correct_flat_field = correct_flat_field

    def crop_stack(self, focal_plane_idx=None):
        """Crop images in the specified channels

        :param int focal_plane_idx: Index of which focal plane acquisition to
         use (2D)
        """

        for tp_idx in self.timepoint_ids:
            tp_dir = os.path.join(self.cropped_dir,
                                  'timepoint_{}'.format(tp_idx))
            os.makedirs(tp_dir, exist_ok=True)
            for channel in self.crop_channels:
                row_idx = get_row_idx(volume_metadata, tp_idx, channel,
                                      focal_plane_idx)
                channel_metadata = self.volume_metadata[row_idx]
                channel_dir = os.path.join(tp_dir,
                                           'channel_{}'.format(channel))
                os.makedirs(channel_dir, exist_ok=True)

                flat_field_image = np.load(
                    os.path.join(self.base_output_dir, 'split_images',
                                 'flat_field_images',
                                 'flat-field_channel-{}.npy'.format(channel)
                                 )
                )
                metadata = []
                for _, row in channel_metadata.iterrows():
                    sample_fname = row['fname']
                    cur_image = np.load(sample_fname)
                    if self.correct_flat_field:
                        cur_image = apply_flat_field_correction(
                            cur_image, flat_field_image=flat_field_image
                        )
                    cropped_image_data = crop_image(
                        input_image=cur_image, tile_size=self.tile_size,
                        step_size=self.step_size, isotropic=self.isotropic
                    )
                    save_cropped_images(cropped_image_data, row,
                                        channel_dir, metadata)
                save_cropped_meta(metadata, channel, self.cropped_dir)

    def crop_stack_by_indices(self, mask_channels, min_fraction,
                              save_cropped_masks=False, isotropic=False,
                              focal_plane_idx=None):
        """Crop and retain tiles that meet ROI_vf >=  min_fraction

        :param int/list mask_channels: generate mask from the sum of these
         (flurophore) channels
        :param float min_fraction: threshold for using a cropped image for
         training. minimum volume fraction / percent occupied in cropped image
        :param bool save_cropped_masks: bool indicator for saving cropped masks
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param int focal_plane_idx: Index of which focal plane acquisition to
         use
        """

        msg = '0 <= min_fraction <= 1'
        assert min_fraction > 0.0 and min_fraction < 1.0, msg

        if isinstance(mask_channels, int):
            mask_channels = [mask_channels]
        mask_dir_name = '-'.join(map(str, mask_channels))
        mask_dir_name = 'mask_{}_vf-{}'.format(mask_dir_name, min_fraction)

        cropped_dir = '{}_vf-{}'.format(self.cropped_dir, min_fraction)

        for tp_idx in self.timepoint_ids:
            tp_dir = os.path.join(cropped_dir,
                                  'timepoint_{}'.format(tp_idx))
            os.makedirs(tp_dir, exist_ok=True)

            crop_indices_fname = os.path.join(
                self.base_output_dir, 'split_images',
                'timepoint_{}'.format(tp_idx),
                '{}_vf-{}.pkl'.format(mask_dir_name, min_fraction)
            )

            if not os.path.exists(crop_indices_fname):
                mask_gen_obj = MaskProcessor(
                    os.path.join(self.base_output_dir, 'split_images'),
                    tp_idx, mask_channels
                )
                cropped_mask_dir = os.path.join(self.cropped_dir,
                                                mask_dir_name)
                mask_gen_obj.get_crop_indices(min_fraction, self.tile_size,
                                              self.step_size, cropped_mask_dir,
                                              save_cropped_masks, isotropic)

            with open(crop_indices_fname, 'rb') as f:
                crop_indices_dict = pickle.load(f)

            for channel in self.crop_channels:
                row_idx = get_row_idx(volume_metadata, tp_idx, channel,
                                      focal_plane_idx)
                channel_metadata = self.volume_metadata[row_idx]
                channel_dir = os.path.join(tp_dir,
                                           'channel_{}'.format(channel))
                os.makedirs(channel_dir, exist_ok=True)

                flat_field_image = np.load(
                    os.path.join(self.base_output_dir, 'split_images',
                                 'flat_field_images',
                                 'flat-field_channel-{}.npy'.format(channel)
                                 )
                )

                metadata = []
                for _, row in channel_metadata.iterrows():
                    sample_fname = row['fname']
                    cur_image = np.load(sample_fname)
                    if self.correct_flat_field:
                        cur_image = apply_flat_field_correction(
                            cur_image, flat_field_image=flat_field_image
                        )
                    _, fname = os.path.split(sample_fname)
                    cropped_image_data = crop_at_indices(
                        cur_image, crop_indices_dict[fname], isotropic
                    )
                    save_cropped_images(cropped_image_data, row,
                                             channel_dir, metadata)
                save_cropped_meta(metadata, channel, cropped_dir)
