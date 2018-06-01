"""Generate masks from sum of flurophore channels"""

import glob
import numpy as np
import os
import pandas as pd
import pickle
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_opening

from micro_dl.utils.aux_utils import get_row_idx, validate_tp_channel
from micro_dl.utils.image_utils import apply_flat_field_correction, resize_mask


def process_cropped_masks(cropped_mask, min_fraction, sample_index_list,
                          crop_index, sample_idx, op_dir,
                          save_cropped_mask=False, isotropic=False):
    """Saves the cropped mask to op_dir.

    :param np.array cropped_mask: cropped mask with shape = tile_size
    :param float min_fraction: threshold for using a cropped image for
     training. minimum volume fraction / percent occupied in cropped image
    :param list sample_index_list: list that holds crop indices for the
     current image
    :param list crop_index: indices used for cropping
    :param int sample_idx: sample number
    :param str op_dir: dir to save cropped images
    :param bool save_cropped_masks: bool indicator for saving cropped masks
    :param bool isotropic: bool indicator for isotropic resolution (if 3D)
    """

    roi_vf = np.sum(cropped_mask) / np.prod(cropped_mask.shape)
    if roi_vf >= min_fraction:
        sample_index_list.append(crop_index)
        if save_cropped_mask:
            img_id = 'n{}_x{}_{}_y{}_{}'.format(
                sample_idx, crop_index[0], crop_index[1], crop_index[2],
                crop_index[3]
            )
            if len(cropped_mask.shape) == 3:
                img_id = '{}_z{}-{}.npy'.format(img_id, crop_index[4],
                                                crop_index[5])
                if isotropic:
                    cropped_mask = resize_mask(cropped_mask,
                                               [cropped_mask.shape[0], ] * 3)
            else:
                img_id = '{}.npy'.format(img_id)

            cropped_mask_fname = os.path.join(op_dir, img_id)
            np.save(cropped_mask_fname, cropped_mask,
                    allow_pickle=True, fix_imports=True)


class MaskProcessor:
    """Generate masks and get crop indices based on min vol fraction"""

    def __init__(self, split_dir, mask_channels, timepoint_ids=-1):
        """Init.

        :param str split_dir: dir with split images from stack (or individual
         sample images)
        :param list/int timepoint_ids: timepoints to consider
        :param int/list mask_channels: generate mask from the sum of these
         (flurophore) channels
        """

        self.mask_channels = mask_channels
        self.split_dir = split_dir
        self.timepoint_ids = timepoint_ids

        if isinstance(mask_channels, int):
            mask_channels = [mask_channels]
        mask_dir_name = '-'.join(map(str, mask_channels))
        self.mask_dir_name = 'mask_{}'.format(mask_dir_name)

    def generate_masks(self, focal_plane_idx=None, correct_flat_field=False):
        """Generate masks from flat-field corrected flurophore images.

        :param int focal_plane_idx: Index of which focal plane acquisition to
         use
        :param bool correct_flat_field: bool indicator to correct for flat
         field or not
        """

        meta_fname = os.path.join(self.split_dir, 'split_images_info.csv')
        try:
            volume_metadata = pd.read_csv(meta_fname)
        except IOError as e:
            e.args += 'cannot read split image info'
            raise

        tp_channel_ids = validate_tp_channel(volume_metadata,
                                             timepoint_ids=self.timepoint_ids,
                                             channel_ids=self.mask_channels)
        self.timepoint_ids = tp_channel_ids['timepoints']
        self.mask_channels = tp_channel_ids['channels']

        for tp_idx in self.timepoint_ids:
            row_idx = get_row_idx(volume_metadata, tp_idx,
                                  self.mask_channels[0], focal_plane_idx)
            metadata = volume_metadata[row_idx]
            tp_dir = os.path.join(self.split_dir,
                                  'timepoint_{}'.format(tp_idx))
            mask_dir = os.path.join(tp_dir, self.mask_dir_name)
            os.makedirs(mask_dir, exist_ok=True)

            fnames = [os.path.split(row['fname'])[1]
                      for _, row in metadata.iterrows()]
            for fname in fnames:
                for idx, channel in enumerate(self.mask_channels):
                    cur_fname = os.path.join(
                        tp_dir, 'channel_{}'.format(channel), fname
                    )
                    cur_image = np.load(cur_fname)
                    if correct_flat_field:
                        cur_image = apply_flat_field_correction(
                            self.split_dir, channel, cur_image
                        )
                    if idx == 0:
                        summed_image = cur_image
                    else:
                        summed_image += cur_image
                thr = threshold_otsu(summed_image, nbins=512)
                str_elem = disk(5)
                thr_image = binary_opening(summed_image >= thr, str_elem)
                mask = binary_fill_holes(thr_image)
                np.save(os.path.join(mask_dir, fname), mask,
                        allow_pickle=True, fix_imports=True)

    def get_crop_indices(self, min_fraction, tile_size, step_size,
                         cropped_mask_dir=None, save_cropped_masks=False,
                         isotropic=False):
        """Get crop indices and save mask for tiles with roi_vf >= min_fraction

        Crops and saves mask to mask_output_dir. saves a dict with fname as
        keys and list of indices as values.

        :param float min_fraction: threshold for using a cropped image for
         training. minimum volume fraction / percent occupied in cropped image
        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In case
         of no overlap, the step size is tile_size. If overlap, step_size <
         tile_size
        :param str cropped_mask_dir: directory to save the cropped masks
        :param bool save_cropped_masks: bool indicator for saving cropped masks
        :param bool isotropic: if 3D, make the grid/shape isotropic
        """

        msg = '0 <= min_fraction <= 1'
        assert min_fraction > 0.0 and min_fraction < 1.0, msg
        msg = 'tile and step size are not of same length'
        assert len(tile_size) == len(step_size), msg
        if save_cropped_masks:
            assert cropped_mask_dir is not None
        if isotropic:
            isotropic_shape = [tile_size[0], ] * len(tile_size)
            msg = 'tile size is not isotropic'
            assert list(tile_size) == isotropic_shape, msg

        for tp_idx in self.timepoint_ids:
            mask_ip_dir = os.path.join(self.split_dir,
                                       'timepoint_{}'.format(tp_idx),
                                       self.mask_dir_name)
            masks_in_dir = glob.glob(os.path.join(mask_ip_dir, '*.npy'))
            index_dict = {}

            for mask_idx, mask_fname in enumerate(masks_in_dir):
                _, fname = os.path.split(mask_fname)
                mask = np.load(mask_fname)
                size_x = mask.shape[0]
                size_y = mask.shape[1]
                n_dim = len(mask.shape)
                if n_dim == 3:
                    size_z = mask.shape[2]

                sample_num = int(fname.split('_')[1][1:])
                cur_index_list = []
                for x in range(0, size_x - tile_size[0] + 1, step_size[0]):
                    for y in range(0, size_y - tile_size[1] + 1, step_size[1]):
                        if n_dim == 3:
                            for z in range(0, size_z - tile_size[2] + 1,
                                           step_size[2]):
                                cropped_mask = mask[x: x + tile_size[0],
                                                    y: y + tile_size[1],
                                                    z: z + tile_size[2]]
                                cur_index = [x, x + tile_size[0],
                                             y, y + tile_size[1],
                                             z, z + tile_size[2]]
                                process_cropped_masks(
                                    cropped_mask, min_fraction, cur_index_list,
                                    cur_index, sample_num, cropped_mask_dir,
                                    save_cropped_masks, isotropic
                                )
                        else:
                            cropped_mask = mask[x: x + tile_size[0],
                                                y: y + tile_size[1]]
                            cur_index = [x, x + tile_size[0],
                                         y, y + tile_size[1]]
                            process_cropped_masks(
                                cropped_mask, min_fraction, cur_index_list,
                                cur_index, sample_num, cropped_mask_dir,
                                save_cropped_masks
                            )
                index_dict[fname] = cur_index_list
            dict_fname = os.path.join(
                self.split_dir, 'timepoint_{}'.format(tp_idx),
                '{}_vf-{}.pkl'.format(self.mask_dir_name, min_fraction)
            )
            with open(dict_fname, 'wb') as f:
                pickle.dump(index_dict, f)