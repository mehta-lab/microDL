"""Generate masks from sum of flurophore channels"""

import glob
import numpy as np
import os
import pandas as pd
import pickle

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils


class MaskProcessor:
    """Generate masks and get crop indices based on min vol fraction"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 channel_ids,
                 flat_field_dir=None,
                 timepoint_ids=-1,
                 int2str_len=3):
        """Init.

        :param str input_dir: Directory with image frames
        :param str output_dir: Directory where masks will be saved
        :param str flatfield_dir: Directory with flatfield images if flatfield correction
            is applied
        :param int/list channel_ids: generate mask from the sum of these
         (flurophore) channel indices
        :param list/int timepoint_ids: timepoints to consider
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
            channel_ids=channel_ids)
        self.timepoint_ids = metadata_ids['timepoint_ids']
        self.mask_channels = metadata_ids['channel_ids']

        self.input_dir = input_dir
        self.mask_dir_name = output_dir
        self.flat_field_dir = flat_field_dir
        self.int2str_len = int2str_len

    @staticmethod
    def _process_cropped_masks(cropped_mask,
                               min_fraction,
                               sample_index_list,
                               crop_index,
                               sample_idx,
                               output_dir,
                               save_cropped_mask=False,
                               isotropic=False):
        """Saves the cropped masks to output_dir.

        :param np.array cropped_mask: cropped mask same shape as images
        :param float min_fraction: threshold for using a cropped image for
         training. minimum volume fraction / percent occupied in cropped image
        :param list sample_index_list: list that holds crop indices for the
         current image
        :param list crop_index: indices used for cropping
        :param int sample_idx: sample number
        :param str output_dir: dir to save cropped images
        :param bool save_cropped_mask: bool indicator for saving cropped masks
        :param bool isotropic: bool indicator for isotropic resolution (if 3D)
        """

        roi_vf = np.mean(cropped_mask)
        if roi_vf >= min_fraction:
            sample_index_list.append(crop_index)
            if save_cropped_mask:
                img_id = 'n{}_r{}-{}_c{}-{}'.format(
                    sample_idx, crop_index[0], crop_index[1], crop_index[2],
                    crop_index[3]
                )
                if len(cropped_mask.shape) == 3:
                    img_id = '{}_sl{}-{}.npy'.format(img_id, crop_index[4],
                                                     crop_index[5])
                    if isotropic:
                        cropped_mask = image_utils.resize_mask(
                            cropped_mask, [cropped_mask.shape[0], ] * 3
                        )
                else:
                    img_id = '{}.npy'.format(img_id)

                cropped_mask_fname = os.path.join(output_dir, img_id)
                np.save(cropped_mask_fname, cropped_mask,
                        allow_pickle=True, fix_imports=True)

    def generate_masks(self,
                       focal_plane_idx=None,
                       correct_flat_field=False,
                       str_elem_radius=5):
        """
        Generate masks from flat-field corrected flurophore images.
        The sum of flurophore channels is thresholded to generate a foreground
        mask.

        :param int focal_plane_idx: Index of which focal plane acquisition to
         use
        :param bool correct_flat_field: bool indicator to correct for flat
         field or not
        :param int str_elem_radius: Radius of structuring element for morphological
            operations
        """
        # Get only the focal plane if specified
        if focal_plane_idx is not None:
            focal_plane_ids = [focal_plane_idx]
        else:
            focal_plane_ids = self.frames_metadata['slice_idx'].unique()

        # Loop through all the indices and create masks
        for slice_idx in focal_plane_ids:
            for time_idx in self.timepoint_ids:
                for pos_idx in np.unique(self.frames_metadata["pos_idx"]):
                    mask_images = []
                    for channel_idx in self.mask_channels:
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
                        cur_image = image_utils.read_image(file_path)
                        if correct_flat_field:
                            cur_image = image_utils.apply_flat_field_correction(
                                cur_image,
                                flat_field_dir=self.flat_field_dir,
                                channel_idx=channel_idx,
                            )
                        mask_images.append(cur_image)
                    # Combine channel images and generate mask
                    summed_image = np.sum(np.stack(mask_images), axis=0)
                    summed_image = summed_image.astype('float32')
                    mask = image_utils.create_mask(summed_image, str_elem_radius)
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

    def get_crop_indices(self,
                         min_fraction,
                         tile_size,
                         step_size,
                         cropped_mask_dir=None,
                         save_cropped_masks=False,
                         isotropic=False):
        """Get crop indices and save mask for tiles with roi_vf >= min_fraction

        Tiles an image and retains tiles that have minimum ROI / foreground.
        Saves the tiles to mask_output_dir. Saves a dict with fname as
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

        msg = 'min_fraction is expected to be within 5-50 %'
        assert min_fraction > 0.05 and min_fraction < 0.5, msg

        msg = 'tile and step size are not of same length'
        assert len(tile_size) == len(step_size), msg

        if save_cropped_masks:
            assert cropped_mask_dir is not None

        if isotropic:
            isotropic_shape = [tile_size[0], ] * len(tile_size)
            msg = 'tile size is not isotropic'
            assert list(tile_size) == isotropic_shape, msg

        for tp_idx in self.timepoint_ids:
            mask_ip_dir = os.path.join(self.input_dir,
                                       'timepoint_{}'.format(tp_idx),
                                       self.mask_dir_name)
            masks_in_dir = glob.glob(os.path.join(mask_ip_dir, '*.npy'))
            index_dict = {}

            for mask_idx, mask_fname in enumerate(masks_in_dir):
                _, fname = os.path.split(mask_fname)
                mask = np.load(mask_fname)
                n_rows = mask.shape[0]
                n_cols = mask.shape[1]
                n_dim = len(mask.shape)
                if n_dim == 3:
                    n_slices = mask.shape[2]

                sample_num = int(fname.split('_')[1][1:])
                cur_index_list = []
                for r in range(0, n_rows - tile_size[0] + 1, step_size[0]):
                    for c in range(0, n_cols - tile_size[1] + 1, step_size[1]):
                        if n_dim == 3:
                            for sl in range(0, n_slices - tile_size[2] + 1,
                                            step_size[2]):
                                cropped_mask = mask[r: r + tile_size[0],
                                                    c: c + tile_size[1],
                                                    sl: sl + tile_size[2]]
                                cur_index = [r, r + tile_size[0],
                                             c, c + tile_size[1],
                                             sl, sl + tile_size[2]]
                                self._process_cropped_masks(
                                    cropped_mask, min_fraction, cur_index_list,
                                    cur_index, sample_num, cropped_mask_dir,
                                    save_cropped_masks, isotropic
                                )
                        else:
                            cropped_mask = mask[r: r + tile_size[0],
                                                c: c + tile_size[1]]
                            cur_index = [r, r + tile_size[0],
                                         c, c + tile_size[1]]
                            self._process_cropped_masks(
                                cropped_mask, min_fraction, cur_index_list,
                                cur_index, sample_num, cropped_mask_dir,
                                save_cropped_masks
                            )
                index_dict[fname] = cur_index_list
            dict_fname = os.path.join(
                self.input_dir, 'timepoint_{}'.format(tp_idx),
                '{}_vf-{}.pkl'.format(self.mask_dir_name, min_fraction)
            )
            with open(dict_fname, 'wb') as f:
                pickle.dump(index_dict, f)
