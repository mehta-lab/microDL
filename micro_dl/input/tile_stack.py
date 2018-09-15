"""Tile images for training"""

import cv2
import numpy as np
import os
import pandas as pd
import pickle

from micro_dl.input.gen_crop_masks import MaskProcessor
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.normalize import hist_clipping, zscore
from micro_dl.utils.aux_utils import save_tile_meta, get_meta_idx
import micro_dl.utils.image_utils as image_utils


class ImageStackTiler:
    """Tiles all images images in a stack"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_size,
                 step_size,
                 timepoint_ids=-1,
                 tile_channels=-1,
                 flat_field_dir=None,
                 isotropic=False,
                 meta_path=None):
        """
        Normalizes images using z-score, then tiles them.
        Isotropic here refers to the same dimension/shape along row, col, slice
        and not really isotropic resolution in mm.

        :param str input_dir: Directory with frames to be tiled
        :param str output_dir: Directory for storing the tiled images
        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In case
         of no overlap, the step size is tile_size. If overlap, step_size <
         tile_size
        :param list/int timepoint_ids: timepoints to consider
        :param list/int tile_channels: tile images in the given channels.
         default=-1, tile all channels
        :param str flat_field_dir: Flatfield directory. None if no flatfield correction
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param meta_path: If none, assume metadata csv is in output_dir
            + split_images/ and is named split_images_info.csv
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        if meta_path is None:
            frames_metadata = pd.read_csv(os.path.join(
                self.input_dir, 'frames_meta.csv'
            ))
        else:
            frames_metadata = pd.read_csv(meta_path)
        self.frames_metadata = frames_metadata

        metadata_ids = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=timepoint_ids,
            channel_ids=tile_channels,
        )

        self.channels_ids = metadata_ids['channels']
        self.timepoint_ids = metadata_ids['timepoints']

        self.tile_size = tile_size
        self.step_size = step_size
        self.isotropic = isotropic
        self.flat_field_dir = flat_field_dir

    @staticmethod
    def _save_tiled_images(cropped_image_info, meta_row,
                           channel_dir, cropped_meta):
        """Save cropped images for individual/sample image
        :param list cropped_image_info: a list with tuples (cropped image id
         of the format rrmin-rmax_ccmin-cmax_slslmin-slmax and cropped image)
         for the current image
        :param pd.DataFrame(row) meta_row: row of metadata from split images
        :param str channel_dir: dir to save cropped images
        :param list cropped_meta: list of tuples with (cropped image id of the
        format rrmin-rmax_ccmin-cmax_slslmin-slmax, cropped image) for all
        images in current channel
        """

        for id_img_tuple in cropped_image_info:
            rcsl_idx = id_img_tuple[0]
            img_fname = 'n{}_{}'.format(meta_row['sample_num'], rcsl_idx)
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

    def _tile_channel(self,
                      tile_function,
                      channel_dir,
                      channel_metadata,
                      flat_field_image,
                      hist_clip_limits,
                      metadata,
                      crop_indices=None):
        """
        Tiles and saves tiles for one channel

        :param function tile_function: tile function. Either image_utils functions
            tile_image or crop_at_indices
        :param str channel_dir: dir for saving tiled images
        :param pd.DataFrame channel_metadata: DF with meta info for the current
         channel
        :param np.array flat_field_image: flat_filed image for this channel
        :param list hist_clip_limits: lower and upper hist clipping limits
        :param list metadata: list of tuples with tiled info (cropped image id
        of the format rrmin-rmax_ccmin-cmax_slslmin-slmax, cropped image)
        :param dict of lists crop_indices: dict with key as fname and values
         are list of crop indices
        """

        for _, row in channel_metadata.iterrows():
            sample_fname = row['fname']
            # Read npy or image
            if sample_fname[-3:] == 'npy':
                cur_image = np.load(sample_fname)
            else:
                cur_image = cv2.imread(sample_fname, cv2.IMREAD_ANYDEPTH)

            if self.correct_flat_field:
                cur_image = image_utils.apply_flat_field_correction(
                    cur_image, flat_field_image=flat_field_image
                )
            # normalize
            if hist_clip_limits is not None:
                cur_image = hist_clipping(cur_image,
                                          hist_clip_limits[0],
                                          hist_clip_limits[1])
            cur_image = zscore(cur_image)
            if tile_function == image_utils.tile_image:
                cropped_image_data = tile_function(
                    input_image=cur_image,
                    tile_size=self.tile_size,
                    step_size=self.step_size,
                    isotropic=self.isotropic,
                )
            elif tile_function == image_utils.crop_at_indices:
                assert crop_indices is not None
                _, fname = os.path.split(sample_fname)
                cropped_image_data = tile_function(
                    input_image=cur_image,
                    crop_indices=crop_indices[fname],
                    isotropic=self.isotropic,
                )
            else:
                raise ValueError('tile function invalid')
            self._save_tiled_images(
                cropped_image_data,
                row,
                channel_dir,
                metadata,
            )

    def tile_stack(self,
                   focal_plane_idx=None,
                   hist_clip_limits=None):
        """
        Tiles images in the specified channels.

        Saves a csv with columns ['time_idx', 'channel_idx', 'pos_idx',
        'slice_idx', 'file_name'] for all the tiles

        :param int focal_plane_idx: Index of which focal plane acquisition to
         use (2D).
        :param list hist_clip_limits: lower and upper percentiles used for
         histogram clipping.
        """
        # Get only the focal plane if specified
        if focal_plane_idx is not None:
            focal_plane_ids = [focal_plane_idx]
        else:
            focal_plane_ids = self.frames_metadata['slice_idx'].unique()
        # TODO: preallocating dataframe shape would save some time
        tiled_metadata = pd.DataFrame(columns=[
            "channel_idx",
            "slice_idx",
            "time_idx",
            "channel_name",
            "file_name",
            "pos_idx"])
        for channel_idx in self.channels_ids:
            # Perform flatfield correction if flatfield dir is specified
            flat_field_image = None
            if self.flat_field_dir is not None:
                flat_field_image = np.load(
                    os.path.join(
                        self.flat_field_dir,
                        'flat-field_channel-{}.npy'.format(channel_idx),
                    )
                )
            for slice_idx in focal_plane_ids:
                for time_idx in self.timepoint_ids:
                    for pos_idx in np.unique(self.frames_metadata["pos_idx"]):
                        frame_idx = get_meta_idx(
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
                        channel_name = self.frames_metadata.loc[frame_idx, "channel_name"]
                        cur_image = image_utils.read_image(file_path)
                        if self.flat_field_dir is not None:
                            cur_image = image_utils.apply_flat_field_correction(
                                cur_image,
                                flat_field_image=flat_field_image,
                            )
                        # normalize
                        if hist_clip_limits is not None:
                            cur_image = hist_clipping(
                                cur_image,
                                hist_clip_limits[0],
                                hist_clip_limits[1],
                            )
                        cur_image = zscore(cur_image)
                        # Now to the actual tiling
                        tiled_image_data = image_utils.tile_image(
                            input_image=cur_image,
                            tile_size=self.tile_size,
                            step_size=self.step_size,
                            isotropic=self.isotropic,
                        )
                        # Loop through all the tiles, write and add to metadata
                        for data_tuple in tiled_image_data:
                            rcsl_idx = data_tuple[0]
                            file_name = aux_utils.get_im_name(
                                time_idx=time_idx,
                                channel_idx=channel_idx,
                                slice_idx=slice_idx,
                                pos_idx=pos_idx,
                                extra_field=rcsl_idx,
                            )
                            np.save(os.path.join(self.output_dir, file_name),
                                    data_tuple[1],
                                    allow_pickle=True,
                                    fix_imports=True)
                            tiled_metadata = tiled_metadata.append(
                                {"channel_idx": channel_idx,
                                 "slice_idx": slice_idx,
                                 "time_idx": time_idx,
                                 "channel_name": channel_name,
                                 "file_name": file_name,
                                 "pos_idx": pos_idx,
                                 },
                                ignore_index=True,
                            )
        # Finally, save all the metadata
        tiled_metadata.to_csv(
            os.path.join(self.output_dir, "frames_meta.csv"),
            sep=",",
        )

    def tile_stack_with_vf_constraint(self,
                                      mask_channels,
                                      min_fraction,
                                      save_cropped_masks=False,
                                      isotropic=False,
                                      focal_plane_idx=None,
                                      hist_clip_limits=None):
        """
        Crop and retain tiles that have minimum foreground

        Minimum foreground is defined as the percent of non-zero pixels/ volume
        fraction in a mask which is a thresholded sum of flurophore image.

        :param int/list mask_channels: generate mask from the sum of these
         (flurophore) channels
        :param float min_fraction: threshold for using a cropped image for
         training. minimum volume fraction / percent occupied in cropped image
        :param list hist_clip_limits: lower and upper percentiles used for
         histogram clipping.
        :param bool save_cropped_masks: bool indicator for saving cropped masks
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param int focal_plane_idx: Index of which focal plane acquisition to
         use
        """

        if isinstance(mask_channels, int):
            mask_channels = [mask_channels]
        mask_dir_name = '-'.join(map(str, mask_channels))
        mask_dir_name = 'mask_{}_vf-{}'.format(mask_dir_name, min_fraction)

        tiled_dir = '{}_vf-{}'.format(self.tiled_dir, min_fraction)

        os.makedirs(tiled_dir, exist_ok=True)

        for tp_idx in self.timepoint_ids:
            tp_dir = os.path.join(tiled_dir,
                                  'timepoint_{}'.format(tp_idx))
            os.makedirs(tp_dir, exist_ok=True)

            crop_indices_fname = os.path.join(
                self.base_tile_dir, 'split_images',
                'timepoint_{}'.format(tp_idx),
                '{}.pkl'.format(mask_dir_name)
            )

            if not os.path.exists(crop_indices_fname):
                mask_gen_obj = MaskProcessor(
                    os.path.join(self.base_output_dir, 'split_images'),
                    mask_channels,
                    tp_idx
                )
                if save_cropped_masks:
                    cropped_mask_dir = os.path.join(tp_dir, mask_dir_name)
                    os.makedirs(cropped_mask_dir, exist_ok=True)
                else:
                    cropped_mask_dir = None
                mask_gen_obj.get_crop_indices(min_fraction, self.tile_size,
                                              self.step_size, cropped_mask_dir,
                                              save_cropped_masks, isotropic)

            with open(crop_indices_fname, 'rb') as f:
                crop_indices_dict = pickle.load(f)

            for channel in self.tile_channels:
                row_idx = aux_utils.get_row_idx(
                    self.frames_metadata, tp_idx, channel, focal_plane_idx
                )
                channel_metadata = self.frames_metadata[row_idx]
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
                self._tile_channel(image_utils.crop_at_indices,
                                   channel_dir, channel_metadata,
                                   flat_field_image, hist_clip_limits,
                                   metadata, crop_indices_dict)
                save_tile_meta(metadata, channel, tiled_dir)
