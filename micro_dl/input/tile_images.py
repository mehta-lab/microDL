"""Tile images for training"""

import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as normalize
import micro_dl.utils.image_utils as image_utils
from micro_dl.utils.image_utils import crop_at_indices_save


def multiprocessing(fn_args, workers):
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(crop_at_indices_save, *zip(*fn_args))
    return list(res)


class ImageTiler:
    """Tiles all images images in a dataset"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_dict,
                 tile_size=[256, 256],
                 step_size=[64, 64],
                 depths=1,
                 mask_depth=1,
                 time_ids=-1,
                 channel_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 hist_clip_limits=None,
                 flat_field_dir=None,
                 isotropic=False,
                 data_format='channels_first',
                 uniform_structure=True,
                 num_workers=1):
        """
        Normalizes images using z-score, then tiles them.
        Isotropic here refers to the same dimension/shape along row, col, slice
        and not really isotropic resolution in mm.

        :param str input_dir: Directory with frames to be tiled
        :param str output_dir: Base output directory
        :param list tile_size: size of the blocks to be cropped
         from the image
        :param list step_size: size of the window shift. In case
         of no overlap, the step size is tile_size. If overlap, step_size <
         tile_size
        :param int/list depths: The z depth for generating stack training data
            Default 1 assumes 2D data for all channels to be tiled.
            For cases where input and target shapes are not the same (e.g. stack
             to 2D) you should specify depths for each channel in tile.channels.
        :param int mask_depth: Depth for mask channel
        :param list/int time_ids: Tile given timepoint indices
        :param list/int tile_channels: Tile images in the given channel indices
         default=-1, tile all channels
        :param int slice_ids: Index of which focal plane acquisition to
         use (for 2D). default=-1 for the whole z-stack
        :param int pos_ids: Position (FOV) indices to use
        :param list hist_clip_limits: lower and upper percentiles used for
         histogram clipping.
        :param str flat_field_dir: Flatfield directory. None if no flatfield
            correction
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param bool uniform_structure: indicator if all sub-units have same
         number of unique values
        :param int num_workers: number of cores to use for multiprocessing
        :param str data_format: Channels first or last
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        if 'depths' in tile_dict:
            depths = tile_dict['depths']
        if 'mask_depth' in tile_dict:
            mask_depth = tile_dict['mask_depth']
        if 'tile_size' in tile_dict:
            tile_size = tile_dict['tile_size']
        if 'step_size' in tile_dict:
            step_size = tile_dict['step_size']
        if 'isotropic' in tile_dict:
            isotropic = tile_dict['isotropic']
        if 'channels' in tile_dict:
            channel_ids = tile_dict['channels']
        if 'positions' in tile_dict:
            pos_ids = tile_dict['positions']
        if 'hist_clip_limits' in tile_dict:
            hist_clip_limits = tile_dict['hist_clip_limits']
        if 'data_format' in tile_dict:
            data_format = tile_dict['data_format']
            assert data_format in {'channels_first', 'channels_last'},\
                "Data format must be channels_first or channels_last"
        self.depths = depths
        self.mask_depth = mask_depth
        self.tile_size = tile_size
        self.step_size = step_size
        self.isotropic = isotropic
        self.hist_clip_limits = hist_clip_limits
        self.data_format = data_format

        self.str_tile_step = 'tiles_{}_step_{}'.format(
            '-'.join([str(val) for val in tile_size]),
            '-'.join([str(val) for val in step_size]),
        )
        self.tile_dir = os.path.join(
            output_dir,
            self.str_tile_step,
        )
        self.uniform_structure = uniform_structure
        self.num_workers = num_workers

        # If tile dir already exist, things could get messy because we don't
        # have any checks in place for how to add to existing tiles
        try:
            os.makedirs(self.tile_dir, exist_ok=False)
        except FileExistsError as e:
            print("You're trying to write to existing dir. ", e)
            raise

        self.tile_mask_dir = None
        self.flat_field_dir = flat_field_dir
        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        # Get metadata indices
        metadata_ids, nested_id_dict = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
            uniform_structure=uniform_structure
        )
        self.channel_ids = metadata_ids['channel_ids']
        self.time_ids = metadata_ids['time_ids']
        self.slice_ids = metadata_ids['slice_ids']
        self.pos_ids = metadata_ids['pos_ids']
        if nested_id_dict:
            self.nested_id_dict = nested_id_dict

        # If more than one depth is specified, they must match channel ids
        if isinstance(self.depths, list):
            assert len(self.depths) == len(self.channel_ids),\
             "depths ({}) and channels ({}) length mismatch".format(
                len(self.depths), len(self.channel_ids)
            )
            # Get max of all specified depths
            max_depth = max(max(self.depths), self.mask_depth)
            # Convert channels + depths to dict for lookup
            self.channel_depth = dict(zip(self.channel_ids, self.depths))
        else:
            max_depth = max(self.depths, self.mask_depth)
            self.channel_depth = dict(zip(
                self.channel_ids,
                [self.depths] * len(self.channel_ids)),
            )

        self.margin = 0
        if max_depth > 1:
            margin = max_depth // 2
            nbr_slices = len(self.slice_ids)
            assert nbr_slices > 2 * margin,\
                "Insufficient slices ({}) for max depth {}".format(
                    nbr_slices, max_depth)
            assert self.slice_ids[-1] - self.slice_ids[0] + 1 == nbr_slices,\
                "Slice indices are not contiguous"
            # TODO: use itertools.groupby if non-contiguous data is a thing
            # np.unique is sorted so we can just remove first and last ids
            self.slice_ids = self.slice_ids[margin:-margin]
            self.margin = margin

    def get_tile_dir(self):
        """
        Return directory containing tiles
        :return str tile_dir: Directory with tiles
        """
        return self.tile_dir

    def get_tile_mask_dir(self):
        """
        Return directory containing tiles of mask
        :return str tile_mask_dir: Directory with tiled mask
        """
        return self.tile_mask_dir

    def _preprocess_im(self,
                       time_idx,
                       channel_idx,
                       slice_idx,
                       pos_idx,
                       flat_field_im=None,
                       hist_clip_limits=None):
        """
        Preprocess image given by indices: flatfield correction, histogram
        clipping and z-score normalization is performed.

        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position (FOV) index
        :param np.array flat_field_im: Flat field image for channel
        :param list hist_clip_limits: Limits for histogram clipping (size 2)
        :return np.array im: 2D preprocessed image
        :return str channel_name: Channel name
        """

        depth = self.channel_depth[channel_idx]
        margin = 0 if depth == 1 else depth // 2
        im_stack = []
        for z in range(slice_idx - margin, slice_idx + margin + 1):
            meta_idx = aux_utils.get_meta_idx(
                self.frames_metadata,
                time_idx,
                channel_idx,
                z,
                pos_idx,
            )
            channel_name = self.frames_metadata.loc[meta_idx, "channel_name"]
            file_path = os.path.join(
                self.input_dir,
                self.frames_metadata.loc[meta_idx, "file_name"],
            )
            im = image_utils.read_image(file_path)
            if flat_field_im is not None:
                im = image_utils.apply_flat_field_correction(
                    im,
                    flat_field_image=flat_field_im,
                )
            im_stack.append(im)
        # Stack images
        im_stack = np.stack(im_stack, axis=2)
        # normalize
        if hist_clip_limits is not None:
            im_stack = normalize.hist_clipping(
                im_stack,
                hist_clip_limits[0],
                hist_clip_limits[1],
            )
        return normalize.zscore(im_stack), channel_name

    def _get_input_fnames(self,
                          time_idx,
                          channel_idx,
                          slice_idx,
                          pos_idx):
        """Get input_fnames

        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position (FOV) index
        :return: list of input fnames
        """

        depth = self.channel_depth[channel_idx]
        margin = 0 if depth == 1 else depth // 2
        im_fnames = []
        for z in range(slice_idx - margin, slice_idx + margin + 1):
            meta_idx = aux_utils.get_meta_idx(
                self.frames_metadata,
                time_idx,
                channel_idx,
                z,
                pos_idx,
            )
            file_path = os.path.join(
                self.input_dir,
                self.frames_metadata.loc[meta_idx, "file_name"],
            )
            im_fnames.append(file_path)
        return im_fnames

    def _write_tiled_data(self,
                          tiled_data,
                          save_dir,
                          time_idx=None,
                          channel_idx=None,
                          slice_idx=None,
                          pos_idx=None,
                          tile_indices=None,
                          tiled_metadata=None,
                          ):
        """
        Loops through tuple and writes all tile image data. Adds row to metadata
        dataframe as well if that is present.

        :param list of tuples tiled_data: Tile name and np.array
        :param str save_dir: Directory where tiles will be written
        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position (FOV) index
        :param list of tuples tile_indices: Tile indices
        :param dataframe tiled_metadata: Dataframe containing metadata for all
         tiles
        :return dataframe tiled_metadata: Metadata with rows added to it
        """
        for i, data_tuple in enumerate(tiled_data):
            rcsl_idx = data_tuple[0]
            file_name = aux_utils.get_im_name(
                time_idx=time_idx,
                channel_idx=channel_idx,
                slice_idx=slice_idx,
                pos_idx=pos_idx,
                extra_field=rcsl_idx,
            )
            tile = data_tuple[1]
            # Check and potentially flip dimensions for 3D data
            if self.data_format == 'channels_first' and len(tile.shape) > 2:
                tile = np.transpose(tile, (2, 0, 1))
            np.save(os.path.join(save_dir, file_name),
                    tile,
                    allow_pickle=True,
                    fix_imports=True)
            tile_idx = tile_indices[i]
            if tiled_metadata is not None:
                tiled_metadata = tiled_metadata.append(
                    {"channel_idx": channel_idx,
                     "slice_idx": slice_idx,
                     "time_idx": time_idx,
                     "file_name": file_name,
                     "pos_idx": pos_idx,
                     "row_start": tile_idx[0],
                     "col_start": tile_idx[2]},
                    ignore_index=True
                )
        return tiled_metadata

    def _get_flat_field(self, channel_idx):
        """
        Get flat field image for a given channel index

        :param int channel_idx: Channel index
        :return np.array flat_field_im: flat field image for channel
        """
        flat_field_im = None
        if self.flat_field_dir is not None:
            flat_field_im = np.load(
                os.path.join(
                    self.flat_field_dir,
                    'flat-field_channel-{}.npy'.format(channel_idx),
                )
            )
        return flat_field_im

    def _get_dataframe(self):
        """
        Creates an empty dataframe with metadata column names for tiles. It's
        the same names as for frames, but with channel_name removed and with
        the addition of row_start and col_start.
        TODO: Should I also save row_end and col_end while I'm at it?
        Might be useful if we want to recreate tiles from a previous preprocessing
        with mask run... Or just retrieve tile_size from preprocessing_info...
        This is one of the functions that will have to be adapted once tested on
        3D data.

        :return dataframe tiled_metadata
        """
        return pd.DataFrame(columns=[
            "channel_idx",
            "slice_idx",
            "time_idx",
            "file_name",
            "pos_idx",
            "row_start",
            "col_start"])

    def _tile_image_serially(self,
                             time_idx,
                             channel_idx,
                             slice_idx,
                             pos_idx,
                             flat_field_im,
                             tiled_metadata,
                             tile_indices=None):
        """Tile image one at a time """

        im, channel_name = self._preprocess_im(
            time_idx,
            channel_idx,
            slice_idx,
            pos_idx,
            flat_field_im=flat_field_im,
            hist_clip_limits=self.hist_clip_limits
        )
        if tile_indices is None:
            tiled_image_data, tile_indices = \
                image_utils.tile_image(
                    input_image=im,
                    tile_size=self.tile_size,
                    step_size=self.step_size,
                    isotropic=self.isotropic,
                    return_index=True,
                )
        else:
            tiled_image_data = image_utils.crop_at_indices(
                input_image=im,
                crop_indices=tile_indices,
                isotropic=self.isotropic,
            )
        tiled_metadata = self._write_tiled_data(
            tiled_image_data,
            save_dir=self.tile_dir,
            time_idx=time_idx,
            channel_idx=channel_idx,
            slice_idx=slice_idx,
            pos_idx=pos_idx,
            tile_indices=tile_indices,
            tiled_metadata=tiled_metadata,
        )
        return tiled_metadata, tile_indices

    def _tile_stack_parallel(self,
                             time_idx,
                             channel_idx,
                             slice_idx,
                             pos_idx,
                             tile_indices,
                             fn_args):
        """ """

        input_fnames = self._get_input_fnames(
            time_idx=time_idx,
            channel_idx=channel_idx,
            slice_idx=slice_idx,
            pos_idx=pos_idx
        )
        flat_field_fname = None
        if self.flat_field_dir is not None:
            flat_field_fname = os.path.join(
                self.flat_field_dir,
                'flat-field_channel-{}.npy'.format(channel_idx)
            )
        hist_clip_limits = None
        if self.hist_clip_limits is not None:
            hist_clip_limits = tuple(
                self.hist_clip_limits
            )

        # all args to mp should be hashable :-(
        cur_args = (tuple(input_fnames),
                    flat_field_fname,
                    hist_clip_limits,
                    time_idx,
                    channel_idx,
                    pos_idx,
                    slice_idx,
                    tuple(tile_indices),
                    self.data_format,
                    self.isotropic,
                    self.tile_dir)
        fn_args.append(cur_args)

    def tile_stack(self):
        """
        Tiles images in the specified channels.

        https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

        Saves a csv with columns
        ['time_idx', 'channel_idx', 'pos_idx','slice_idx', 'file_name']
        for all the tiles
        """

        tiled_metadata = self._get_dataframe()
        tile_indices = None
        fn_args = []
        if not self.uniform_structure:
            for tp_idx, tp_dict in self.nested_id_dict.items():
                for ch_idx, ch_dict in tp_dict.items():
                    flat_field_im = self._get_flat_field(channel_idx=ch_idx)
                    for pos_idx, sl_idx_list in ch_dict.items():
                        if self.margin > 0:
                            cur_sl_idx_list = \
                                sl_idx_list[self.margin:-self.margin]
                        else:
                            cur_sl_idx_list = sl_idx_list
                        for sl_idx in cur_sl_idx_list:
                            if self.num_workers == 1:
                                tiled_metadata, tile_indices = \
                                    self._tile_image_serially(
                                        tp_idx,
                                        ch_idx,
                                        sl_idx,
                                        pos_idx,
                                        flat_field_im,
                                        tiled_metadata,
                                        tile_indices)
                            elif self.num_workers > 1:
                                if tile_indices is None:
                                    im, channel_name = self._preprocess_im(
                                        tp_idx,
                                        ch_idx,
                                        sl_idx,
                                        pos_idx,
                                        flat_field_im=flat_field_im,
                                        hist_clip_limits=self.hist_clip_limits
                                    )
                                    tiled_image_data, tile_indices = \
                                        image_utils.tile_image(
                                            input_image=im,
                                            tile_size=self.tile_size,
                                            step_size=self.step_size,
                                            isotropic=self.isotropic,
                                            return_index=True,
                                        )
                                self._tile_stack_parallel(tp_idx,
                                                          ch_idx,
                                                          sl_idx,
                                                          pos_idx,
                                                          tile_indices,
                                                          fn_args)
        else:
            for channel_idx in self.channel_ids:
                # Perform flatfield correction if flatfield dir is specified
                flat_field_im = self._get_flat_field(channel_idx=channel_idx)
                for slice_idx in self.slice_ids:
                    for time_idx in self.time_ids:
                        for pos_idx in self.pos_ids:
                            if self.num_workers == 1:
                                tiled_metadata, tile_indices = \
                                    self._tile_image_serially(
                                                     time_idx,
                                                     channel_idx,
                                                     slice_idx,
                                                     pos_idx,
                                                     flat_field_im,
                                                     tiled_metadata,
                                                     tile_indices)
                            elif self.num_workers > 1:
                                if tile_indices is None:
                                    im, channel_name = self._preprocess_im(
                                        time_idx,
                                        channel_idx,
                                        slice_idx,
                                        pos_idx,
                                        flat_field_im=flat_field_im,
                                        hist_clip_limits=self.hist_clip_limits
                                    )
                                    tiled_image_data, tile_indices = \
                                        image_utils.tile_image(
                                            input_image=im,
                                            tile_size=self.tile_size,
                                            step_size=self.step_size,
                                            isotropic=self.isotropic,
                                            return_index=True,
                                        )
                                # change fn_args to a mp.queue
                                self._tile_stack_parallel(time_idx,
                                                          channel_idx,
                                                          slice_idx,
                                                          pos_idx,
                                                          tile_indices,
                                                          fn_args)
        if self.num_workers > 1:
            tiled_meta_df_list = multiprocessing(fn_args,
                                                 workers=self.num_workers)
            tiled_metadata = pd.concat(tiled_meta_df_list, ignore_index=True)
        # Finally, save all the metadata
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )

    def _get_mask(self, time_idx, mask_channel, slice_idx, pos_idx, mask_dir):
        """
        Load a mask image or an image stack, depending on depth

        :param int time_idx: Time index
        :param str mask_channel: Channel index for mask
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position index
        :param str mask_dir: Directory containing masks
        :return np.array im_stack: Mask image/stack
        """
        margin = self.mask_depth // 2
        im_stack = []
        for z in range(slice_idx - margin, slice_idx + margin + 1):
            file_name = aux_utils.get_im_name(
                time_idx=time_idx,
                channel_idx=mask_channel,
                slice_idx=z,
                pos_idx=pos_idx,
            )
            file_path = os.path.join(
                mask_dir,
                file_name,
            )
            im_stack.append(image_utils.read_image(file_path))
        # Stack images
        return np.stack(im_stack, axis=2)

    # break this into uniform_struct: with and w/o mp, not uniform_struct: with
    # and w/o uniform struct

    def tile_mask_stack(self,
                        mask_dir=None,
                        save_tiled_masks=None,
                        mask_channel=None,
                        min_fraction=None,
                        isotropic=False):
        """
        Tiles images in the specified channels assuming there are masks
        already created in mask_dir. Only tiles above a certain fraction
        of foreground in mask tile will be saved and added to metadata.

        Saves a csv with columns ['time_idx', 'channel_idx', 'pos_idx',
        'slice_idx', 'file_name'] for all the tiles

        :param str mask_dir: Directory containing masks
        :param str save_tiled_masks: How/if to save mask tiles. If None, don't
            save masks.
            If 'as_channel', save masked tiles as a channel given
            by mask_channel in tile_dir.
            If 'as_masks', create a new tile_mask_dir and save them there
        :param str mask_channel: Channel number assigned to mask
        :param float min_fraction: Minimum fraction of foreground in tiled masks
        :param bool isotropic: Indicator of isotropy
        """
        if save_tiled_masks == 'as_masks':
            self.tile_mask_dir = os.path.join(
                self.output_dir,
                'mask_' + '-'.join(map(str, self.channel_ids)) +
                self.str_tile_step,
            )
            os.makedirs(self.tile_mask_dir, exist_ok=True)
        elif save_tiled_masks == 'as_channel':
            self.tile_mask_dir = self.tile_dir

        tiled_metadata = self._get_dataframe()
        mask_metadata = self._get_dataframe()
        # Load flatfield images if flatfield dir is specified
        flat_field_im = None
        if self.flat_field_dir is not None:
            flat_field_ims = []
            for channel_idx in self.channel_ids:
                flat_field_ims.append(self._get_flat_field(channel_idx))

        for slice_idx in self.slice_ids:
            for time_idx in self.time_ids:
                for pos_idx in np.unique(self.frames_metadata["pos_idx"]):
                    # Since masks are generated across channels, we only need
                    # load them once across channels
                    mask_image = self._get_mask(
                        time_idx=time_idx,
                        mask_channel=mask_channel,
                        slice_idx=slice_idx,
                        pos_idx=pos_idx,
                        mask_dir=mask_dir)
                    tiled_mask_data, tile_indices = image_utils.tile_image(
                        input_image=mask_image,
                        min_fraction=min_fraction,
                        tile_size=self.tile_size,
                        step_size=self.step_size,
                        isotropic=isotropic,
                        return_index=True,
                    )
                    # Loop through all the mask tiles, write tiled masks
                    mask_metadata = self._write_tiled_data(
                        tiled_data=tiled_mask_data,
                        save_dir=self.tile_mask_dir,
                        time_idx=time_idx,
                        channel_idx=mask_channel,
                        slice_idx=slice_idx,
                        pos_idx=pos_idx,
                        tile_indices=tile_indices,
                        tiled_metadata=mask_metadata,
                    )
                    # Loop through all channels and tile from indices
                    for i, channel_idx in enumerate(self.channel_ids):

                        if self.flat_field_dir is not None:
                            flat_field_im = flat_field_ims[i]

                        im, channel_name = self._preprocess_im(
                            time_idx,
                            channel_idx,
                            slice_idx,
                            pos_idx,
                            flat_field_im=flat_field_im,
                            hist_clip_limits=self.hist_clip_limits,
                        )
                        # Now to the actual tiling of data
                        tiled_image_data = image_utils.crop_at_indices(
                            input_image=im,
                            crop_indices=tile_indices,
                            isotropic=self.isotropic,
                        )
                        # Loop through all the tiles, write and add to metadata
                        tiled_metadata = self._write_tiled_data(
                            tiled_data=tiled_image_data,
                            save_dir=self.tile_dir,
                            time_idx=time_idx,
                            channel_idx=channel_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            tile_indices=tile_indices,
                            tiled_metadata=tiled_metadata,
                        )

        # Finally, save all the metadata
        if self.tile_mask_dir == self.tile_dir:
            tiled_metadata = tiled_metadata.append(
                mask_metadata,
                ignore_index=True,
            )
        else:
            mask_metadata.to_csv(
                os.path.join(self.tile_mask_dir, "frames_meta.csv"),
                sep=",",
            )
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )
