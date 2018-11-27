import copy
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
from micro_dl.input.tile_images_uni_struct import ImageTilerUniform
from micro_dl.utils.mp_utils import mp_tile_save, mp_crop_at_indices_save


class ImageTilerNonUniform(ImageTilerUniform):
    """Tiles all images images in a dataset"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_dict,
                 tile_size=[256, 256],
                 step_size=[64, 64],
                 depths=1,
                 time_ids=-1,
                 channel_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 hist_clip_limits=None,
                 flat_field_dir=None,
                 isotropic=False,
                 data_format='channels_first',
                 num_workers=4,
                 int2str_len=3):
        """Init

        Please ref to init of ImageTilerUniform
        """

        super().__init__(input_dir,
                         output_dir,
                         tile_dict,
                         tile_size,
                         step_size,
                         depths,
                         time_ids,
                         channel_ids,
                         slice_ids,
                         pos_ids,
                         hist_clip_limits,
                         flat_field_dir,
                         isotropic,
                         data_format,
                         num_workers,
                         int2str_len)
        # Get metadata indices
        metadata_ids, nested_id_dict = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
            uniform_structure=False
        )
        self.nested_id_dict = nested_id_dict
        # self.tile_dir is already created in super(). Check if frames_meta
        # exists in self.tile_dir
        meta_path = os.path.join(self.tile_dir, 'frames_meta.csv')
        assert not os.path.exists(meta_path), 'Tile dir exists. ' \
                                              'cannot add to existing dir'

    def tile_stack(self):
        """
        Tiles images in the specified channels.

        Saves a csv with columns
        ['time_idx', 'channel_idx', 'pos_idx','slice_idx', 'file_name']
        for all the tiles
        """

        # Get or create tiled metadata and tile indices
        fn_args = []
        for tp_idx, tp_dict in self.nested_id_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                flat_field_im = self._get_flat_field(channel_idx=ch_idx)
                for pos_idx, sl_idx_list in ch_dict.items():
                    tile_indices = None
                    cur_sl_idx_list = aux_utils.adjust_slice_margins(
                        sl_idx_list, self.channel_depth[ch_idx]
                    )
                    for sl_idx in cur_sl_idx_list:
                        cur_args, tile_indices = \
                            super().gather_tiling_fn_calls(tile_indices,
                                                           ch_idx,
                                                           tp_idx,
                                                           sl_idx,
                                                           pos_idx,
                                                           flat_field_im)
                        fn_args.append(cur_args)
        tiled_meta_df_list = mp_crop_at_indices_save(fn_args,
                                                     workers=self.num_workers)
        tiled_metadata = pd.concat(tiled_meta_df_list, ignore_index=True)
        # Finally, save all the metadata
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )

    def tile_mask_stack(self,
                        mask_dir,
                        mask_channel,
                        min_fraction,
                        mask_depth=1):
        """
        Tiles images in the specified channels assuming there are masks
        already created in mask_dir. Only tiles above a certain fraction
        of foreground in mask tile will be saved and added to metadata.

        Saves a csv with columns ['time_idx', 'channel_idx', 'pos_idx',
        'slice_idx', 'file_name'] for all the tiles

        :param str mask_dir: Directory containing masks
        :param int mask_channel: Channel number assigned to mask
        :param float min_fraction: Min fraction of foreground in tiled masks
        :param int mask_depth: Depth for mask channel
        """

        # mask depth has to match input or ouput channel depth
        assert mask_depth <= max(self.channel_depth)
        self.mask_depth = mask_depth

        ch0_ids = []
        # nested_id_dict had no info on mask channel if channel_ids != -1.
        # Assuming structure is same across channels. Get time, pos and slice
        # indices for ch_idx=0
        mask_ch_ind = mask_channel in self.channel_ids
        tmp_ch = mask_channel if mask_ch_ind else self.channel_ids[0]

        # create a copy of nested_id_dict to remove the entries of the mask
        # channel
        if mask_ch_ind:
            nested_id_dict_1 = copy.deepcopy(self.nested_id_dict)

        for tp_idx, tp_dict in self.nested_id_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                if ch_idx == tmp_ch:
                    cur_idx = [tp_idx, ch_idx, ch_dict]
                    ch0_ids.append(cur_idx)
                    if mask_ch_ind:
                        del nested_id_dict_1[tp_idx][ch_idx]

        # tile mask channel and use to get indices for other channels
        mask_fn_args = []
        for tp_idx, _, ch_dict in ch0_ids:
            # ignore channel_idx, replace it with mask_channel
            for pos_idx, sl_idx_list in ch_dict.items():
                cur_sl_idx_list = aux_utils.adjust_slice_margins(
                    sl_idx_list, self.mask_depth
                )
                for sl_idx in cur_sl_idx_list:
                    cur_args = super().mask_tiling_fn_calls(
                        mask_channel=mask_channel,
                        time_idx=tp_idx,
                        slice_idx=sl_idx,
                        pos_idx=pos_idx,
                        mask_dir=mask_dir,
                        min_fraction=min_fraction
                    )
                    mask_fn_args.append(cur_args)
        # tile_image uses min_fraction assuming input_image is a bool
        mask_meta_df_list = mp_tile_save(mask_fn_args,
                                         workers=self.num_workers)
        mask_meta_df = pd.concat(mask_meta_df_list, ignore_index=True)
        # Finally, save all the metadata
        mask_meta_df = mask_meta_df.sort_values(by=['file_name'])
        mask_meta_df.to_csv(os.path.join(self.tile_dir, 'frames_meta.csv'),
                            sep=",")

        # Load flatfield images if flatfield dir is specified
        flat_field_im = None
        if self.flat_field_dir is not None:
            flat_field_ims = []
            for channel_idx in self.channel_ids:
                flat_field_ims.append(super()._get_flat_field(channel_idx))

        nested_dict = nested_id_dict_1 if mask_ch_ind else self.nested_id_dict
        # tile the rest
        fn_args = []
        for tp_idx, tp_dict in nested_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                if ch_idx == mask_channel:
                    break
                if self.flat_field_dir is not None:
                    flat_field_im = flat_field_ims[ch_idx]
                for pos_idx, sl_idx_list in ch_dict.items():
                    cur_sl_idx_list = aux_utils.adjust_slice_margins(
                        sl_idx_list, self.channel_depth[ch_idx]
                    )
                    for sl_idx in cur_sl_idx_list:
                        cur_tile_indices = super()._get_tile_indices(
                            tiled_meta=mask_meta_df,
                            time_idx=tp_idx,
                            channel_idx=mask_channel,
                            pos_idx=pos_idx,
                            slice_idx=sl_idx
                        )
                        cur_args, _ = super().gather_tiling_fn_calls(
                            cur_tile_indices,
                            ch_idx,
                            tp_idx,
                            sl_idx,
                            pos_idx,
                            flat_field_im=flat_field_im)
                        fn_args.append(cur_args)

        tiled_meta_df_list = mp_crop_at_indices_save(fn_args,
                                                     workers=self.num_workers)
        tiled_metadata = pd.concat(tiled_meta_df_list, ignore_index=True)
        tiled_metadata = pd.concat([mask_meta_df, tiled_metadata],
                                   ignore_index=True)

        # Finally, save all the metadata
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )
