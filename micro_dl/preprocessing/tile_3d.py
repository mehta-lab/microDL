import numpy as np
import os
import pandas as pd

from micro_dl.preprocessing.tile_uniform_images import ImageTilerUniform
import micro_dl.utils.tile_utils as tile_utils
from micro_dl.utils import aux_utils as aux_utils


class ImageTilerUniform3D(ImageTilerUniform):
    """Tiles all volumes in a dataset"""

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
                 image_format='zyx',
                 num_workers=4,
                 int2str_len=3):
        """Init

        Please ref to init of ImageTilerUniform.
        Assuming slice_ids are contiguous
        Depth here is not used. slice_idx is used to store slice_start_idx.

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
                         image_format,
                         num_workers,
                         int2str_len)
        del self.depths

        if isinstance(self.tile_size):
            assert len(self.tile_size) == 3, \
                'tile size missing for some dimensions'

        if isinstance(self.step_size):
            assert len(self.step_size) == 3, \
                'step size missing for some dimensions'

        assert len(self.slice_ids) >= tile_size[2], \
            'Insufficient number of slices: {} < {}'.format(
                len(self.slice_ids), tile_size[2]
            )

        self.frames_metadata.rename(columns={'slice_start_idx': 'slice_idx'},
                                    inplace=True)

    def _get_tile_indices(self, tiled_meta,
                          time_idx,
                          channel_idx,
                          pos_idx,
                          slice_idx):
        """Get the tile indices from saved meta data

        :param pd.DataFrame tiled_meta: DF with image level meta info
        :param int time_idx: time index for current image
        :param int channel_idx: channel index for current image
        :param int pos_idx: position / sample index for current image
        :param int slice_idx: starting slice index for this volume
        :return list tile_indices: list of tile indices
        """

        c = tiled_meta['channel_idx'] == channel_idx
        z = tiled_meta['slice_idx'] == slice_idx
        p = tiled_meta['pos_idx'] == pos_idx
        t = tiled_meta['time_idx'] == time_idx

        channel_meta = tiled_meta[c & z & p & t]
        # Get tile_indices
        tile_indices = pd.concat([
            channel_meta['row_start'],
            channel_meta['row_start'].add(self.tile_size[0]),
            channel_meta['col_start'],
            channel_meta['col_start'].add(self.tile_size[1]),
            channel_meta['slice_idx'],
            channel_meta['slice_idx'].add(self.tile_size[2]),
        ], axis=1)
        # Match list format similar to tile_image
        tile_indices = tile_indices.values.tolist()
        return tile_indices

    def tile_stack(self):
        """
        Tiles images in the specified channels.

        https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

        Saves a csv with columns
        ['time_idx', 'channel_idx', 'pos_idx','slice_idx', 'file_name']
        for all the tiles
        """
        # Get or create tiled metadata and tile indices
        prev_tiled_metadata, tile_indices = self._get_tiled_data()
        tiled_meta0 = None
        fn_args = []
        for channel_idx in self.channel_ids:
            # Perform flatfield correction if flatfield dir is specified
            flat_field_im = self._get_flat_field(channel_idx=channel_idx)
            for time_idx in self.time_ids:
                for pos_idx in self.pos_ids:
                    for slice_idx in range(self.slice_ids[0],
                                           self.slice_ids[-1],
                                           self.step_size[2]):
                        if tile_indices is None:
                            # tile and save first image
                            # get meta data and tile_indices
                            im = tile_utils.preprocess_volume(
                                frames_metadata=self.frames_metadata,
                                input_dir=self.input_dir,
                                time_idx=time_idx,
                                channel_idx=channel_idx,
                                slice_idx=slice_idx,
                                pos_idx=pos_idx,
                                flat_field_im=flat_field_im,
                                hist_clip_limits=self.hist_clip_limits
                            )
                            save_dict = {
                                'time_idx': time_idx,
                                'channel_idx': channel_idx,
                                'pos_idx': pos_idx,
                                'slice_start_idx': slice_idx,
                                'save_dir': self.tile_dir,
                                'image_format': self.image_format,
                                'int2str_len': self.int2str_len
                            }
                            tiled_meta0, tile_indices = \
                                tile_utils.tile_image(
                                    input_image=im,
                                    tile_size=self.tile_size,
                                    step_size=self.step_size,
                                    return_index=True,
                                    save_dict=save_dict
                                )
                        else:
                            cur_args = self.get_crop_tile_args(
                                channel_idx,
                                time_idx,
                                slice_idx,
                                pos_idx,
                                task_type='crop',
                                tile_indices=tile_indices
                            )
                            fn_args.append(cur_args)
        tiled_meta_df_list = mp_crop_save(fn_args,
                                          workers=self.num_workers)
        if tiled_meta0 is not None:
            tiled_meta_df_list.append(tiled_meta0)
        tiled_metadata = pd.concat(tiled_meta_df_list, ignore_index=True)
        if self.tiles_exist:
            tiled_metadata.reset_index(drop=True, inplace=True)
            prev_tiled_metadata.reset_index(drop=True, inplace=True)
            tiled_metadata = pd.concat([prev_tiled_metadata, tiled_metadata],
                                       ignore_index=True)
        # Finally, save all the metadata
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )





