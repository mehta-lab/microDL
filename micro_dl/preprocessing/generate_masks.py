"""Generate masks from sum of flurophore channels"""
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as im_utils
import micro_dl.utils.io_utils as io_utils
from micro_dl.utils.mp_utils import mp_create_save_mask
from skimage.filters import threshold_otsu


class MaskProcessor:
    """
    Appends Masks to zarr directories
    """

    def __init__(
        self,
        zarr_dir,
        channel_ids,
        flatfield_name=None,
        time_ids=-1,
        slice_ids=-1,
        pos_ids=-1,
        int2str_len=3,
        uniform_struct=True,
        num_workers=4,
        mask_type="otsu",
    ):
        """
        :param str zarr_dir: directory of HCS zarr store to pull data from.
                            Note: data in store is assumed to be stored in
                            (time, channel, z, y, x) format.
        :param int mask_channel: channel number assigned to generated masks. Masks
                                appended to data array and and stored in unique arrays
                                under this name. By default, mask_name is
        :param list[int] channel_ids: Channel indices to be masked (typically
            just one)
        :param str flat_field_dir: Directory with flatfield images if
            flatfield correction is applied
        :param int/list channel_ids: generate mask from the sum of these
            (flurophore) channel indices
        :param list/int time_ids: timepoints to consider
        :param int slice_ids: Index of which focal plane (z)
            acquisition to use (default -1 includes all slices)
        :param int pos_ids: Position (FOV) indices to use
        :param int int2str_len: Length of str when converting ints
        :param bool uniform_struct: bool indicator for same structure across
            pos and time points
        :param int num_workers: number of workers for multiprocessing
        :param str mask_type: method to use for generating mask. Needed for
            mapping to the masking function
        """
        self.zarr_dir = zarr_dir
        self.flatfield_name = flatfield_name
        self.num_workers = num_workers

        # Create a unique mask channel number so masks can be treated
        # as a new channel

        metadata_ids, nested_id_dict = aux_utils.validate_metadata_indices(
            zarr_dir=zarr_dir,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
        )

        self.channel_ids = metadata_ids["channel_ids"]

        self.int2str_len = int2str_len
        self.uniform_struct = uniform_struct
        self.nested_id_dict = nested_id_dict

        assert mask_type in [
            "otsu",
            "unimodal",
            "dataset otsu",
            "borders_weight_loss_map",  # not sure if we want to support this one
        ], "Masking method invalid, 'otsu', 'unimodal', 'dataset otsu', 'borders_weight_loss_map'\
             are currently supported"
        self.mask_type = mask_type
        self.ints_metadata = None
        self.channel_thr_df = None
        if mask_type == "dataset otsu":
            self.ints_metadata = aux_utils.read_meta(
                self.input_dir, "intensity_meta.csv"
            )
            self.channel_thr_df = self.get_channel_thr_df()

        self.modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir)

    def get_channel_thr_df(self):
        ints_meta_sub = self.ints_metadata.loc[
            self.ints_metadata["channel_idx"].isin(self.channel_ids),
            ["dir_name", "channel_idx", "intensity"],
        ]
        # channel_thr_df = ints_meta_sub.groupby(['dir_name', 'channel_idx']).agg(get_unimodal_threshold).reset_index()
        channel_thr_df = (
            ints_meta_sub.groupby(["dir_name", "channel_idx"])
            .agg(threshold_otsu)
            .reset_index()
        )
        channel_thr_df["intensity"] = channel_thr_df["intensity"]
        return channel_thr_df

    def get_mask_dir(self):
        """
        Return mask directory
        :return str mask_dir: Directory where masks are stored
        """
        return self.mask_dir

    def get_mask_channel(self):
        """
        Return mask channel
        :return int mask_channel: Assigned channel number for mask
        """
        return self.mask_channel

    def _get_flatfield_slice(self, time_index, channel_index, position_index, z_index):
        """
        Get slice of flatfield image given a position and a channel.
        Channel must be

        Note: flatfields do not vary between channels, but they are stored
        separately in each channel to accommodate gunpowder.

        :param int time_index: time id to use for selecting flatfield slice
        :param int channel_index: channel id to use for selecting flatfield slice
        :param int position_index: position id to use for selecting flatfield slice
        :param int z_index: z-stack depth id to use for selecting flatfield slice

        :return np.ndarray slice: 2D flatfield
        """
        # TODO: normalization duplicates this code. fix duplication by moving this
        #       into the zarr modifier class

        position_metadata = self.modifier.get_position_meta(position_index)
        if "flatfield" in position_metadata:
            ff_name = position_metadata["flatfield"]["array_name"]
            ff_channels = position_metadata["flatfield"]["channel_ids"]
        else:
            return None

        flatfield = self.modifier.get_untracked_array(
            position=position_index, name=ff_name
        )

        # flatfield array might have collapsed indices
        ff_channel_pos = ff_channels.index(channel_index)
        flatfield_slice = flatfield[time_index, ff_channel_pos, 0, :, :]

        return flatfield_slice

    def generate_masks(self, structure_elem_radius=5):
        """
        Generate masks from flat-field corrected flurophore images.
        The sum of flurophore channels is thresholded to generate a foreground
        mask.

        Masks are saved as an additional channel in each data array for each
        specified position. If certain channels are not specified, gaps are
        filled with arrays of zeros.

        Masks are also saved as an additional untracked array named "mask" and
        tracked in the "mask" metadata field.

        :param int structure_elem_radius: Radius of structuring element for
         morphological operations
        """
        fn_args = []

        # Gather function arguments for each index pair at each position
        shape = self.modifier.shape

        all_times = list(range(shape[0]))
        all_positions = list(self.modifier.position_map)
        all_channels = list(range(shape[2]))
        all_z_indices = list(range(shape[3]))
        
        for time in all_times:
            for channel 

        channel_thrs = None
        if self.uniform_struct:
            for id_row in id_df.to_numpy():
                dir_name, time_idx, pos_idx, slice_idx = id_row
                channels_meta_sub = aux_utils.get_sub_meta(
                    frames_metadata=self.frames_metadata,
                    time_ids=time_idx,
                    channel_ids=self.channel_ids,
                    slice_ids=slice_idx,
                    pos_ids=pos_idx,
                )
                ff_fnames = self._get_ff_paths(
                    channel_ids=self.channel_ids,
                )
                if self.mask_type == "dataset otsu":
                    channel_thrs = self.channel_thr_df.loc[
                        self.channel_thr_df["dir_name"] == dir_name, "intensity"
                    ].to_numpy()
                cur_args = (
                    channels_meta_sub,
                    ff_fnames,
                    structure_elem_radius,
                    self.mask_dir,
                    self.mask_channel,
                    self.int2str_len,
                    self.mask_type,
                    self.mask_ext,
                    self.input_dir,
                    channel_thrs,
                )
                fn_args.append(cur_args)
        else:
            for tp_idx, tp_dict in self.nested_id_dict.items():
                mask_channel_dict = tp_dict[self.channel_ids[0]]
                for pos_idx, sl_idx_list in mask_channel_dict.items():
                    for sl_idx in sl_idx_list:
                        channels_meta_sub = aux_utils.get_sub_meta(
                            frames_metadata=self.frames_metadata,
                            time_ids=tp_idx,
                            channel_ids=self.channel_ids,
                            slice_ids=sl_idx,
                            pos_ids=pos_idx,
                        )
                        ff_fnames = self._get_ff_paths(
                            channel_ids=self.channel_ids,
                        )
                        cur_args = (
                            channels_meta_sub,
                            ff_fnames,
                            structure_elem_radius,
                            self.mask_dir,
                            self.mask_channel,
                            self.int2str_len,
                            self.mask_type,
                            self.mask_ext,
                            self.input_dir,
                        )
                        fn_args.append(cur_args)

        mask_meta_list = mp_create_save_mask(fn_args, self.num_workers)
        mask_meta_df = pd.DataFrame.from_dict(mask_meta_list)
        mask_meta_df = mask_meta_df.sort_values(by=["file_name"])
        mask_meta_df["dir_name"] = self.mask_dir
        mask_meta_df.to_csv(os.path.join(self.mask_dir, "frames_meta.csv"), sep=",")
        # update fg_frac field in image frame_meta.csv
        cols_to_merge = self.frames_metadata.columns[
            self.frames_metadata.columns != "fg_frac"
        ]
        self.frames_metadata = pd.merge(
            self.frames_metadata[cols_to_merge],
            mask_meta_df[["pos_idx", "time_idx", "slice_idx", "fg_frac"]],
            how="left",
            on=["pos_idx", "time_idx", "slice_idx"],
        )
        self.frames_metadata.to_csv(
            os.path.join(self.input_dir, "frames_meta.csv"),
            sep=",",
        )
