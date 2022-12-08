"""Generate masks from sum of flurophore channels"""
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as im_utils
import micro_dl.utils.io_utils as io_utils
from micro_dl.utils.mp_utils import mp_create_and_write_mask
from skimage.filters import threshold_otsu


class MaskProcessor:
    """
    Appends Masks to zarr directories
    """

    def __init__(
        self,
        zarr_dir,
        channel_ids,
        time_ids=-1,
        pos_ids=-1,
        num_workers=4,
        mask_type="otsu",
        output_channel_index=None,
    ):
        """
        :param str zarr_dir: directory of HCS zarr store to pull data from.
                            Note: data in store is assumed to be stored in
                            (time, channel, z, y, x) format.
        :param list[int] channel_ids: Channel indices to be masked (typically
            just one)
        :param int/list channel_ids: generate mask from the sum of these
            (flurophore) channel indices
        :param list/int time_ids: timepoints to consider
        :param int pos_ids: Position (FOV) indices to use
        :param int num_workers: number of workers for multiprocessing
        :param str mask_type: method to use for generating mask. Needed for
            mapping to the masking function. One of:
                {'otsu', 'unimodal', 'borders_weight_loss_map'}
        :param int/None output_channel_index: specific channel to write to,
                overwriting the existing data and metadata in this channel
        """
        self.zarr_dir = zarr_dir
        self.num_workers = num_workers

        # Validate that given indices are available.
        metadata_ids = aux_utils.validate_metadata_indices(
            zarr_dir=zarr_dir,
            time_ids=time_ids,
            channel_ids=channel_ids,
            pos_ids=pos_ids,
        )
        self.time_ids = metadata_ids["time_ids"]
        self.channel_ids = metadata_ids["channel_ids"]
        self.position_ids = metadata_ids["pos_ids"]

        assert mask_type in [
            "otsu",
            "unimodal",
            "borders_weight_loss_map",
        ], "Masking method invalid, 'otsu', 'unimodal', 'borders_weight_loss_map'\
             are currently supported"
        self.mask_type = mask_type
        self.ints_metadata = None
        self.channel_thr_df = None

        self.modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir)

        if output_channel_index > self.modifier.channels:
            print("Mask output channel beyond channel range, appending instead")
            output_channel_index = None
        elif output_channel_index < self.modifier.channels:
            channel_name = self.modifier.channel_names[output_channel_index]
            print(
                f"Received mask output_channel_index is {output_channel_index}. "
                f"This channel is currently populated by {channel_name}. Overwriting "
                "after mask computation completes."
            )
        self.output_channel_index = output_channel_index

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

        # Gather function arguments for each index pair at each position
        all_positions = list(self.modifier.position_map)
        mp_mask_creator_args = []

        for position in all_positions:
            mp_mask_creator_args.append(
                tuple(
                    [
                        self.zarr_dir,
                        position,
                        self.time_ids,
                        self.channel_ids,
                        structure_elem_radius,
                        self.mask_type,
                        "_".join(["mask", self.mask_type]),
                        self.output_channel_index,
                    ]
                )
            )

        # create and write masks and metadata using multiprocessing
        mp_create_and_write_mask(mp_mask_creator_args, workers=self.num_workers)
