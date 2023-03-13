"""Script for preprocessing stack"""
import argparse
import time

from micro_dl.preprocessing.estimate_flat_field import FlatFieldEstimator2D
from micro_dl.preprocessing.generate_masks import MaskProcessor
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.meta_utils as meta_utils
import micro_dl.utils.io_utils as io_utils
import iohub.ngff as ngff


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="path to yaml configuration file",
    )
    args = parser.parse_args()
    return args


def flatfield_correct(zarr_dir, flatfield_channels, flatfield_slice_ids, block_size):
    """
    Estimate flatfield_images in given channels. Store flat field estimations
    new array type at each position, and update image-level metadata to reflect
    that.

    :param str zarr_dir: path to HCS-compatible zarr store with input data
    :param list flatfield_channels: Channels in which to estimate flatfields
    :param int block_size: Specify block size if different from default (32 pixels)
    :param list flatfield_slice_ids: list of slice indices to include in flatfield
                                estimation
    """
    flatfield_inst = FlatFieldEstimator2D(
        zarr_dir=zarr_dir,
        channel_ids=flatfield_channels,
        slice_ids=flatfield_slice_ids,
        block_size=block_size,
    )
    flatfield_inst.estimate_flat_field()


def pre_process(torch_config):
    """
    Preprocess data. Possible options are:

    correct_flatfield: Perform flatfield correction (2D only currently)
    normalize: Calculate values for on-the-fly normalization on a FOV &
                dataset level
    create_masks: Generate binary masks from given input channels

    This script will preprocess your dataset, save auxilary data and
    associated metadata for on-the-fly processing during training. Masks
    will be saved both as an additional channel and as an array tracked in
    custom metadata. Flatfields will be saved as an array tracked in custom
    metadata.

    :param dict torch_config: 'master' torch config with subfields for all steps
                            of data analysis
    :raises AssertionError: If 'masks' in preprocess_config contains both channels
     and mask_dir (the former is for generating masks from a channel)
    """
    time_start = time.time()
    plate = ngff.open_ome_zarr()
    io_utils.HCSZarrModifier(zarr_file=torch_config["zarr_dir"])
    preprocess_config = torch_config["preprocessing"]

    # -----------------Estimate flat field images--------------------

    flatfield_channels = []
    if "flatfield" in preprocess_config:
        print("Estimating Flatfield: ------------- \n")
        # collect params
        flatfield_config = preprocess_config["flatfield"]

        flatfield_channels = -1
        if "channel_ids" in flatfield_config:
            flatfield_channels = flatfield_config["channel_ids"]

        flatfield_slices = -1
        if "slice_ids" in flatfield_config:
            flatfield_slices = flatfield_config["slice_ids"]

        flatfield_block_size = 32
        if "block_size" in flatfield_config:
            flatfield_block_size = flatfield_config["block_size"]

        # validate
        if isinstance(flatfield_channels, list):
            assert set(flatfield_channels).issubset(
                list(range(len(modifier.channel_names)))
            ), "Flatfield channels {} is not a subset of channel_ids".format(
                flatfield_channels
            )

        # estimate flatfields
        flatfield_correct(
            zarr_dir=torch_config["zarr_dir"],
            flatfield_channels=flatfield_channels,
            flatfield_slice_ids=flatfield_slices,
            block_size=flatfield_block_size,
        )

    # ----------------- Generate normalization values -----------------
    if "normalize" in preprocess_config:
        print("Computing Normalization Values: ------------- \n")
        # collect params
        normalize_config = preprocess_config["normalize"]

        norm_num_workers = 4
        if "num_workers" in normalize_config:
            norm_num_workers = normalize_config["num_workers"]

        norm_channel_ids = -1
        if "channel_ids" in normalize_config:
            norm_channel_ids = normalize_config["channel_ids"]

        norm_block_size = 32
        if "block_size" in normalize_config:
            norm_block_size = normalize_config["block_size"]

        meta_utils.generate_normalization_metadata(
            zarr_dir=torch_config["zarr_dir"],
            num_workers=norm_num_workers,
            channel_ids=norm_channel_ids,
            grid_spacing=norm_block_size,
        )

    # ------------------------Generate masks-------------------------
    if "masks" in preprocess_config:
        print("Generating Masks: ------------- \n")
        # collect params
        mask_config = preprocess_config["masks"]

        mask_channel_ids = -1
        if "channel_ids" in mask_config:
            mask_channel_ids = mask_config["channel_ids"]

        mask_time_ids = -1
        if "time_ids" in mask_config:
            mask_time_ids = mask_config["time_ids"]

        mask_pos_ids = -1

        mask_num_workers = 4
        if "num_workers" in mask_config:
            mask_num_workers = mask_config["num_workers"]

        mask_type = "unimodal"
        if "thresholding_type" in mask_config:
            mask_type = mask_config["thresholding_type"]

        mask_output_channel = None
        if "output_channel" in mask_config:
            mask_output_channel = mask_config["output_channel"]

        structuring_radius = 5
        if "structure_element_radius" in mask_config:
            structuring_radius = mask_config["structure_element_radius"]

        # validate
        if mask_type not in {
            "unimodal",
            "otsu",
            "edge_detection",
            "borders_weight_loss_map",
        }:
            raise ValueError(
                f"Thresholding type {mask_type} must be one of: "
                f"{['unimodal', 'otsu', 'edge_detection', 'borders_weight_loss_map']}"
            )

        # generate masks
        mask_generator = MaskProcessor(
            zarr_dir=torch_config["zarr_dir"],
            channel_ids=mask_channel_ids,
            time_ids=mask_time_ids,
            pos_ids=mask_pos_ids,
            num_workers=mask_num_workers,
            mask_type=mask_type,
            output_channel_index=mask_output_channel,
        )
        mask_generator.generate_masks(structure_elem_radius=structuring_radius)

    # ----------------------Generate weight map-----------------------
    # TODO: determine if weight map generation should be offered in simultaneity
    #      with binary mask generation

    return time.time() - time_start


if __name__ == "__main__":
    args = parse_args()
    torch_config = aux_utils.read_config(args.config)
    runtime = pre_process(torch_config)
    print(f"Preprocessing complete. Runtime: {runtime:.2f} seconds")
