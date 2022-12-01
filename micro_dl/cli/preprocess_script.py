"""Script for preprocessing stack"""
import argparse
import numpy as np
import os
import pandas as pd
import time
import warnings

from micro_dl.preprocessing.estimate_flat_field import FlatFieldEstimator2D
from micro_dl.preprocessing.generate_masks import MaskProcessor
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.meta_utils as meta_utils


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


def get_required_params(preprocess_config):
    """
    Create a dictionary with required parameters for preprocessing
    from the preprocessing config. Required parameters are:
        'input_dir': Directory containing input image data
        'output_dir': Directory to write preprocessed data
        'slice_ids': Slice indices
        'time_ids': Time indices
        'pos_ids': Position indices
        'channel_ids': Channel indices
        'uniform_struct': (bool) If images are uniform
        'int2strlen': (int) How long of a string to convert integers to
        'normalize_channels': (list) Containing bools the length of channels
        'num_workers': Number of workers for multiprocessing
        'normalize_im': (str) Normalization scheme
            (stack, dataset, slice, volume)
        'zarr_file': Zarr file name in case of zarr file (as opposed to tiffs)

    :param dict preprocess_config: Preprocessing config
    :return dict required_params: Required parameters
    """
    input_dir = preprocess_config["input_dir"]
    output_dir = preprocess_config["output_dir"]
    slice_ids = -1
    if "slice_ids" in preprocess_config:
        slice_ids = preprocess_config["slice_ids"]

    time_ids = -1
    if "time_ids" in preprocess_config:
        time_ids = preprocess_config["time_ids"]

    pos_ids = -1
    if "pos_ids" in preprocess_config:
        pos_ids = preprocess_config["pos_ids"]

    channel_ids = -1
    if "channel_ids" in preprocess_config:
        channel_ids = preprocess_config["channel_ids"]

    uniform_struct = True
    if "uniform_struct" in preprocess_config:
        uniform_struct = preprocess_config["uniform_struct"]

    int2str_len = 3
    if "int2str_len" in preprocess_config:
        int2str_len = preprocess_config["int2str_len"]

    num_workers = 4
    if "num_workers" in preprocess_config:
        num_workers = preprocess_config["num_workers"]

    normalize_im = "stack"
    normalize_channels = -1
    if "normalize" in preprocess_config:
        if "normalize_im" in preprocess_config["normalize"]:
            normalize_im = preprocess_config["normalize"]["normalize_im"]
        if "normalize_channels" in preprocess_config["normalize"]:
            normalize_channels = preprocess_config["normalize"]["normalize_channels"]
            if isinstance(channel_ids, list):
                assert len(channel_ids) == len(
                    normalize_channels
                ), "Nbr channels {} and normalization {} mismatch".format(
                    channel_ids,
                    normalize_channels,
                )

    required_params = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "slice_ids": slice_ids,
        "time_ids": time_ids,
        "pos_ids": pos_ids,
        "channel_ids": channel_ids,
        "uniform_struct": uniform_struct,
        "int2strlen": int2str_len,
        "normalize_channels": normalize_channels,
        "num_workers": num_workers,
        "normalize_im": normalize_im,
    }
    return required_params


def flat_field_correct(required_params, block_size, flat_field_channels):
    """
    Estimate flat_field_images in given channels. Store flat field estimations
    new array type at each position, and update image-level metadata to reflect
    that.

    :param dict required_params: dict with keys: input_dir, output_dir, time_ids,
     channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param int block_size: Specify block size if different from default (32 pixels)
    :param list flat_field_channels: Channels in which to estimate flatfields.
    :return str flat_field_dir: full path of dir with flat field correction
     images
    """
    flat_field_inst = FlatFieldEstimator2D(
        input_dir=required_params["input_dir"],
        output_dir=required_params["output_dir"],
        channel_ids=flat_field_channels,
        slice_ids=required_params["slice_ids"],
        block_size=block_size,
    )
    flat_field_inst.estimate_flat_field()


def generate_masks(
    required_params,
    mask_from_channel,
    str_elem_radius,
    flat_field_dir,
    mask_type,
    mask_channel,
    mask_ext,
    mask_dir=None,
):
    """
    Generate masks per image or volume

    :param dict required_params: dict with keys: input_dir, output_dir, time_ids,
        channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param int/list mask_from_channel: generate masks from sum of these
        channels
    :param int str_elem_radius: structuring element size for morphological
        opening
    :param str/None flat_field_dir: dir with flat field correction images
    :param str mask_type: string to map to masking function. otsu or unimodal
        or borders_weight_loss_map
    :param int/None mask_channel: channel num assigned to mask channel. I
    :param str mask_ext: 'npy' or 'png'. Save the mask as uint8 PNG or
         NPY files
    :param str/None mask_dir: If creating weight maps from mask directory,
        specify mask dir
    :return str mask_dir: Directory with created masks
    :return int mask_channel: Channel number assigned to masks
    """
    assert mask_type in {
        "otsu",
        "unimodal",
        "dataset otsu",
        "borders_weight_loss_map",
    }, (
        "Supported mask types: 'otsu', 'unimodal', 'dataset otsu', 'borders_weight_loss_map'"
        + ", not {}".format(mask_type)
    )

    # If generating weights map, input dir is the mask dir
    input_dir = required_params["input_dir"]
    if mask_dir is not None:
        input_dir = mask_dir
    # Instantiate channel to mask processor
    mask_processor_inst = MaskProcessor(
        input_dir=input_dir,
        output_dir=required_params["output_dir"],
        channel_ids=mask_from_channel,
        flat_field_dir=flat_field_dir,
        time_ids=required_params["time_ids"],
        slice_ids=required_params["slice_ids"],
        pos_ids=required_params["pos_ids"],
        int2str_len=required_params["int2strlen"],
        uniform_struct=required_params["uniform_struct"],
        num_workers=required_params["num_workers"],
        mask_type=mask_type,
        mask_channel=mask_channel,
        mask_ext=mask_ext,
    )

    mask_processor_inst.generate_masks(
        str_elem_radius=str_elem_radius,
    )
    mask_dir = mask_processor_inst.get_mask_dir()
    mask_channel = mask_processor_inst.get_mask_channel()
    return mask_dir, mask_channel


def generate_zscore_table(required_params, norm_dict, mask_dir):
    """
    Compute z-score parameters and update frames_metadata based on the normalize_im
    :param dict required_params: Required preprocessing parameters
    :param dict norm_dict: Normalization scheme (preprocess_config['normalization'])
    :param str mask_dir: Directory containing masks
    """
    assert (
        "min_fraction" in norm_dict
    ), "normalization part of config must contain min_fraction"
    frames_metadata = aux_utils.read_meta(required_params["input_dir"])
    ints_metadata = aux_utils.read_meta(
        required_params["input_dir"],
        meta_fname="intensity_meta.csv",
    )
    mask_metadata = aux_utils.read_meta(mask_dir)
    cols_to_merge = ints_metadata.columns[ints_metadata.columns != "fg_frac"]
    ints_metadata = pd.merge(
        ints_metadata[cols_to_merge],
        mask_metadata[["pos_idx", "time_idx", "slice_idx", "fg_frac"]],
        how="left",
        on=["pos_idx", "time_idx", "slice_idx"],
    )
    _, ints_metadata = meta_utils.compute_zscore_params(
        frames_meta=frames_metadata,
        ints_meta=ints_metadata,
        input_dir=required_params["input_dir"],
        normalize_im=required_params["normalize_im"],
        min_fraction=norm_dict["min_fraction"],
    )
    ints_metadata.to_csv(
        os.path.join(required_params["input_dir"], "intensity_meta.csv"),
        sep=",",
    )


def save_config(cur_config, runtime):
    """
    Save the current config (cur_config) or append to existing config.

    :param dict cur_config: Current config
    :param float runtime: Run time for preprocessing
    """

    # Read preprocessing.json if exists in input dir
    parent_dir = cur_config["input_dir"].split(os.sep)[:-1]
    parent_dir = os.sep.join(parent_dir)

    prior_config_fname = os.path.join(parent_dir, "preprocessing_info.json")
    prior_preprocess_config = None
    if os.path.exists(prior_config_fname):
        prior_preprocess_config = aux_utils.read_json(prior_config_fname)

    meta_path = os.path.join(cur_config["output_dir"], "preprocessing_info.json")

    processing_info = [{"processing_time": runtime, "config": cur_config}]
    if prior_preprocess_config is not None:
        prior_preprocess_config.append(processing_info[0])
        processing_info = prior_preprocess_config
    os.makedirs(cur_config["output_dir"], exist_ok=True)
    aux_utils.write_json(processing_info, meta_path)


def pre_process(preprocess_config):
    """
    Preprocess data. Possible options are:

    correct_flat_field: Perform flatfield correction (2D only currently)
    resample: Resize 2D images (xy-plane) according to a scale factor,
        e.g. to match resolution in z. Resize 3d images
    create_masks: Generate binary masks from given input channels
    do_tiling: Split frames (stacked frames if generating 3D tiles) into
    smaller tiles with tile_size and step_size.

    This script will preprocess your dataset, save tiles and associated
    metadata. Then in the train_script, a dataframe for training data
    will be assembled based on the inputs and target you specify.

    :param dict preprocess_config: dict with key options:
    [input_dir, output_dir, slice_ids, time_ids, pos_ids
    correct_flat_field, use_masks, masks, tile_stack, tile]
    :param dict required_params: dict with commom params for all tasks
    :raises AssertionError: If 'masks' in preprocess_config contains both channels
     and mask_dir (the former is for generating masks from a channel)
    """
    time_start = time.time()
    required_params = get_required_params(preprocess_config)

    # ------------------Create metadata---------------------
    # Create metadata (ignore old metadata)
    order = "cztp"
    name_parser = "parse_sms_name"
    if "metadata" in preprocess_config:
        if "order" in preprocess_config["metadata"]:
            order = preprocess_config["metadata"]["order"]
        if "name_parser" in preprocess_config["metadata"]:
            name_parser = preprocess_config["metadata"]["name_parser"]
    # Create metadata from file names instead
    file_format = "zarr"
    if "file_format" in preprocess_config:
        file_format = preprocess_config["file_format"]
    meta_utils.frames_meta_generator(
        input_dir=required_params["input_dir"],
        file_format=file_format,
        order=order,
        name_parser=name_parser,
    )

    # -----------------Estimate flat field images--------------------
    flat_field_dir = None
    flat_field_channels = []
    if "flat_field" in preprocess_config:
        # If flat_field_channels aren't specified, correct all channel_ids
        flat_field_channels = required_params["channel_ids"]
        if "flat_field_channels" in preprocess_config["flat_field"]:
            flat_field_channels = preprocess_config["flat_field"]["flat_field_channels"]
        # Check that flatfield channels is subset of channel_ids
        assert set(flat_field_channels).issubset(
            required_params["channel_ids"]
        ), "Flatfield channels {} is not a subset of channel_ids".format(
            flat_field_channels
        )
        #  Method options: 'estimate' (from input) or 'from_file' (load pre-saved)
        flat_field_method = "estimate"
        if "method" in preprocess_config["flat_field"]:
            flat_field_method = preprocess_config["flat_field"]["method"]
        assert flat_field_method in {
            "estimate",
            "from_file",
        }, "Method should be estimate or from_file (use existing)"
        if flat_field_method is "estimate":
            assert (
                "flat_field_dir" not in preprocess_config["flat_field"]
            ), "estimate_flat_field or use images in flat_field_dir."
            block_size = None
            if "block_size" in preprocess_config["flat_field"]:
                block_size = preprocess_config["flat_field"]["block_size"]
            flat_field_dir = flat_field_correct(
                required_params,
                block_size,
                flat_field_channels,
            )
            preprocess_config["flat_field"]["flat_field_dir"] = flat_field_dir

        elif flat_field_method is "from_file":
            assert (
                "flat_field_dir" in preprocess_config["flat_field"]
            ), "flat_field_dir must exist if using from_file as flat_field method."
            flat_field_dir = preprocess_config["flat_field"]["flat_field_dir"]
            # Check that all flatfield channels are present
            existing_channels = []
            for ff_name in os.listdir(flat_field_dir):
                # Naming convention is: flat-field-channel_c.npy
                if ff_name[:10] == "flat-field":
                    existing_channels.append(int(ff_name[-5]))
            assert set(existing_channels) == set(flat_field_channels), (
                "Expected flatfield channels {}, and saved channels {} "
                "mismatch".format(flat_field_channels, existing_channels)
            )

    # ----------------- Generate normalization values -----------------
    if required_params["normalize_im"] in ["dataset", "volume", "slice"]:
        block_size = None
        if "metadata" in preprocess_config:
            if "block_size" in preprocess_config["metadata"]:
                block_size = preprocess_config["metadata"]["block_size"]
            meta_utils.ints_meta_generator(
                input_dir=required_params["input_dir"],
                num_workers=required_params["num_workers"],
                block_size=block_size,
                flat_field_dir=flat_field_dir,
                channel_ids=required_params["channel_ids"],
            )

    # ------------------------Generate masks-------------------------
    mask_dir = None
    mask_channel = None
    if "masks" in preprocess_config:
        # Automatically assign existing masks the next available channel number
        frames_meta = aux_utils.read_meta(required_params["input_dir"])
        mask_channel = frames_meta["channel_idx"].max() + 1
        if "channels" in preprocess_config["masks"]:
            # Generate masks from channel
            assert (
                "mask_dir" not in preprocess_config["masks"]
            ), "Don't specify a mask_dir if generating masks from channel"
            mask_from_channel = preprocess_config["masks"]["channels"]
            str_elem_radius = 5
            if "str_elem_radius" in preprocess_config["masks"]:
                str_elem_radius = preprocess_config["masks"]["str_elem_radius"]
            mask_type = "otsu"
            if "mask_type" in preprocess_config["masks"]:
                mask_type = preprocess_config["masks"]["mask_type"]
            mask_ext = ".png"
            if "mask_ext" in preprocess_config["masks"]:
                mask_ext = preprocess_config["masks"]["mask_ext"]

            mask_dir, mask_channel = generate_masks(
                required_params=required_params,
                mask_from_channel=mask_from_channel,
                flat_field_dir=flat_field_dir,
                str_elem_radius=str_elem_radius,
                mask_type=mask_type,
                mask_channel=mask_channel,
                mask_ext=mask_ext,
            )
        elif "mask_dir" in preprocess_config["masks"]:
            assert (
                "channels" not in preprocess_config["masks"]
            ), "Don't specify channels to mask if using pre-generated masks"
            mask_dir = preprocess_config["masks"]["mask_dir"]
            # Get preexisting masks from directory and match to input dir
            mask_meta = meta_utils.mask_meta_generator(
                mask_dir,
            )
            mask_meta["channel_idx"] = mask_channel
            # Write metadata
            mask_meta_fname = os.path.join(mask_dir, "frames_meta.csv")
            mask_meta.to_csv(mask_meta_fname, sep=",")
        else:
            raise ValueError(
                "If using masks, specify either mask_channel", "or mask_dir."
            )
        preprocess_config["masks"]["mask_dir"] = mask_dir
        preprocess_config["masks"]["mask_channel"] = mask_channel

    # ---------------------Generate z score table---------------------
    if required_params["normalize_im"] in ["dataset", "volume", "slice"]:
        assert (
            mask_dir is not None
        ), "'dataset', 'volume', 'slice' normalization requires masks"
        generate_zscore_table(
            required_params,
            preprocess_config["normalize"],
            mask_dir,
        )

    # ----------------------Generate weight map-----------------------
    weights_dir = None
    weights_channel = None
    if "make_weight_map" in preprocess_config and preprocess_config["make_weight_map"]:
        # Must have mask dir and mask channel defined to generate weight map
        assert mask_dir is not None, "Must have mask dir to generate weights"
        assert mask_channel is not None, "Must have mask channel to generate weights"
        mask_type = "borders_weight_loss_map"
        # Mask channel should be highest channel value in dataset at this point
        weights_channel = mask_channel + 1
        # Generate weights
        weights_dir, _ = generate_masks(
            required_params=required_params,
            mask_from_channel=mask_channel,
            flat_field_dir=None,
            str_elem_radius=5,
            mask_type=mask_type,
            mask_channel=weights_channel,
            mask_ext=".npy",
            mask_dir=mask_dir,
        )
        preprocess_config["weights"] = {
            "weights_dir": weights_dir,
            "weights_channel": weights_channel,
        }

    # Write in/out/mask/tile paths and config to json in output directory
    time_el = time.time() - time_start
    return preprocess_config, time_el


if __name__ == "__main__":
    args = parse_args()
    preprocess_config = aux_utils.read_config(args.config)
    preprocess_config, runtime = pre_process(preprocess_config)
    save_config(preprocess_config, runtime)
