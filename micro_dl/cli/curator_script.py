# %% script to generate your ground truth directory for microDL prediction evaluation
# After inference, the predictions generated are stored as zarr store.
# Evaluation metrics can be computed by comparison of prediction to human proof read ground truth.
#
import numpy as np
import os
from PIL import Image
import imagio
import iohub.ngff as ngff
import micro_dl.inference.evaluation_metrics as metrics
import cellpose
import math
import argparse

import micro_dl.utils.aux_utils as aux_utils

# %% read the below details from the config file


def parse_args():
    """
    Parse command line arguments
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


def main(config):

    """
    pick the focus slice from n_pos number of positions, segment, and save as tifs
    Also store the infomartion as csv file, where you can add the evaluation metrics results.
    Info to be stored: 
        1. position no, 
        2. focus slice number (it will be the center slice as the images are aligned)
        3. time point
        4. chan name for evaluation (DAPI if nucleus)
    """

    torch_config = aux_utils.read_config(config)

    zarr_dir = torch_config["data"]["data_path"]
    pred_dir = torch_config["evaluation_metrics"]["pred_dir"]
    ground_truth_chan = torch_config["data"]["target_channel"]
    labelFree_chan = torch_config["data"]["source_channel"]
    PosList = torch_config["evaluation_metrics"]["PosList"]
    cp_model = torch_config["evaluation_metrics"]["cp_model"]
    
    ground_truth_subdir = "ground_truth"
    path_split_head_tail = os.path.split(zarr_dir)
    target_zarr_dir = path_split_head_tail[0]
    zarr_name = path_split_head_tail[1]

    os.mkdir(
        os.path.join(target_zarr_dir, ground_truth_subdir)
    )  # create dir to store single page tifs
    plate = ngff.open_ome_zarr(store_path=zarr_dir, mode="r+")
    im = plate.data
    chan_names = plate.channel_names
    out_shape = im.shape
    # zarr_pos_len = reader.get_num_positions()
    try:
        assert len(PosList) > out_shape[0]
    except AssertionError:
        print(
            "number of positions listed in config exceeds number of positions in dataset"
        )

    for pos in range(PosList):
        raw_data = im[pos]
        target_data = raw_data[
            0, ground_truth_chan[0], ...
        ]
        Z, Y, X = np.ndarray.shape(target_data)
        focus_idx_target = math.round((Z + 1) / 2)
        target_focus_slice = target_data[focus_idx_target, :, :]  # FL focus slice image

        im_target = Image.fromarray(
            target_focus_slice.astype(np.uint8)
        )  # save focus slice as single page tif
        save_name = (
            "_p" + str(format(pos, "03d")) + "_z" + str(format(target_focus_slice, "03d"))
        )
        im_target.save(
            os.path.join(
                target_zarr_dir,
                ground_truth_subdir,
                chan_names(ground_truth_chan[0]) + save_name + ".tif",
            )
        )

        source_focus_slice = raw_data[
            0, chan_names.index(labelFree_chan), focus_idx_target, :, :
        ]  # lable-free focus slice image
        im_source = Image.fromarray(
            source_focus_slice.astype(np.uint8)
        )  # save focus slice as single page tif
        im_source.save(
            os.path.join(
                target_zarr_dir,
                ground_truth_subdir,
                chan_names(labelFree_chan[0]) + save_name + ".tif",
            )
        )  # save for reference

        cp_mask = metrics.cpmask_array(
            target_focus_slice, cp_model
        )  # cellpose segmnetation for binary mask
        imagio.imwrite(
            os.path.join(
                target_zarr_dir,
                ground_truth_subdir,
                labelFree_chan + save_name + "_cp_mask.png",
            ),
            cp_mask,
        )  # save binary mask as numpy or png


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
