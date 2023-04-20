# %% script to generate your ground truth directory for microDL prediction evaluation
# After inference, the predictions generated are stored as zarr store.
# Evaluation metrics can be computed by comparison of prediction to human proof read ground truth.
#
import numpy as np
import os
from PIL import Image
import imageio as iio
import iohub.ngff as ngff
import argparse

import micro_dl.inference.evaluation_metrics as metrics
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.mp_utils import add_channel
from waveorder.focus import focus_from_transverse_band

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
    pick the focus slice from n_pos number of positions, cellpose segment, and save as tifs
    also save segmentation input and label-free image as tifs for ground truth curation
    segment fluorescence predictions and store mask as new channel
    """

    torch_config = aux_utils.read_config(config)

    zarr_dir = torch_config["data"]["data_path"]
    pred_dir = torch_config["evaluation_metrics"]["pred_dir"]
    ground_truth_chans = torch_config["data"]["target_channel"]
    labelFree_chan = torch_config["data"]["source_channel"]
    PosList = torch_config["evaluation_metrics"]["PosList"]
    z_list = torch_config["evaluation_metrics"]["z_list"]
    cp_model = torch_config["evaluation_metrics"]["cp_model"]

    # if torch_config["evaluation_metrics"]["NA_det"] is None:
    #     NA_det = 1.3
    #     lambda_illu = 0.4
    #     pxl_sz = 0.103
    # else:
    #     NA_det = torch_config["evaluation_metrics"]["NA_det"]
    #     lambda_illu = torch_config["evaluation_metrics"]["lambda_illu"]
    #     pxl_sz = torch_config["evaluation_metrics"]["pxl_sz"]

    ground_truth_subdir = "ground_truth"
    path_split_head_tail = os.path.split(pred_dir)
    target_zarr_dir = path_split_head_tail[0]
    zarr_name = path_split_head_tail[1]

    if not os.path.exists(os.path.join(target_zarr_dir, ground_truth_subdir)):
        os.mkdir(
            os.path.join(target_zarr_dir, ground_truth_subdir)
        )  # create dir to store single page tifs
        plate = ngff.open_ome_zarr(store_path=zarr_dir, mode="r+")
        chan_names = plate.channel_names

        for position, pos_data in plate.positions():
            im = pos_data.data
            # im = plate.data
            out_shape = im.shape
            # zarr_pos_len = reader.get_num_positions()
            try:
                assert len(PosList) > out_shape[0]
            except AssertionError:
                print(
                    "number of positions listed in config exceeds number of positions in dataset"
                )
            pos = int(position.split("/")[2])
            for gt_chan in ground_truth_chans:
                if pos in PosList:
                    target_data = im[0, chan_names.index(gt_chan), ...]
                    Z, Y, X = target_data.shape
                    focus_idx_target = z_list[pos]
                    # focus_idx_target = focus_from_transverse_band(
                    #     target_data, NA_det, lambda_illu, pxl_sz
                    # )
                    target_focus_slice = target_data[
                        focus_idx_target, :, :
                    ]  # FL focus slice image

                    im_target = Image.fromarray(
                        target_focus_slice
                    )  # save focus slice as single page tif
                    save_name = (
                        "_p" + str(format(pos, "03d")) + "_z" + str(focus_idx_target)
                    )
                    im_target.save(
                        os.path.join(
                            target_zarr_dir,
                            ground_truth_subdir,
                            gt_chan + save_name + ".tif",
                        )
                    )

                    source_focus_slice = im[
                        0, chan_names.index(labelFree_chan[0]), focus_idx_target, :, :
                    ]  # lable-free focus slice image
                    im_source = Image.fromarray(
                        source_focus_slice
                    )  # save focus slice as single page tif
                    im_source.save(
                        os.path.join(
                            target_zarr_dir,
                            ground_truth_subdir,
                            labelFree_chan[0] + save_name + ".tif",
                        )
                    )  # save for reference

                    cp_mask = metrics.cpmask_array(
                        target_focus_slice, cp_model
                    )  # cellpose segmnetation for binary mask
                    iio.imwrite(
                        os.path.join(
                            target_zarr_dir,
                            ground_truth_subdir,
                            gt_chan + save_name + "_cp_mask.png",
                        ),
                        cp_mask,
                    )  # save binary mask as numpy or png

    # segment prediction and add mask as channel to pred_dir
    pred_plate = ngff.open_ome_zarr(store_path=pred_dir, mode="r+")
    # im_pred = pred_plate.data
    chan_names = pred_plate.channel_names

    
    for position, pos_data in pred_plate.positions():
        for channel_name in chan_names:
            raw_data = pos_data.data
            target_data = raw_data[0, chan_names.index(channel_name), ...]

            Z, Y, X = target_data.shape
            new_channel_array = np.zeros((1, Z, Y, X))
            for z_slice in range(Z):
                target_slice = target_data[z_slice]
                cp_mask = metrics.cpmask_array(target_slice, cp_model)
                new_channel_array[0, z_slice] = cp_mask

            new_channel_name = channel_name + "_cp_mask"
            add_channel(
                pos_data,
                new_channel_array,
                new_channel_name,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
