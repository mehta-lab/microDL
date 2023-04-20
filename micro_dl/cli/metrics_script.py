# %% script to generate your ground truth directory for microDL prediction evaluation
# After inference, the predictions generated are stored as zarr store.
# Evaluation metrics can be computed by comparison of prediction to human proof read ground truth.
#
import os
import imageio as iio
import iohub.ngff as ngff
import argparse
import glob
import pandas as pd

import micro_dl.inference.evaluation_metrics as metrics
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
    pick focus slice mask from pred_zarr from slice number stored on png mask name
    input pred mask & corrected ground truth mask to metrics computation
    store the metrics values as csv file to corresponding positions in list
    Info to be stored:
        1. position no,
        2. eval metrics values
    """

    torch_config = aux_utils.read_config(config)

    pred_dir = torch_config["evaluation_metrics"]["pred_dir"]
    metric_channel = torch_config["data"]["metric_channel"]
    PosList = torch_config["evaluation_metrics"]["PosList"]
    metrics_list = torch_config["evaluation_metrics"]["metrics"]

    ground_truth_subdir = "ground_truth"

    available_metrics = [
        "ssim",
        "corr",
        "r2",
        "mse",
        "mae",
        "acc",
        "dice",
        "IoU",
        "VI",
        "POD",
    ]
    d_pod = [
        "OD_true_positives",
        "OD_false_positives",
        "OD_false_negatives",
        "OD_precision",
        "OD_recall",
        "OD_f1_score",
    ]

    metric_map = {
        "mae_metric": metrics.mae_metric,
        "mse_metric": metrics.mse_metric,
        "r2_metric": metrics.r2_metric,
        "corr_metric": metrics.corr_metric,
        "ssim_metric": metrics.ssim_metric,
        "acc_metric": metrics.accuracy_metric,
        "dice_metric": metrics.dice_metric,
        "IoU_metric": metrics.IOU_metric,
        "VI_metric": metrics.VOI_metric,
        "POD_metric": metrics.POD_metric,
    }

    pred_plate = ngff.open_ome_zarr(store_path=pred_dir, mode="r+")
    im_pred = pred_plate.data

    metric_chan_mask = metric_channel + "_cp_mask"
    path_split_head_tail = os.path.split(pred_dir)
    target_zarr_dir = path_split_head_tail[0]
    ground_truth_dir = os.path.join(target_zarr_dir, ground_truth_subdir)

    df_metrics = pd.DataFrame(columns=available_metrics[:-1] + d_pod[:-1])

    for i, (_, position) in enumerate(pred_plate.positions()):
        raw_data = im_pred[position]
        target_data = raw_data[0, metric_chan_mask.index, ...]

        gt_mask_save_name = (
            "^" + metric_channel + "_p" + str(format(PosList(i), "03d")) + "_z"
        )
        z_slice_no = int(
            glob.glob(ground_truth_dir + "/" + gt_mask_save_name + "*_cp_mask.png")
        )
        gt_mask = iio.imread(
            ground_truth_dir
            + "/"
            + gt_mask_save_name
            + str(z_slice_no)
            + "_cp_mask.png"
        )
        pred_mask = target_data[z_slice_no]

        pos_metric_list = [z_slice_no]
        for metric_name in metrics_list:
            metric_fn = metric_map[metric_name]
            cur_metric_list = metric_fn(
                target=gt_mask,
                prediction=pred_mask,
            )
            pos_metric_list.append(cur_metric_list)

        df_metrics.loc[len(df_metrics.index)] = pos_metric_list

    csv_filename = os.path.join(ground_truth_dir, "GT_metrics.csv")
    df_metrics.to_csv(csv_filename)


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
