
"""
script to generate your ground truth directory for microDL prediction evaluation
After inference, the predictions generated are stored as zarr store. 
Evaluation metrics can be computed by comparison of prediction to human proof read ground truth.
"""

import numpy as np
import os
from PIL import Image
import imagio
import waveorder as wo
from waveorder.io import WaveorderReader, WaveorderWriter
from waveorder.focus import focus_from_transverse_band
import micro_dl.inference.evaluation_metrics as metrics
import cellpose
import math
import argparse

import micro_dl.utils.aux_utils as aux_utils


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
    torch_config = aux_utils.read_config(config)

    zarr_dir = torch_config["evaluation_metrics"]["pred_dir"]
    ground_truth_chan = torch_config["dataset"]["target_channels"]
    labelFree_chan = torch_config["dataset"]["input_channels"]
    PosList = torch_config["evaluation_metrics"]["PosList"]
    model = torch_config["evaluation_metrics"]["model"]
    boundary_eval = torch_config["evaluation_metrics"]["boundary_eval"]

    # zarr_dir = '/hpc/projects/CompMicro/projects/infected_cell_imaging/VirtualStaining/VirtualStain_NuclMem_A549_2023_02_07/Input_Nucl/A549_20X.zarr' 
    # ground_truth_chan = 'DAPI' # read from ["dataset"]["target_channels"]
    # labelFree_chan = 'Phase3D' # to store label-free channel for proof-reading validation, read from ["dataset"]["input_channels"]  
    # PosList = [0,2,3,4,5,7,9,10,11,13,18,19,20]  # position list for evaluation, can be copied from data_split.yml  
    # model = 'CP_20220915_A549Mem'
    # boundary_eval = 'True'
    # NA_obj = 0.55           # NA of objective
    # lambda_illu = 0.532     # illumination wavelength in um
    # ps_f = 6.5/20           # image pixel size = (camera pixel size)/mag

    """
    pick the focus slice from n_pos number of positions, segment, and save as tifs
    Also store the infomartion as csv file, where you can add the evaluation metrics results.
    Info to be stored: 
        1. position no, 
        2. focus slice number (it will be the center slice as the images are aligned)
        3. time point
        4. chan name for evaluation (DAPI if nucleus)
    """

    time_point = 0          # time point of data to be used
    ground_truth_subdir = 'ground_truth'
    path_split_head_tail = os.path.split(zarr_dir)
    FL_zarr_dir = path_split_head_tail[0]
    zarr_name = path_split_head_tail[1]

    os.mkdir(os.path.join(FL_zarr_dir,ground_truth_subdir)) # create dir to store single page tifs
    reader = WaveorderReader(os.path.join(FL_zarr_dir + zarr_name)) 
    zarr_pos_len = reader.get_num_positions()
    try:
        assert len(PosList) > zarr_pos_len
    except AssertionError:
        print("number of positions in user-defined position list exceeds number of positions in dataset")

    for pos in range(PosList):
        raw_data = reader.get_array(pos)
        FL_data = raw_data[time_point,ground_truth_chan[0],...] # extract required FL image stack
        # focus_idx_FL = focus_from_transverse_band(FL_data, NA_det=NA_obj, lambda_ill=lambda_illu, pixel_size=ps_f) # focus slice index
        Z,Y,X = np.ndarray.shape(FL_data)
        focus_idx_FL = math.round((Z+1)/2)
        FL_focus_slice = FL_data[focus_idx_FL,:,:] # FL focus slice image

        im_FL = Image.fromarray(FL_focus_slice.astype(np.uint8)) # save focus slice as single page tif
        save_name =  '_t' + str(format(time_point, '03d')) + '_p' + str(format(pos, '03d')) + '_z' + str(format(FL_focus_slice, '03d'))
        im_FL.save(os.path.join(FL_zarr_dir,ground_truth_subdir,reader.chan_names(ground_truth_chan[0])+save_name+'.tif'))

        LF_focus_slice = raw_data[time_point,reader.chan_names.index(labelFree_chan),focus_idx_FL,:,:] # lable-free focus slice image
        im_LF = Image.fromarray(LF_focus_slice.astype(np.uint8)) # save focus slice as single page tif
        im_LF.save(os.path.join(FL_zarr_dir,ground_truth_subdir,reader.chan_names(labelFree_chan[0])+save_name+'.tif')) # save for reference

        cp_mask = metrics.cpmask_array(FL_focus_slice,model) # cellpose segmnetation for binary mask
        
        if boundary_eval == 'True':  # switch to boundary mask if signal is just boundary ,eg., membrane
            bin_mask = cellpose.utils.masks_to_outlines(cp_mask)
        else:
            bin_mask = metrics.binarize_array(cp_mask)
        
        imagio.imwrite(os.path.join(FL_zarr_dir,ground_truth_subdir,labelFree_chan+save_name+'_cp_mask.png'),bin_mask) # save binary mask as numpy or png


if __name__ == "__main__":
    args = parse_args()
    main(args.config)