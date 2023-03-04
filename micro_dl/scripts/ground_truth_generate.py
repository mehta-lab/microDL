
# %% script to generate your ground truth directory for microDL prediction evaluation
# After inference, the predictions generated are stored as zarr store. 
# Evaluation metrics can be computed by comparison of prediction to human proof read ground truth.
# 
import numpy as np
import os
from PIL import Image
import imagio
import waveorder as wo
from waveorder.io import WaveorderReader, WaveorderWriter
from waveorder.focus import focus_from_transverse_band
import micro_dl.inference.evaluation_metrics as metrics
import cellpose

# %% enter required details here

FL_zarr_dir = '/hpc/projects/CompMicro/projects/infected_cell_imaging/VirtualStaining/VirtualStain_NuclMem_A549_2023_02_07/Input_Nucl/'
zarr_name = 'A549_20X.zarr'
NA_obj = 0.55           # NA of objective
lambda_illu = 0.532     # illumination wavelength in um
ps_f = 6.5/20           # image pixel size = (camera pixel size)/mag
ground_truth_chan = 'DAPI'
labelFree_chan = 'Phase3D' # to store label-free channel for proof-reading validation
no_Pos = 13             # no of position, can be handpicked by user
time_point = 0          # time point of data to be used
model = 'CP_20220915_A549Mem'
boundary_eval = 'True'

# %% pick the focus slice from n_pos number of positions, segment, and save as tifs

PosList = list(range(0,no_Pos)) # or PosList = [0,2,3,4,5,7,9,10,11,13,18,19,20]
os.mkdir(os.path.join(FL_zarr_dir,'ground_truth')) # create dir to store single page tifs
reader = WaveorderReader(os.path.join(FL_zarr_dir + zarr_name)) 
zarr_pos_len = reader.get_num_positions()
try:
    assert len(PosList) > zarr_pos_len
except AssertionError:
    print("no of positions in user-defined position list exceeds positions in dataset")

for pos in range(PosList):
    raw_data = reader.get_array(pos)
    FL_data = raw_data[time_point,reader.chan_names.index(ground_truth_chan),...] # extract required FL image stack
    focus_idx_FL = focus_from_transverse_band(FL_data, NA_det=NA_obj, lambda_ill=lambda_illu, pixel_size=ps_f) # focus slice index
    FL_focus_slice = FL_data[focus_idx_FL,:,:] # FL focus slice image

    im = Image.fromarray(FL_focus_slice.astype(np.uint8)) # save focus slice as single page tif
    save_name =  '_t' + str(format(time_point, '03d')) + '_p' + str(format(pos, '03d')) + '_z' + str(format(FL_focus_slice, '03d'))
    im.save(os.path.join(os.path.join(FL_zarr_dir,'ground_truth'),(ground_truth_chan+save_name+'.tif')))

    LF_focus_slice = raw_data[time_point,reader.chan_names.index(labelFree_chan),focus_idx_FL,:,:] # lable-free focus slice image
    im.save(os.path.join(os.path.join(FL_zarr_dir,'ground_truth'),(ground_truth_chan+save_name+'.tif'))) # save for reference

    cp_mask = metrics.cpmask_array(FL_focus_slice,model) # cellpose segmnetation for binary mask
    
    if boundary_eval == 'True':  # switch to boundary mask if signal is just boundary ,eg., membrane
        bin_mask = cellpose.utils.masks_to_outlines(cp_mask)
    else:
        bin_mask = metrics.binarize_array(cp_mask)
    
    imagio.imwrite(os.path.join(os.path.join(FL_zarr_dir,'ground_truth'),(labelFree_chan+save_name+'_cp_mask.png')),bin_mask) # save binary mask as numpy or png
