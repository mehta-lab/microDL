#%%
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")
import zarr
import numpy as np
import zarr.core as core
import os
from pathlib import Path
import matplotlib.pyplot as plt

import micro_dl.utils.io_utils as io_utils
from micro_dl.preprocessing.estimate_flat_field import FlatFieldEstimator2D
from micro_dl.utils.meta_utils import generate_normalization_metadata

zarr_dir = (
    "/home/christian.foley/virtual_staining/data_visualization/"
    "A549PhaseFLDeconvolution_63X_pos0.zarr"
)
zarr_dir = (
    "/hpc/projects/CompMicro/projects/virtualstaining/"
    "torch_microDL/data/2022_03_31_GOLGA2_nuc_mem_LF_63x_04NA_HEK/"
    "test_no_pertubation.zarr"
)
field_name = "preprocessing"
metadata = {
    field_name: {
        "flat_field": {
            "chans": [0, 1, 2],
            "array_name": "ones",
        },
        "mask": {
            "input_chans": [0, 1, 2],
            "output_chan": 4,
        },
        "normalization_val": 0.75,
        "registration": {
            "x_shift": 23,
            "y_shift": 9,
        },
    },
}
metadata = metadata[field_name]
data = np.ones((1, 3, 1, 2048, 2048))
name = "flat_field"
# %%

# zarr_dir = "/hpc/projects/CompMicro/rawdata/hummingbird/Janie/2022_03_15_orgs_nuc_mem_63x_04NA/all_21_3.zarr"
zarr_dir = "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/data/2022_11_01_VeroMemNuclStain/output.zarr"
reader = io_utils.ZarrReader(zarrfile=zarr_dir)
modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir, enable_creation=True)
# %%
if zarr_dir[:5] in "/home/":
    modifier.init_untracked_array(data_array=data, position=0, name="ones")
    modifier.write_meta_field(position=0, metadata=metadata, field_name=field_name)

# %%
print("Performing flatfield estimation:")
estimator = FlatFieldEstimator2D(
    zarr_dir=zarr_dir,
    channel_ids=-1,
    slice_ids=-1,
    flat_field_array_name="flatfield",
)

estimator.estimate_flat_field()
# %%
print("\nCalculating normalization statistics:")
generate_normalization_metadata(
    zarr_dir=zarr_dir,
    channel_ids=-1,
)
# %%
