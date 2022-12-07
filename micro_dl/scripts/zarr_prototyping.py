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
zarr_dir = "/home/christian.foley/virtual_staining/data_visualization/A549PhaseFLDeconvolution_63X_pos0.zarr"
reader = io_utils.ZarrReader(zarrfile=zarr_dir)
modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir, enable_creation=True)
# %%
if zarr_dir[:5] in "/home/":
    print("writing untracked array and in-code metadata test")
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
modifier.get_position_meta(0)

# %%
dummy_new_channel = np.ones(tuple([1, 1, *modifier.shape[2:]]))

dummy_meta = {
    "active": False,
    "coefficient": 1.0,
    "color": "FFFFFF",
    "family": "linear",
    "inverted": False,
    "label": "ones_for_test",
    "window": {"end": 100000000000.0, "max": 1000, "min": -1000, "start": 500.0},
}


# %%
modifier.add_channel(
    new_channel_array=dummy_new_channel,
    position=0,
    omero_metadata=dummy_meta,
)
# %%
array = modifier.get_array(position=0)

plt.imshow(array[0, 3, 0])
# %%
