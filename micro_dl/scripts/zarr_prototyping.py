#%%
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")
# %%
import zarr
import numpy as np
import zarr.core as core
import os
from pathlib import Path
import matplotlib.pyplot as plt

import micro_dl.utils.io_utils as io_utils


zarr_dir = "/home/christian.foley/virtual_staining/data_visualization/A549PhaseFLDeconvolution_63X_pos0.zarr"
field_name = "preprocessing"
metadata = {
    field_name: {
        "flat_field": {
            "chans": [1, 2, 3],
            "array_name": "ones",
        },
        "mask": {
            "input_chans": [1, 2, 3],
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

# zarr_dir = "/hpc/projects/CompMicro/rawdata/hummingbird/Janie/2022_03_15_orgs_nuc_mem_63x_04NA/all_21_3.zarr/"
reader = io_utils.ZarrReader(zarrfile=zarr_dir)

# %%
modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir, enable_creation=True)
modifier.init_untracked_array(data_array=data, position=0, name="ones")
# %%
modifier.write_meta_field(position=0, metadata=metadata, field_name=field_name)

# %%
block_sampler_args = []
for position in modifier.position_map:
    ff_path = None
    position_group = modifier.get_position_group(position)
    position_path = position_group.path
    array_path = os.path.join(zarr_dir, position_path, modifier.arr_name)

    preprocessing_metadata = position_group.attrs.asdict()["preprocessing"]
    if "flat_field" in preprocessing_metadata:
        ff_name = preprocessing_metadata["flat_field"]["array_name"]
        ff_path = os.path.join(zarr_dir, position_path, ff_name)

    block_sampler_args.append((array_path, ff_path))

#%%
mp_grid_sampler_args = []
for position in modifier.position_map:
    ff_name = None
    position_group = modifier.get_position_group(position)

    preprocessing_metadata = position_group.attrs.asdict()["preprocessing"]
    if "flat_field" in preprocessing_metadata:
        ff_name = preprocessing_metadata["flat_field"]["array_name"]

    mp_grid_sampler_args.append((position, ff_name, zarr_dir))
# %%
print(mp_grid_sampler_args)
print(modifier.get_array(position).shape)
print(modifier.get_untracked_array(position, "ones").shape)
# %%
