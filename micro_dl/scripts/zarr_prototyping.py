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

from micro_dl.utils.meta_utils import add_zarr_meta_field
import micro_dl.utils.io_utils as io_utils


zarr_dir = "/home/christian.foley/virtual_staining/data_visualization/A549PhaseFLDeconvolution_63X_pos0.zarr"
field_name = "preprocessing"
metadata = {
    field_name: {
        "flat_field_chans": [1, 2, 3],
        "mask_chans": [1, 2, 3],
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
for position in modifier.position_map:

    print(channels)

    z = modifier.get_position_group(position).attrs.asdict()
    print(z)
    # hcs_meta = modifier.hcs_meta
    # print(list(hcs_meta))
# %%
