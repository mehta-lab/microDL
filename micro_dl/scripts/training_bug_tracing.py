# %%
import argparse
import yaml
import os
import sys
import torch

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.torch_unet.utils.training as train

# %%
config_25D = (
    "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/"
    "config_files/2022_09_27_A549_NuclStain/"
    "11_02_2022_gunpowder_testing/torch_config_25D.yml"
)
config_2D = (
    "/hpc/projects/compmicro/projects/virtualstaining/torch_microDL/"
    "config_files/2022_09_27_A549_NuclStain/10_12_2022_15_22/"
    "torch_config_2D.yml"
)
config_test = (
    "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/"
    "config_files/2019_02_15_KidneyTissue_full_dataset/"
    "11_02_2022_parameter_tuning/25D_Unet/z5_ret_actin/"
    "Stack5_fltr16_256_do20_otus_masked_MAE_1chan_ret_actin_pix_iqr_norm_tf10_pt20_torch_config.yml"
)
torch_config = aux_utils.read_config(config_25D)

# Instantiate training object
trainer = train.TorchTrainer(torch_config)

# generate dataloaders and init model
trainer.generate_dataloaders()
trainer.load_model()

# train
trainer.train()

# %%
