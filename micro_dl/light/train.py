import warnings

import torch
from numcodecs import blosc
from lightning.pytorch.cli import LightningCLI

from micro_dl.light.data import HCSDataModule
from micro_dl.light.engine import PhaseToNuc25D


def main():
    # https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
    blosc.use_threads = False
    torch.set_float32_matmul_precision("high")
    # TODO: remove this after MONAI 1.2 release
    # https://github.com/Project-MONAI/MONAI/pull/6105
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage")
        _ = LightningCLI(PhaseToNuc25D, HCSDataModule)


if __name__ == "__main__":
    main()
