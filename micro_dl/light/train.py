import warnings
from datetime import datetime

import torch
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from numcodecs import blosc

from micro_dl.light.data import HCSDataModule
from micro_dl.light.engine import PhaseToNuc25D


class VSLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    TensorBoardLogger,
                    save_dir="",
                    version=datetime.now().strftime(r"%Y%m%d-%H%M%S"),
                    log_graph=True,
                )
            }
        )


def main():
    # https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
    blosc.use_threads = False
    torch.set_float32_matmul_precision("high")
    # TODO: remove this after MONAI 1.2 release
    # https://github.com/Project-MONAI/MONAI/pull/6105
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage")
        _ = VSLightningCLI(
            PhaseToNuc25D,
            HCSDataModule,
        )


if __name__ == "__main__":
    main()
