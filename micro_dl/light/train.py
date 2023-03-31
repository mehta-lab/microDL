import warnings

from lightning.pytorch.cli import LightningCLI

from micro_dl.light.data import HCSDataModule
from micro_dl.light.engine import PhaseToNuc25D


def main():
    # TODO: remove this after MONAI 1.2 release
    # https://github.com/Project-MONAI/MONAI/pull/6105
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage")
        _ = LightningCLI(PhaseToNuc25D, HCSDataModule)


if __name__ == "__main__":
    main()
