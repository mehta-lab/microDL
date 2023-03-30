from lightning.pytorch.cli import LightningCLI

from micro_dl.light.data import HCSDataModule
from micro_dl.light.engine import PhaseToNuc25D


def main():
    cli = LightningCLI(PhaseToNuc25D, HCSDataModule)


if __name__ == "__main__":
    main()
