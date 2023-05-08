# script to profile dataloading

from micro_dl.light.data import HCSDataModule
from numcodecs import blosc
from profilehooks import profile

blosc.use_threads = False

dm = HCSDataModule(
    "/hpc/scratch/group.comp.micro/microdl_input/HEK_2022_03_15_Phase5e-4_Denconv_Nuc8e-4_Mem8e-4_pad15_bg50_32slicesHCS.zarr/",
    "Phase3D",
    "Deconvolved-Nuc",
    5,
    0.8,
    batch_size=20,
    num_workers=0,
    augment=True,
)

dm.setup("fit")


@profile(immediate=True, sort="time", dirs=True)
def load_batch(n=1):
    for i, batch in enumerate(dm.train_dataloader()):
        print(batch["source"].device)
        if i == n:
            break


load_batch()
