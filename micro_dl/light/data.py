from typing import Callable, Literal, Union

import numpy as np
import torch
from iohub.ngff import ImageArray, Plate, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms import (
    CenterSpatialCrop,
    Compose,
    RandAdjustContrast,
    RandAffine,
    RandGaussianSmooth,
    RandSpatialCrop,
    ScaleIntensityRangePercentiles,
)
from torch.utils.data import DataLoader, Dataset, Subset


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        plate: Plate,
        source_ch_idx: int,
        target_ch_idx: int,
        z_window_size: int,
        transform: Callable = None,
        source_transform: Callable = None,
        target_center_slice_only: bool = True,
    ) -> None:
        super().__init__()
        self.plate = plate
        self.source_ch_idx = source_ch_idx
        self.target_ch_idx = target_ch_idx
        self.z_window_size = z_window_size
        self.transform = transform
        self.source_transform = source_transform
        self.target_center_slice_only = target_center_slice_only
        self._count_windows()

    def _count_windows(self) -> None:
        w = 0
        self.windows = {}
        for _, fov in self.plate.positions():
            ts = fov["0"].frames
            zs = fov["0"].slices - self.z_window_size + 1
            w += ts * zs
            self.windows[w] = fov
        self._max_window = w

    def _find_window(self, index: int) -> tuple[int, int]:
        window_keys = list(self.windows.keys())
        window_idx = sorted(window_keys + [index + 1]).index(index + 1)
        w = window_keys[window_idx]
        tz = index - window_keys[window_idx - 1] if window_idx > 0 else index
        return w, tz

    def _read_img_window(self, img: ImageArray, ch_idx: int, tz: int) -> torch.Tensor:
        zs = img.slices - self.z_window_size + 1
        t = (tz + zs) // zs - 1
        z = tz - t * zs
        if ch_idx == self.target_ch_idx and self.target_center_slice_only:
            z_slice = int(z + self.z_window_size // 2)
            pre_dim = 2
        else:
            z_slice = slice(z, z + self.z_window_size)
            pre_dim = 1
        selection = (int(t), int(ch_idx), z_slice)
        data = img[selection][(np.newaxis,) * pre_dim]
        return torch.from_numpy(data)

    def __len__(self) -> int:
        return self._max_window

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        w, tz = self._find_window(index)
        img = self.windows[w]["0"]
        source = self._read_img_window(img, self.source_ch_idx, tz)
        target = self._read_img_window(img, self.target_ch_idx, tz)
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)
        if self.source_transform:
            source = self.transform(source)
        return {"source": source, "target": target}

    def __del__(self):
        self.plate.close()


class HCSDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        source_channel_name: str,
        target_channel_name: str,
        z_window_size: int,
        split_ratio: float,
        batch_size: int = 16,
        num_workers: int = 8,
        yx_patch_size: tuple[int, int] = (256, 256),
        augment: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.z_window_size = z_window_size
        self.yx_patch_size = yx_patch_size
        self.augment = augment
        # read metadata in the main process
        with open_ome_zarr(data_path, mode="r") as plate:
            self.num_fovs = len(list(plate.positions()))
            self.sep = int(self.num_fovs * split_ratio)
            self.source_ch_idx = plate.get_channel_index(source_channel_name)
            self.target_ch_idx = plate.get_channel_index(target_channel_name)

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        # train/val split
        if stage in (None, "fit"):
            # set training stage transforms
            fit_transform = self._fit_transform()
            train_transform = Compose(self._train_transform() + fit_transform)
            train_source_transform = None
            if self.augment:
                train_source_transform = Compose(self._train_source_transform())
            plate = open_ome_zarr(self.data_path, mode="r")
            whole_train_dataset = SlidingWindowDataset(
                plate,
                source_ch_idx=self.source_ch_idx,
                target_ch_idx=self.target_ch_idx,
                z_window_size=self.z_window_size,
                transform=train_transform,
                source_transform=train_source_transform,
            )
            whole_val_dataset = SlidingWindowDataset(
                plate,
                source_ch_idx=self.source_ch_idx,
                target_ch_idx=self.target_ch_idx,
                z_window_size=self.z_window_size,
                transform=Compose(fit_transform),
            )
            # randomness is handled globally
            indices = torch.randperm(self.num_fovs)
            self.train_dataset = Subset(whole_train_dataset, indices[: self.sep])
            self.val_dataset = Subset(whole_val_dataset, indices[self.sep :])
        # test stage
        if stage in (None, "test"):
            raise NotImplementedError(f"{stage} stage")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def _fit_transform(self):
        return [
            CenterSpatialCrop(
                (
                    -1,
                    self.yx_patch_size[0],
                    self.yx_patch_size[1],
                )
            ),
            ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1),
        ]

    def _train_transform(self) -> list[Callable]:
        transforms = [
            RandSpatialCrop(
                roi_size=(
                    -1,
                    self.yx_patch_size[0] * 2,
                    self.yx_patch_size[1] * 2,
                ),
                random_size=False,
            )
        ]
        if self.augment:
            transforms.append(
                RandAffine(
                    prob=0.5,
                    rotate_range=(0, np.pi, np.pi),
                    shear_range=(0, 0.1, 0.1),
                    scale_range=(0, 0.5, 0.5),
                )
            )
        return transforms

    def _train_source_transform(self):
        transforms = [
            RandAdjustContrast(prob=0.1, gamma=(0.5, 2.0)),
            RandGaussianSmooth(
                prob=0.2,
                sigma_x=(0.05, 0.25),
                sigma_y=(0.05, 0.25),
                sigma_z=(0.05, 0.25),
            ),
        ]
        return transforms
