import logging
import os
import tempfile
from typing import Any, Callable, Literal, Union

import numpy as np
import torch
import zarr
from iohub.ngff import ImageArray, Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    MapTransform,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianSmoothd,
    RandWeightedCropd,
)
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset


class NormalizeTargetd(MapTransform):
    def __init__(self, keys, plate: Plate, target_channel: str) -> None:
        super().__init__(keys, allow_missing_keys=False)
        norm_meta = plate.zattrs["normalization"]
        self.iqr = norm_meta[target_channel]["dataset_statistics"]["iqr"]
        self.median = norm_meta[target_channel]["dataset_statistics"]["median"]

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] - self.median) / self.iqr
        return d


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        source_channel: str,
        target_channel: str,
        z_window_size: int,
        transform: Callable = None,
    ) -> None:
        super().__init__()
        self.positions = positions
        self.source_ch_idx = positions[0].get_channel_index(source_channel)
        self.target_ch_idx = positions[0].get_channel_index(target_channel)
        self.z_window_size = z_window_size
        self.transform = transform
        self._get_windows()

    def _get_windows(self) -> None:
        w = 0
        self.window_keys = []
        self.window_arrays = []
        for fov in self.positions:
            img_arr = fov["0"]
            ts = img_arr.frames
            zs = img_arr.slices - self.z_window_size + 1
            w += ts * zs
            self.window_keys.append(w)
            self.window_arrays.append(img_arr)
        self._max_window = w

    def _find_window(self, index: int) -> tuple[int, int]:
        window_idx = sorted(self.window_keys + [index + 1]).index(index + 1)
        w = self.window_keys[window_idx]
        tz = index - self.window_keys[window_idx - 1] if window_idx > 0 else index
        return self.window_arrays[self.window_keys.index(w)], tz

    def _read_img_window(
        self, img: Union[ImageArray, NDArray], ch_idx: int, tz: int
    ) -> torch.Tensor:
        zs = img.shape[-3] - self.z_window_size + 1
        t = (tz + zs) // zs - 1
        z = tz - t * zs
        selection = (int(t), int(ch_idx), slice(z, z + self.z_window_size))
        data = img[selection][np.newaxis]
        return torch.from_numpy(data)

    def __len__(self) -> int:
        return self._max_window

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img, tz = self._find_window(index)
        source = self._read_img_window(img, self.source_ch_idx, tz)
        target = self._read_img_window(img, self.target_ch_idx, tz)
        sample = {"source": source, "target": target}
        if self.transform:
            sample = self.transform(sample)
        if isinstance(sample, list):
            return sample[0]
        return sample

    def __del__(self):
        self.positions[0].zgroup.store.close()


class HCSDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        source_channel: str,
        target_channel: str,
        z_window_size: int,
        split_ratio: float,
        batch_size: int = 16,
        num_workers: int = 8,
        architecture: Literal["2.5D", "2D", "3D"] = "2.5D",
        yx_patch_size: tuple[int, int] = (256, 256),
        augment: bool = True,
        caching: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.source_channel = source_channel
        self.target_channel = target_channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_2d = True if architecture == "2.5D" else False
        self.z_window_size = z_window_size
        self.split_ratio = split_ratio
        self.yx_patch_size = yx_patch_size
        self.augment = augment
        self.caching = caching

    def _cache(self, lazy_plate: Plate) -> Plate:
        self.tmp_zarr = os.path.join(
            tempfile.gettempdir(), os.path.basename(self.data_path)
        )
        logging.info(f"Caching dataset at {self.tmp_zarr}.")
        mem_store = zarr.NestedDirectoryStore(self.tmp_zarr)
        _, skipped, _ = zarr.copy(
            lazy_plate.zgroup,
            zarr.open(mem_store, mode="a"),
            name="/",
            log=logging.debug,
            if_exists="skip_initialized",
            compressor=None,
        )
        if skipped > 0:
            logging.warning(
                f"Skipped {skipped} items when caching. Check debug log for details."
            )
        return Plate(group=zarr.open(mem_store, mode="r"))

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        if stage in (None, "fit", "validate"):
            plate = open_ome_zarr(self.data_path, mode="r")
            if self.caching:
                plate = self._cache(plate)
            # disable metadata tracking in MONAI for performance
            set_track_meta(False)
            # define training stage transforms
            normalize_transform = [
                NormalizeTargetd("target", plate, self.target_channel)
            ]
            fit_transform = self._fit_transform()
            train_transform = Compose(
                normalize_transform + self._train_transform() + fit_transform
            )
            val_transform = Compose(normalize_transform + fit_transform)
            # shuffle positions, randomness is handled globally
            positions = [pos for _, pos in plate.positions()]
            shuffled_indices = torch.randperm(len(positions))
            positions = list(positions[i] for i in shuffled_indices)
            num_train_fovs = int(len(positions) * self.split_ratio)
            # train/val split
            dataset_settings = dict(
                source_channel=self.source_channel,
                target_channel=self.target_channel,
                z_window_size=self.z_window_size,
            )
            self.train_dataset = SlidingWindowDataset(
                positions[:num_train_fovs],
                transform=train_transform,
                **dataset_settings,
            )
            self.val_dataset = SlidingWindowDataset(
                positions[num_train_fovs:],
                transform=val_transform,
                **dataset_settings,
            )
        # test/predict stage
        else:
            raise NotImplementedError(f"{stage} stage")

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.target_2d and not isinstance(batch, torch.Tensor):
            batch["target"] = batch["target"][:, :, self.z_window_size // 2][:, :, None]
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def _fit_transform(self):
        return [
            CenterSpatialCropd(
                keys=["source", "target"],
                roi_size=(
                    -1,
                    self.yx_patch_size[0],
                    self.yx_patch_size[1],
                ),
            )
        ]

    def _train_transform(self) -> list[Callable]:
        transforms = [
            RandWeightedCropd(
                keys=["source", "target"],
                w_key="target",
                spatial_size=(-1, self.yx_patch_size[0] * 2, self.yx_patch_size[1] * 2),
                num_samples=1,
            )
        ]
        if self.augment:
            transforms.extend(
                [
                    RandAffined(
                        keys=["source", "target"],
                        prob=0.5,
                        rotate_range=(np.pi, 0, 0),
                        shear_range=(0, (0.05), (0.05)),
                        scale_range=(0, 0.2, 0.2),
                    ),
                    RandAdjustContrastd(keys=["source"], prob=0.1, gamma=(0.75, 1.5)),
                    RandGaussianSmoothd(
                        keys=["source"],
                        prob=0.2,
                        sigma_x=(0.05, 0.25),
                        sigma_y=(0.05, 0.25),
                        sigma_z=((0.05, 0.25)),
                    ),
                ]
            )
        return transforms
