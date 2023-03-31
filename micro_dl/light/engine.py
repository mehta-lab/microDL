from matplotlib.cm import get_cmap
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from torch.optim.lr_scheduler import ConstantLR
from typing import Literal

from micro_dl.torch_unet.networks.Unet25D import Unet25d
from micro_dl.torch_unet.utils.model import ModelDefaults25D, define_model


class PhaseToNuc25D(LightningModule):
    def __init__(
        self,
        model_config: dict = {},
        batch_size: int = 16,
        loss_function: nn.Module = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
    ) -> None:
        """Regression U-Net module for virtual staining.

        Parameters
        ----------
        model : nn.Module
            U-Net model
        batch_size : int, optional
            Batch size, by default 16
        max_epochs : int, optional
            Max epochs in fitting, by default 100
        loss_function : nn.Module, optional
            Loss function module, by default L2
        lr : float, optional
            Learning rate, by default 1e-3
        schedule: Literal["WarmupCosine", "Constant"], optional
            Learning rate scheduler, by default 'Constant'
        """
        super().__init__()
        self.model = define_model(Unet25d, ModelDefaults25D(), model_config)
        self.batch_size = batch_size
        self.loss_function = loss_function if loss_function else nn.MSELoss()
        self.lr = lr
        self.schedule = schedule
        self.validation_step_outputs = []

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        target = batch["target"]
        pred = self.forward(batch["source"])
        loss = self.loss_function(pred, target)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        source = batch["source"]
        target = batch["target"]
        pred = self.forward(source)
        loss = self.loss_function(pred, target)
        self.log("val_loss", loss, batch_size=self.batch_size)
        if batch_idx % 10 == 0:
            self.validation_step_outputs.append(
                [
                    np.squeeze(img[0].cpu().numpy().max(axis=(0, 1)))
                    for img in (source, target, pred)
                ]
            )

    def on_validation_epoch_end(self):
        """Plot and log sample images"""
        images_grid = [[]] * len(self.validation_step_outputs)
        for row, imgs in enumerate(self.validation_step_outputs):
            for im, cm_name in zip(imgs, ["gray"] + ["inferno"] * 2):
                rendered_im = get_cmap(cm_name)(im, bytes=True)[..., :3]
                images_grid[row].append(rendered_im.transpose(1,2,0))
            images_grid[row] = np.concatenate(images_grid[row], axis=1)
        grid = np.concatenate(images_grid, axis=0)
        self.logger.experiment.add_image("val_samples", grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.schedule == "WarmupCosine":
            scheduler = WarmupCosineSchedule(
                optimizer, warmup_steps=3, t_total=self.trainer.max_epochs
            )
        elif self.schedule == "Constant":
            scheduler = ConstantLR(
                optimizer, factor=1, total_iters=self.trainer.max_epochs
            )
        return [optimizer], [scheduler]
