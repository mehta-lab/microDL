import numpy as np
import torch
import torch.nn as nn
from monai.optimizers import WarmupCosineSchedule
from lightning.pytorch import LightningModule

from micro_dl.torch_unet.networks.Unet25D import Unet25d
from micro_dl.torch_unet.utils.model import ModelDefaults25D, define_model


class PhaseToNuc25D(LightningModule):
    def __init__(
        self,
        model_config: dict = {},
        batch_size: int = 16,
        max_epochs: int = 100,
        loss_function: nn.Module = None,
        lr: float = 1e-3,
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
        """
        super().__init__()
        self.model = define_model(Unet25d, ModelDefaults25D(), model_config)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss_function = loss_function if loss_function else nn.MSELoss()
        self.lr = lr

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
            return np.concatenate(
                [
                    img[0].numpy().max(axis=(0, 1))
                    for img in (source, target, pred)
                ],
                axis=1,
            )

    def on_validation_epoch_end(self, validation_step_outputs):
        grid = np.concatenate(validation_step_outputs, axis=0)
        self.logger.experiment.add_image("val_samples", grid)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=3, t_total=self.max_epochs
        )
        return [optimizer], [scheduler]
