"""Unet 2D"""
from micro_dl.networks.base_unet import BaseUNet


class UNet2D(BaseUNet):
    """2D UNet"""

    def ___init__(self, network_config):
        """Init"""

        super().__init__(network_config=network_config)
        print('init', self.UpSampling)

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        shape = (
            self.config['num_input_channels'],
            self.config['height'],
            self.config['width']
        )
        return shape
