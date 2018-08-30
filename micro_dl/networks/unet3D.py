"""Unet 3D"""
from micro_dl.networks.base_unet import BaseUNet


class UNet3D(BaseUNet):
    """3D UNet"""

    def __init__(self, network_config):
        """Init"""

        super().__init__(network_config=network_config)

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        if self.config['data_format'] == 'channels_first':
            shape = (self.config['num_input_channels'],
                     self.config['width'],
                     self.config['height'],
                     self.config['depth'])
        else:
            shape = (self.config['width'],
                     self.config['height'],
                     self.config['depth'],
                     self.config['num_input_channels'])
        return shape
