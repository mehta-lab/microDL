from keras import backend as K
from keras.layers import MaxPooling2D, AveragePooling2D, Concatenate

from micro_dl.networks.base_unet import BaseUNet


class UNet2D(BaseUNet):
    """2D UNet"""

    def ___init__(self, network_config):
        """Init"""

        super().__init__(network_config=network_config)

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        shape = (
            self.config['num_input_channels'],
            self.config['height'],
            self.config['width']
        )
        return shape

    def _set_pooling_type(self):
        """Set the pooling type"""

        pool = self.config['pooling_type']
        self.Pooling = {
            'max': MaxPooling2D,
            'average': AveragePooling2D
        }[pool]

    @staticmethod
    def _pad_channels(input_layer, num_desired_channels,
                      final_layer, channel_axis):
        """Zero pad along channels before residual/skip merge"""

        input_zeros = K.zeros_like(final_layer)
        num_input_layers = int(input_layer.get_shape()[channel_axis])
        new_zero_channels = int((num_desired_channels - num_input_layers) / 2)
        if num_input_layers % 2 == 0:
            zero_pad_layers = input_zeros[:, :new_zero_channels, :, :]
            layer_padded = Concatenate(axis=channel_axis)(
                [zero_pad_layers, input_layer, zero_pad_layers]
            )
        else:
            zero_pad_layers = input_zeros[:, :new_zero_channels+1, :, :]
            layer_padded = Concatenate(axis=channel_axis)(
                [zero_pad_layers, input_layer, zero_pad_layers[:, :-1, :, :]]
            )
        return layer_padded