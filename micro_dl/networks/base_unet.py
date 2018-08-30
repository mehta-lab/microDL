"""Base class for U-net"""
import tensorflow as tf
from keras.layers import Activation, Input, Lambda, UpSampling2D, UpSampling3D
from keras.layers.merge import Add, Concatenate

from micro_dl.networks.base_conv_net import BaseConvNet
from micro_dl.networks.conv_blocks import conv_block, pad_channels, \
    residual_conv_block, residual_downsample_conv_block
from micro_dl.utils.aux_utils import get_channel_axis, import_class, \
    validate_config
from micro_dl.utils.network_utils import get_keras_layer


class BaseUNet(BaseConvNet):
    """Base U-net implementation

    1) Unet: https://arxiv.org/pdf/1505.04597.pdf
    2) residual Unet: https://arxiv.org/pdf/1711.10684.pdf
    border_mode='same' preferred over 'valid'. Else have to interpolate the
    last block to match the input image size.
    """

    def __init__(self, network_config):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        super().__init__(network_config)
        req_params = ['num_filters_per_block',
                      'num_convs_per_block',
                      'skip_merge_type',
                      'upsampling',
                      'num_target_channels',
                      'residual',
                      'block_sequence']

        param_check, msg = validate_config(network_config, req_params)
        if not param_check:
            raise ValueError(msg)
        self.config = network_config

        num_down_blocks = len(network_config['num_filters_per_block']) - 1

        width = network_config['width']
        feature_width_at_last_block = width / (2 ** num_down_blocks)
        msg = 'network depth is incompatible with the input size'
        assert feature_width_at_last_block >= 2, msg

        #  keras upsampling repeats the rows and columns in data. leads to
        #  checkerboard in upsampled images. repeat - use keras builtin
        #  nearest_neighbor, bilinear: interpolate using custom layers
        upsampling = network_config['upsampling']
        msg = 'invalid upsampling, not in repeat/bilinear/nearest_neighbor'
        assert upsampling in ['bilinear', 'nearest_neighbor', 'repeat'], msg

        self.config = network_config
        if 'depth' in network_config:
            self.config['num_dims'] = 3
            if upsampling == 'repeat':
                self.UpSampling = UpSampling3D
            else:
                self.UpSampling = import_class('networks',
                                               'InterpUpSampling3D')
        else:
            self.config['num_dims'] = 2
            if upsampling == 'repeat':
                self.UpSampling = UpSampling2D
            else:
                self.UpSampling = import_class('networks',
                                               'InterpUpSampling2D')

        self.num_down_blocks = num_down_blocks

    def _upsampling_block(self,
                          input_layers,
                          skip_layers,
                          block_idx,
                          upsampling_shape=None):
        """Upsampling blocks of U net

        The skip layers could be either concatenated or added

        :param keras.layes input_layers: input layer to be upsampled
        :param keras.layers skip_layers: skip layers from the downsampling path
        :param int block_idx: block in the downsampling path to be used for
         skip connection
        :param tuple upsampling_shape: allows for anisotropic upsampling
        :return: keras.layers after upsampling, skip-merge, conv block
        """

        if upsampling_shape is None:
            upsampling_shape = (2, ) * self.config['num_dims']

        # upsampling
        if self.config['upsampling'] == 'repeat':
            layer_upsampled = self.UpSampling(
                size=upsampling_shape,
                data_format=self.config['data_format']
            )(input_layers)
        else:
            layer_upsampled = self.UpSampling(
                size=upsampling_shape,
                data_format=self.config['data_format'],
                interp_type=self.config['upsampling']
            )(input_layers)

        # skip-merge
        channel_axis = get_channel_axis(self.config['data_format'])
        if self.config['skip_merge_type'] == 'concat':
            layer = Concatenate(axis=channel_axis)([layer_upsampled,
                                                    skip_layers])
        else:
            skip_layers = Lambda(
                pad_channels,
                arguments={'final_layer': layer_upsampled,
                           'channel_axis': channel_axis})(skip_layers)
            layer = Add()([layer_upsampled, skip_layers])

        # conv
        if self.config['residual']:
            layer = residual_conv_block(layer=layer,
                                        network_config=self.config,
                                        block_idx=block_idx)
        else:
            layer = conv_block(layer=layer,
                               network_config=self.config,
                               block_idx=block_idx)
        return layer

    def _downsampling_branch(self,
                             input_layer,
                             filter_shape=None,
                             downsample_shape=None):
        """Downsampling half of U-net

        :param keras.layer input_layer: must be the output of Input layer
        :param tuple filter_shape: filter size is an int for most cases.
         filter_shape enables passing anisotropic filter shapes
        :return keras.layer layer: output layer of bridge/middle block
         skip_layers_list: list of all skip layers
        """

        if filter_shape is not None:
            self.config['filter_size'] = filter_shape

        if downsample_shape is None:
            downsample_shape = (2,) * self.config['num_dims']

        skip_layers_list = []
        for block_idx in range(self.num_down_blocks + 1):
            block_name = 'down_block_{}'.format(block_idx + 1)
            with tf.name_scope(block_name):
                if self.config['residual']:
                    layer = residual_downsample_conv_block(
                        layer=input_layer,
                        network_config=self.config,
                        block_idx=block_idx,
                        downsample_shape=downsample_shape
                    )
                    skip_layers_list.append(layer)
                else:
                    layer = conv_block(layer=input_layer,
                                       network_config=self.config,
                                       block_idx=block_idx)
                    skip_layers_list.append(layer)
                    if block_idx < self.num_down_blocks:
                        pool_object = get_keras_layer(
                            type=self.config['pooling_type'],
                            num_dims=self.config['num_dims']
                        )
                        layer = pool_object(
                            pool_size=downsample_shape,
                            data_format=self.config['data_format']
                        )(layer)
            input_layer = layer
        del skip_layers_list[-1]
        return layer, skip_layers_list

    def build_net(self):
        """Assemble the network"""

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling + middle blocks ---------
        input_layer, skip_layers_list = self._downsampling_branch(
            input_layer=input_layer
        )

        # ------------- Upsampling / decoding blocks -------------
        for block_idx in reversed(range(self.num_down_blocks)):
            cur_skip_layers = skip_layers_list[block_idx]
            block_name = 'up_block_{}'.format(block_idx)
            with tf.name_scope(block_name):
                layer = self._upsampling_block(input_layers=input_layer,
                                               skip_layers=cur_skip_layers,
                                               block_idx=block_idx)
            input_layer = layer

        # ------------ output block ------------------------
        final_activation = self.config['final_activation']
        num_output_channels = self.config['num_target_channels']
        conv_object = get_keras_layer(type='conv',
                                      num_dims=self.config['num_dims'])
        with tf.name_scope('output'):
            layer = conv_object(
                filters=num_output_channels,
                kernel_size=(1,) * self.config['num_dims'],
                padding=self.config['padding'],
                kernel_initializer=self.config['init'],
                data_format=self.config['data_format'])(input_layer)
            outputs = Activation(final_activation)(layer)
        return inputs, outputs