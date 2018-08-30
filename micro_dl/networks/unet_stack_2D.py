"""Predict the center slice from a stack of 3-5 slices"""
import tensorflow as tf
import warnings
from keras.layers import Activation, Conv3D, Input, UpSampling3D

from micro_dl.networks.base_unet import BaseUNet
from micro_dl.utils.aux_utils import import_class, validate_config


class UNetStackTo2D(BaseUNet):
    """Implements a U-net that takes a stack and predicts the center slice"""

    def __init__(self, network_config):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        super().__init__(network_config)
        req_params = ['depth']
        param_check, msg = validate_config(network_config, req_params)
        if not param_check:
            raise ValueError(msg)

        if network_config['depth'] > 5:
            warnings.warn('using more than 5 slices to predict center slice',
                          Warning)
        self.config['num_dims'] = 3
        if self.config['upsampling'] == 'repeat':
            self.UpSampling = UpSampling3D
        else:
            self.UpSampling = import_class('networks',
                                           'InterpUpSampling3D')

    def _skip_block(self, input_layer, num_slices, num_filters):
        """Converts skip layers from 3D to 2D: 1x1 along Z

        The contracting path of this U-net uses 3D images of shape
        [x, y, depth]. The expanding path reduces the shape to [x, y, 1]

        :param keras.layers input_layer: layers to be used in skip connection
        :param int num_slices: as named
        :param int num_filters: as named
        :return: convolved layer with valid padding
        """

        filter_shape = (1, 1, num_slices)
        layer = Conv3D(filters=num_filters,
                       kernel_size=filter_shape,
                       padding='valid',
                       data_format=self.config['data_format'])(input_layer)
        return layer

    def build_net(self):
        """Assemble the network

        Treat the downsampling blocks as 3D and the upsampling blocks as 2D.
        All blocks use 3D filters: either 3x3x3 or 3x3x1
        """

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling + middle blocks ---------
        filter_size = self.config['filter_size']
        num_slices = self.config['depth']
        filter_shape = (filter_size, filter_size, num_slices)

        input_layer, skip_layers_list = super()._downsampling_branch(
            input_layer=input_layer,
            filter_shape=filter_shape,
            downsample_shape=(2, 2, 1)
        )

        #  ---------- skip block before upsampling ---------
        block_name = 'skip_block_{}'.format(
            len(self.config['num_filters_per_block'])
        )
        with tf.name_scope(block_name):
            layer = self._skip_block(
                input_layer=input_layer,
                num_slices=num_slices,
                num_filters=self.config['num_filters_per_block'][-1]
            )
        input_layer = layer

        # ------------- Upsampling / decoding blocks -------------
        upsampling_shape = (2, 2, 1)
        self.config['filter_size'] = (filter_size, filter_size, 1)
        for block_idx in reversed(range(self.num_down_blocks)):
            cur_skip_layers = skip_layers_list[block_idx]
            cur_skip_layers = self._skip_block(
                input_layer=cur_skip_layers,
                num_slices=num_slices,
                num_filters=self.config['num_filters_per_block'][block_idx]
            )
            block_name = 'up_block_{}'.format(block_idx)
            with tf.name_scope(block_name):
                layer = super()._upsampling_block(
                    input_layers=input_layer,
                    skip_layers=cur_skip_layers,
                    block_idx=block_idx,
                    upsampling_shape=upsampling_shape
                )
            input_layer = layer

        # ------------ output block ------------------------
        final_activation = self.config['final_activation']
        with tf.name_scope('output'):
            layer = Conv3D(filters=1,
                           kernel_size=(1, 1, 1),
                           padding='same',
                           kernel_initializer='he_normal',
                           data_format=self.config['data_format'])(input_layer)
            outputs = Activation(final_activation)(layer)
        return inputs, outputs

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        if self.config['data_format'] == 'channels_first':
            shape = (1,
                     self.config['width'],
                     self.config['height'],
                     self.config['depth'])
        else:
            shape = (self.config['width'],
                     self.config['height'],
                     self.config['depth'], 1)
        return shape
