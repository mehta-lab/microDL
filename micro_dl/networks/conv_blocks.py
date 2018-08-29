"""Collection of different conv blocks typically used in conv nets"""
import keras.backend as K
from keras.layers import BatchNormalization, Conv2D, Conv3D, Dropout, Lambda
from keras.layers.merge import Add, Concatenate
import tensorflow as tf

from micro_dl.utils.aux_utils import get_channel_axis
from micro_dl.utils.network_utils import create_activation_layer,\
    create_layer_sequence, get_keras_layer


def conv_block(layer, network_config, block_idx):
    """Convolution block

    Allowed block-seq: [conv-BN-activation, conv-activation-BN,
     BN-activation-conv]
    To accommodate params of advanced activations, activation is a dict with
     keys 'type' and 'params'.

    :param keras.layers layer: current input layer
    :param dict network_config: dict with the following keys
     int num_convs_per_block: as named
     int num_filters: as named
     tuple filter_size: as named
     dict activation: keys: type: activation type, and params: other advanced
      activation related params
     str init: method used for initializing weights
     str padding: as named
     bool batch_norm: indicator for batch norm
     float dropout_prob: as named
     str data_format: as named. [channels_last, channels_first]
     int num_dims: dimensionality of the filter
     str block_sequence: order of conv, BN and activation
     float dropout: dropout probablility ...
    :param int block_idx: block index in the network
    :return: keras.layers after performing operations in block-sequence
     repeated for num_convs_per_block times
    """

    conv = get_keras_layer(type='conv', num_dims=network_config['num_dims'])
    activation_layer_instance = create_activation_layer(
        network_config['activation']
    )
    bn_instance = BatchNormalization(axis=get_channel_axis(
                network_config['data_format']
    ))
    block_sequence = create_layer_sequence(network_config['block_sequence'],
                                           conv,
                                           bn_instance,
                                           activation_layer_instance)
    for _ in range(network_config['num_convs_per_block']):
        for cur_layer in block_sequence:
            if isinstance(cur_layer, (Conv2D, Conv3D)):
                layer = conv(filters=network_config['num_filters'][block_idx],
                             kernel_size=network_config['filter_size'],
                             padding=network_config['padding'],
                             kernel_initializer=network_config['init'],
                             data_format=network_config['data_format'])(layer)
            elif isinstance(cur_layer, BatchNormalization) and \
                    network_config['batch_norm']:
                layer = bn_instance(layer)
            else:
                layer = activation_layer_instance(layer)

        if network_config['dropout']:
            layer = Dropout(network_config['dropout'])(layer)
    return layer


def downsample_conv_block(layer, network_config, block_idx):
    """Conv-BN-activation block

    :param keras.layers layer: current input layer
    :param dict network_config: please check conv_block()
    :param int block_idx: block index in the network
    :return: keras.layers after downsampling and conv_block
    """

    conv = get_keras_layer(type='conv', num_dims=network_config['num_dims'])
    activation_layer_instance = create_activation_layer(
        network_config['activation']
    )
    bn_instance = BatchNormalization(axis=get_channel_axis(
                network_config['data_format']
    ))
    block_sequence = create_layer_sequence(network_config['block_sequence'],
                                           conv,
                                           bn_instance,
                                           activation_layer_instance)
    for conv_idx in range(network_config['num_convs_per_block']):
        for cur_layer in block_sequence:
            if isinstance(cur_layer, (Conv2D, Conv3D)):
                if block_idx > 0 and conv_idx == 0:
                    stride = (2, ) * network_config['num_dims']
                else:
                    stride = (1, ) * network_config['num_dims']
                layer = conv(filters=network_config['num_filters'][block_idx],
                             kernel_size=network_config['filter_size'],
                             strides=stride,
                             padding=network_config['padding'],
                             kernel_initializer=network_config['init'],
                             data_format=network_config['data_format'])(layer)
            elif isinstance(cur_layer, BatchNormalization) and \
                    network_config['batch_norm']:
                layer = bn_instance(layer)
            else:
                layer = activation_layer_instance(layer)

        if network_config['dropout']:
            layer = Dropout(network_config['dropout'])(layer)
    return layer


def _pad_channels(input_layer, final_layer, channel_axis):
    """Zero pad along channels before residual/skip merge

    :param keras.layers input_layer:
    :param keras.layers final_layer:
    :param int channel_axis: dimension along which to pad
    """

    num_input_layers = int(input_layer.get_shape()[channel_axis])
    num_final_layers = int(final_layer.get_shape()[channel_axis])
    num_zero_channels = num_final_layers - num_input_layers
    tensor_zeros = K.zeros_like(final_layer)
    tensor_zeros, _ = tf.split(tensor_zeros,
                               [num_zero_channels, num_input_layers],
                               axis=channel_axis)
    if num_zero_channels % 2 == 0:
        top_block, bottom_block = tf.split(
            tensor_zeros,
            [int(num_zero_channels/2), int(num_zero_channels/2)],
            axis=channel_axis
        )
    else:
        top_block, bottom_block = tf.split(
            tensor_zeros,
            [int((num_zero_channels + 1) / 2),
             int((num_zero_channels - 1) / 2)],
            axis=channel_axis
        )
    layer_padded = Concatenate(axis=channel_axis)(
        [top_block, input_layer, bottom_block]
    )
    return layer_padded


def _merge_residual(final_layer,
                    input_layer,
                    data_format,
                    num_dims,
                    kernel_init):
    """Add residual connection from input to last layer

    :param keras.layers final_layer: last layer
    :param keras.layers input_layer: input_layer
    :param str data_format: [channels_first, channels_last]
    :param int num_dims: as named
    :param str kernel_init: kernel initializer from config
    :return: input_layer 1x1 / padded to match the shape of final_layer
     and added
    """

    channel_axis = get_channel_axis(data_format)
    conv_object = get_keras_layer(type='conv',
                                  num_dims=num_dims)
    num_final_layers = int(final_layer.get_shape()[channel_axis])
    num_input_layers = int(input_layer.get_shape()[channel_axis])
    if num_input_layers > num_final_layers:
        # use 1x 1 to get to the desired num of feature maps
        input_layer = conv_object(
            filters=num_final_layers,
            kernel_size=(1, ) * num_dims,
            padding='same',
            kernel_initializer=kernel_init,
            data_format=data_format)(input_layer)
    elif num_input_layers < num_final_layers:
        # padding with zeros along channels
        input_layer = Lambda(
                      _pad_channels,
                      arguments={'num_desired_channels': num_final_layers,
                                 'final_layer': final_layer,
                                 'channel_axis': channel_axis})(input_layer)
    layer = Add()([final_layer, input_layer])
    return layer


def residual_conv_block(layer, network_config, block_idx):
    """Convolution block where the last layer is merged (+) with input layer

    :param keras.layers layer: current input layer
    :param dict network_config: please check conv_block()
    :param int block_idx: block index in the network
    :return: keras.layers after conv-block and residual merge
    """

    input_layer = layer
    final_layer = conv_block(layer, network_config, block_idx)
    layer = _merge_residual(final_layer=final_layer,
                            input_layer=input_layer,
                            data_format=network_config['data_format'],
                            num_dims=network_config['num_dims'],
                            kernel_init=network_config['init'])
    return layer


def residual_downsample_conv_block(layer, network_config, block_idx):
    """Convolution block where the last layer is merged (+) with input layer

    :param keras.layers layer: current input layer
    :param dict network_config: please check conv_block()
    :param int block_idx: block index in the network
    :return: keras.layers after conv-block and residual merge
    """

    input_layer = layer
    if block_idx == 0:
        final_layer = conv_block(layer, network_config, block_idx)
    else:
        final_layer = downsample_conv_block(layer, network_config, block_idx)
        pool_layer = get_keras_layer(type=network_config['pooling_type'],
                                     num_dims=network_config['num_dims'])
        pool_size = (2,) * network_config['num_dims']
        downsampled_input_layer = pool_layer(
            pool_size=pool_size,
            data_format=network_config['data_format']
        )
        input_layer = downsampled_input_layer

    layer = _merge_residual(final_layer=final_layer,
                            input_layer=input_layer,
                            data_format=network_config['data_format'],
                            num_dims=network_config['num_dims'],
                            kernel_init=network_config['init'])
    return layer
