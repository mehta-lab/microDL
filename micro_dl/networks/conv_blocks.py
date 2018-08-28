"""Collection of different conv blocks typically used in conv nets"""
from keras.layers import BatchNormalization, Dropout

from micro_dl.utils.network_utils import create_activation_layer,\
    get_channel_axis, get_keras_layer


def conv_bn_activation_block(layer, network_config):
    """Conv-BN-activation block

    To accommodate params of advanced activations, activation is a dict with
    keys 'type' and 'params'

    :param keras.layers layer: current input layer
    :param dict network_config: dict with the following keys
     int num_convs_per_block: as named
     int num_filters: as named
     tuple filter_size: as named
     str activation: activation type, and other advanced activation related
      params
     str init: method used for initializing weights
     str padding: as named
     bool batch_norm: indicator for batch norm
     float dropout_prob: as named
     str data_format: as named. [channels_last, channels_first]
     int num_dims: dimensionality of the filter
     str layer_order: order of conv, BN and activation
    :return: keras.layers after convolution->BN->activ
    """

    conv = get_keras_layer(type='conv', num_dims=network_config['num_dims'])
    activation_layer_instance = create_activation_layer(
        network_config['activation']
    )

    for _ in range(network_config['num_convs_per_block']):
        layer = conv(filters=network_config['num_filters'],
                     kernel_size=network_config['filter_size'],
                     padding=network_config['padding'],
                     kernel_initializer=network_config['init'],
                     data_format=network_config['data_format'])(layer)
        if network_config['batch_norm']:
            layer = BatchNormalization(axis=get_channel_axis(
                network_config['data_format']
            ))(layer)

        layer = create_activation_layer(layer, network_config['activation'])
        if network_config['dropout_prob']:
            layer = Dropout(network_config['dropout_prob'])(layer)
    return layer


def conv_activation_bn_block(layer, network_config):
    """Conv-Activation-BN block

    Rationale to put activation before BN is that 'BN after activation will
    normalize the positive features without statistically biasing them with
    features that do not make it through to the next convolutional layer'
    https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
    :param layer, network_config: please refer to conv_bn_activation_block()
    """

    conv = get_keras_layer(type='conv', num_dims=network_config['num_dims'])
    for _ in range(network_config['num_convs_per_block']):
        layer = conv(filters=network_config['num_filters'],
                     kernel_size=network_config['filter_size'],
                     padding=network_config['padding'],
                     kernel_initializer=network_config['init'],
                     data_format=network_config['data_format'])(layer)

        layer = create_activation_layer(layer, network_config['activation'])

        if network_config['batch_norm']:
            layer = BatchNormalization(axis=get_channel_axis(
                network_config['data_format']
            ))(layer)

        if network_config['dropout_prob']:
            layer = Dropout(network_config['dropout_prob'])(layer)
    return layer


