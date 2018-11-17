import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Add, Concatenate

from micro_dl.utils.aux_utils import get_channel_axis
from micro_dl.utils.network_utils import get_keras_layer


def pad_channels(input_layer, final_layer, channel_axis):
    """Zero pad along channels before residual/skip merge

    :param keras.layers input_layer: input layer to be padded with zeros / 1x1
    to match shape of final layer
    :param keras.layers final_layer: layer whose shape has to be matched
    :param int channel_axis: dimension along which to pad
    :return: keras.layer layer_padded - layer with the same shape as final
     layer
    """

    num_input_layers = tf.shape(input_layer)[channel_axis]
    num_final_layers = tf.shape(final_layer)[channel_axis]
    num_zero_channels = num_final_layers - num_input_layers
    tensor_zeros = K.zeros_like(final_layer)
    tensor_zeros, _ = tf.split(tensor_zeros,
                               [num_zero_channels, num_input_layers],
                               axis=channel_axis)
    if num_zero_channels % 2 == 0:
        delta = 0
    else:
        delta = 1

    top_block, bottom_block = tf.split(
        tensor_zeros,
        [(num_zero_channels + delta) // 2,
         (num_zero_channels - delta) // 2],
        axis=channel_axis
    )
    layer_padded = tf.concat([top_block, input_layer, bottom_block],
                             axis=channel_axis)
    op_shape = final_layer.get_shape().as_list()
    layer_padded.set_shape(tuple(op_shape))
    return layer_padded


def _crop_layer(input_layer, final_layer, data_format, num_dims, padding):
    """Crop input layer to match shape of final layer

    ONLY SYMMETRIC CROPPING IS HANDLED HERE!

    :param keras.layers final_layer: last layer of conv block or skip layers
     in Unet
    :param keras.layers input_layer: input_layer to the block
    :param str data_format: [channels_first, channels_last]
    :param int num_dims: as named
    :param str padding: same or valid
    :return: keras.layer, input layer cropped if shape is different than final
     layer, else input layer as is
    """
    if padding == 'same':
        return input_layer

    input_shape = tf.shape(input_layer)
    final_shape = tf.shape(final_layer)
    # offsets for the top left corner of the crop
    if data_format == 'channels_first':
        offsets = [0, 0, (input_shape[2] - final_shape[2]) // 2,
                   (input_shape[3] - final_shape[3]) // 2]
        crop_shape = [-1, final_shape[1], final_shape[2], final_shape[3]]
        if num_dims == 3:
            offsets.append((input_shape[4] - final_shape[4]) // 2)
            crop_shape.append(final_shape[4])
    else:
        offsets = [0, (input_shape[1] - final_shape[1]) // 2,
                   (input_shape[2] - final_shape[2]) // 2]
        crop_shape = [-1, final_shape[1], final_shape[2]]
        if num_dims == 3:
            offsets.append((input_shape[3] - final_shape[3]) // 2)
            crop_shape.append(final_shape[3])
        offsets.append(0)
        crop_shape.append(final_shape[-1])
    # https://github.com/tensorflow/tensorflow/issues/19376
    input_cropped = tf.slice(input_layer, offsets, crop_shape)

    op_shape = final_layer.get_shape().as_list()
    channel_axis = get_channel_axis(data_format)
    op_shape[channel_axis] = input_layer.get_shape().as_list()[channel_axis]
    input_cropped.set_shape(tuple(op_shape))

    return input_cropped


def _merge_residual(final_layer,
                    input_layer,
                    data_format,
                    num_dims,
                    kernel_init,
                    padding):
    """Add residual connection from input to last layer

    :param keras.layers final_layer: last layer
    :param keras.layers input_layer: input_layer
    :param str data_format: [channels_first, channels_last]
    :param int num_dims: as named
    :param str kernel_init: kernel initializer from config
    :param str padding: same or valid
    :return: input_layer 1x1 / padded to match the shape of final_layer
     and added
    """

    channel_axis = get_channel_axis(data_format)
    conv_object = get_keras_layer(type='conv',
                                  num_dims=num_dims)
    num_final_layers = int(final_layer.get_shape()[channel_axis])
    num_input_layers = int(input_layer.get_shape()[channel_axis])
    # crop input if padding='valid'
    input_layer = Lambda(_crop_layer,
                         arguments={'final_layer': final_layer,
                                    'data_format': data_format,
                                    'num_dims': num_dims,
                                    'padding': padding})(input_layer)

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
            pad_channels,
                      arguments={'final_layer': final_layer,
                                 'channel_axis': channel_axis})(input_layer)
    layer = Add()([final_layer, input_layer])
    return layer


def skip_merge(skip_layers,
               upsampled_layers,
               skip_merge_type,
               data_format,
               num_dims,
               padding):
    """Skip connection concatenate/add to upsampled layer

    :param keras.layer skip_layers: as named
    :param keras.layer upsampled_layers: as named
    :param str skip_merge_type: [add, concat]
    :param str data_format: [channels_first, channels_last]
    :param int num_dims: as named
    :param str padding: same or valid
    :return: keras.layer skip merged layer
    """

    channel_axis = get_channel_axis(data_format)

    # crop input if padding='valid'
    skip_layers = Lambda(_crop_layer,
                         arguments={'final_layer': upsampled_layers,
                                    'data_format': data_format,
                                    'num_dims': num_dims,
                                    'padding': padding})(skip_layers)

    if skip_merge_type == 'concat':
        layer = Concatenate(axis=channel_axis)([upsampled_layers,
                                                skip_layers])
    else:
        skip_layers = Lambda(
            pad_channels,
            arguments={'final_layer': upsampled_layers,
                       'channel_axis': channel_axis})(skip_layers)
        layer = Add()([upsampled_layers, skip_layers])
    return layer