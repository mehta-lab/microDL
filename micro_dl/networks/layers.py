"""Bilinear interpolation for upsampling"""
import keras.backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf


class BilinearUpSampling2D(Layer):
    """Interpolates the feature map for upsampling"""

    def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
        """Init

        https://github.com/aurora95/Keras-FCN/blob/master/utils/BilinearUpSampling.py
        https://keunwoochoi.wordpress.com/2016/11/18/for-beginners-writing-a-custom-keras-layer/
        https://keras.io/layers/writing-your-own-keras-layers/
        resizing by int only!

        :param int/tuple size: upsampling factor
        :param str data_format: allowed options are 'channels_last',
         'channels_first'
        """

        msg = 'data_format is not in channels_first/last'
        assert data_format in ['channels_last', 'channels_first'], msg

        if isinstance(size, int):
            size = (size, ) * 2
        self.data_format = data_format
        self.size = size
        self.input_spec = [InputSpec(ndim=4)]
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build layer

        There are no weights for bilinear interpolation
        """

        self.input_spec = [InputSpec(shape=input_shape)]
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute shape of output

        :param tuple input_shape:
        """

        input_shape = np.array(input_shape)
        if self.data_format == 'channels_first':
            # convert to channels_last
            input_shape = input_shape[[0, 2, 3, 1]]

        width = int(self.size[0] * input_shape[1]
                    if input_shape[2] is not None else None)
        height = int(self.size[1] * input_shape[2]
                     if input_shape[3] is not None else None)

        if self.data_format == 'channels_first':
            #  switching back
            input_shape = input_shape[[0, 3, 1, 2]]
            return tuple(input_shape[0], input_shape[1], width, height)

        if self.data_format == 'channels_last':
            return tuple(input_shape[0], width, height, input_shape[3])

    def call(self, x, mask=None):
        """Layer's logic

        tf.image.resize_bilinear uses channels_last and has border issues!
        https://github.com/tensorflow/tensorflow/issues/6720
        """

        original_shape = K.int_shape(x)
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 2, 3, 1])

        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array(self.size).astype('int32'))
        x = tf.image.resize_bilinear(x, new_shape, align_corners=True)

        if self.data_format == 'channels_last':
            x.set_shape((None,
                         original_shape[1] * self.size[0],
                         original_shape[2] * self.size[1]),
                        None)
            return x

        if self.data_format == 'channels_first':
            #  switch back to channels_first
            x = tf.transpose(x, [0, 3, 1, 2])
            x.set_shape((None, None,
                         original_shape[2] * self.size[0],
                         original_shape[3] * self.size[1]))
            return x

    def get_config(self):
        """Return config"""

        base_config = super().get_config()
        base_config['size'] = self.size
        base_config['data_format'] = self.data_format
        return base_config


class BilinearUpSampling3D(Layer):
    """Interpolates the feature map for upsampling"""

    def __init__(self, size=(2, 2, 2), data_format='channels_last', **kwargs):
        """Init

        :param int/tuple size: upsampling factor
        :param str data_format: allowed options are 'channels_last',
         'channels_first'
        """

        msg = 'data_format is not in channels_first/last'
        assert data_format in ['channels_last', 'channels_first'], msg

        if isinstance(size, int):
            size = (size, ) * 3
        self.data_format = data_format
        self.size = size
        self.input_spec = [InputSpec(ndim=5)]
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build layer

        There are no weights for bilinear interpolation
        """

        self.input_spec = [InputSpec(shape=input_shape)]
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute shape of output

        :param tuple input_shape:
        """

        input_shape = np.array(input_shape)
        if self.data_format == 'channels_first':
            # convert to channels_last
            input_shape = input_shape[[0, 2, 3, 4, 1]]

        width = int(self.size[0] * input_shape[1]
                    if input_shape[1] is not None else None)
        height = int(self.size[1] * input_shape[2]
                     if input_shape[2] is not None else None)
        depth = int(self.size[2] * input_shape[3]
                    if input_shape[3] is not None else None)

        if self.data_format == 'channels_last':
            return tuple(input_shape[0], width, height, depth, input_shape[4])

        if self.data_format == 'channels_first':
            #  switch back
            input_shape = input_shape[[0, 4, 1, 2, 3]]
            return tuple(input_shape[0], input_shape[1], width, height, depth)

    def call(self, x, mask=None):
        """Layer's logic

        https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
        https://stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images
        """

        if self.data_format == 'channels_last':
            b_size, x_size, y_size, z_size, c_size = x.shape.as_list()
        elif self.data_format == 'channels_first':
            b_size, c_size, x_size, y_size, z_size = x.shape.as_list()
        else:
            raise ValueError('Invalid data_format: ' + self.data_format)
        x_size_new, y_size_new, z_size_new = self.size

        if (x_size == x_size_new) and (y_size == y_size_new) and (
                z_size == z_size_new):
            # already in the target shape
            return x

        if self.data_format == 'channels_first':
            #  convert to channels_last
            x = tf.transpose(x, [0, 2, 3, 4, 1])

        # resize y-z
        squeeze_b_x = tf.reshape(x, [-1, y_size, z_size, c_size])
        resize_b_x = tf.image.resize_bilinear(
            squeeze_b_x, [y_size_new, z_size_new], align_corners=True
        )
        resume_b_x = tf.reshape(
            resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size]
        )

        # resize x
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size])
        resize_b_z = tf.image.resize_bilinear(
            squeeze_b_z, [y_size_new, x_size_new], align_corners=True
        )
        resume_b_z = tf.reshape(
            resize_b_z,
            [b_size, z_size_new, y_size_new, x_size_new, c_size]
        )
        output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])

        if self.data_format == 'channels_last':
            return output_tensor

        if self.data_format == 'channels_first':
            output_tensor = tf.transpose(output_tensor, [0, 4, 1, 2, 3])
            return output_tensor
