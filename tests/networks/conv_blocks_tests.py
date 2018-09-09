"""Tests for  conv_blocks"""
import keras.backend as K
from keras import Model
from keras import layers as k_layers
import nose.tools
import numpy as np
import unittest

from micro_dl.networks import conv_blocks as conv_blocks
from micro_dl.utils.aux_utils import get_channel_axis
from micro_dl.utils.network_utils import get_keras_layer


class TestConvBlocks(unittest.TestCase):

    def setUp(self):
        """Set up input shapes for both 2D and 3D and network_config"""

        self.ip_shape_2d = (1, 64, 64)
        self.ip_shape_3d = (1, 64, 64, 64)
        self.network_config = {'num_dims': 2,
                               'block_sequence': 'conv-bn-activation',
                               'num_filters_per_block': [16, 32],
                               'num_convs_per_block': 2,
                               'filter_size': 3,
                               'init': 'he_normal',
                               'padding': 'same',
                               'activation': {'type': 'relu'},
                               'batch_norm': True,
                               'dropout': 0.2,
                               'data_format': 'channels_first',
                               'num_input_channels': 1,
                               'residual': True,
                               'pooling_type': 'max'}

    def _test_op_shape(self, layers_list, exp_shapes):
        """Checks for the shape of output layers

        :param list layers_list: output of functional blocks of a NN stored as
         a list
        :param list exp_shapes: list of lists with the shape of each output
         layer
        """

        num_blocks = len(layers_list)
        assert len(layers_list) == len(exp_shapes), 'num of layers != ' \
                                                    'num of exp shapes'
        for block_idx in range(num_blocks):
            np.testing.assert_array_equal(
                layers_list[block_idx].get_shape().as_list()[1:],
                exp_shapes[block_idx]
            )

    def _test_num_trainable_params(self, weight_arrays):
        """Check for number of trainable weights

        :param list weight_arrays: list of np.arrays
        """

        nose.tools.assert_equal(
            len(weight_arrays),
            (len(self.network_config['num_filters_per_block']) *
             self.network_config['num_convs_per_block'] * 4)
        )

    def _test_weight_shape(self, weight_arrays):
        """Check for shape of filters + bias and BN beta and gamma

        :param list weight_arrays: list of np.arrays
        """

        num_filters_per_block = \
            self.network_config['num_filters_per_block'].copy()
        expected_shapes = []
        num_filters_per_block.insert(
            0, self.network_config['num_input_channels']
        )

        for block_idx, block_num_filters in \
                enumerate(self.network_config['num_filters_per_block']):
            for conv_idx in \
                    range(self.network_config['num_convs_per_block']):
                if self.network_config['num_dims'] == 2:
                    # filter shape
                    expected_shapes.append(
                        (self.network_config['filter_size'],
                         self.network_config['filter_size'],
                         num_filters_per_block[conv_idx + block_idx],
                         block_num_filters)
                    )
                else:  # 3D
                    expected_shapes.append(
                        (self.network_config['filter_size'],
                         self.network_config['filter_size'],
                         self.network_config['filter_size'],
                         num_filters_per_block[conv_idx + block_idx],
                         block_num_filters)
                    )
                # filter bias
                expected_shapes.append((block_num_filters,))
                # bn_gamma
                expected_shapes.append((block_num_filters,))
                # bn_beta
                expected_shapes.append((block_num_filters,))

        for idx, weight in enumerate(weight_arrays):
            nose.tools.assert_equal(weight.shape, expected_shapes[idx])

    def _create_model(self, ip_shape, block_function):
        """Create a model with the functional blocks

        :param tuple ip_shape: as named
        :param function block_function: function from conv_blocks
        :return:
         op_layers: list of keras layers, output of the blocks being tested
         weight_arrays: list of np.arrays with trainable weights of the model
        """

        ip_layer = k_layers.Input(shape=ip_shape, dtype='float32')
        op_layer1 = block_function(ip_layer, self.network_config, 0)
        op_layer2 = block_function(op_layer1, self.network_config, 1)
        op_layers = [op_layer1, op_layer2]

        model = Model(ip_layer, op_layer2)
        weight_tensors = model.trainable_weights
        sess = K.get_session()
        weight_arrays = sess.run(weight_tensors)
        return op_layers, weight_arrays

    def test_conv_block(self):
        """Test conv_block()"""

        for idx, ip_shape in enumerate([self.ip_shape_2d, self.ip_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            num_filters_per_block = \
                self.network_config['num_filters_per_block'].copy()

            exp_op_shapes = []
            for block_idx in range(len(num_filters_per_block)):
                cur_op_shape = list(ip_shape)
                cur_op_shape[0] = num_filters_per_block[block_idx]
                exp_op_shapes.append(cur_op_shape)

            op_layers, weight_arrays = self._create_model(
                ip_shape, conv_blocks.conv_block
            )

            # test for op layer shape
            self._test_op_shape(layers_list=op_layers,
                                exp_shapes=exp_op_shapes)

            # test for num of trainable weights
            self._test_num_trainable_params(weight_arrays)

            # test for shape of trainable weights
            self._test_weight_shape(weight_arrays)

    def test_downsample_conv_block(self):
        """Test downsample_conv_block()"""

        for idx, ip_shape in enumerate([self.ip_shape_2d, self.ip_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            num_filters_per_block = \
                self.network_config['num_filters_per_block'].copy()

            exp_op_shapes = []
            for block_idx in range(len(num_filters_per_block)):
                cur_op_shape = np.array(ip_shape)
                cur_op_shape[0] = num_filters_per_block[block_idx]
                if block_idx > 0:
                    cur_op_shape[1:] = cur_op_shape[1:] / 2
                exp_op_shapes.append(cur_op_shape)

            op_layers, weight_arrays = self._create_model(
                ip_shape, conv_blocks.downsample_conv_block
            )

            # test for op layer shape
            self._test_op_shape(layers_list=op_layers,
                                exp_shapes=exp_op_shapes)

            # test for num of trainable weights
            self._test_num_trainable_params(weight_arrays)

            # test for shape of trainable weights
            self._test_weight_shape(weight_arrays)

    def test_pad_channels(self):
        """Test pad_channels()"""

        for idx, ip_shape in enumerate([self.ip_shape_2d, self.ip_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            ip_layer = k_layers.Input(shape=ip_shape, dtype='float32')
            conv_layer = get_keras_layer('conv',
                                         self.network_config['num_dims'])
            op_layer = conv_layer(
                filters=self.network_config['num_filters_per_block'][0],
                kernel_size=self.network_config['filter_size'],
                padding='same',
                data_format=self.network_config['data_format']
            )(ip_layer)

            channel_axis = get_channel_axis(self.network_config['data_format'])
            layer_padded = k_layers.Lambda(
                conv_blocks.pad_channels,
                arguments={'final_layer': op_layer,
                           'channel_axis': channel_axis})(ip_layer)
            model = Model(ip_layer, layer_padded)
            test_shape = list(ip_shape)
            test_shape.insert(0, 1)
            test_image = np.ones(shape=test_shape)
            sess = K.get_session()
            # forward pass
            op = model.predict(test_image, batch_size=1)
            # test shape
            ip_shape = list(ip_shape)
            ip_shape[0] = self.network_config['num_filters_per_block'][0]
            np.testing.assert_array_equal(op_layer.get_shape().as_list()[1:],
                                          ip_shape)
            op = np.squeeze(op)
            # only slice 8 is not zero
            nose.tools.assert_equal(np.sum(op), np.sum(op[8]))
            nose.tools.assert_equal(np.sum(op[8]),
                                    64 ** self.network_config['num_dims'])

    def test_merge_residual(self):
        """Test _merge_residual()"""

        for ip_shape in [(1, 16, 16), (24, 16, 16)]:
            ip_layer = k_layers.Input(shape=ip_shape, dtype='float32')
            op_layer = k_layers.Conv2D(
                filters=self.network_config['num_filters_per_block'][0],
                kernel_size=self.network_config['filter_size'],
                kernel_initializer='Ones',
                padding='same',
                data_format=self.network_config['data_format']
            )(ip_layer)
            res_layer = conv_blocks._merge_residual(
                final_layer=op_layer,
                input_layer=ip_layer,
                data_format=self.network_config['data_format'],
                num_dims=2,
                kernel_init='Ones'
            )
            model = Model(ip_layer, res_layer)
            test_shape = list(ip_shape)
            test_shape.insert(0, 1)
            test_image = np.ones(shape=test_shape)
            sess = K.get_session()
            # forward pass
            op = model.predict(test_image, batch_size=1)
            if ip_shape[0] == 1:
                # a convolution with ones, unique values 9, 6(edges),
                # 4(corners). only the center slice is res added with input
                np.testing.assert_array_equal(op[:, 8, :, :],
                                              op[:, 7, :, :] + test_image[0])
            if ip_shape[0] == 24:
                # input -> 1x1 to match the num of layers. res_layer must be >
                # input center slices
                np.testing.assert_array_less(test_image[:, 4:20, :, :], op)

    def test_residual_conv_block(self):
        """Test residual_conv_block()

        Adding a residual connection doesn't increase the number of trainable
        params.
        """

        for idx, ip_shape in enumerate([self.ip_shape_2d, self.ip_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            num_filters_per_block = \
                self.network_config['num_filters_per_block'].copy()

            exp_op_shapes = []
            for block_idx in range(len(num_filters_per_block)):
                cur_op_shape = list(ip_shape)
                cur_op_shape[0] = num_filters_per_block[block_idx]
                exp_op_shapes.append(cur_op_shape)

            op_layers_res, weight_arrays_res = self._create_model(
                ip_shape, conv_blocks.residual_conv_block
            )

            # test for op layer shape
            self._test_op_shape(layers_list=op_layers_res,
                                exp_shapes=exp_op_shapes)

            # test for num of trainable weights
            self._test_num_trainable_params(weight_arrays_res)

            # test for shape of trainable weights
            self._test_weight_shape(weight_arrays_res)

    def test_residual_downsample_conv_block(self):
        """Test residual_downsample_conv_block()"""

        for idx, ip_shape in enumerate([self.ip_shape_2d, self.ip_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            num_filters_per_block = \
                self.network_config['num_filters_per_block'].copy()

            exp_op_shapes = []
            for block_idx in range(len(num_filters_per_block)):
                cur_op_shape = np.array(ip_shape)
                cur_op_shape[0] = num_filters_per_block[block_idx]
                if block_idx > 0:
                    cur_op_shape[1:] = cur_op_shape[1:] / 2
                exp_op_shapes.append(cur_op_shape)

            op_layers, weight_arrays = self._create_model(
                ip_shape, conv_blocks.residual_downsample_conv_block
            )

            # test for op layer shape
            self._test_op_shape(layers_list=op_layers,
                                exp_shapes=exp_op_shapes)

            # test for num of trainable weights
            self._test_num_trainable_params(weight_arrays)

            # test for shape of trainable weights
            self._test_weight_shape(weight_arrays)
