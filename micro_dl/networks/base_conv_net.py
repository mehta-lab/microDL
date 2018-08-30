"""Base class for all networks"""
from abc import ABCMeta, abstractmethod

from micro_dl.utils.aux_utils import validate_config


class BaseConvNet(metaclass=ABCMeta):
    """Base class for all networks"""

    @abstractmethod
    def __init__(self, network_config):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        req_params = ['batch_norm', 'pooling_type', 'height', 'width',
                      'data_format', 'num_input_channels', 'final_activation']
        param_check, msg = validate_config(network_config, req_params)

        if not param_check:
            raise ValueError(msg)
        self.config = network_config

        assert network_config['height'] == network_config['width'], \
            'The network expects a square image'

        assert network_config['data_format'] in ['channels_first',
                                                 'channels_last'], \
            'invalid data format. Not in [channels_first or channels_last]'

        # fill in default values
        if 'filter_size' not in network_config:
            network_config['filter_size'] = 3
        if 'activation' not in network_config:
            network_config['activation']['type'] = 'relu'
        if 'padding' not in network_config:
            network_config['padding'] = 'same'
        if 'init' not in network_config:
            network_config['init'] = 'he_normal'
        if 'dropout' not in network_config:
            network_config['dropout'] = 0.0

        dropout_prob = network_config['dropout']
        assert 0.0 <= dropout_prob < 0.7, 'invalid dropout probability'
        self.dropout_prob = dropout_prob

    @abstractmethod
    def build_net(self):
        """Assemble/build the network from layers"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _get_input_shape(self):
        """Return the shape of the input"""
        raise NotImplementedError
