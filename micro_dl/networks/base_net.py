"""Base class for all networks"""
from abc import ABCMeta, abstractmethod


class BaseNet(metaclass=ABCMeta):
    """Base class for all networks"""

    @abstractmethod
    def __init__(self, network_config):
        """Init

        :param yaml network_config: yaml with all network associated parameters
        """
        raise NotImplementedError

    @abstractmethod
    def build_net(self):
        """Assemble/build the network from layers"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _get_input_shape(self):
        """Return the shape of the input"""
        raise NotImplementedError
