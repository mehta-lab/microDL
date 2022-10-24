# %%
import collections
import sys
sys.path.insert(0, '/home/christian.foley/virtual_staining/microDL/')

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import itertools
import unittest
from micro_dl.torch_unet.utils.dataset import ToTensor

from micro_dl.torch_unet.utils.training import TorchTrainer

class TestTraining(unittest.TestCase):
    def SetUp(self):
        '''
        Set up configuration for testing TorchTrainer and module
        '''
        self.network_config = {
            'model': {
                'architecture': '2.5D',
                'in_channels': 1,
                'out_channels': 1,
                'residual': True,
                'task': 'reg',
                'model_dir': None},
            'training': {
                'epochs': 40,
                'learning_rate': 0.0045,
                'optimizer': 'adam',
                'loss': 'mse',
                'testing_stride': 1,
                'save_model_stride': 1,
                'save_dir': '',
                'mask': False,
                'mask_type': 'unimodal',
                'device': 0
                }
        }
        self.archs = ['2D', '2.5D']
        self.data_dims = [(16,1,512,512), (16,1,5,512,512)]
    
    def _random_dataloaders(self, size):
        """_Creates torch dataloaders which load from random normally
        distributed datasets of size 'size'

        :param int size: size of datasets to generate
        :return list dataloaders: list of random pytorch dataloaders for each arch
        """
        assert self.archs, 'Must run SetUp first'
        dataloaders = []
        
        for i in range(len(self.archs)):
            dim = self.data_dims[i]
            
            tensors = []
            for i in range(size):
                random = ToTensor()(np.random.randn(*dim))
                tensors.append(random.to(torch.device('cuda')))

            dataset = TensorDataset(*tensors)
            dataloaders.append(DataLoader(dataset))
        return dataloaders, dataset
    
    def _loss_evaluation(self):
        pass

    #-------------- Tests -----------------#
    
    def test_loss_closeness(self):
        """Test functionality of test_cycle loss versus training loss
        
        Fails if loss from randomly generated test dataset is too
        far from randomly generated train dataset
        """
        self._all_test_configurations(test = 'residual')

# %%
tester = TestTraining()
tester.SetUp()
output = tester._random_dataloaders(5)[0]
# %%
a = output[0]
len(a)
next(iter(a))[0].shape
# %%
for sample in a:
    print(len(sample))
    #print(sample.shape, sample.shape)
# %%
