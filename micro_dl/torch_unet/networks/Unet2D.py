# Last edit: Christian Foley, 08/26/2022

import torch
import torch.nn as nn
from layers.ConvBlock import *

class Unet2d(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, out_xy = (256,256), residual = False, down = 'avgpool', up = 'bilinear',
                 activation = 'relu', num_blocks = 4, task = 'seg', num_filters = []):
        '''
        Instance of 2D Unet. Implemented with variable input/output channels and depth (block numbers).
        Follows 2D UNet Architecture: 
            1) Unet: https://arxiv.org/pdf/1505.04597.pdf
            2) residual Unet: https://arxiv.org/pdf/1711.10684.pdf
        
        TODO: parameter documentation
        '''
        
        super(Unet2d, self).__init__()
        self.num_blocks = num_blocks
        self.residual = residual
        self.task = task
        self.out_xy = out_xy
        
        #----- Standardize Filter Sequence -----#
        if len(num_filters) != 0:
            assert len(num_filters) == num_blocks, 'Length of num_filters must be equal to num_blocks + 1 (number of convolutional blocks per path).'
            self.num_filters = num_filters
        else:
            self.num_filters = [pow(2,i)*16 for i in range(num_blocks + 1)]

            
        #----- Downsampling steps -----#
        self.down_list = []
        if down == 'maxpool':
            for i in range(num_blocks):
                self.down_list.append(nn.MaxPool2d(kernel_size=2))
        elif down == 'avgpool':
            for i in range(num_blocks):
                self.down_list.append(nn.AvgPool2d(kernel_size=2))
        elif down == 'conv':
            raise NotImplementedError('Not yet implemented!')
            #TODO: implement.
        
        #----- Upsampling steps -----#
        self.up_list = []
        if up == 'bilinear':
            for i in range(num_blocks):
                self.up_list.append(lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2))
        elif up == 'conv':
            raise NotImplementedError('Not yet implemented!')
            #TODO: implement
        elif up == 'tconv':
            raise NotImplementedError('Not yet implemented!')
            #TODO: implement
        
        #----- Convolutional blocks -----# Forward Filters [16, 32, 64, 128, 256] -> Backward Filters [128+256, 64+128, 32+64, 16+32, 1]
        self.down_conv_blocks = []
        forward_filters = [in_channels] + self.num_filters
        for i in range(num_blocks):
            self.down_conv_blocks.append(ConvBlock(forward_filters[i], forward_filters[i+1], residual = self.residual, activation = activation))
        
        self.bottom_transition_block = ConvBlock(self.num_filters[-2], self.num_filters[-1])

        self.up_conv_blocks = []
        backward_filters = [self.num_filters[-(i+1)] + self.num_filters[-(i+2)] for i in range(len(self.num_filters)) if i < len(self.num_filters) - 1] + [out_channels]
        for i in range(num_blocks):
            self.up_conv_blocks.append(ConvBlock(backward_filters[i], forward_filters[-(i+2)], residual = self.residual, activation = activation))
            
        #----- Network-level residual-----#
        if self.residual:
            self.conv_resid = ConvBlock(self.num_filters[0], out_channels, residual = self.residual, activation = activation, num_layers = 1)
        else:
            self.conv_resid = ConvBlock(self.num_filters[0], out_channels, residual = self.residual, activation = activation, num_layers = 1)
        
        #----- Terminal Block and Activation Layer -----#
        if self.task == 'reg':
            self.terminal_block = ConvBlock(forward_filters[1], out_channels, residual = self.residual, activation = 'linear', num_layers = 1)
            self.linear_activation = nn.Linear(*self.out_xy)
        else:
            self.terminal_block = ConvBlock(forward_filters[1], out_channels, residual = self.residual, activation = activation, num_layers = 1)
        
    def forward(self, x):
        '''
        Forward call of network
            - x -> Torch.tensor: input image stack
            
        Call order:
            => num_block 2D convolutional blocks, with downsampling in between (encoder)
            => num_block 2D convolutional blocks, with upsampling between them (decoder)
            => skip connections between corresponding blocks on encoder and decoder paths
            => terminal block collapses to output dimensions

        '''
        #encoder
        skip_tensors = []
        for i in range(self.num_blocks):
            x = self.down_conv_blocks[i](x)
            skip_tensors.append(x)
            x = self.down_list[i](x)
        
        #transition block
        x = self.bottom_transition_block(x)
        
        #decoder
        for i in range(self.num_blocks):
            x = self.up_list[i](x)
            x = torch.cat([x, skip_tensors[-1*(i+1)]], 1)
            x = self.up_conv_blocks[i](x)
        
        # output channel collapsing layer
        x = self.terminal_block(x)
        if self.task == 'reg':
            x = self.linear_activation(x)
        
        return x
    
    def model(self):
        '''
        Allows calling of parameters inside ConvBlock object: 'model.model().parameters()'

        Sequential order:
            => num_block 2D convolutional blocks, with downsampling in between (encoder)
            => num_block 2D convolutional blocks, with upsampling between them (decoder)
            => skip connections between corresponding blocks on encoder and decoder paths
            => terminal block compresses to output dimensions
        
        We can make a list of layer modules and unpack them into nn.Sequential.
            Note: this is distinct from the forward call because we want to use the forward call with addition, since this
                  is a residual network. The forward call performs the resid calculation, and all the parameters can be seen
                  by the optimizer when given this model.        
        '''

        
        def get_block_layers(conv_block):
            block_layers = conv_block.model().children()
            return list(block_layers)
        
        layers = []

        #encoder blocks
        for i in range(self.num_blocks):
            layers.extend(get_block_layers(self.down_conv_blocks[i]))
            layers.append(self.down_list[i])
            
        #transition block
        layers.extend(get_block_layers(self.bottom_transition_block))
        
        #decoder blocks
        for i in range(self.num_blocks):
            #layers.append(self.up_list[i]) <-- We don't include the upsampling because it is a functional interpolation, not a module
            layers.extend(get_block_layers(self.up_conv_blocks[i]))
        
        #terminal block & activation
        layers.extend(get_block_layers(self.terminal_block))
        
        if self.task == 'reg':
            layers.append(self.linear_activation)
        
        
        return nn.Sequential(*layers)