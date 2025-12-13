# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
#
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, batch_norm=False, instance_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, 
                                     output_padding=output_padding))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))   
    layers.append(nn.ReLU(True))     
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=False, instance_norm=True, init_zero_weights=False, leaky=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if not leaky:
        layers.append(nn.ReLU(True))
    else:
        layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        conv_layer = [nn.ReflectionPad2d(1)]
        conv_layer += [conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=0)]
        conv_layer += [nn.Dropout(0.5)]
        conv_layer += [nn.ReflectionPad2d(1)]
        conv_layer += [conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=0)]
        self.conv_layer = nn.Sequential(*conv_layer)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False):
        # super(CycleGenerator, self).__init__()
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        # c7s1-64,d128,d256,
        # R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
        # 1. Define the encoder part of the generator (that extracts features from the input image)
        model = []
        model  += [nn.ReflectionPad2d(3)]
        model += [conv(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0)]
        model += [conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)]
        model += [conv(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)]
        
        # 2. Define the transformation part of the generator
        for i in range(6):
            model.append(ResnetBlock(256))

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        model += [deconv(in_channels=256, out_channels=128, kernel_size=3)]
        model += [deconv(in_channels=128, out_channels=64, kernel_size=3)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 128 x 128

            Output
            ------
                out: BS x 3 x 128 x 128
        """
        return self.model(x)


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64):##delete parameter
        # super(DCDiscriminator, self).__init__()
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        # C64-C128-C256-C512
        model = []
        model += [conv(in_channels=3, out_channels=64, kernel_size=4, stride=2, instance_norm=False, leaky=True)]
        model += [conv(in_channels=64, out_channels=128, kernel_size=4, stride=2, instance_norm=True, leaky=True)]
        model += [conv(in_channels=128, out_channels=256, kernel_size=4, stride=2, instance_norm=True, leaky=True)]
        model += [conv(in_channels=256, out_channels=512, kernel_size=4, stride=1, instance_norm=True, leaky=True)]

        model += [nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

