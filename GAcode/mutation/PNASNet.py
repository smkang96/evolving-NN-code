from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from basic_blocks import *



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.basic_block_6 = PNAS_basic_block_6(3,44)
        self.layer1_6 = PNAS_make_layer_6(44, 44, num_cells=6)
        self.layer2_6 = PNAS_downsample_6(44, 88)
        self.layer3_6 = PNAS_make_layer_6(88,88, num_cells=6)
        self.layer4_6 = PNAS_downsample_6(88,176)
        self.layer5_6 = PNAS_make_layer_6(176,176, num_cells=6)
        #forward


    def forward(self, x):
        out = self.basic_block_6(x)
        out = self.layer1_6(out)
        out = self.layer2_6(out)
        out = self.layer3_6(out)
        out = self.layer4_6(out)
        out = self.layer5_6(out)
        #return


