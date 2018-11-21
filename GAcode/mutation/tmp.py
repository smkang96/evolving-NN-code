from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import sys
import math
import argparse
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from basic_blocks import *

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv1_5 = nn.Conv2d(3, 24, kernel_size=3, padding=1,bias=False)
        self.dense1_5 = make_dense_5(24,'+192', 12, 16, True)
        self.trans1_5 = Transition_5(216, 108)
        self.dense2_5 = make_dense_5(108,'+192',12,16,True)
        self.trans2_5 = Transition_5(300, 150)
        self.layerM_03 = ConvDW_mobilenet(150, 128, 1)
        self.dense3_5 = make_dense_5(128, '+192',12, 16, True)
        self.simple_linear_5 = dense_simple_linear_block_5(320,'+0')
        #forward 
    def forward(self, x):
        out = self.conv1_5(x)
        out = self.dense1_5(out)
        out = self.trans1_5(out)
        out = self.dense2_5(out)
        out = self.trans2_5(out)
        out = self.layerM_03(out)
        out = self.dense3_5(out)
        out = self.simple_linear_5(out)
        # return out 
        return out