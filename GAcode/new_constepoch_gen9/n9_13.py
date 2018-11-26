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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layerM_01_0_0_0_0_0_0 = ConvBN_mobilenet(1, 32, 2)
        self.layerM_13_1 = ConvDW_mobilenet(32, 1024, 2)
        self.layer2_1_1_1_2 = VGGBlock(1024, 128, 2)
        self.simple_linear_5_2_2_2_3 = dense_simple_linear_block_5(128,'+0')
#forward
    def forward(self, x):
        out = self.layerM_01_0_0_0_0_0_0(x)
        out = self.layerM_13_1(out)
        out = self.layer2_1_1_1_2(out)
        out = self.simple_linear_5_2_2_2_3(out)
#return
