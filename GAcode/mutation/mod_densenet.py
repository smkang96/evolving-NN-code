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
        self.conv1_5 = nn.Conv2d(3, 24, kernel_size=3, padding=1,bias=False)
        self.dense1_5 = make_dense_5(24,'+192', 12, 16, True)
        self.trans1_5 = Transition_5(216, 108)
        self.dense2_5 = make_dense_5(108,'+192',12,16,True)
        self.trans2_5 = Transition_5(300, 150)
        self.dense3_5 = make_dense_5(150, '+192',12, 16, True)
        self.simple_linear_5 = dense_simple_linear_block_5(342,'+0')
        #forward 
        self.avgpool_end = nn.AvgPool2d(8, stride=1)
        self.linear_end = nn.Linear(342,10)
    def forward(self, x):
        out = self.conv1_5(x)
        out = self.dense1_5(out)
        out = self.trans1_5(out)
        out = self.dense2_5(out)
        out = self.trans2_5(out)
        out = self.dense3_5(out)
        out = self.simple_linear_5(out)
        #return
        out = self.avgpool_end(out)
        out = out.view(out.size(0), -1)
        out = self.linear_end(out)
        return out
