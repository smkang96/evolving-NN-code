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
        self.layer1_0_0_0_0_0_0 = VGGBlock(1, 64, 2)
#forward
    def forward(self, x):
        out = self.layer1_0_0_0_0_0_0(x)
#return
