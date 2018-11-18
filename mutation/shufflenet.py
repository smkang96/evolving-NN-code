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


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.basic_block_3 = shuffle_basic_block_3()
        self.layer1_3 = shuffle_layer_3(24,200, 4, 2)
        self.layer2_3 = shuffle_layer_3(200,400, 8, 2)
        self.layer3_3 = shuffle_layer_3(400, 800, 4, 2)
        #forward 
        self.linear_3 = nn.Linear(800, 10)
        self.avg_pool_2 = nn.AvgPool2d(4)

    def forward(self, x):
        out = self.basic_block_3(x)
        out = self.layer1_3(out)
        out = self.layer2_3(out)
        out = self.layer3_3(out)
        #return
        out = self.avg_pool_2(out)
        out = out.view(out.size(0), -1)
        out = self.linear_3(out)
        return out

