'''GoogLeNet with PyTorch.'''
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
        self.pre_layers_2 = google_basic_block_2(1, 192)
        self.a3_2 = Inception_2(192, 256, 64,  96, 128, 16, 32, 32)
        self.b3_2 = Inception_2(256, 480,128, 128, 192, 32, 96, 64)
        self.maxpool_2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4_2 = Inception_2(480, 512,192,  96, 208, 16,  48,  64)
        self.b4_2 = Inception_2(512, 512,160, 112, 224, 24,  64,  64)
        self.c4_2 = Inception_2(512, 512,128, 128, 256, 24,  64,  64)
        self.d4_2 = Inception_2(512, 528,112, 144, 288, 32,  64,  64)
        self.e4_2 = Inception_2(528, 832,256, 160, 320, 32, 128, 128)
        self.a5_2 = Inception_2(832, 832,256, 160, 320, 32, 128, 128)
        self.b5_2 = Inception_2(832, 1024,384, 192, 384, 48, 128, 128)
        #forward
        self.avgpool_end = nn.AvgPool2d(7, stride=1)
        self.linear_end = nn.Linear(1024,10)
    def forward(self, x):
        out = self.pre_layers_2(x)
        out = self.a3_2(out)
        out = self.b3_2(out)
        out = self.maxpool_2(out)
        out = self.a4_2(out)
        out = self.b4_2(out)
        out = self.c4_2(out)
        out = self.d4_2(out)
        out = self.e4_2(out)
        out = self.maxpool_2(out)
        out = self.a5_2(out)
        out = self.b5_2(out)
        #return 



        out = self.avgpool_end(out)
        out = out.view(out.size(0), -1)
        out = self.linear_end(out)
        return out
