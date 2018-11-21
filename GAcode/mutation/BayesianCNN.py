import torch.nn as nn
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from basic_blocks import *



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = BayesBlock_4(3,32,5)
        self.block2 = BayesBlock_4(32, 64,5)
        self.block3 = BayesBlock_4(64, 128, 5)
        #forward

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        #return out


