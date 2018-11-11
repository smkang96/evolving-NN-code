import torch.nn as nn
import math
import torch
from torch import nn
from torch.nn import Parameter
class BBBConv2d(nn.Module):
    def __init__(self,in_channels, out_chaneels,kernel_size, stride, padding, dilation =1, groups = 1):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_chaneels  = out_chaneels
        self.kernel_size = kernel_size
        self.stride  = stride,
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv_qw_mean = F.conv2d(input = self.input, weight )
        self.conv2dMean = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,stride = self.stride,padding = self.padding,dilation = self.dilation)
        self.conv2dSi = nn.Conv2d(self.in_channels, self.out_channels,self.kernel_size, stride = self.stride, padding= self.padding,diation = self.dilation)
    def forward(self, x):
        conv_mean = self.conv2dMean(x)
        conv_si = torch.sqrt(self.conv2dSi(x.pow(2)))
        out = conv_mean + conv_si * (torch.randn(conv_mean.size()))
        return out
class BaysBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel):
        super(BaysBlock,self).__init__()
        self.conv1 = BBBConv2d(inputs, outputs, 5, stride=1, padding=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    def forward(self,x):
        out = self.conv1(x)
        out = self.soft1(out)
        out = self.pool1(out)
        return out



class BBB3Conv(nn.Module):
    def __init__(self):
        super(BBB3Conv, self).__init__()
        self.block1 = BaysBlock(inputs,32,5)
        self.block2 = BaysBlock(32, 64,5)
        self.block3 = BaysBlock(64, 128, 5)


    def forward(self, x):
        out = self.block1(x)
        out = self.block2(x)
        out = self.block3(x)
        return out 