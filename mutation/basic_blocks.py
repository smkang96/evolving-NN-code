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


class Bottleneck_5(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck_5, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer_5(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer_5, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition_5(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition_5, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class _make_dense_5(nn.Module):
    def __init__(self, nChannels,output_size,  growthRate, nDenseBlocks, bottleneck):
        super(_make_dense_5,self).__init__()
        self.nChannels = nChannels
        self.growthRate = growthRate
        self.nDenseBlocks = nDenseBlocks
        self.bottleneck = bottleneck

        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck_5(nChannels, growthRate))
            else:
                layers.append(SingleLayer_5(nChannels, growthRate))
            nChannels += growthRate
        self.layers = nn.Sequential(*layers)
    def forward(self,x):
        out = self.layers(x)
        return out


class dense_simple_linear_block_5(nn.Module):
    def __init__(self,input, output):
        super(dense_simple_linear_block_5,self).__init__()
        self.bn1 = nn.BatchNorm2d(input)
    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        return out




class Inception_2(nn.Module):
    def __init__(self, in_planes,out_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception_2, self).__init__()
        # 1x1 conv branch
        self.b1_2 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1),nn.BatchNorm2d(n1x1),nn.ReLU(True),)
        # 1x1 conv -> 3x3 conv branch
        self.b2_2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1),nn.BatchNorm2d(n3x3red),nn.ReLU(True),nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),nn.BatchNorm2d(n3x3),nn.ReLU(True),)
        # 1x1 conv -> 5x5 conv branch
        self.b3_2 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1),nn.BatchNorm2d(n5x5red),nn.ReLU(True),nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),nn.BatchNorm2d(n5x5),nn.ReLU(True),nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),nn.BatchNorm2d(n5x5),nn.ReLU(True),)
        # 3x3 pool -> 1x1 conv branch
        self.b4_2 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),nn.Conv2d(in_planes, pool_planes, kernel_size=1),nn.BatchNorm2d(pool_planes),nn.ReLU(True),)

    def forward(self, x):
        y1 = self.b1_2(x)
        y2 = self.b2_2(x)
        y3 = self.b3_2(x)
        y4 = self.b4_2(x)
        return torch.cat([y1,y2,y3,y4], 1)

class google_basic_block_2(nn.Module):
    def __init__(self, input, output):
        super(google_basic_block_2,self).__init__()
        self.conv  = nn.Conv2d(input, output,kernel_size = 3, padding = 1)
        self.batch = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(True)
    def forward(self,x):
        out = self.conv(x)
        out = self.batch(out)
        out = self.relu(out)
        return out



class BBBConv2d_4(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size, stride, padding, dilation =1, groups = 1):
        super(BBBConv2d_4, self).__init__()
        self.in_channels = in_channels
        self.out_channels  = out_channels
        self.kernel_size = kernel_size
        self.stride  = stride,
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv2dMean = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,stride = self.stride,padding = self.padding)
        self.conv2dSi = nn.Conv2d(self.in_channels, self.out_channels,self.kernel_size, stride = self.stride, padding= self.padding)
    def forward(self, x):
        conv_mean = self.conv2dMean(x)
        conv_si = self.conv2dSi(x)
        out = conv_mean + conv_si * torch.randn(conv_mean.size()).to('cuda')
        return out

class BayesBlock_4(nn.Module):
    def __init__(self, inputs, outputs, kernel):
        super(BayesBlock_4,self).__init__()
        self.conv1 = BBBConv2d(inputs, outputs, 5, stride=1, padding=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self,x):
        out = self.conv1(x)
        out = self.soft1(out)
        out = self.pool1(out)
        return out

class SepConv_6(nn.Module):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(SepConv_6, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size, stride,
                               padding=(kernel_size-1)//2,
                               bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA_6(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(CellA_6, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv_6(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y2 = self.bn1(self.conv1(y2))
        return F.relu(y1+y2)


class PNAS_make_layer_6(nn.Module):
    def __init__(self, inputsize, planes, num_cells):
        super(PNAS_make_layer_6, self).__init__()
        layers = []
        self.in_planes = inputsize
        for _ in range(num_cells):
            layers.append(CellA_6(self.in_planes, planes, stride=1))
            self.in_planes = planes
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        out = self.layers(x)
        return out

class PNAS_downsample_6(nn.Module):
    def __init__(self, inputsize, planes):
        super(PNAS_downsample_6,self).__init__()
        layer = CellA_6(inputsize , planes, stride=2)
        self.layer = layer 
    def forward(self, x):
        out = self.layer(x)
        return out
class PNAS_basic_block_6(nn.Module):
    def __init__(self,inputsize, outputsize):
        super (PNAS_basic_block_6,self).__init__()
        self.conv1 = nn.Conv2d(inputsize, outputsize, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputsize)
    def forward(self, x):
        out = out = F.relu(self.bn1(self.conv1(x)))
        return out

class ShuffleBlock_3(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock_3, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class Bottleneck_3(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck_3, self).__init__()
        self.stride = stride

        mid_planes = out_planes//4
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock_3(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out

class shuffle_layer_3(nn.Module):
    def __init__(self,in_planes, out_planes, num_blocks, groups):
        super(shuffle_layer_3,self).__init__()
        layers = []
        self.in_planes = in_planes
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck_3(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        out = self.layers(x)
        return out 
class shuffle_basic_block_3(nn.Module):
    def __init__(self):
        super(shuffle_basic_block_3, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out




