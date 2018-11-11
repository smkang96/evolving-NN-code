'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_2(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1_2 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2_2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3_2 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4_2 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1_2(x)
        y2 = self.b2_2(x)
        y3 = self.b3_2(x)
        y4 = self.b4_2(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers_2 = nn.Sequential(nn.Conv2d(3, 192, kernel_size=3, padding=1),nn.BatchNorm2d(192),nn.ReLU(True))
        self.a3_2 = Inception_2(192,  64,  96, 128, 16, 32, 32)
        self.b3_2 = Inception_2(256, 128, 128, 192, 32, 96, 64)
        self.maxpool_2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4_2 = Inception_2(480, 192,  96, 208, 16,  48,  64)
        self.b4_2 = Inception_2(512, 160, 112, 224, 24,  64,  64)
        self.c4_2 = Inception_2(512, 128, 128, 256, 24,  64,  64)
        self.d4_2 = Inception_2(512, 112, 144, 288, 32,  64,  64)
        self.e4_2 = Inception_2(528, 256, 160, 320, 32, 128, 128)
        self.a5_2 = Inception_2(832, 256, 160, 320, 32, 128, 128)
        self.b5_2 = Inception_2(832, 384, 192, 384, 48, 128, 128)
        self.avgpool_2 = nn.AvgPool2d(8, stride=1)
        self.linear_2 = nn.Linear(1024, 10)

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
        out = self.avgpool_2(out)
        #out = out.view(out.size(0), -1)
        out = self.linear_2(out)
        return out



# test()