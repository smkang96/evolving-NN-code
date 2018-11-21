'''translation of NN code to samplable format: ACGAN'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvDrop_ACGAN(3, 16, 2, 1, bn=False)
        self.conv2 = ConvDrop_ACGAN(16, 32, 1, 0)
        self.conv3 = ConvDrop_ACGAN(32, 64, 2, 1)
        self.conv4 = ConvDrop_ACGAN(64, 128, 1, 0)
        self.conv5 = ConvDrop_ACGAN(128, 256, 2, 1)
        self.conv6 = ConvDrop_ACGAN(256, 512, 1, 0)
        # forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        #return
