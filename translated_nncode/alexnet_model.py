import torch
import torch.nn as nn
import torch.nn.functional as F

class Alexlayer(nn.Module):

	def __init__(self, num_classes=1000):
		self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        #nn.ReLU(inplace=True),
        #nn.MaxPool2d(kernel_size=3, stride=2),
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        #nn.ReLU(inplace=True),
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        #nn.ReLU(inplace=True),
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #nn.ReLU(inplace=True),
        #nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout()
        self.linear1 = self.Linear(256 * 6 * 6, 4096)
        #nn.ReLU
        #nn.dropout
        self.linear2 = self.linear(4096, 4096)
        #nn.Relu
        self.linear3 = self.linear(4096, num_classes)

    def forward(self, x):
    	x = self.conv1(x)
    	out = self.relu(out)
    	out = self.maxpool(out)
    	out = self.conv2(out)
    	out = self.relu(out)
    	out = self.maxpool(out)
    	out = self.conv3(out)
    	out = self.relu(out)
    	out = self.conv4(out)
    	out = self.relu(out)
    	out = self.conv5(out)
    	out = self.relu(out)
    	out = self.maxpool(out)
    	out = self.dropou(out)
    	out = self.linear(out)
    	out = self.relu(out)
    	out = self.dropout(out)
    	out = self.linear2(out)
    	out = self.relu(out)
    	out = self.linear3(out)
    	return out

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.layer = Alexlayer()

	def forward(self, x):
		out = self.layer(x)
		return out