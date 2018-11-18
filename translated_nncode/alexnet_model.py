import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexBlock_7(nn.Module):
	def __init__(self, inp, outp, k_size, stride=1, padding):
		self.block = nn.Sequential(
			nn.Conv2d(inp, outp, kernel_size = k_size, stride = stride, padding=padding),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		out = self.block(x)
		return x

class AlexClassifier_7(nn.Module):
	def __init__(self, num_classes=10):
		self.classifier = nn.Sequential(
			nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.layer1_7 = AlexBlock(3, 64, 11, 4, 2)
		self.maxpool_7 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.layer2_7 = AlexBlock(64, 192, 5, padding=2)
		self.layer3_7 = AlexBlock(192, 384, 3, padding=1)
		self.layer4_7 = AlexBlock(384, 256, 3, padding=1)
		self.layer5_7 = AlexBlock(256, 256, 3, padding=1)
		self.classifier = AlexClassifier(10)

	def forward(self, x):
		out = self.layer1_7(x)
		out = self.maxpool_7(out)
		out = self.layer2_7(out)
		out = self.maxpool_7(out)
		out = self.layer3_7(out)
		out = self.layer4_7(out)
		out = self.layer5_7(out)
		out = self.maxpool_7(out)
		out = self.classifier_7(out)
		return out