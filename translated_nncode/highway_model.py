import torch.nn as nn
import torch.nn.functional as F

class HighwayBlock(nn.Module):
	def __init__(self, size, f=F.softmax):
		super(Highway, self).__init__()


		self.nonlinear = nn.ModuleList([nn.Linear(size, size)])

		self.linear = nn.ModuleList([nn.Linear(size, size)])

		self.gate = nn.ModuleList([nn.Linear(size, size)])

		self.f = f

	def forward(self, x):
		
		gate = F.sigmoid(self.gate[layer](x))
		nonlinear = self.f(self.nonlinear[layer](x))
		linear = self.linear[layer](x)

		x = gate * nonlinear + (1 - gate) * linear

		return x

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.layer1 = HighwayBlock(64)
	def forward(self, x):
		out = self.layer1(x)
		out = self.layer1(x)
		out = self.layer1(x)
		out = self.layer1(x)
		out = self.layer1(x)
		out = self.layer1(x)
		out = self.layer1(x)
		out = self.layer1(x)
		out = self.layer1(x)
		out = self.layer1(x)
		return out
