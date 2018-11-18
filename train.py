'''Training code to evaluate NNs'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
cudnn.benchmark = True
import torchvision
import torchvision.transforms as transforms

## Get Model Architecture /TODO: import from various sources of code
from translated_nncode.stupidnet import Net

import time

## Evaluation settings
allowed_train_time = 2 * 60 # in seconds


## Data Loaders
# from pytorch tutorial
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Network and Optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

## Training: from pytorch tutorial
train_start = time.time()
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d|%5d] loss: %.3f, time left: %.3fs' %
                  (epoch + 1, i + 1, running_loss / 10, train_start-time.time()+allowed_train_time))
            running_loss = 0.0
        
        if time.time() - train_start > allowed_train_time:
            break
    if time.time() - train_start > allowed_train_time:
        break

correct = 0
total = 0
start = time.time()
with torch.no_grad():
    for d_i, data in enumerate(testloader):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if d_i % 10 == 9:
            print('working...')
end = time.time()

accuracy = 100. * correct / total
time = end - start
        
print('Accuracy: %.3f %%' % (accuracy,))
print('Time: %.3f %%' % (time,))
            
print('Finished Training')