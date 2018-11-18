'''Training code to evaluate NNs'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
cudnn.benchmark = True
import torchvision
import torchvision.transforms as transforms

# from translated_nncode.stupidnet import Net

import imp

## Evaluation settings
allowed_train_time = 2 * 15 # in seconds

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

def indiv_evaluator(filename):
    import time # ok, really weird error
    module = imp.load_source('module', filename) # imports net structure

    ## Network and Optimizer
    net = module.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    ## Training: from pytorch tutorial
    print 'training...'
    train_start = time.time()
    for epoch in range(10):
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

            if time.time() - train_start > allowed_train_time:
                break
        if time.time() - train_start > allowed_train_time:
            break

    correct = 0
    total = 0
    print 'evaluating...'
    start = time.time()
    with torch.no_grad():
        for d_i, data in enumerate(testloader):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end = time.time()

    accuracy = 100. * correct / total
    time = end - start
    return accuracy, time

def evaluator(dirname, filenames):
    results = []
    for filename in filenames:
        results.append(indiv_evaluator(dirname + filename))
    return results