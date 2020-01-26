from typing import Tuple
import os
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import numpy as np

import torchvision
import torchvision.transforms as transforms

def cifar10_dataloaders(datadir:str, train_num_workers=4, test_num_workers=4) \
        ->Tuple[DataLoader, DataLoader]:
    if 'pydevd' in sys.modules:
        train_num_workers = test_num_workers = 0
        print('Debugger detected: num_workers=0')

    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    aug_transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    norm_transf = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(aug_transf + norm_transf)
    test_transform = transforms.Compose(norm_transf)

    trainset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=train_num_workers)

    testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=test_transform)
    test_dl = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=test_num_workers)

    return train_dl, test_dl

# Training
def train_epoch(epoch, net, train_dl, device, criterion, optimizer)->float:
    correct, total = 0, 0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0*correct/total

def test(net, test_dl, device)->float:
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0*correct/total

def train(epochs, train_dl, net, device, criterion, optimizer)->None:
    for epoch in range(epochs):
        acc = train_epoch(epoch, net, train_dl, device, criterion, optimizer)
        print(f'Train Epoch={epoch}, Acc={acc}')

def main():
    seed, epochs, datadir = 42, 10, os.path.expanduser('~/torchvision_data_dir')
    cudnn.enabled = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    device = torch.device('cuda')

    net = torchvision.models.resnet18()
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    train_dl, test_dl = cifar10_dataloaders(datadir)

    train(epochs, train_dl, net, device, criterion, optimizer)
    acc = test(net, test_dl, device)

if __name__ == '__main__':
    main()