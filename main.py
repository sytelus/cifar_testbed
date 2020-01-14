from typing import Optional, Tuple
import os
import sys
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import numpy as np

import torchvision
import torchvision.transforms as transforms

from timing import MeasureTime, print_all_timings

@MeasureTime
def cifar10_dataloaders(datadir:str, train_num_workers=4, test_num_workers=4) \
        ->Tuple[DataLoader, DataLoader]:
    if 'pydevd' in sys.modules:
        train_num_workers = test_num_workers = 0
        logging.info('debugger=true, num_workers=0')

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
@MeasureTime
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

@MeasureTime
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

@MeasureTime
def train(epochs, train_dl, net, device, criterion, optimizer)->None:
    for epoch in range(epochs):
        acc = train_epoch(epoch, net, train_dl, device, criterion, optimizer)
        logging.info(f'train_epoch={epoch}, prec1={acc}')


def param_size(model:torch.nn.Module)->int:
    """count all parameters excluding auxiliary"""
    return sum(v.numel() for name, v in model.named_parameters() \
        if "auxiliary" not in name)

def full_path(path:str)->str:
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return os.path.abspath(path)

def setup_logging(filepath:Optional[str]=None, name:Optional[str]=None, level=logging.DEBUG) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False # otherwise root logger prints things again

    if filepath:
        fh = logging.FileHandler(filename=full_path(filepath))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def setup_cuda(seed):
    # setup cuda
    cudnn.enabled = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

@MeasureTime
def train_test(exp_name:str, exp_desc:str, epochs:int, model_name:str, seed:int)->float:
    # config
    lr, momentum, weight_decay = 0.1, 0.0, 0.0

    # dirs
    datadir = full_path('~/torchvision_data_dir')
    expdir = full_path(os.path.join('.', 'results', exp_name))
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(expdir, exist_ok=True)
    setup_logging(filepath=os.path.join(expdir, 'logs.log'))

    # log config for reference
    logging.info(f'exp_name={exp_name}, exp_desc={exp_desc}')
    logging.info(f'model_name={model_name}, seed={seed}, epochs={epochs}')
    logging.info(f'lr={lr}, momentum={momentum}, weight_decay={weight_decay}')
    logging.info(f'datadir={datadir}')
    logging.info(f'expdir={expdir}')

    setup_cuda(seed)

    device = torch.device('cuda')

    model_class = getattr(torchvision.models, model_name)
    net = model_class()
    logging.info(f'param_size_m={param_size(net)/1E6:.1f}')
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    train_dl, test_dl = cifar10_dataloaders(datadir)

    train(epochs, train_dl, net, device, criterion, optimizer)
    return test(net, test_dl, device)

def main():
    parser = argparse.ArgumentParser(description='Pytorch cifasr testbed')
    parser.add_argument('--experiment-name', '-n', default='throwaway')
    parser.add_argument('--experiment-description', '-d', default='throwaway')
    parser.add_argument('--epochs', '-e', type=int, default=35)
    parser.add_argument('--model-name', '-m', default='resnet34')
    parser.add_argument('--seed', '-s', type=int, default=42)

    args = parser.parse_args()

    acc = train_test(args.experiment_name, args.experiment_description,
                     args.epochs, args.model_name, args.seed)
    print_all_timings()
    logging.info(f'test_accuracy={acc}')

if __name__ == '__main__':
    main()