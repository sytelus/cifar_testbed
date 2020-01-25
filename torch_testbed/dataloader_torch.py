import logging
from typing import List, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from . import utils
from .timing import MeasureTime, print_all_timings, print_timing, get_timing
from .cutout import CutoutDefault


def cifar10_transform(aug:bool, cutout=0):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]

    transf = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    if aug:
        aug_transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        transf = aug_transf + transf

    if cutout > 0: # must be after normalization
        transf += [CutoutDefault(cutout)]

    return transforms.Compose(transf)

@MeasureTime
def cifar10_dataloaders(datadir:str, train_batch_size=128, test_batch_size=4096,
                    cutout=0, train_num_workers=4, test_num_workers=4)\
                        ->Tuple[DataLoader, DataLoader]:
    if utils.is_debugging():
        train_num_workers = test_num_workers = 0
        logging.info('debugger=true, num_workers=0')

    train_transform = cifar10_transform(aug=True, cutout=cutout)
    trainset = torchvision.datasets.CIFAR10(root=datadir, train=True,
        download=True, transform=train_transform)
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
        shuffle=True, num_workers=train_num_workers, pin_memory=True)

    test_transform = cifar10_transform(aug=False, cutout=0)
    testset = torchvision.datasets.CIFAR10(root=datadir, train=False,
        download=True, transform=test_transform)
    test_dl = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
        shuffle=False, num_workers=test_num_workers, pin_memory=True)

    return train_dl, test_dl


class PrefetchDataLoader:
    def __init__(self, dataloader, device):
        self.loader = dataloader
        self.iter = None
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data =None

    def __len__(self):
        return len(self.loader)

    def async_prefech(self):
        try:
            self.next_data = next(self.iter)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            if isinstance(self.next_data, torch.Tensor):
                self.next_data = self.next_data.to(device=self.device, non_blocking=True)
            elif isinstance(self.next_data, (list, tuple)):
                self.next_data = [t.to(device=self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in self.next_data]

    def __iter__(self):
        self.iter = iter(self.loader)
        self.async_prefech()
        while self.next_data is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            data = self.next_data
            self.async_prefech()
            yield data