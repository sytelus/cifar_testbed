import logging

import torch
import numpy as np
from torch_testbed import utils, cifar10_models
from torch_testbed.timing import MeasureTime, print_all_timings, print_timing, get_timing
from torch_testbed.dataloader_dali import cifar10_dataloaders


utils.setup_logging()
utils.setup_cuda(42)

batch_size = 512
half = True

datadir = utils.full_path('~/torchvision_data_dir')
train_dl, test_dl = cifar10_dataloaders(datadir,
    train_batch_size=batch_size, test_batch_size=1024,
    cutout=0)

model = cifar10_models.resnet18().cuda()
lr, momentum, weight_decay = 0.025, 0.9, 3.0e-4
optim = torch.optim.SGD(model.parameters(),
                        lr, momentum=momentum, weight_decay=weight_decay)
crit = torch.nn.CrossEntropyLoss().cuda()

if half:
    model = model.half()
    crit = crit.half()

@MeasureTime
def iter_dl(dl):
    i, d = 0, 0
    for x, l in dl:
        x, l = x.cuda().half() if half else x.cuda(), l.cuda()
        y = model(x)
        loss = crit(y, l)
        optim.zero_grad()
        loss.backward()
        optim.step()
        i += 1
        d += len(x)
    return i, d

for _ in range(5):
    i,d = iter_dl(train_dl)

print_all_timings()
print(i, d)

exit(0)