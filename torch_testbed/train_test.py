from typing import List, Tuple, Any
from collections import OrderedDict
import os
import logging
import subprocess

import torch


from torch_testbed.timing import MeasureTime
from torch_testbed import cifar10_models
from torch_testbed.dataloader import cifar10_dataloaders
from torch_testbed import utils

# Training
@MeasureTime
def train_epoch(epoch, net, train_dl, device, crit, optim,
                sched, sched_on_epoch, half)->float:
    correct, total = 0, 0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs = inputs.to(device, non_blocking=False)
        targets = targets.to(device)

        if half:
            inputs = inputs.half()

        outputs = net(inputs)
        loss = crit(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if sched and not sched_on_epoch:
            sched.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if sched and sched_on_epoch:
        sched.step()
    return 100.0*correct/total

@MeasureTime
def test(net, test_dl, device, half)->float:
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs = inputs.to(device, non_blocking=False)
            targets = targets.to(device)

            if half:
                inputs = inputs.half()

            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0*correct/total

@MeasureTime
def train(epochs, train_dl, net, device, crit, optim,
          sched, sched_on_epoch, half)->float:
    if half:
        net.half()
        crit.half()
    acc = 0.0
    for epoch in range(epochs):
        lr = optim.param_groups[0]['lr']
        acc = train_epoch(epoch, net, train_dl, device, crit, optim,
                          sched, sched_on_epoch, half)
        logging.info(f'train_epoch={epoch}, prec1={acc}, lr={lr:.4g}')
    return acc

def param_size(model:torch.nn.Module)->int:
    """count all parameters excluding auxiliary"""
    return sum(v.numel() for name, v in model.named_parameters() \
        if "auxiliary" not in name)


@MeasureTime
def train_test(exp_name:str, exp_desc:str, epochs:int, model_name:str,
               train_batch_size:int, seed:int, half:bool, test_batch_size:int, cutout:int,
               sched_type:str, optim_type:str)->Tuple[float, float]:
    # dirs
    datadir = utils.full_path('~/torchvision_data_dir')
    expdir = utils.full_path(os.path.join('~/logdir/cifar_testbed/', exp_name))
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(expdir, exist_ok=True)
    utils.setup_logging(filepath=os.path.join(expdir, 'logs.log'))

    # log config for reference
    logging.info(f'exp_name="{exp_name}", exp_desc="{exp_desc}"')
    logging.info(f'model_name="{model_name}", seed={seed}, epochs={epochs}')
    logging.info(f'half={half}, cutout={cutout}, sched_type={sched_type}')
    logging.info(f'datadir="{datadir}"')
    logging.info(f'expdir="{expdir}"')

    if not utils.is_debugging():
        sysinfo_filepath = os.path.join(expdir, 'sysinfo.txt')
        subprocess.Popen([f'./sysinfo.sh "{expdir}" > "{sysinfo_filepath}"'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         shell=True)

    utils.setup_cuda(seed)

    device = torch.device('cuda')

    model_class = getattr(cifar10_models, model_name)
    net = model_class()
    logging.info(f'param_size_m={param_size(net):.1e}')
    net = net.to(device)

    crit = torch.nn.CrossEntropyLoss().to(device)

    if optim_type=='darts':
        lr, momentum, weight_decay = 0.025, 0.9, 3.0e-4
        optim = torch.optim.SGD(net.parameters(),
                                lr, momentum=momentum, weight_decay=weight_decay)
        logging.info(f'optim_type={optim_type}, '
                     f'lr={lr}, momentum={momentum}, weight_decay={weight_decay}')
    elif optim_type=='resnet':
        lr, momentum, weight_decay = 0.1, 0.9, 1.0e-4
        optim = torch.optim.SGD(net.parameters(),
                                lr, momentum=momentum, weight_decay=weight_decay)
        logging.info(f'optim_type={optim_type}, '
                     f'lr={lr}, momentum={momentum}, weight_decay={weight_decay}')
    elif optim_type=='sc': # super convergence
        lr, betas, weight_decay = 3.0e-3, (0.95,0.85), 1.2e-6
        optim = torch.optim.AdamW(net.parameters(), lr=lr, betas=betas, eps=1.0e-08,
                          weight_decay=weight_decay, amsgrad=False)
        logging.info(f'optim_type={optim_type}, '
                     f'lr={lr}, betas={betas}, weight_decay={weight_decay}')
    else:
        raise RuntimeError(f'Unsupported LR scheduler type: {sched_type}')

    # load data just before train start so any errors so far is not delayed
    train_dl, test_dl = cifar10_dataloaders(datadir,
        train_batch_size=train_batch_size, test_batch_size=test_batch_size,
        cutout=cutout)
    #train_dl = PrefetchDataLoader(train_dl, device)

    if sched_type=='darts':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
            T_max=epochs, eta_min=0.001) # darts paper
        sched_on_epoch = True
    elif sched_type=='resnet':
        sched = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=[100, 150]) # resnet original paper
        sched_on_epoch = True
    elif sched_type=='sc':
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, 0.0001, epochs=epochs, steps_per_epoch=len(train_dl),
            pct_start=5.0/epochs, anneal_strategy='linear'
        )
        sched_on_epoch = False
    else:
        raise RuntimeError(f'Unsupported LR scheduler type: {sched_type}')

    train_acc = train(epochs, train_dl, net, device, crit, optim,
          sched, sched_on_epoch, half)
    test_acc = test(net, test_dl, device, half)
    return train_acc, test_acc

