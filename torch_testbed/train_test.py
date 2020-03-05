from argparse import ArgumentError
from typing import List, Mapping, Tuple, Any
from collections import OrderedDict
import os
import logging
import subprocess
import numpy as np
import itertools
import copy

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _Loss

import yaml

from torch_testbed.timing import MeasureTime, MeasureBlockTime
from torch_testbed import cifar10_models
from torch_testbed import utils
from . import optims

# Training
@MeasureTime
def train_epoch(epoch, net, train_dl, device, crit, optim,
                sched, sched_on_epoch, half)->Tuple[float, float]:
    correct, total, loss_total = 0, 0, 0.0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if half:
            inputs = inputs.half()

        outputs, loss = train_step(net, crit, optim, sched, sched_on_epoch,
                                   inputs, targets)
        loss_total += loss

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if sched and sched_on_epoch:
        sched.step(epoch=epoch)
    return 100.0*correct/total, loss_total

@MeasureTime
def train_step(net:torch.nn.Module,
               crit:_Loss, optim:Optimizer, sched:_LRScheduler, sched_on_epoch:bool,
               inputs:torch.Tensor, targets:torch.Tensor)->Tuple[torch.Tensor, float]:
    with MeasureBlockTime('forward'):
        outputs = net(inputs)

    loss = crit(outputs, targets)
    optim.zero_grad()
    with MeasureBlockTime('backward'):
        loss.backward()

    optim.step()
    if sched and not sched_on_epoch:
        sched.step()
    return outputs, loss.item()

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
def train(epochs, train_dl, test_dl, net, device, crit, optim,
          sched, sched_on_epoch, half, quiet)->List[Mapping]:
    if half:
        net.half()
        crit.half()
    train_acc, test_acc = 0.0, 0.0
    metrics = []
    for epoch in range(epochs):
        lr = optim.param_groups[0]['lr']
        train_acc, loss = train_epoch(epoch, net, train_dl, device, crit, optim,
                          sched, sched_on_epoch, half)
        test_acc = test(net, test_dl, device, half)
        metrics.append({'test_top1':test_acc, 'train_top1':train_acc, 'lr':lr, 'epoch': epoch, 'train_loss': loss})
        if not quiet:
            logging.info(f'train_epoch={epoch}, test_top1={test_acc}, train_top1={train_acc}, lr={lr:.4g}')
    return metrics

def param_size(model:torch.nn.Module)->int:
    """count all parameters excluding auxiliary"""
    return sum(v.numel() for name, v in model.named_parameters() \
        if "auxiliary" not in name)


@MeasureTime
def train_test(datadir:str, expdir:str,
               exp_name:str, exp_desc:str, epochs:int, model_name:str,
               train_batch_size:int, loader_workers:int, seed:int, half:bool, test_batch_size:int,
               loader:str, cutout:int, sched_optim:str)->Tuple[List[Mapping], int]:

    if loader=='torch':
        import torch_testbed.dataloader_torch as dlm
    elif loader=='dali':
        import torch_testbed.dataloader_dali as dlm
    else:
        raise ArgumentError(f'data loader type "{loader}" is not recognized')

    # dirs
    datadir = utils.full_path(datadir)
    os.makedirs(datadir, exist_ok=True)

    utils.setup_logging(filepath=os.path.join(expdir, 'logs.log'))

    # log config for reference
    logging.info(f'exp_name="{exp_name}", exp_desc="{exp_desc}"')
    logging.info(f'model_name="{model_name}", seed={seed}, epochs={epochs}')
    logging.info(f'half={half}, cutout={cutout}, train_batch_size={train_batch_size}')
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

    logging.info(f'optim_type={sched_optim}')
    optim, sched, sched_on_epoch, batch_size = getattr(optims, sched_optim).optim_sched(
        epochs, net)
    if train_batch_size <= 0:
        train_batch_size=batch_size
    logging.info(f'train_batch_size={train_batch_size}')

    # load data just before train start so any errors so far is not delayed
    train_dl, test_dl = dlm.cifar10_dataloaders(datadir,
        train_batch_size=train_batch_size, test_batch_size=test_batch_size,
        train_num_workers=loader_workers, test_num_workers=loader_workers,
        cutout=cutout)
    #train_dl = PrefetchDataLoader(train_dl, device)

    metrics = train(epochs, train_dl, test_dl, net, device, crit, optim,
          sched, sched_on_epoch, half, False)

    with open(os.path.join(expdir, 'metrics.yaml'), 'w') as f:
        yaml.dump(metrics, f)

    return metrics, train_batch_size


def generate_sched_trials():
    #lrs = np.array([10**p for p in range(-8, 1)])
    #lrs = np.concatenate((lrs, lrs*2.5, lrs*5, lrs*7.5))
    lrs = np.array([10**p for p in range(-5, 1)])
    lrs = np.concatenate((lrs,))


    #moms = np.concatenate(([1-10**p for p in range(-3, -1)],[p/10 for p in range(1, 10,)], [0.85]))
    moms = np.concatenate(([1-10**p for p in range(-2, -1)],[p/10 for p in range(1, 10,2)]))

    # wd = np.array([10**p for p in range(-6, 0)])
    # wd = np.concatenate((wd, wd*2.5, wd*5, wd*7.5))
    wd = np.array([10**p for p in range(-6, -2)])
    wd = np.concatenate((wd, wd*5))

    return list(itertools.product(lrs.tolist(), moms.tolist(), wd.tolist()))

def sched_trial_epoch(net_orig, sched_trial, train_dl, test_dl, epochs, device,
                      crit, half)->Tuple[torch.nn.Module, List[Mapping]]:
    net = copy.deepcopy(net_orig)
    optim = torch.optim.SGD(net.parameters(),
        sched_trial[0], momentum=sched_trial[1], weight_decay=sched_trial[2])
    metrics = train(epochs, train_dl, test_dl, net, device, crit, optim,
          sched=None, sched_on_epoch=False, half=half, quiet=True)
    return net, metrics

@MeasureTime
def ideal_sched(datadir:str, expdir:str,
               exp_name:str, exp_desc:str, epochs:int, model_name:str,
               train_batch_size:int, loader_workers:int, seed:int, half:bool, test_batch_size:int,
               loader:str, cutout:int, sched_optim:str):

    if loader=='torch':
        import torch_testbed.dataloader_torch as dlm
    elif loader=='dali':
        import torch_testbed.dataloader_dali as dlm
    else:
        raise ArgumentError(f'data loader type "{loader}" is not recognized')

    # dirs
    datadir = utils.full_path(datadir)
    os.makedirs(datadir, exist_ok=True)

    utils.setup_logging(filepath=os.path.join(expdir, 'logs.log'))

    # log config for reference
    logging.info(f'exp_name="{exp_name}", exp_desc="{exp_desc}"')
    logging.info(f'model_name="{model_name}", seed={seed}, epochs={epochs}')
    logging.info(f'half={half}, cutout={cutout}, train_batch_size={train_batch_size}')
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
    train_batch_size = 512 if train_batch_size <=0 else train_batch_size

    # load data just before train start so any errors so far is not delayed
    train_dl, test_dl = dlm.cifar10_dataloaders(datadir,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        train_num_workers=loader_workers, test_num_workers=loader_workers,
        cutout=cutout)
    #train_dl = PrefetchDataLoader(train_dl, device)

    lookahead = 12

    sched_trials = generate_sched_trials()
    run_results = []
    for epoch in range(epochs):
        # for this epoch we will conduct several trials
        best_net, best_max_acc, best_epoch_acc, best_sched_trial = None, -1, -1, None
        trial_results = []
        run_results.append((epoch, trial_results))
        for sched_trial in sched_trials:
            # get trained net and metrics for this trial
            trained_net, metrics = sched_trial_epoch(net, sched_trial, train_dl, test_dl, lookahead, device, crit, half)
            # extract test top1 for this trial for the first epoch
            epoch_acc = metrics[0]['test_top1']
            # extract best test top1 from the lookaheah
            max_acc = max(metrics, key=lambda m:m['test_top1'])['test_top1']
            # if this trial produced the best then record it
            if max_acc > best_max_acc:
                best_net, best_max_acc, best_epoch_acc, best_sched_trial = trained_net, max_acc, epoch_acc, sched_trial

            # update trial results
            trial_results.append((max_acc, sched_trial, metrics, epoch_acc))
            # keep best trial at the top and save it
            trial_results.sort(key=lambda t: t[0], reverse=True)  # keep sorted as we are saving
            with open(os.path.join(expdir, 'sched_trials.yaml'), 'w') as f:
                yaml.dump(run_results, f)

        # retrain for 1 epoch
        net = sched_trial_epoch(net, best_sched_trial, train_dl, test_dl, 1, device, crit, half)

        logging.info(f'train_epoch={epoch}, best_max={best_max_acc}, epoch_acc={best_epoch_acc}, sched={best_sched_trial}')

    return run_results[-1][1][0][2], train_batch_size




