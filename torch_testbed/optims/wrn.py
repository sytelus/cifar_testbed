import logging
import torch

from .piecewise_lr import PiecewiseLR

def optim_sched(epochs, net, *kargs, **kvargs):
        lr, momentum, weight_decay = 0.1, 0.9, 1.0e-5
        optim = torch.optim.SGD(net.parameters(),
                                lr, momentum=momentum, weight_decay=weight_decay)
        logging.info(f'lr={lr}, momentum={momentum}, weight_decay={weight_decay}')

        sched = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=[60, 120, 160, 200, 300, 400, 600], gamma=0.2)
        sched_on_epoch = True

        return optim, sched, sched_on_epoch, 128