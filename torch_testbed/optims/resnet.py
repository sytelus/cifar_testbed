import logging
import torch

def optim_sched(epochs, net, *kargs, **kvargs):
        lr, momentum, weight_decay = 0.1, 0.9, 1.0e-4
        optim = torch.optim.SGD(net.parameters(),
                                lr, momentum=momentum, weight_decay=weight_decay)
        logging.info(f'optim_type=resnet, '
                     f'lr={lr}, momentum={momentum}, weight_decay={weight_decay}')

        sched = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=[100, 150]) # resnet original paper
        sched_on_epoch = True

        return optim, sched, sched_on_epoch