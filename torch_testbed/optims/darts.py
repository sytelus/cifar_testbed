import logging
import torch

def optim_sched(epochs, net, *kargs, **kvargs):
        lr, momentum, weight_decay = 0.025, 0.9, 3.0e-4
        optim = torch.optim.SGD(net.parameters(),
                                lr, momentum=momentum, weight_decay=weight_decay)
        logging.info(f'lr={lr}, momentum={momentum}, weight_decay={weight_decay}')

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
            T_max=epochs, eta_min=0.001) # darts paper
        sched_on_epoch = True

        return optim, sched, sched_on_epoch, 128
