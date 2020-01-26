import logging
import torch

def optim_sched(epochs, net, *kargs, **kvargs):
        batch_size, train_size = 512, 50000
        steps_per_epoch = round(train_size / batch_size)
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * 15 # first 15 epochs

        lr, momentum, weight_decay = 0.1, 0.9, 8e-4
        optim = torch.optim.SGD(net.parameters(),
                                lr, momentum=momentum, weight_decay=weight_decay)
        logging.info(f'lr={lr}, momentum={momentum}, weight_decay={weight_decay}')

        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=warmup_steps/total_steps, anneal_strategy='linear',
            cycle_momentum=False, div_factor=100000.0,
            final_div_factor=100000.0
        )
        sched_on_epoch = False

        return optim, sched, sched_on_epoch, batch_size