import logging
import torch

def optim_sched(epochs, net, *kargs, **kvargs):
        batch_size, train_size = 512, 50000
        total_steps = round(train_size / batch_size)
        steps_per_epoch = round(total_steps / epochs)
        warmup_steps = steps_per_epoch * 5 # first 5 epochs

        lr, momentum, weight_decay = 0.4, 0.9, 0.000125 * 512
        optim = torch.optim.SGD(net.parameters(),
                                lr, momentum=momentum, weight_decay=weight_decay)
        logging.info(f'lr={lr}, momentum={momentum}, weight_decay={weight_decay}')

        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=lr/batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=warmup_steps/total_steps, anneal_strategy='linear',
            cycle_momentum=False, div_factor=100000.0,
            final_div_factor=100000.0
        )
        sched_on_epoch = False

        return optim, sched, sched_on_epoch, 512