import logging
import torch

def optim_sched(epochs, net, train_dl, *kargs, **kvargs):
        lr, betas, weight_decay = 3.0e-3, (0.95,0.85), 1.2e-6
        optim = torch.optim.AdamW(net.parameters(), lr=lr, betas=betas, eps=1.0e-08,
                          weight_decay=weight_decay, amsgrad=False)
        logging.info(f'optim_type=superconv, '
                     f'lr={lr}, betas={betas}, weight_decay={weight_decay}')

        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, 0.0001, epochs=epochs, steps_per_epoch=len(train_dl),
            pct_start=5.0/epochs, anneal_strategy='linear'
        )
        sched_on_epoch = False

        return optim, sched, sched_on_epoch