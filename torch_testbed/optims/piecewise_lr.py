from typing import List, Optional

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

class PiecewiseLR(_LRScheduler):
    def __init__(self, optimizer,
                 epochs:List[int], lrs:List[float],
                 steps_per_epoch:int, last_epoch=-1):
        assert len(epochs)==len(lrs) and len(epochs)>0 and epochs[0]==0
        assert all(x<y for x, y in zip(epochs, epochs[1:])) # monotonicity

        self._epochs = epochs
        self._lrs = lrs
        self._steps_per_epoch = steps_per_epoch
        self._i = 0
        self.epoch = 0
        self.steps = -1

        self._min_epoch = epochs[0]
        self._max_epoch = epochs[1] if len(epochs)>1 else epochs[0]
        self._min_lr = lrs[0]
        self._max_lr = lrs[1] if len(lrs)>1 else lrs[0]

        # super will save members in state_dict
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        self.steps += 1
        self.epoch = self.steps / self._steps_per_epoch
        if epoch is not None and epoch > self.epoch:
            self.epoch = epoch
            self.steps = epoch * self._steps_per_epoch

        super().step(epoch)

    def get_lr(self):
        if self._i < len(self._epochs)-1: # room for i to advance
            if self._epochs[self._i+1] <= self.epoch: # exceeded next epoch
                self._min_epoch = self._i # set min to current, max to next
                self._min_lr = self._lrs[self._i]
                self._i += 1
                self._max_epoch = self._i
                self._max_lr = self._lrs[self._i]
            # else leave i at the end

            assert self.epoch >= self._min_epoch # sanity check
            if self.epoch == self._min_epoch:
                frac = 0.0
            else:
                frac = (self.epoch-self._min_epoch)/(self._max_epoch-self._min_epoch)
            assert frac >= 0.0 and frac <= 1.0 # sanity check
            lr = (self._max_lr-self._min_lr)*frac + self._min_lr
        else:
            lr = self._lrs[self._i]

        return [lr for base_lr in self.base_lrs]