from typing import Any, Sequence

import numpy as np
from dpipe.train import ValuePolicy
from torch.optim.lr_scheduler import CyclicLR


class CyclicScheduler(ValuePolicy):
    def __init__(self, optimizer):
        super().__init__(None)
        self.sched = CyclicLR(optimizer,base_lr=1e-4,max_lr=1e-2,step_size_up=50,step_size_down=150,mode='triangular2')

    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        self.sched.step()

    @property
    def value(self):
        return self.sched.get_last_lr()[0]
    @value.setter
    def value(self,val):
        pass





class DecreasingOnPlateauOfVal(ValuePolicy):
    """
    Policy that traces average train loss and if it didn't decrease according to ``atol``
    or ``rtol`` for ``patience`` epochs, multiply `value` by ``multiplier``.
    """

    def __init__(self, *, initial: float, multiplier: float):
        super().__init__(initial)
        self.last_best = [0,0]
        self.lr_dec_mul = multiplier

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        if self.detect_plateau(metrics):
            self.value *= self.lr_dec_mul
            print(f'current lr is {self.value}')
            self.last_best = [0,0]

    def detect_plateau(self,metrics):
        curr_metric = metrics['sdice_score']
        if curr_metric < self.last_best[1]*1.001 and self.last_best[1]< self.last_best[0]*1.001:
            to_ret = True
        else:
            to_ret = False
        self.last_best.append(curr_metric)
        print('last scores')
        print(self.last_best)
        self.last_best = self.last_best[-2:]
        return to_ret
