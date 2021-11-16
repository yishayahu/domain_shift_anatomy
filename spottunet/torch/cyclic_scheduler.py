from typing import Any

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

