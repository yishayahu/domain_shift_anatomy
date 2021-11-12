from collections import Sequence

import torch
from dpipe.train import Policy
from torch import nn

class DummyPolicy(Policy):
    def __init__(self,*args,**kwargs):
        pass

class FineTunePolicy(Policy):
    def __init__(self,architecture,optimizer):
        self.optimizer = optimizer
        self.layers = [x for x in architecture.named_modules() if isinstance(x[1], nn.Conv2d) or isinstance(x[1], nn.BatchNorm2d) or isinstance(x[1],nn.ConvTranspose2d)]
        self.unfreezed_layers = {0}
        self.optimizer.param_groups[0]['params'] = list(self.layers[0][1].parameters())
        self.grad_per_layer = torch.zeros(len(self.layers))
        self.last_best = [0,0]
        for layer_index,(n1,m1) in enumerate(self.layers):
            m1.register_full_backward_hook(self.collect_grads(layer_index))


    def epoch_started(self, epoch: int):
        self.grad_per_layer = torch.zeros(len(self.layers))
    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        if self.detect_plateau(metrics):
            layer_to_unfreeze_name,layer_to_unfreeze = self.layers[torch.argmax(self.grad_per_layer)]
            print(f'unfreezing {layer_to_unfreeze_name} current state is {layer_to_unfreeze.weight.requires_grad}')
            self.optimizer.param_groups[0]['params'].extend(list(layer_to_unfreeze.parameters()))


    def detect_plateau(self,metrics):
        curr_metric = metrics['sdice_score']

        if curr_metric < self.last_best[0] and curr_metric < self.last_best[1]:
            to_ret = True
        else:
            to_ret = False
        self.last_best.append(curr_metric)
        self.last_best = self.last_best[-2:]
        return to_ret


    def collect_grads(self,layer_index):
        def hook(_,__,grad_output):
            assert len(grad_output) ==1
            if layer_index not in self.unfreezed_layers:
                self.grad_per_layer[layer_index] += torch.mean(grad_output[0].cpu())

        return hook
