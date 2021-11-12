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
        self.unfreezed_layers = {}
        self.optimizer.param_groups[0]['params'] = []
        self.grad_per_layer = torch.zeros(len(self.layers))
        self.last_best = [0,0]
        for layer_index,(n1,m1) in enumerate(self.layers):
            if 'init_path.0' in n1 or 'init_path.1' in n1:
                self.unfreezed_layers[layer_index] = n1
                self.optimizer.param_groups[0]['params'].extend(list(m1.parameters()))
            m1.register_full_backward_hook(self.collect_grads(layer_index))
        print(f'current unfreeze {self.unfreezed_layers}')

    def epoch_started(self, epoch: int):
        self.grad_per_layer = torch.zeros(len(self.layers))
    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        if self.detect_plateau(metrics):
            layer_index = torch.argmax(self.grad_per_layer)
            layer_to_unfreeze_name,layer_to_unfreeze = self.layers[layer_index]
            print(f'unfreezing {layer_to_unfreeze_name}')
            self.unfreezed_layers[layer_index] = layer_to_unfreeze_name
            print(f'current unfreeze {self.unfreezed_layers}')
            self.optimizer.param_groups[0]['params'].extend(list(layer_to_unfreeze.parameters()))


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


    def collect_grads(self,layer_index):
        def hook(_,__,grad_output):
            assert len(grad_output) ==1
            if layer_index not in self.unfreezed_layers:
                self.grad_per_layer[layer_index] += torch.mean(grad_output[0].cpu())

        return hook
