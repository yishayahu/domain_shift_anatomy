from collections import Sequence
from typing import Any

import torch
from dpipe.train import Policy
from torch import nn

class DummyPolicy(Policy):
    def __init__(self,*args,**kwargs):
        pass

class FineTunePolicy(Policy):
    def __init__(self,return_to_ckpt,architecture,optimizer):
        self.layers = ['init_path.0', 'init_path.1', 'init_path.2', 'init_path.3', 'shortcut0', 'down1.0', 'down1.1', 'down1.2', 'down1.3', 'shortcut1', 'down2.0', 'down2.1', 'down2.2', 'down2.3', 'shortcut2', 'bottleneck.0', 'bottleneck.1', 'bottleneck.2', 'bottleneck.3', 'bottleneck.4', 'up2.0', 'up2.1', 'up2.2', 'up2.3', 'up1.0', 'up1.1', 'up1.2', 'up1.3', 'out_path.0', 'out_path.1', 'out_path.2', 'out_path.3', 'out_path.4']
        self.layers = {n1:[] for n1 in self.layers}
        self.architecture = architecture
        self.optimizer = optimizer
        self.optimizer.param_groups[0]['params'] = []
        for n1,m1 in architecture.named_modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1,nn.ConvTranspose2d):
                 for n2 in self.layers.keys():
                    if n2 in n1:
                        self.layers[n2].append(m1)
            if isinstance(m1, nn.BatchNorm2d):
                for n2 in self.layers.keys():
                    if n2 in n1:
                        self.layers[n2].append(m1)
                # continue
                # self.optimizer.param_groups[0]['params'].extend(list(m1.parameters()))



        self.return_to_ckpt = return_to_ckpt

        self.unfreezed_layers = {}

        self.grad_per_layer = torch.zeros((len(self.layers),3))
        self.last_best = [0,0]
        self.ckpt = [None,None]
        self.index_to_layer = {}
        for layer_index,(n1,m_list) in enumerate(self.layers.items()):
            self.index_to_layer[layer_index] = n1
            for m1 in m_list:
                if 'init_path.0' in n1 or 'init_path.1' in n1:
                    self.unfreezed_layers[layer_index] = n1
                    self.optimizer.param_groups[0]['params'].extend(list(m1.parameters()))
                    if 'init_path.0' in n1:
                        m1.register_full_backward_hook(self.collect_grads())
        print(f'current unfreeze {self.unfreezed_layers}')

    def epoch_started(self, epoch: int):
        if self.return_to_ckpt:
            self.ckpt.append(self.architecture.state_dict())
            self.ckpt = self.ckpt[-2:]
        self.grad_per_layer = torch.zeros(len(self.layers),3)

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        if self.detect_plateau(metrics):
            layer_index = int(torch.argmax(self.grad_per_layer[:,0]))
            layer_to_unfreeze_name = self.index_to_layer[layer_index]
            layer_to_unfreeze_list = self.layers[layer_to_unfreeze_name]
            print(f'unfreezing {layer_to_unfreeze_name}')
            self.unfreezed_layers[layer_index] = layer_to_unfreeze_name
            print(f'current unfreeze {self.unfreezed_layers}')
            self.last_best = [0,0]
            if self.return_to_ckpt:
                self.architecture.load_state_dict(self.ckpt[0])
                self.ckpt = [None,None]
            for m1 in layer_to_unfreeze_list:
                self.optimizer.param_groups[0]['params'].extend(list(m1.parameters()))

    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        for layer_index in range(len(self.layers)):
            if layer_index not in self.unfreezed_layers:
                if self.grad_per_layer[layer_index][2] != 0:
                    self.grad_per_layer[layer_index][0] += self.grad_per_layer[layer_index][1] / self.grad_per_layer[layer_index][2]
                    self.grad_per_layer[layer_index][1] = 0
                    self.grad_per_layer[layer_index][2] = 0



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


    def collect_grads(self):
        def hook(_,__,___):
            for layer_index in range(len(self.layers)):
                if layer_index not in self.unfreezed_layers:
                    m1_list = self.layers[self.index_to_layer[layer_index]]
                    for m1 in m1_list:
                        self.grad_per_layer[layer_index][1] += torch.sum(torch.abs(m1.weight.grad.cpu()))
                        self.grad_per_layer[layer_index][2] += m1.weight.numel()
                        if m1.bias is not None:
                            self.grad_per_layer[layer_index][1] +=  torch.sum(torch.abs(m1.bias.grad.cpu()))
                            self.grad_per_layer[layer_index][2] += m1.bias.numel()
        return hook


