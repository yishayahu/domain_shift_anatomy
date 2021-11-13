from collections import Sequence

import torch
from dpipe.train import Policy
from torch import nn

class DummyPolicy(Policy):
    def __init__(self,*args,**kwargs):
        pass

class FineTunePolicy(Policy):
    def __init__(self,architecture,optimizer):
        self.layers = ['init_path.0', 'init_path.1', 'init_path.2', 'init_path.3', 'shortcut0', 'down1.0', 'down1.1', 'down1.2', 'down1.3', 'shortcut1', 'down2.0', 'down2.1', 'down2.2', 'down2.3', 'shortcut2', 'bottleneck.0', 'bottleneck.1', 'bottleneck.2', 'bottleneck.3', 'bottleneck.4', 'up2.0', 'up2.1', 'up2.2', 'up2.3', 'up1.0', 'up1.1', 'up1.2', 'up1.3', 'out_path.0', 'out_path.1', 'out_path.2', 'out_path.3', 'out_path.4']
        self.layers = {n1:[] for n1 in self.layers}
        for n1,m1 in architecture.named_modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d) or isinstance(m1,nn.ConvTranspose2d):
                 for n2 in self.layers.keys():
                    if n2 in n1:
                        self.layers[n2].append(m1)
        # todo: add full layer each time, check spottune,return to ckpt
        self.optimizer = optimizer
        self.unfreezed_layers = {}
        self.optimizer.param_groups[0]['params'] = []
        self.grad_per_layer = torch.zeros(len(self.layers))
        self.last_best = [0,0]
        self.index_to_layer = {}
        for layer_index,(n1,m_list) in enumerate(self.layers.items()):
            self.index_to_layer[layer_index] = n1
            for m1 in m_list:
                if 'init_path.0' in n1 or 'init_path.1' in n1:
                    self.unfreezed_layers[layer_index] = n1
                    self.optimizer.param_groups[0]['params'].extend(list(m1.parameters()))
                m1.register_full_backward_hook(self.collect_grads(layer_index))
        print(f'current unfreeze {self.unfreezed_layers}')

    def epoch_started(self, epoch: int):
        self.grad_per_layer = torch.zeros(len(self.layers))
    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        if self.detect_plateau(metrics):
            layer_index = int(torch.argmax(self.grad_per_layer))
            layer_to_unfreeze_name,layer_to_unfreeze_list = self.layers[self.index_to_layer[layer_index]]
            print(f'unfreezing {layer_to_unfreeze_name}')
            self.unfreezed_layers[layer_index] = layer_to_unfreeze_name
            print(f'current unfreeze {self.unfreezed_layers}')
            for m1 in layer_to_unfreeze_list:
                self.optimizer.param_groups[0]['params'].extend(list(m1.parameters()))


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
