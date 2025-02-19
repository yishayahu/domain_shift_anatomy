from collections import Sequence
from functools import partial
from typing import Any

import torch
from dpipe.train import Policy
from torch import nn

class DummyPolicy(Policy):
    def __init__(self,*args,**kwargs):
        pass

class PreDefinedFineTunePolicy(Policy):
    def __init__(self,architecture,optimizer):
        self.layers_to_unfreeze = ['shortcut0', 'out_path.4', 'down1.1', 'init_path.1', 'bottleneck.1', 'init_path.2', 'down2.2', 'init_path.0']
        self.layers = ['init_path.0', 'init_path.1', 'init_path.2', 'init_path.3', 'shortcut0', 'down1.0', 'down1.1', 'down1.2', 'down1.3', 'shortcut1', 'down2.0', 'down2.1', 'down2.2', 'down2.3', 'shortcut2', 'bottleneck.0', 'bottleneck.1', 'bottleneck.2', 'bottleneck.3', 'bottleneck.4', 'up2.0', 'up2.1', 'up2.2', 'up2.3', 'up1.0', 'up1.1', 'up1.2', 'up1.3', 'out_path.0', 'out_path.1', 'out_path.2', 'out_path.3', 'out_path.4']

        self.layers = {n1:[] for n1 in self.layers}
        self.optimizer = optimizer
        self.optimizer.param_groups.append({k:v for k,v in self.optimizer.param_groups[0].items() if k != 'params'})
        self.optimizer.param_groups[1]['params'] = []
        self.optimizer.param_groups[0]['lr'] = 0
        self.last_best = [0,0]
        self.unfreezed_layers = set()
        for n1,m1 in architecture.named_modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1,nn.ConvTranspose2d):
                for n2 in self.layers.keys():
                    if n2 in n1:
                        self.layers[n2].append(m1)
            if isinstance(m1, nn.BatchNorm2d):
                for n2 in self.layers.keys():
                    if n2 in n1:
                        self.layers[n2].append(m1)

        self.transfer_params(self.layers_to_unfreeze.pop(0))
    def transfer_params(self,name1):
        m_list = self.layers[name1]
        for m1 in m_list:
            ind = self.index_in_optimizer(m1.weight)
            self.optimizer.param_groups[1]['params'].append(self.optimizer.param_groups[0]['params'].pop(ind))
            if m1.bias is not None:
                ind = self.index_in_optimizer(m1.bias)
                self.optimizer.param_groups[1]['params'].append(self.optimizer.param_groups[0]['params'].pop(ind))
        print(f'unfreezing {name1}')
        self.unfreezed_layers.add(name1)

    def index_in_optimizer(self,p1):
        for i,param in enumerate(self.optimizer.param_groups[0]['params']):
            if param is p1:
                return i
        assert False

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        if self.detect_plateau(metrics) and len(self.layers_to_unfreeze) > 0:
            self.last_best = [0,0]
            self.transfer_params(self.layers_to_unfreeze.pop(0))
            print(f'current unfreeze {self.unfreezed_layers}')
        if epoch == 100:
            self.optimizer.param_groups[1]['lr'] = 1e-4

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


class FineTunePolicy(Policy):
    def __init__(self,return_to_ckpt,architecture,optimizer):
        self.layers = ['init_path.0', 'init_path.1', 'init_path.2', 'init_path.3', 'shortcut0', 'down1.0', 'down1.1', 'down1.2', 'down1.3', 'shortcut1', 'down2.0', 'down2.1', 'down2.2', 'down2.3', 'shortcut2', 'bottleneck.0', 'bottleneck.1', 'bottleneck.2', 'bottleneck.3', 'bottleneck.4', 'up2.0', 'up2.1', 'up2.2', 'up2.3', 'up1.0', 'up1.1', 'up1.2', 'up1.3', 'out_path.0', 'out_path.1', 'out_path.2', 'out_path.3', 'out_path.4']
        self.layers = {n1:[] for n1 in self.layers}
        self.architecture = architecture
        self.optimizer = optimizer
        self.optimizer.param_groups[0]['lr'] = 1e-10
        for n1,m1 in architecture.named_modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1,nn.ConvTranspose2d):
                 for n2 in self.layers.keys():
                    if n2 in n1:
                        self.layers[n2].append(m1)
            if isinstance(m1, nn.BatchNorm2d):
                for n2 in self.layers.keys():
                    if n2 in n1:
                        self.layers[n2].append(m1)
        self.return_to_ckpt = return_to_ckpt
        self.layer_per_group = {}
        # self.unfreezed_layers = {}

        self.grad_per_layer = torch.zeros((len(self.layers),3))
        self.last_best = [0,0]
        self.ckpt = [None,None]
        self.index_to_layer = {}
        for layer_index,(n1,m_list) in enumerate(self.layers.items()):
            self.index_to_layer[layer_index] = n1
            for m1 in m_list:
                # if 'init_path.1' in n1:
                #     self.unfreezed_layers[layer_index] = n1
                #     self.optimizer.param_groups[0]['params'].extend(list(m1.parameters()))
                if 'init_path.0' in n1:
                    m1.register_backward_hook(self.collect_grads())
        for i in range(10):
            for n1 in ['init_path.1','init_path.2']:
                self.transfer_params(name1=n1,m_list=self.layers[n1])
        # print(f'current unfreeze {self.unfreezed_layers}')

    def epoch_started(self, epoch: int):
        if self.return_to_ckpt:
            self.ckpt.append(self.architecture.state_dict())
            self.ckpt = self.ckpt[-2:]
        self.grad_per_layer = torch.zeros(len(self.layers),3)

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        if self.detect_plateau(metrics):
            counter = 0
            while True:
                layer_index = int(torch.argmax(self.grad_per_layer[:,0]))
                self.grad_per_layer[layer_index,0] = 0
                layer_to_unfreeze_name = self.index_to_layer[layer_index]
                layer_to_unfreeze_list = self.layers[layer_to_unfreeze_name]
                sucsess = self.transfer_params(layer_to_unfreeze_name,layer_to_unfreeze_list)
                if sucsess:
                    counter+=1
                    print(f'unfreezing {layer_to_unfreeze_name}')
                    if counter==3:
                        break
                else:
                    print(f'unfreezing {layer_to_unfreeze_name} failed')
            print(self.layer_per_group)
            self.last_best = [0,0]
            if self.return_to_ckpt:
                self.architecture.load_state_dict(self.ckpt[0])
                self.ckpt = [None,None]


    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        for layer_index in range(len(self.layers)):
            # if layer_index not in self.unfreezed_layers:
            if self.grad_per_layer[layer_index][2] != 0:
                self.grad_per_layer[layer_index][0] += self.grad_per_layer[layer_index][1] / self.grad_per_layer[layer_index][2]
                self.grad_per_layer[layer_index][1] = 0
                self.grad_per_layer[layer_index][2] = 0

    @staticmethod
    def index_in_optimizer(p1, group):
        for i,param in enumerate(group['params']):
            if param is p1:
                return i
        assert False

    def transfer_params(self,name1,m_list):
        factor = 10
        source_group_lr = self.layer_per_group.get(name1,1e-10)
        if source_group_lr > 1e-4:
            return False
        current_group_source = [x for x in self.optimizer.param_groups if x['lr'] == source_group_lr]
        current_group_dest = [x for x in self.optimizer.param_groups if x['lr'] == source_group_lr*factor]
        assert len(current_group_source) == 1
        current_group_source = current_group_source[0]
        if len(current_group_dest) == 0:
            self.optimizer.param_groups.append({k:v for k,v in self.optimizer.param_groups[0].items() if k != 'params'})
            self.optimizer.param_groups[-1]['params'] = []
            self.optimizer.param_groups[-1]['lr'] = source_group_lr * factor
            current_group_dest = self.optimizer.param_groups[-1]
        else:
            current_group_dest = current_group_dest[0]
        for m1 in m_list:
            ind = self.index_in_optimizer(m1.weight,current_group_source)
            current_group_dest['params'].append(current_group_source['params'].pop(ind))
            if m1.bias is not None:
                ind = self.index_in_optimizer(m1.bias,current_group_source)
                current_group_dest['params'].append(current_group_source['params'].pop(ind))
        self.layer_per_group[name1] = source_group_lr*factor
        return True


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
        def hook(_,grads,___):
            for layer_index in range(len(self.layers)):
                # if layer_index not in self.unfreezed_layers:
                m1_list = self.layers[self.index_to_layer[layer_index]]
                for m1 in m1_list:
                    if m1.weight.grad is not None:
                        self.grad_per_layer[layer_index][1] += torch.sum(torch.abs(m1.weight.grad.cpu()))
                    else:
                        assert 'init_path.0' in self.index_to_layer[layer_index]
                        self.grad_per_layer[layer_index][1] += torch.sum(torch.abs(grads[1].cpu()))
                    self.grad_per_layer[layer_index][2] += m1.weight.numel()
                    if m1.bias is not None:
                        self.grad_per_layer[layer_index][1] +=  torch.sum(torch.abs(m1.bias.grad.cpu()))
                        self.grad_per_layer[layer_index][2] += m1.bias.numel()
        return hook




class FineTunePolicyUsingDist(Policy):
    def __init__(self,architecture,reference_architecture,optimizer):
        self.layers = ['init_path.0', 'init_path.1', 'init_path.2', 'init_path.3', 'shortcut0', 'down1.0', 'down1.1', 'down1.2', 'down1.3', 'shortcut1', 'down2.0', 'down2.1', 'down2.2', 'down2.3', 'shortcut2', 'bottleneck.0', 'bottleneck.1', 'bottleneck.2', 'bottleneck.3', 'bottleneck.4', 'up2.0', 'up2.1', 'up2.2', 'up2.3', 'up1.0', 'up1.1', 'up1.2', 'up1.3', 'out_path.0', 'out_path.1', 'out_path.2', 'out_path.3', 'out_path.4']
        self.layers = {n1:[] for n1 in self.layers}
        self.architecture = architecture
        self.optimizer = optimizer
        self.optimizer.param_groups.append({k:v for k,v in self.optimizer.param_groups[0].items() if k != 'params'})
        self.optimizer.param_groups[1]['params'] = []
        self.optimizer.param_groups[0]['lr'] = 1e-10

        self.unfreezed_layers = set()
        self.last_best = [0,0]

        for (n1,m1),(n2,m2) in zip(architecture.named_modules(),reference_architecture.named_modules()):
            assert n1 == n2
            if isinstance(m1, nn.Conv2d) or isinstance(m1,nn.ConvTranspose2d) or isinstance(m1, nn.BatchNorm2d):
                for n3 in self.layers.keys():
                    if n3 in n1:
                        self.layers[n3].append((m1,m2))
        m_list = self.layers['init_path.1']
        self.transfer_params('init_path.1',m_list)
        print(f'current unfreeze {self.unfreezed_layers}')

    def transfer_params(self,name1,m_list):
        for m1,_ in m_list:
            ind = self.index_in_optimizer(m1.weight)
            self.optimizer.param_groups[1]['params'].append(self.optimizer.param_groups[0]['params'].pop(ind))
            if m1.bias is not None:
                ind = self.index_in_optimizer(m1.bias)
                self.optimizer.param_groups[1]['params'].append(self.optimizer.param_groups[0]['params'].pop(ind))
        self.unfreezed_layers.add(name1)

    def index_in_optimizer(self,p1):
        for i,param in enumerate(self.optimizer.param_groups[0]['params']):
            if param is p1:
                return i
        assert False
    @staticmethod
    def max_aux(unfreezed,x):
        n1,m_list = x
        if n1 in unfreezed:
            return 0
        dist = 0
        num_elem = 0
        for m1,m2 in m_list:
            dist+=torch.sum(torch.abs((m1.weight.detach().cpu() - m2.weight.cpu())))
            num_elem+=m1.weight.numel()
            if m1.bias is not None:
                dist+=torch.sum(torch.abs((m1.bias.detach().cpu() - m2.bias.cpu())))
                num_elem+=m1.bias.numel()
        return dist / num_elem


    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        if self.detect_plateau(metrics):
            self.last_best = [0,0]
            n1,m_list = max(self.layers.items(),key=partial(self.max_aux,self.unfreezed_layers))
            print(f'unfreezing {n1}')
            self.transfer_params(n1,m_list)
            print(f'current unfreeze {self.unfreezed_layers}')

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
