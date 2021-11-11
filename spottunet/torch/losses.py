import numpy as np
import torch
from dpipe.torch import weighted_cross_entropy_with_logits
from torch import nn


class FineRegularizedLoss:
    def __init__(self, architecture, reference_architecture,base_criterion=weighted_cross_entropy_with_logits, max_weight=np.array(7).astype('float32'), beta=1.5):
        self.architecture = architecture
        self.reference_architecture = reference_architecture
        self.reference_architecture.to(next(self.architecture.parameters()).device)
        self.max_weight = max_weight
        self.beta = beta
        self.base_criterion = base_criterion
    def __call__(self, outputs,targets):
        loss = weighted_cross_entropy_with_logits(outputs,targets)
        losses_dict = {'base_loss':loss}
        custom_loss = 0.0

        amount_of_params = len([x for x in self.architecture.named_modules() if isinstance(x[1], nn.Conv2d) or isinstance(x[1], nn.BatchNorm2d)])
        i = 0
        for ((n1,p1), (n2,p2)) in zip(self.architecture.named_modules(), self.reference_architecture.named_modules()):
            assert n1 == n2
            if isinstance(p1, nn.Conv2d) or isinstance(p1, nn.BatchNorm2d):
                temp_loss = torch.sqrt(torch.sum(((p1.weight - p2.weight) ** 2)))
                cur_weight = torch.tensor(self.max_weight) - self.beta * torch.log(torch.tensor(float(amount_of_params-i), requires_grad=True)).to(temp_loss.device)
                custom_loss+=cur_weight * temp_loss
                i+=1
        losses_dict['reg_loss'] = custom_loss
        if custom_loss == 0:
            print('eeee')
            losses_dict['total_loss'] = loss
        else:
            losses_dict['total_loss'] = custom_loss+loss
        t = ''
        for k,v in losses_dict.items():
            t+= f'{k}_{v}'
        print('losses',t)

        return losses_dict