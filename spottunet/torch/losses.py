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

    @staticmethod
    def module_condition(module1):
        module_name, module1 = module1
        if isinstance(module1, nn.Conv2d) or isinstance(module1, nn.ConvTranspose2d):
            if 'out_path.3' not in module_name and  'out_path.4' not in module_name:
                return True
        return False
    def __call__(self, outputs,targets):
        loss = weighted_cross_entropy_with_logits(outputs,targets)
        losses_dict = {'base_loss':loss}
        custom_loss = 0.0

        amount_of_params = len([x for x in self.architecture.named_modules() if self.module_condition(x)])
        i = 0
        for ((n1,p1), (n2,p2)) in zip(self.architecture.named_modules(), self.reference_architecture.named_modules()):
            assert n1 == n2
            if self.module_condition((n1,p1)):
                weight_distance = torch.sqrt((p1.weight - p2.weight) ** 2).flatten()
                if p1.bias is not None:
                    bias_distance = torch.sqrt((p1.bias - p2.bias) ** 2).flatten()
                    weight_distance = torch.cat([weight_distance,bias_distance])
                temp_loss = torch.mean(weight_distance)
                cur_weight = torch.tensor(self.max_weight) - self.beta * torch.log(torch.tensor(float(amount_of_params-i), requires_grad=True)).to(temp_loss.device)
                custom_loss+=cur_weight * temp_loss
                i+=1
        losses_dict['reg_loss'] = custom_loss
        if custom_loss == 0:

            losses_dict['total_loss'] = loss
        else:
            losses_dict['total_loss'] = custom_loss+loss
        if torch.any(torch.isnan(losses_dict['total_loss'])):
            raise Exception('loss is nan')


        return losses_dict