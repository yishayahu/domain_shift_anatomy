import numpy as np
import torch
from dpipe.torch import weighted_cross_entropy_with_logits



class FineRegularizedLoss:
    def __init__(self, architecture, reference_architecture,base_criterion=weighted_cross_entropy_with_logits, max_weight=np.array(8).astype('float32'), beta=1.5):
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

        amount_of_params = len(list(self.architecture.named_parameters()))
        for i,((n1,p1), (n2,p2)) in enumerate(zip(self.architecture.named_parameters(), self.reference_architecture.named_parameters())):
            assert n1 == n2
            if 'shortcut' in n1:
                continue
            else:
                temp_loss = torch.sum(((p1 - p2) ** 2)) ** 0.5
                cur_weight = torch.tensor(self.max_weight) - self.beta * torch.log(torch.tensor(float(amount_of_params-i), requires_grad=True)).to(temp_loss.device)
                custom_loss+=cur_weight * temp_loss

        losses_dict['reg_loss'] = custom_loss
        losses_dict['total_loss'] = custom_loss+loss

        return losses_dict