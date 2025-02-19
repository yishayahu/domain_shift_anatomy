from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.nn import Module
from torch.optim import Optimizer

from dpipe.im.utils import identity, dmap
from dpipe.torch.utils import *
from dpipe.torch.model import *

from spottunet.torch.functional import gumbel_softmax

layers = ['init_path.0', 'init_path.1', 'init_path.2', 'init_path.3', 'shortcut0', 'down1.0', 'down1.1', 'down1.2',
          'down1.3', 'shortcut1', 'down2.0', 'down2.1', 'down2.2', 'down2.3', 'shortcut2', 'bottleneck.0',
          'bottleneck.1', 'bottleneck.2', 'bottleneck.3', 'bottleneck.4', 'up2.0', 'up2.1', 'up2.2', 'up2.3', 'up1.0',
          'up1.1', 'up1.2', 'up1.3', 'out_path.0', 'out_path.1', 'out_path.2', 'out_path.3', 'out_path.4']
prev_step = -1


def train_step(*inputs, architecture, criterion, optimizer, n_targets=1, loss_key=None,
               alpha_l2sp=None, reference_architecture=None, train_step_logger=None,batch_iter_step=None, **optimizer_params):
    architecture.train()
    if n_targets >= 0:
        n_inputs = len(inputs) - n_targets
    else:
        n_inputs = -n_targets

    assert 0 <= n_inputs <= len(inputs)
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:n_inputs], inputs[n_inputs:]
    if alpha_l2sp is not None:
        if reference_architecture is None:
            raise ValueError('`reference_architecture` should be provided for L2-SP regularization.')

        w_diff = torch.tensor(0., requires_grad=True, dtype=torch.float32)
        w_diff.to(get_device(architecture))
        for p1, p2 in zip(architecture.parameters(), reference_architecture.parameters()):
            w_diff = w_diff + torch.sum((p1 - p2) ** 2)
        loss = criterion(architecture(*inputs), *targets) + alpha_l2sp * w_diff
    else:
        loss = criterion(architecture(*inputs), *targets)

    global prev_step
    if train_step_logger is not None and train_step_logger._experiment.step > prev_step and reference_architecture is not None:
        prev_step = train_step_logger._experiment.step




    if loss_key is not None:
        optimizer_step(optimizer, loss[loss_key], **optimizer_params)
        return dmap(to_np, loss)
    if type(loss) == dict:
        optimizer_step(optimizer, loss['total_loss_'], **optimizer_params)
        loss = {k: float(v) for (k, v) in loss.items()}
    else:
        optimizer_step(optimizer, loss, **optimizer_params)
        loss = to_np(loss)
    return loss


def train_step_spottune(*inputs: np.ndarray, architecture_main, architecture_policy, k_reg, reg_mode, temperature,
                        criterion: Callable, optimizer_main: Optimizer, optimizer_policy: Optimizer,
                        n_targets: int = 1, loss_key: str = None, k_reg_source=None, with_source=False, soft=False,
                        **optimizer_params) -> np.ndarray:
    architecture_main.train()
    architecture_policy.train()

    inputs = sequence_to_var(*inputs, device=architecture_main)

    if with_source:
        inputs_target = inputs[:2]
        inputs_source = inputs[2:]
        inputs_target, targets_target = inputs_target[0], inputs_target[1]
        inputs_source, targets_source = inputs_source[0], inputs_source[1]

        #  getting the policy (source)
        probs_source = architecture_policy(inputs_source)  # [batch, 16]
        action_source = gumbel_softmax(probs_source.view(probs_source.size(0), -1, 2), soft=soft,
                                       temperature=temperature)  # [batch, 8, 2]
        policy_source = action_source[:, :, 1]  # [batch, 8]

        # getting the policy (target)
        probs = architecture_policy(inputs_target)  # [batch, 16]
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2), soft=soft, temperature=temperature)  # [batch, 8, 2]
        policy = action[:, :, 1]  # [batch, 8]

        # forward (target)
        outputs = architecture_main.forward(inputs_target, policy)
        loss = criterion(outputs, targets_target) + reg_policy(policy=policy, k=k_reg, mode=reg_mode) + \
               reg_policy(policy=policy_source, k=k_reg_source)

    else:
        inputs, targets = inputs[0], inputs[1]

        # getting the policy (target)
        probs = architecture_policy(inputs)  # [32, 16]
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2), soft=soft, temperature=temperature)  # [32, 8, 2]
        policy = action[:, :, 1]  # [32, 8]

        # forward (target)
        outputs = architecture_main.forward(inputs, policy)
        loss = criterion(outputs, targets) + reg_policy(policy=policy, k=k_reg, mode=reg_mode)

    optimizer_step_spottune(optimizer_main, optimizer_policy, loss, **optimizer_params)

    return to_np(loss)


def inference_step_spottune(*inputs: np.ndarray, architecture_main: Module, architecture_policy: Module, temperature,
                            use_gumbel, activation: Callable = identity, soft=False) -> np.ndarray:
    """
    Returns the prediction for the given ``inputs``.

    Notes
    -----
    Note that both input and output are **not** of type ``torch.Tensor`` - the conversion
    to and from ``torch.Tensor`` is made inside this function.
    """

    architecture_main.eval()
    architecture_policy.eval()

    net_input = sequence_to_var(*inputs, device=architecture_main)

    probs = architecture_policy(*net_input)
    action = gumbel_softmax(probs.view(probs.size(0), -1, 2), soft=soft, use_gumbel=use_gumbel, temperature=temperature)
    policy = action[:, :, 1]

    with torch.no_grad():
        return to_np(activation(architecture_main(*net_input, policy)))


def optimizer_step_spottune(optimizer_main: Optimizer, optimizer_policy: Optimizer,
                            loss: torch.Tensor, **params) -> torch.Tensor:
    """
    Performs the backward pass with respect to ``loss``, as well as a gradient step.

    ``params`` is used to change the optimizer's parameters.

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=1)
    >>> optimizer_step(optimizer, loss) # perform a gradient step
    >>> optimizer_step(optimizer, loss, lr=1e-3) # set lr to 1e-3 and perform a gradient step
    >>> optimizer_step(optimizer, loss, betas=(0, 0)) # set betas to 0 and perform a gradient step

    Notes
    -----
    The incoming ``optimizer``'s parameters are not restored to their original values.
    """
    lr_main, lr_policy = params['lr_main'], params['lr_policy']

    set_params(optimizer_main, lr=lr_main)
    set_params(optimizer_policy, lr=lr_policy)

    optimizer_main.zero_grad()
    optimizer_policy.zero_grad()

    loss.backward()
    optimizer_main.step()
    optimizer_policy.step()

    return loss


def reg_policy(policy, k, mode='l1'):
    if mode == 'l1':
        reg = k * (1 - policy).sum() / torch.numel(policy)  # shape(policy) [batch_size, n_blocks]
    elif mode == 'l2':
        reg = k * torch.sqrt(((1 - policy) ** 2).sum()) / torch.numel(policy)
    else:
        raise ValueError(f'`mode` should be either `l1` or `l2`; but `{mode}` is given')
    return reg
