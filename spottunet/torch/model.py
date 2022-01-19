import random
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from dpipe.train import Checkpoints, Logger, Policy, EarlyStopping, ValuePolicy
from dpipe.train.base import _DummyCheckpoints, _DummyLogger, _build_context_manager
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import Module
from torch.optim import Optimizer

from dpipe.im.utils import identity, dmap
from dpipe.torch.utils import *
from dpipe.torch.model import *

from clustering.ds_wrapper import DsWrapper
from spottunet.torch.functional import gumbel_softmax
from spottunet.torch.utils import tensor_to_image

layers = ['init_path.0', 'init_path.1', 'init_path.2', 'init_path.3', 'shortcut0', 'down1.0', 'down1.1', 'down1.2',
          'down1.3', 'shortcut1', 'down2.0', 'down2.1', 'down2.2', 'down2.3', 'shortcut2', 'bottleneck.0',
          'bottleneck.1', 'bottleneck.2', 'bottleneck.3', 'bottleneck.4', 'up2.0', 'up2.1', 'up2.2', 'up2.3', 'up1.0',
          'up1.1', 'up1.2', 'up1.3', 'out_path.0', 'out_path.1', 'out_path.2', 'out_path.3', 'out_path.4']
prev_step = -1


def train_step(*inputs, architecture, criterion, optimizer, n_targets=1, loss_key=None,
               alpha_l2sp=None, reference_architecture=None, train_step_logger=None,use_clustering_curriculum=False,batch_iter_step=None, **optimizer_params):
    architecture.train()
    if n_targets >= 0:
        n_inputs = len(inputs) - n_targets
    else:
        n_inputs = -n_targets

    assert 0 <= n_inputs <= len(inputs)
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets,domains,patient_ids,slices = inputs[0:1], inputs[1:2],inputs[2].flatten(),inputs[3].flatten(),inputs[4].flatten()
    loss = criterion(architecture(*inputs), *targets)
    if use_clustering_curriculum:
        if type(batch_iter_step.dataset) == DsWrapper:
            loss = batch_iter_step.dataset.send_loss(loss,batch_iter_step)
        else:
            loss = torch.mean(loss)
    global prev_step
    if train_step_logger is not None and train_step_logger._experiment.step > prev_step and reference_architecture is not None:
        prev_step = train_step_logger._experiment.step
        dist_pet_layer = []
        param_size_per_layer = []
        normalize_dist_per_layer = []
        dist_pet_layer_bn = []
        param_size_per_layer_bn = []
        normalize_dist_per_layer_bn = []
        names = []
        names_bn = []
        for (n1, p1), (n2, p2) in zip(architecture.named_parameters(), reference_architecture.named_parameters()):
            assert n1 == n2
            if 'out_path.4' in n1 or 'out_path.3.layer.bias' in n1:
                continue
            if 'bn' in n1:
                dist_pet_layer_bn.append(float(torch.mean(torch.abs(p1.detach().cpu() - p2.cpu()))))
                param_size_per_layer_bn.append(float(torch.mean(torch.abs(p1))))
                normalize_dist_per_layer_bn.append(dist_pet_layer[-1] / param_size_per_layer[-1])
                names_bn.append(n1)
            else:
                dist_pet_layer.append(float(torch.mean(torch.abs(p1.detach().cpu() - p2.cpu()))))
                param_size_per_layer.append(float(torch.mean(torch.abs(p1))))
                normalize_dist_per_layer.append(dist_pet_layer[-1] / param_size_per_layer[-1])
                names.append(n1)
        if False:
            for k, v in {'dist': dist_pet_layer, 'param_size': param_size_per_layer,
                         'relative_dist': normalize_dist_per_layer, 'dist_bn': dist_pet_layer_bn,
                         'param_size_bn': param_size_per_layer_bn, 'relative_dist_bn': normalize_dist_per_layer_bn}.items():
                im_path = f'{train_step_logger._experiment.name}_{k}.png'
                if 'bn' in k:
                    curr_names = names_bn
                else:
                    curr_names = names
                print(f'max for {k} is in layer {curr_names[np.argmax(v)]} and its values is {np.max(v)}')

                plt.plot(list(range(len(v))), v)
                plt.savefig(im_path)

                log_log = {f'{k}': wandb.Image(im_path)}
                if len(optimizer.param_groups) > 1 and k == 'dist':
                    for i in range(len(optimizer.param_groups)):
                        print(f"lr_group_{i}: {optimizer.param_groups[i]['lr']}")
                        log_log[f'lr_group_{i}'] = optimizer.param_groups[i]['lr']
                wandb.log(log_log, step=train_step_logger._experiment.step)
                plt.cla()
                plt.clf()

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

def permutation_recursion(dists):
    n_clusters = len(dists)
    def permutation_recursion_aux(l1,acc_dist,min_dist):
        if len(l1) == n_clusters:
            return acc_dist,l1
        min_l = []
        current_order = list(range(n_clusters))
        random.shuffle(current_order)
        for clus in current_order:
            if len(l1) == 0:
                print(f"in clus: {clus}")
            if clus not in l1:
                curr_target = len(l1)
                curr_l = l1.copy()+[clus]
                curr_acc_dist = acc_dist+dists[clus][curr_target]
                if curr_acc_dist > min_dist:
                    continue
                curr_acc_dist,curr_l = permutation_recursion_aux(curr_l,curr_acc_dist,min_dist)
                if curr_acc_dist > min_dist:
                    continue
                if curr_acc_dist < min_dist:
                    min_l = curr_l
                    min_dist = curr_acc_dist

        return min_dist,min_l
    return permutation_recursion_aux([],0,np.inf)[1]


def get_best_match(sc, tc):
    dists = np.full((sc.shape[0],tc.shape[0]),fill_value=np.inf)
    for i in range(sc.shape[0]):
        for j in range(tc.shape[0]):
            dists[i][j] = np.mean(np.abs(sc[i]-tc[j]))
    print('looking for best match')
    best_match = permutation_recursion(dists)
    print('best match found')

    return best_match

def train_unsup(train_step: Callable, batch_iter: Callable, n_epochs: int = np.inf, logger: Logger = None,
          checkpoints: Checkpoints = None, validate: Callable = None, **kwargs):
    """
    Performs a series of train and validation steps.

    Parameters
    ----------
    train_step: Callable
        a function to perform train step.
    batch_iter: Callable
        batch iterator.
    n_epochs: int
        maximal number of training epochs
    logger: Logger, None, optional
    checkpoints: Checkpoints, None, optional
    validate: Callable, None, optional
        a function to calculate metrics on the validation set.
    kwargs
        additional keyword arguments passed to ``train_step``.
        For instances of `ValuePolicy` their `value` attribute is passed.
        Other policies are used for early stopping.

    References
    ----------
    See the :doc:`tutorials/training` tutorial for more details.
    """

    def get_policy_values():
        return {k: v.value for k, v in policies.items() if isinstance(v, ValuePolicy)}

    def broadcast_event(method, *args, **kw):
        for name, policy in policies.items():
            getattr(policy, method.__name__)(*args, **kw)

    if checkpoints is None:
        checkpoints = _DummyCheckpoints()
    if logger is None:
        logger = _DummyLogger()
    if not hasattr(batch_iter, '__enter__'):
        batch_iter = _build_context_manager(batch_iter)

    epoch = checkpoints.restore()
    scalars = {name: value for name, value in kwargs.items() if not isinstance(value, Policy)}
    policies = {name: value for name, value in kwargs.items() if isinstance(value, Policy)}

    slice_to_cluster = None
    n_clusters = 12
    source_clusters = None
    target_clusters = None
    best_matchs = None
    with batch_iter as iterator:
        try:
            while epoch < n_epochs:
                slice_to_feature_source = {}
                slice_to_feature_target = {}
                vizviz = set()
                broadcast_event(Policy.epoch_started, epoch)
                train_losses = []
                for idx, inputs in enumerate(iterator()):
                    broadcast_event(Policy.train_step_started, epoch, idx)

                    loss=train_step(*inputs, **scalars, **get_policy_values(),
                                    slice_to_feature_source=slice_to_feature_source,
                                    slice_to_feature_target=slice_to_feature_target,
                                    slice_to_cluster=slice_to_cluster,
                                    source_clusters=source_clusters,
                                    target_clusters=target_clusters,
                                    best_matchs=best_matchs,
                                    vizviz=vizviz)
                    train_losses.append(loss)
                    broadcast_event(Policy.train_step_finished, epoch, idx, train_losses[-1])

                source_clusters = []
                target_clusters = []
                for i in range(n_clusters):
                    source_clusters.append([])
                    target_clusters.append([])
                p = PCA(n_components=20,random_state=42)
                t = TSNE(n_components=3,learning_rate='auto',init='pca',random_state=42)
                points = np.stack(list(slice_to_feature_source.values()) + list(slice_to_feature_target.values()))
                points = points.reshape(points.shape[0],-1)
                print('doing tsne')
                points = p.fit_transform(points)
                points = t.fit_transform(points)
                source_points,target_points = points[:len(slice_to_feature_source)],points[len(slice_to_feature_source):]
                # source_points,target_points = points[:max(len(slice_to_feature_source),n_clusters)],points[-max(len(slice_to_feature_target),n_clusters):]
                k1 = KMeans(n_clusters=n_clusters,random_state=42)
                print('doing kmean 1')
                sc = k1.fit_predict(source_points)
                k2 = KMeans(n_clusters=n_clusters,random_state=42)
                print('doing kmean 2')
                tc = k2.fit_predict(target_points)
                print('getting best match')

                best_matchs_indexes=get_best_match(k1.cluster_centers_,k2.cluster_centers_)
                slice_to_cluster = {}

                items = list(slice_to_feature_source.items())

                for i in range(len(slice_to_feature_source)):
                    source_clusters[sc[i]].append(items[i][1])
                    slice_to_cluster[items[i][0]] = sc[i]


                items = list(slice_to_feature_target.items())
                for i in range(len(slice_to_feature_target)):
                    target_clusters[tc[i]].append(items[i][1])
                    slice_to_cluster[items[i][0]] = tc[i]
                for i in range(len(source_clusters)):
                    source_clusters[i] = np.mean(source_clusters[i],axis=0)
                    target_clusters[i] = np.mean(target_clusters[i],axis=0)

                im_path = f'{logger._experiment.name}_{epoch}.png'

                colors = ['black','blue','cyan','red','orange'
                        ,'tomato','lime','gold','magenta','dodgerblue'
                          ,'peru','grey','brown','olive','navy'
                          ,'blueviolet','darkgreen','maroon','yellow','cadetblue']


                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for i,(p,marker) in enumerate([(k1.cluster_centers_,'.'),(k2.cluster_centers_,'^')]):
                    if i ==0:
                        ax.scatter(p[:,0],p[:,1],p[:,2],marker = marker,c=colors[:len(p)])
                    else:
                        ax.scatter(p[:,0],p[:,1],p[:,2],marker = marker,c=[colors[best_matchs_indexes[i]] for i in range(len(p))])
                plt.savefig(im_path)
                plt.cla()
                plt.clf()
                log_log = {f'fig': wandb.Image(im_path)}
                wandb.log(log_log, step=logger._experiment.step)
                best_matchs = []
                for i in range(len(best_matchs_indexes)):
                    best_matchs.append(torch.tensor(source_clusters[best_matchs_indexes[i]]))
                logger.train(train_losses, epoch)
                logger.policies(get_policy_values(), epoch)
                broadcast_event(Policy.validation_started, epoch, train_losses)

                metrics = None
                if validate is not None:
                    metrics = validate()
                    logger.metrics(metrics, epoch)

                broadcast_event(Policy.epoch_finished, epoch, train_losses,
                                metrics=metrics, policies=get_policy_values())
                checkpoints.save(epoch, train_losses, metrics)


                epoch += 1


        except EarlyStopping:
            pass

def train_step_unsup(*inputs, architecture, criterion, optimizer, n_targets=1, loss_key=None,
               alpha_l2sp=None,best_matchs, reference_architecture=None,vizviz=None, train_step_logger=None,use_clustering_curriculum=False,batch_iter_step=None,target_domain=None,slice_to_feature_source=None,slice_to_cluster=None,slice_to_feature_target=None,source_clusters=None,target_clusters=None, **optimizer_params):
    architecture.train()
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets,domains,patient_ids,slice_nums = inputs[0:1], inputs[1:2],inputs[2].flatten(),inputs[3].flatten().int(),inputs[4].flatten().int()
    logits,features = architecture(*inputs)
    features = features.mean(1)
    loss_dict = {}
    dist_loss = torch.tensor(0.0,device=logits.device)
    log_log = {}
    dist_loss_counter = 0
    for d,pi,sn,feature,img in zip(domains,patient_ids,slice_nums,features,inputs[0]):
        if d == target_domain:
            slice_to_feature_target[f'{pi}_{sn}'] =feature.detach().cpu().numpy()
            if best_matchs is not None and f'{pi}_{sn}' in slice_to_cluster:
                dist_loss_counter+=1
                dist_loss+= torch.mean(torch.abs(feature - best_matchs[slice_to_cluster[f'{pi}_{sn}']].to(logits.device)))
                if f'target_{slice_to_cluster[f"{pi}_{sn}"]}' not in vizviz:
                    img = tensor_to_image(img)
                    im_path =  f'target_{slice_to_cluster[f"{pi}_{sn}"]}_{train_step_logger._experiment.step}.png'
                    img.save(im_path)
                    log_log[f'{slice_to_cluster[f"{pi}_{sn}"]}/target'] = wandb.Image(im_path)
                    vizviz.add(f'target_{slice_to_cluster[f"{pi}_{sn}"]}')

        else:
            slice_to_feature_source[f'{pi}_{sn}'] =feature.detach().cpu().numpy()
            if best_matchs is not None and f'{pi}_{sn}' in slice_to_cluster:
                if f'source_{slice_to_cluster[f"{pi}_{sn}"]}' not in vizviz:
                    img = tensor_to_image(img)
                    im_path =  f'source_{slice_to_cluster[f"{pi}_{sn}"]}_{train_step_logger._experiment.step}.png'
                    img.save(im_path)
                    log_log[f'{slice_to_cluster[f"{pi}_{sn}"]}/source'] = wandb.Image(im_path)
                    vizviz.add(f'source_{slice_to_cluster[f"{pi}_{sn}"]}')
            # if best_matchs is not None and f'{pi}_{sn}' in slice_to_cluster:
            #     dist_loss+= torch.mean(torch.abs(feature - source_clusters[slice_to_cluster[f'{pi}_{sn}']].to(logits.device)))
    log_log['dist_loss_counter'] = dist_loss_counter
    wandb.log(log_log, step=train_step_logger._experiment.step)
    loss = criterion(logits, *targets)
    loss[domains == target_domain] = 0
    loss = loss.mean()
    loss_dict['dist_loss'] = dist_loss
    loss_dict['loss'] = loss
    loss_dict['total_loss_'] = loss+dist_loss
    loss = loss_dict
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
