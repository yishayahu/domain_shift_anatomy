import random
from typing import Sequence, Callable, Union
import os
from copy import deepcopy

import PIL
import numpy as np
import torch
from dpipe.batch_iter import sample

from dpipe.torch import load_model_state, get_device
from dpipe.itertools import pam, squeeze_first

def load_model_state_cv3_wise(architecture, baseline_exp_path):
    val_path = os.path.abspath('.')
    exp = val_path.split('/')[-1]
    n_val = int(exp.split('_')[-1])

    n_fold = n_val // 15
    n_cv_block = n_val % 3

    path_to_pretrained_model = os.path.join(baseline_exp_path,
                                            f'experiment_{n_fold * 3 + n_cv_block}', 'model.pth')
    load_model_state(architecture, path=path_to_pretrained_model)


def load_model_state_fold_wise(architecture, baseline_exp_path,modify_state_fn=None):
    load_model_state(architecture, path=baseline_exp_path, modify_state_fn=modify_state_fn)


def modify_state_fn_spottune(current_state, state_to_load, init_random=False):
    add_str = '_freezed'
    state_to_load_parallel = deepcopy(state_to_load)
    for key in state_to_load.keys():
        a = key.split('.')
        a[0] = a[0] + add_str
        a = '.'.join(a)
        value_to_load = torch.rand(state_to_load[key].shape).to(state_to_load[key].device) if init_random else \
                        state_to_load[key]
        state_to_load_parallel[a] = value_to_load
    return state_to_load_parallel


def load_two_models_into_spottune(module, path_base, path_post):
    val_path = os.path.abspath('.')
    exp = val_path.split('/')[-1]
    n_val = int(exp.split('_')[-1])
    path_base = os.path.join(path_base, f'experiment_{n_val // 5}', 'model.pth')
    path_post = os.path.join(path_post, f'experiment_{n_val}', 'model.pth')

    state_to_load_base = torch.load(path_base, map_location=get_device(module))
    state_to_load_post = torch.load(path_post, map_location=get_device(module))
    add_key = '_freezed'
    for key in state_to_load_base.keys():
        key_lvls = key.split('.')
        key_lvls[0] = key_lvls[0] + add_key
        key_frzd = '.'.join(key_lvls)
        state_to_load_post[key_frzd] = state_to_load_base[key]
    module.load_state_dict(state_to_load_post)


def freeze_model(model, exclude_layers=('inconv', )):
    for name, param in model.named_parameters():
        requires_grad = False
        for l in exclude_layers:
            if l in name:
                requires_grad = True
        param.requires_grad = requires_grad

def none_func(*args,**kwargs):
    return None
def empty_dict_func(*args,**kwargs):
    return {'sdice_score':50}
def freeze_model_spottune(model):
    for name, param in model.named_parameters():
        if 'freezed' in name:
            requires_grad = False
        else:
            requires_grad = True
        param.requires_grad = requires_grad


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True


def load_by_gradual_id(*loaders: Callable, ids: Sequence, weights: Sequence[float] = None,
                       random_state: Union[np.random.RandomState, int] = None,batches_per_epoch=100,batch_size=16,ts_size=2):
    source_ids = ids[:-ts_size]
    target_ids = ids[-ts_size:]
    source_iter = sample(source_ids, weights, random_state)
    target_iter = sample(target_ids, weights, random_state)
    epoch = 0
    while True:

        for _ in range(batches_per_epoch):
            from_target = min((epoch//4)+ 1,batch_size-1)
            from_source = batch_size - from_target
            for _ in range(from_target):
                yield squeeze_first(tuple(pam(loaders, next(target_iter))))
            for _ in range(from_source):
                yield squeeze_first(tuple(pam(loaders, next(source_iter))))
        epoch+=1


class LoadByClusterId:
    def __init__(self):
        self.cluster_to_ids = {}
        self.current_cluster = None
        self.yileding_by_cluster = False
    def __call__(self,*loaders: Callable, ids: Sequence, weights: Sequence[float] = None,
                          random_state: Union[np.random.RandomState, int] = None):
        """
        Infinitely yield objects loaded by ``loaders`` according to the identifier from ``ids``.
        The identifiers are randomly sampled from ``ids`` according to the ``weights``.

        Parameters
        ----------
        loaders: Callable
            function, which loads object by its id.
        ids: Sequence
            the sequence of identifiers to sample from.
        weights: Sequence[float], None, optional
            The weights associated with each id. If ``None``, the weights are assumed to be equal.
            Should be the same size as ``ids``.
        random_state: int, np.random.RandomState, None, optional
            if not ``None``, used to set the random seed for reproducibility reasons.
        """
        samplers = {'all':sample(ids, weights, random_state),'source':sample(ids[:45], weights, random_state)}
        while True:

            if self.current_cluster is not None:
                self.yileding_by_cluster = True
                id_ = np.random.choice(self.cluster_to_ids[self.current_cluster])

                yield squeeze_first(tuple(pam(loaders, id_)))
                self.yileding_by_cluster = False
                id_ = next(samplers['source'])
                yield squeeze_first(tuple(pam(loaders, id_)))
            else:
                self.yileding_by_cluster = False
                id_ = next(samplers['all'])
                yield squeeze_first(tuple(pam(loaders, id_)))



class GetByClusterSlice:
    def __init__(self):
        self.cluster_id_loader = None
        self.cluster_to_id_slices = {}
        self.current_cluster = None
    def __call__(self,*arrays, interval: int = 1,msm=False):
        if msm:

            slc = np.random.randint(arrays[0].shape[0] // interval) * interval
            if len(arrays) > 2:
                domain,id1 = arrays[2:]
                if self.current_cluster is not None and self.cluster_id_loader.yileding_by_cluster:
                    slc = np.random.choice(self.cluster_to_id_slices[self.current_cluster]['CC0' + str(id1)])
                arrays = arrays[:2]
                return tuple([array[slc] for array in arrays] + [domain,id1,slc])
            else:
                return tuple(array[slc] for array in arrays)
        slc = np.random.randint(arrays[0].shape[-1] // interval) * interval
        if len(arrays) > 2:
            domain,id1 = arrays[2:]
            if self.current_cluster is not None and self.cluster_id_loader.yileding_by_cluster:
                slc = np.random.choice(self.cluster_to_id_slices[self.current_cluster]['CC0' + str(id1)])
            arrays = arrays[:2]
            return tuple([array[..., slc] for array in arrays] + [domain,id1,slc])
        else:
            return tuple(array[..., slc] for array in arrays)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = tensor.detach().cpu()
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze()
    else:
        assert tensor.shape[0] == 3
        tensor= tensor.transpose((1,2,0))
    return PIL.Image.fromarray(tensor)