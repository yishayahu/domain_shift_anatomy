import gc
import random
from typing import Sequence, Callable, Union
import os
from copy import deepcopy

import numpy as np
import pandas as pd
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

from dpipe.torch import load_model_state
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
    return {0:None}
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
                       random_state: Union[np.random.RandomState, int] = None,batches_per_epoch=100,batch_size=16,ts_size=2,keep_source=True):
    source_ids = ids[:-ts_size]
    target_ids = ids[-ts_size:]
    source_iter = sample(source_ids, weights, random_state)
    target_iter = sample(target_ids, weights, random_state)
    epoch = 0
    while True:

        for _ in range(batches_per_epoch):
            if keep_source:
                from_target = min((epoch//4)+ 1,batch_size-1)
            else:
                from_target = min((epoch//4)+ 1,batch_size)
            from_source = batch_size - from_target
            for _ in range(from_target):
                yield squeeze_first(tuple(pam(loaders, next(target_iter))))
            for _ in range(from_source):
                yield squeeze_first(tuple(pam(loaders, next(source_iter))))
        epoch+=1


def curriculum_load_by_gradual_id(*loaders: Callable, ids: Sequence, weights: Sequence[float] = None,
                                  random_state: Union[np.random.RandomState, int] = None,batches_per_epoch=100,batch_size=16,ts_size=2,keep_source=True,csv_path=None,start_epoch=None):
    assert start_epoch is not None
    print(f'curriculum_load_by_gradual_id start from {start_epoch}')
    source_ids = ids[:-ts_size]
    target_ids = ids[-ts_size:]
    target_iter = sample(target_ids, weights, random_state)
    df = pd.read_csv(csv_path)
    df = df[df['label'] == 0]
    df["id"] = df['id'].apply(lambda row: 'CC'+ str(row).zfill(4))
    df = df[df['id'].apply(lambda row: str(row) in source_ids)]
    df = df[df['slice_num'] < 256]
    df = df[df['slice_num'] > 35]
    amount_to_remove_every_epoch = df.shape[0] // 64
    cur_start_index = start_epoch * amount_to_remove_every_epoch
    epoch = start_epoch
    while True:
        gc.collect()
        for _ in range(batches_per_epoch):
            if keep_source:
                from_target = min((epoch//4)+ 1,batch_size-1)
            else:
                from_target = min((epoch//4)+ 1,batch_size)
            from_source = batch_size - from_target
            for _ in range(from_target):
                yield squeeze_first(tuple(pam(loaders, next(target_iter))))
            for _ in range(from_source):
                df_loc = np.random.randint(cur_start_index, df.shape[0])
                row = df.iloc[df_loc]
                id1,slc = row['id'],int(row['slice_num'])
                ret = squeeze_first(tuple(pam(loaders, id1)))
                ret = (x[...,slc] for x in ret)
                yield ret
        epoch+=1
        cur_start_index += amount_to_remove_every_epoch
        if cur_start_index > df.shape[0] * 0.7:
            cur_start_index = int(df.shape[0] * 0.7)



def load_half_from_test(*loaders: Callable, ids: Sequence, weights: Sequence[float] = None,
                        random_state: Union[np.random.RandomState, int] = None,batches_per_epoch=100,batch_size=16,ts_size=2):
    train_ids = ids[-ts_size:]
    test_ids = ids[:-ts_size]
    train_iter = sample(train_ids, weights, random_state)
    test_iter = sample(test_ids, weights, random_state)

    while True:

        for _ in range(batches_per_epoch):
            for _ in range(8):
                yield squeeze_first(tuple(pam(loaders, next(train_iter))))
            for _ in range(8):
                yield squeeze_first(tuple(pam(loaders, next(test_iter))))
