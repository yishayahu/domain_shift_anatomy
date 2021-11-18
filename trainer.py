import argparse
import os
import shutil
from pathlib import Path

import dpipe.commands as commands
import numpy as np
import torch

from functools import partial

import yaml
from dpipe.config import if_missing, lock_dir, run
from dpipe.io import load

from dpipe.train import train, Checkpoints, Policy
from dpipe.train.logging import TBLogger, ConsoleLogger, WANDBLogger
from dpipe.torch import save_model_state, load_model_state, inference_step
from spottunet.torch.module.agent_net import resnet

from spottunet.torch.checkpointer import CheckpointsWithBest
from spottunet.torch.module.spottune_unet_layerwise import SpottuneUNet2D
from spottunet.torch.schedulers import CyclicScheduler, DecreasingOnPlateauOfVal
from spottunet.torch.fine_tune_policy import FineTunePolicy, DummyPolicy, FineTunePolicyUsingDist
from spottunet.torch.losses import FineRegularizedLoss
from spottunet.torch.model import train_step, inference_step_spottune, train_step_spottune
from spottunet.utils import fix_seed, get_pred, sdice, skip_predict
from spottunet.metric import evaluate_individual_metrics_probably_with_ids, compute_metrics_probably_with_ids, \
    aggregate_metric_probably_with_ids, evaluate_individual_metrics_probably_with_ids_no_pred, \
    compute_metrics_probably_with_ids_spottune
from spottunet.split import one2one
from dpipe.dataset.wrappers import apply, cache_methods
from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri
from spottunet.paths import DATA_PATH, BASELINE_PATH
from dpipe.im.metrics import dice_score
from spottunet.batch_iter import slicewise, SPATIAL_DIMS, get_random_slice, sample_center_uniformly, extract_patch
from dpipe.predict import add_extract_dims, divisible_shape
from spottunet.torch.module.unet import UNet2D
from dpipe.train.policy import Schedule, TQDM
from dpipe.torch.functional import weighted_cross_entropy_with_logits
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, multiply
from dpipe.im.shape_utils import prepend_dims
from spottunet.torch.utils import load_model_state_fold_wise, freeze_model, none_func, empty_dict_func, \
    load_by_gradual_id, freeze_model_spottune, modify_state_fn_spottune


class Config:
    def parse(self,raw):
        for k,v in raw.items():
            if type(v) == dict:
                curr_func = v.pop('FUNC')
                return_as_class = v.pop('as_class',False)
                assert curr_func in globals()
                for key,val in v.items():
                    if type(val) == str and val in globals():
                        v[key] = globals()[val]
                v = partial(globals()[curr_func],**v)
                if return_as_class:
                    v = v()
            elif v in globals():
                v = globals()[v]
            setattr(self,k,v)
    def __init__(self, raw):
        self._second_round = raw.pop('SECOND_ROUND') if 'SECOND_ROUND' in raw else {}
        self.parse(raw)

    def second_round(self):
        self.parse(self._second_round)

def get_random_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size):
    sp_dims_2d = (-2, -1)
    center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    x, y = extract_patch((image_slc, segm_slc, center), x_patch_size, y_patch_size, spatial_dims=sp_dims_2d)
    return x, y
if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument("--exp_name", default='debug')
    cli.add_argument("--config")
    cli.add_argument("--device", default='cpu')
    cli.add_argument("--source", default=0)
    cli.add_argument("--target", default=2)
    cli.add_argument("--base_res_dir", default='/home/dsi/shaya/spottune_results/')
    cli.add_argument("--base_split_dir", default='/home/dsi/shaya/data_splits/')
    cli.add_argument("--ts_size", default=2)
    opts = cli.parse_args()
    cfg_path = open(f"configs/Shaya_exp/{opts.config}.yml", "r")
    cfg = Config(yaml.safe_load(cfg_path))
    device = opts.device if torch.cuda.is_available() else 'cpu'
    ## define paths
    exp_dir = os.path.join(opts.base_res_dir,f'ts_size_{opts.ts_size}',f'source_{opts.source}_target_{opts.target}',opts.exp_name)
    Path(exp_dir).mkdir(parents=True,exist_ok=True)
    lock_dir(exp_dir)
    splits_dir =  os.path.join(opts.base_split_dir,f'ts_{opts.ts_size}',f'target_{opts.target}')
    log_path = os.path.join(exp_dir,'train_logs')
    saved_model_path = os.path.join(exp_dir,'model.pth')
    saved_model_path_policy = os.path.join(exp_dir,'model_policy.pth')
    test_predictions_path = os.path.join(exp_dir,'test_predictions')
    test_metrics_path = os.path.join(exp_dir,'test_metrics')
    best_test_metrics_path = os.path.join(exp_dir,'best_test_metrics')
    checkpoints_path = os.path.join(exp_dir,'checkpoints')
    data_path = DATA_PATH
    train_ids = load(os.path.join(splits_dir,'train_ids.json'))
    if getattr(cfg,'ADD_SOURCE_IDS',False):
        train_ids = load(os.path.join(opts.base_split_dir,'sources',f'source_{opts.source}','train_ids.json')) + train_ids
    # todo: add source to gradual
    test_ids = load(os.path.join(splits_dir,'test_ids.json'))
    val_ids = load(os.path.join(splits_dir,'val_ids.json'))
    base_ckpt_path = os.path.join(opts.base_split_dir,'sources',f'source_{opts.source}','model.pth')

    ## training params
    freeze_func = cfg.FREEZE_FUNC
    n_epochs = cfg.NUM_EPOCHS


    batches_per_epoch = getattr(cfg,'BATCHES_PER_EPOCH',100)
    spot = getattr(cfg,'SPOT',False)
    batch_size = 16
    lr_init = 1e-3
    project = f'spot_ts_{opts.ts_size}_s{opts.source}_t{opts.target}'


    if opts.exp_name == 'debug':
        print('debug mode')
        batches_per_epoch = 2
        batch_size = 2
        project = 'spot3'


    print(f'running {opts.exp_name}')


    voxel_spacing = (1, 0.95, 0.95)

    preprocessed_dataset = apply(Rescale3D(CC359(data_path), voxel_spacing), load_image=scale_mri)
    dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    seed = 0xBadCafe


    dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))
    sdice_tolerance = 1

    sdice_metric = lambda x, y, i: sdice(get_pred(x), get_pred(y), dataset.load_spacing(i), sdice_tolerance)
    val_metrics = {'dice_score': partial(aggregate_metric_probably_with_ids, metric=dice_metric),
                   'sdice_score': partial(aggregate_metric_probably_with_ids, metric=sdice_metric)}


    architecture = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16) if not spot else SpottuneUNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16)

    architecture.to(device)
    load_model_state_fold_wise(architecture=architecture, baseline_exp_path=base_ckpt_path,modify_state_fn=None if not spot else modify_state_fn_spottune)

    logger = WANDBLogger(project=project,dir=exp_dir,entity=None,run_name=opts.exp_name,config=cfg_path)

    optimizer = torch.optim.SGD(
        architecture.parameters(),
        lr=lr_init,
        momentum=0.9,
        nesterov=True,
        weight_decay=0
    )
    lr = getattr(cfg,'SCHDULER',Schedule(initial=lr_init, epoch2value_multiplier={45: 0.1, }))
    if type(lr) == partial:
        lr = lr()

    if spot:
        reference_architecture = None
    else:
        reference_architecture = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16)
        load_model_state_fold_wise(architecture=reference_architecture, baseline_exp_path=base_ckpt_path)
    cfg.second_round()
    sample_func = getattr(cfg,'SAMPLE_FUNC',load_by_random_id)
    training_policy = getattr(cfg,'TRAINING_POLICY',DummyPolicy())
    criterion = getattr(cfg,'CRITERION',weighted_cross_entropy_with_logits)



    if spot:
        architecture_policy = resnet(num_class=64)
        architecture_policy.to(device)
        temperature = 0.1
        use_gumbel_inference = False
        @slicewise  # 3D -> 2D iteratively
        @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
        @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
        def predict(image):
            return inference_step_spottune(image, architecture_main=architecture, architecture_policy=architecture_policy,
                                           activation=torch.sigmoid, temperature=temperature, use_gumbel=use_gumbel_inference,soft=cfg.SOFT)
        validate_step = partial(compute_metrics_probably_with_ids_spottune, predict=predict,
                                load_x=dataset.load_image, load_y=dataset.load_segm, ids=val_ids, metrics=val_metrics,
                                architecture_main=architecture)
        lr_init_policy = 0.01
        lr_policy = Schedule(initial=lr_init_policy, epoch2value_multiplier={45: 0.1, })
        optimizer_policy = torch.optim.SGD(
            architecture_policy.parameters(),
            lr=lr_init_policy,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.001
        )

        train_kwargs = dict(lr_main=lr, lr_policy=lr_policy, k_reg=0.005, k_reg_source=None, reg_mode='l1',
                            architecture_main=architecture, architecture_policy=architecture_policy,
                            temperature=temperature, with_source=False, optimizer_main=optimizer,
                            optimizer_policy=optimizer_policy, criterion=criterion,soft=cfg.SOFT)
        checkpoints = CheckpointsWithBest(checkpoints_path, {
            **{k: v for k, v in train_kwargs.items() if isinstance(v, Policy)},
            'model.pth': architecture, 'model_policy.pth': architecture_policy,
            'optimizer.pth': optimizer, 'optimizer_policy.pth': optimizer_policy
        })
        train_step_func = train_step_spottune

    else:
        @slicewise  # 3D -> 2D iteratively
        @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
        @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
        def predict(image):
            return inference_step(image, architecture=architecture, activation=torch.sigmoid)

        validate_step = partial(compute_metrics_probably_with_ids if opts.exp_name != 'debug' else empty_dict_func, predict=predict,
                                load_x=dataset.load_image, load_y=dataset.load_segm, ids=val_ids, metrics=val_metrics)
        lr_policy = None
        architecture_policy = None
        optimizer_policy = None
        train_kwargs = dict(architecture=architecture, optimizer=optimizer, criterion=criterion, reference_architecture=reference_architecture,train_step_logger=logger)
        if lr:
            train_kwargs['lr'] = lr

        checkpoints = CheckpointsWithBest(checkpoints_path, {
            **{k: v for k, v in train_kwargs.items() if isinstance(v, Policy)},
            'model.pth': architecture, 'optimizer.pth': optimizer
        })
        train_step_func = train_step





    ids_sampling_weights = None
    slice_sampling_interval = 1  # 1, 3, 6, 12, 24, 36, 48 todo: change to be configable




    x_patch_size = y_patch_size = np.array([256, 256])
    batch_iter = Infinite(
        sample_func(dataset.load_image, dataset.load_segm, ids=train_ids,
                          weights=ids_sampling_weights, random_state=seed),
        unpack_args(get_random_slice, interval=slice_sampling_interval),
        unpack_args(get_random_patch_2d, x_patch_size=x_patch_size, y_patch_size=y_patch_size),
        multiply(prepend_dims),
        multiply(np.float32),
        batch_size=batch_size, batches_per_epoch=batches_per_epoch
    )
    train_model = partial(train,
        train_step=train_step_func,
        batch_iter=batch_iter,
        n_epochs=n_epochs,
        logger=logger,
        checkpoints=checkpoints,
        validate=validate_step,
        bar=TQDM(),
        training_policy=training_policy,
        **train_kwargs
    )

    predict_to_dir = skip_predict
    final_metrics = {'dice_score': dice_metric, 'sdice_score': sdice_metric}
    evaluate_individual_metrics = partial(
        evaluate_individual_metrics_probably_with_ids_no_pred,
        load_y=dataset.load_segm,
        load_x=dataset.load_image,
        predict=predict,
        metrics=final_metrics,
        test_ids=test_ids,
        logger=logger
    )
    fix_seed(seed=0xBAAAAAAD)
    freeze_func(architecture)
    if spot:
        run_experiment = run(
            if_missing(lambda p_main, p_policy: [train_model(), save_model_state(architecture, p_main),
                                                 save_model_state(architecture_policy, p_policy)],
                       saved_model_path, saved_model_path_policy),
            architecture.save_policy('policy_training_record'),
            load_model_state(architecture, saved_model_path),
            load_model_state(architecture_policy, saved_model_path_policy),
            if_missing(predict_to_dir, output_path=test_predictions_path),
            if_missing(evaluate_individual_metrics, results_path=test_metrics_path),
            architecture.save_policy('policy_inference_record'),
            load_model_state(architecture, checkpoints.best_model_ckpt()),
            load_model_state(architecture_policy, checkpoints.best_policy_ckpt()),
            if_missing(partial(evaluate_individual_metrics,best='_best'), results_path=best_test_metrics_path),
        )

    else:
        run_experiment = run(
            if_missing(lambda p: [train_model(), save_model_state(architecture, p)], saved_model_path),
            load_model_state(architecture, saved_model_path),
            if_missing(predict_to_dir, output_path=test_predictions_path),
            if_missing(evaluate_individual_metrics, results_path=test_metrics_path),
            load_model_state(architecture, checkpoints.best_model_ckpt()),
            if_missing(partial(evaluate_individual_metrics,best='_best'), results_path=best_test_metrics_path),
        )

