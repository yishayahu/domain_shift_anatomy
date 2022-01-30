import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, multiply
from dpipe.dataset.wrappers import cache_methods, apply
from dpipe.im.shape_utils import prepend_dims
from dpipe.io import load
from dpipe.predict import add_extract_dims, divisible_shape
from dpipe.torch import inference_step, weighted_cross_entropy_with_logits, sequence_to_var
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from spottunet.batch_iter import slicewise, SPATIAL_DIMS, get_random_slice, sample_center_uniformly, extract_patch

from spottunet.batch_iter import get_random_slice, slicewise, SPATIAL_DIMS

from spottunet.paths import DATA_PATH, msm_splits_dir, msm_res_dir, st_splits_dir, st_res_dir  # , st_splits_dir
from functools import partial
from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri
from dpipe.im.metrics import dice_score

from spottunet.utils import fix_seed, get_pred, sdice, skip_predict

from spottunet.torch.module.unet import UNet2D
from dpipe.torch import save_model_state, load_model_state
from spottunet.metric import evaluate_individual_metrics_probably_with_ids, compute_metrics_probably_with_ids, \
    aggregate_metric_probably_with_ids, evaluate_individual_metrics_probably_with_ids_no_pred


def get_random_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size):
    sp_dims_2d = (-2, -1)
    center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    x, y = extract_patch((image_slc, segm_slc, center), x_patch_size, y_patch_size, spatial_dims=sp_dims_2d)
    return x, y


def run_adaBN(source, target, device, base_res_dir, base_split_dir, metric):
    data_path = DATA_PATH
    # define and load model
    model = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to(device)
    src_ckpt_path = f'{base_split_dir}/site_{source}/model_sgd.pth'
    load_model_state(model, src_ckpt_path)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    splits_dir = os.path.join(base_split_dir, f'site_{target}')
    voxel_spacing = (1, 0.95, 0.95)
    train_ids = load(os.path.join(splits_dir, 'train_ids.json'))
    test_ids = load(os.path.join(splits_dir, 'test_ids.json'))

    preprocessed_dataset = apply(Rescale3D(CC359(data_path), voxel_spacing), load_image=scale_mri)
    dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    x_patch_size = y_patch_size = np.array([256, 256])
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_segm, ids=train_ids,
                          weights=None, random_state=42),
        unpack_args(get_random_slice, interval=1),
        unpack_args(get_random_patch_2d, x_patch_size=x_patch_size, y_patch_size=y_patch_size),
        multiply(prepend_dims),
        multiply(np.float32),
        batch_size=16, batches_per_epoch=80  # change batch-size if needed
    )
    iter1 = batch_iter()
    model.train()
    for batch_slices in tqdm(iter1):
        with torch.no_grad():
            batch_inp, batch_seg = sequence_to_var(*[batch_slices[0], batch_slices[1]], device=model)
            out = model(batch_inp)
            loss1 = weighted_cross_entropy_with_logits(out, batch_seg)
            print(loss1)
    dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))
    model.eval()
    sdice_tolerance = 1

    @slicewise  # 3D -> 2D iteratively
    @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
    @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
    def predict(image):
        return inference_step(image, architecture=model, activation=torch.sigmoid)

    load_x = dataset.load_image
    load_y = dataset.load_segm
    sdice_metric = lambda x, y, i: sdice(get_pred(x), get_pred(y), dataset.load_spacing(i), sdice_tolerance)
    final_metrics = {'dice_score': dice_metric, 'sdice_score': sdice_metric}
    evaluate_individual_metrics = partial(
        evaluate_individual_metrics_probably_with_ids_no_pred,
        load_y=load_y,
        load_x=load_x,
        predict=predict,
        metrics=final_metrics,
        test_ids=test_ids
    )
    p1 = Path(f'{base_res_dir}/source_{source}_target_{target}/adaBN/best_test_metrics/')
    p1.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),p1/'model.pth')
    evaluate_individual_metrics(results_path=p1,exist_ok=True),


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--device")
    cli.add_argument("--source")
    cli.add_argument("--target")
    cli.add_argument("--msm", default=False)
    opts = cli.parse_args()
    fix_seed(42)
    base_res_dir = st_res_dir if not opts.msm else msm_res_dir
    base_split_dir = st_splits_dir if not opts.msm else msm_splits_dir
    metric = 'sdice_score' if not opts.msm else 'dice'
    run_adaBN(source=opts.source, target=opts.target, device=opts.device, base_res_dir=base_res_dir,
              base_split_dir=base_split_dir, metric=metric)
    if opts.msm:
        raise NotImplemented()


if __name__ == '__main__':
    main()

