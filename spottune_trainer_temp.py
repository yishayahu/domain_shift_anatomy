import os

import dpipe.commands as commands
import numpy as np
import torch

from functools import partial
from dpipe.config import if_missing, lock_dir, run
from dpipe.layout import Flat
from dpipe.train import train, Checkpoints, Policy
from dpipe.train.logging import TBLogger, WANDBLogger
from dpipe.torch import save_model_state, load_model_state
from spottunet.torch.model import train_step, inference_step_spottune, train_step_spottune
from spottunet.utils import fix_seed, get_pred, sdice, skip_predict
from spottunet.metric import evaluate_individual_metrics_probably_with_ids, compute_metrics_probably_with_ids, aggregate_metric_probably_with_ids, compute_metrics_probably_with_ids_spottune, evaluate_individual_metrics_probably_with_ids_no_pred
from spottunet.split import one2one
from dpipe.dataset.wrappers import apply, cache_methods
from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri
from spottunet.paths import DATA_PATH, BASELINE_PATH
from dpipe.im.metrics import dice_score
from spottunet.batch_iter import slicewise, SPATIAL_DIMS, get_random_slice, sample_center_uniformly, extract_patch
from dpipe.predict import add_extract_dims, divisible_shape
from spottunet.torch.module.spottune_unet_layerwise import SpottuneUNet2D as UNet2D
from spottunet.torch.module.agent_net import resnet
from dpipe.train.policy import Schedule, TQDM
from dpipe.torch.functional import weighted_cross_entropy_with_logits
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, multiply
from dpipe.im.shape_utils import prepend_dims
from spottunet.torch.utils import load_model_state_fold_wise, modify_state_fn_spottune, freeze_model_spottune

exp_dir = '/home/dsi/shaya/dart_results/spottune/experiment_0'
log_path = os.path.join(exp_dir,'train_logs')
saved_model_path = os.path.join(exp_dir,'model.pth')
test_predictions_path = os.path.join(exp_dir,'test_predictions')
test_metrics_path = os.path.join(exp_dir,'test_metrics')
checkpoints_path = os.path.join(exp_dir,'checkpoints')
data_path = DATA_PATH

voxel_spacing = (1, 0.95, 0.95)

preprocessed_dataset = apply(Rescale3D(CC359(data_path), voxel_spacing), load_image=scale_mri)
dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
val_size = 2
n_add_ids = 1  # 1, 2, 3
pretrained = True

seed = 0xBadCafe
n_first_exclude = 0
n_exps = 30
split = one2one(dataset.df, val_size=val_size, n_add_ids=n_add_ids,
                train_on_add_only=pretrained, seed=seed)[n_first_exclude:n_exps]
layout = Flat(split)
train_ids = layout.get_ids('train',folder=exp_dir)
test_ids = layout.get_ids('test',folder=exp_dir)
val_ids = layout.get_ids('val',folder=exp_dir)

n_chans_in = 1
n_chans_out = 1

dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))

sdice_tolerance = 1

sdice_metric = lambda x, y, i: sdice(get_pred(x), get_pred(y), dataset.load_spacing(i), sdice_tolerance)
val_metrics = {'dice_score': partial(aggregate_metric_probably_with_ids, metric=dice_metric),
               'sdice_score': partial(aggregate_metric_probably_with_ids, metric=sdice_metric)}

n_filters = 16
architecture_main = UNet2D(n_chans_in=n_chans_in, n_chans_out=n_chans_out, n_filters_init=n_filters)
architecture_policy = resnet(num_class=64)

temperature = 0.1
use_gumbel_inference = False

@slicewise  # 3D -> 2D iteratively
@add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
@divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
def predict(image):
    return inference_step_spottune(image, architecture_main=architecture_main, architecture_policy=architecture_policy,
                                   activation=torch.sigmoid, temperature=temperature, use_gumbel=use_gumbel_inference)


val_predict = predict

load_x = dataset.load_image
load_y = dataset.load_segm

validate_step = partial(compute_metrics_probably_with_ids_spottune, predict=val_predict,
                        load_x=load_x, load_y=load_y, ids=val_ids, metrics=val_metrics,
                        architecture_main=architecture_main)

logger = WANDBLogger(project='spot3',dir=exp_dir,entity=None)


alpha_l2sp = None

reference_architecture = None
lr_init_main = 1e-3
lr_main = Schedule(initial=lr_init_main, epoch2value_multiplier={45: 0.1, })

lr_init_policy = 0.01
lr_policy = Schedule(initial=lr_init_policy, epoch2value_multiplier={45: 0.1, })

k_reg = 0.005

k_reg_source = None
reg_mode = 'l1'
with_source = False
optimizer_main = torch.optim.SGD(
    architecture_main.parameters(),
    lr=lr_init_main,
    momentum=0.9,
    nesterov=True,
    weight_decay=0
)

optimizer_policy = torch.optim.SGD(
    architecture_policy.parameters(),
    lr=lr_init_policy,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.001
)

criterion = weighted_cross_entropy_with_logits

train_kwargs = dict(lr_main=lr_main, lr_policy=lr_policy, k_reg=k_reg, k_reg_source=k_reg_source, reg_mode=reg_mode,
                    architecture_main=architecture_main, architecture_policy=architecture_policy,
                    temperature=temperature, with_source=with_source, optimizer_main=optimizer_main,
                    optimizer_policy=optimizer_policy, criterion=criterion, alpha_l2sp=alpha_l2sp)

checkpoints = Checkpoints(checkpoints_path, {
    **{k: v for k, v in train_kwargs.items() if isinstance(v, Policy)},
    'model_main.pth': architecture_main, 'model_policy.pth': architecture_policy,
    'optimizer_main.pth': optimizer_main, 'optimizer_policy.pth': optimizer_policy
})

ids_sampling_weights = None
slice_sampling_interval = 1  # 1, 3, 6, 12, 24, 36

def get_random_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size):
    sp_dims_2d = (-2, -1)
    center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    x, y = extract_patch((image_slc, segm_slc, center), x_patch_size, y_patch_size, spatial_dims=sp_dims_2d)
    return x, y


x_patch_size = y_patch_size = np.array([256, 256])
batch_size = 16

batches_per_epoch = 100
batch_iter = Infinite(
    load_by_random_id(dataset.load_image, dataset.load_segm, ids=train_ids,
                      weights=ids_sampling_weights, random_state=seed),
    unpack_args(get_random_slice, interval=slice_sampling_interval),
    unpack_args(get_random_patch_2d, x_patch_size=x_patch_size, y_patch_size=y_patch_size),
    multiply(prepend_dims),
    multiply(np.float32),
    batch_size=batch_size, batches_per_epoch=batches_per_epoch
)
n_epochs = 60

train_model = partial(train,
                      train_step=train_step_spottune,
                      batch_iter=batch_iter,
                      n_epochs=n_epochs,
                      logger=logger,
                      checkpoints=checkpoints,
                      validate=validate_step,
                      bar=TQDM(),
                      **train_kwargs
                      )

predict_to_dir = skip_predict

final_metrics = {'dice_score': dice_metric, 'sdice_score': sdice_metric}
evaluate_individual_metrics = partial(
    evaluate_individual_metrics_probably_with_ids_no_pred,
    load_y=load_y,
    load_x=load_x,
    predict=predict,
    metrics=final_metrics,
    test_ids=test_ids,
    logger=logger,
)
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

baseline_exp_path = BASELINE_PATH

saved_model_path_main = 'model_main.pth'
saved_model_path_policy = 'model_policy.pth'

run_experiment = run(
    fix_seed(seed=0xBAAAAAAD),
    lock_dir(),

    load_model_state_fold_wise(architecture=architecture_main, baseline_exp_path=baseline_exp_path,
                               modify_state_fn=modify_state_fn_spottune, n_folds=len(dataset.df.fold.unique()),
                               n_first_exclude=n_first_exclude),

    freeze_model_spottune(architecture_main),
    architecture_main.to(device),
    architecture_policy.to(device),

    if_missing(lambda p_main, p_policy: [train_model(), save_model_state(architecture_main, p_main),
                                         save_model_state(architecture_policy, p_policy)],
               saved_model_path_main, saved_model_path_policy),

    architecture_main.save_policy('policy_training_record'),

    load_model_state(architecture_main, saved_model_path_main),
    load_model_state(architecture_policy, saved_model_path_policy),

    if_missing(predict_to_dir, output_path=test_predictions_path),
    if_missing(evaluate_individual_metrics, results_path=test_metrics_path),

    architecture_main.save_policy('policy_inference_record')
)
debugging_mode = False
