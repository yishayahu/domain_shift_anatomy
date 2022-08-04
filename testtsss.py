import json
import os
from functools import partial

import numpy as np
import torch.cuda
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.io import load
from dpipe.predict import add_extract_dims, divisible_shape
from dpipe.torch import load_model_state, inference_step
from spottunet.torch.module.agent_net import resnet

from spottunet.torch.model import inference_step_spottune

from spottunet.torch.module.spottune_unet_layerwise import SpottuneUNet2D
from tqdm import tqdm

from spottunet.batch_iter import slicewise, SPATIAL_DIMS
from spottunet.dataset.cc359 import scale_mri, CC359, Rescale3D
from spottunet.metric import evaluate_individual_metrics_probably_with_ids_no_pred
from spottunet.paths import DATA_PATH, st_splits_dir
from spottunet.torch.module.unet import UNet2D
from spottunet.utils import sdice, get_pred


def calc_stats(model_path,site,spot=False):
    data_path = DATA_PATH
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    n_chans_in = 1
    sdice_tolerance = 1
    if spot:
        model = SpottuneUNet2D(n_chans_in=n_chans_in, n_chans_out=1, n_filters_init=16)
        architecture_policy = resnet(num_class=64,in_chans=n_chans_in)
        architecture_policy.load_state_dict(torch.load(model_path.replace('model.pth','model_policy.pth')))
        @slicewise  # 3D -> 2D iteratively
        @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
        @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
        def predict(image):
            return inference_step_spottune(image, architecture_main=model, architecture_policy=architecture_policy,
                                           activation=torch.sigmoid, temperature=0.1, use_gumbel=False,soft=False)
    else:
        model = UNet2D(n_chans_in=n_chans_in, n_chans_out=1, n_filters_init=16).to(device)

        @slicewise  # 3D -> 2D iteratively
        @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
        @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
        def predict(image):
            return inference_step(image, architecture=model, activation=torch.sigmoid)
    load_model_state(model, model_path)
    splits_dir = os.path.join(st_splits_dir, f'sources',f'source_{site}')
    voxel_spacing = (1, 0.95, 0.95)

    val_ids = load(os.path.join(splits_dir, 'val_ids.json'))
    model.eval()

    preprocessed_dataset = apply(Rescale3D(CC359(data_path), voxel_spacing), load_image=scale_mri)
    dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    load_x = dataset.load_image
    load_y = dataset.load_segm
    sdice_metric = lambda x, y, i: sdice(get_pred(x), get_pred(y), dataset.load_spacing(i), sdice_tolerance)

    evaluate_individual_metrics = partial(
        evaluate_individual_metrics_probably_with_ids_no_pred,
        load_y=load_y,
        load_x=load_x,
        predict=predict,
        metrics={'sdice':sdice_metric},
        test_ids=val_ids
    )
    evaluate_individual_metrics(results_path='aaaaaaa',exist_ok=True)
    return np.mean(list(load('aaaaaaa/sdice.json').values()))
def main():
    numnum = 1
    if numnum == 1:
        methods = ['spottune','posttrain']
    elif numnum == 2:
        methods =['posttrain_continue_optimizer','gradual_tl']
    else:
        methods =['unfreeze_first','l2sp']
    dd = {}
    sizes = [0,1,2,4]
    bar = tqdm(total=len(methods)*len(sizes)* 31)
    print(methods)
    for s in range(6):
        src_ckpt_path = f'/home/dsi/shaya/spottune_results/source_{s}/only_source_sgd/model.pth'
        src_res = calc_stats(src_ckpt_path,s)
        bar.update(1)
        for t in range(6):
            if s ==t:
                continue
            for method in methods:
                if method not in dd:
                    dd[method] = {}
                for size in sizes:
                    if size not in dd[method]:
                        dd[method][size] = {}
                    tgt_ckpt_path = f'/home/dsi/shaya/spottune_results/ts_size_{size}/source_{s}_target_{t}/{method}/model.pth'
                    if not os.path.exists(tgt_ckpt_path):
                        continue
                    try:
                        tgt_res = calc_stats(tgt_ckpt_path,s,spot=method=='spottune')

                        dd[method][size][f'{s}_{t}'] = (src_res,tgt_res)
                        bar.update(1)
                        json.dump(dd,open(f'dd_dd_{numnum}.json','w'))
                    except:
                        print(f'error in metohd {method} size {size} source {s} target {t}')

if __name__ == '__main__':
    main()





#
# import json
# import os
# from functools import partial
#
# import numpy as np
# import torch.cuda
# from dpipe.dataset.wrappers import apply, cache_methods
# from dpipe.io import load
# from dpipe.predict import add_extract_dims, divisible_shape
# from dpipe.torch import load_model_state, inference_step
# from tqdm import tqdm
#
# from spottunet.batch_iter import slicewise, SPATIAL_DIMS
# from spottunet.dataset.cc359 import scale_mri, CC359, Rescale3D
# from spottunet.metric import evaluate_individual_metrics_probably_with_ids_no_pred
# from spottunet.msm_utils import ComputeMetricsMsm
# from spottunet.paths import DATA_PATH, st_splits_dir, MSM_DATA_PATH, msm_splits_dir
# from spottunet.torch.module.unet import UNet2D
# from spottunet.utils import sdice, get_pred
#
#
# def calc_stats(model_path,site):
#
#     device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
#     n_chans_in = 3
#     model = UNet2D(n_chans_in=n_chans_in, n_chans_out=1, n_filters_init=16).to(device)
#     load_model_state(model, model_path)
#     splits_dir = os.path.join(msm_splits_dir, f'sources',f'source_{site}')
#     voxel_spacing = (1, 0.95, 0.95)
#
#     val_ids = load(os.path.join(splits_dir, 'val_ids.json'))
#     model.eval()
#
#     def predict(image):
#         res =  inference_step(image, architecture=model, activation=torch.sigmoid)
#         return res
#     msm_metrics_computer = ComputeMetricsMsm(val_ids=val_ids,test_ids=val_ids,logger=None)
#     msm_metrics_computer.predict = predict
#
#     evaluate_individual_metrics = partial(msm_metrics_computer.test_metrices)
#     evaluate_individual_metrics(results_path='aaaaaaa')
#     return load('aaaaaaa/dice.json')
# def main():
#     for size in [1,2,4]:
#         print(size)
#         dd = {}
#         bar = tqdm(total=12)
#         for s in range(6):
#             src_ckpt_path = f'/home/dsi/shaya/data_split_msm2/sources/source_{0}/model_adam.pth'
#             src_res = calc_stats(src_ckpt_path,s)
#             bar.update(1)
#             for t in range(6):
#                 if s !=t:
#                     continue
#                 tgt_ckpt_path = f'/home/dsi/shaya/msm_results/ts_size_{size}/source_{s}_target_{t}/gradual_tl_msm_adam/model.pth'
#                 if not os.path.exists(tgt_ckpt_path):
#                     continue
#                 tgt_res = calc_stats(tgt_ckpt_path,s)
#                 dd[f'{s}_{t}'] = (src_res,tgt_res)
#                 bar.update(1)
#         print(np.mean([x[0]for x in dd.values()]))
#         print(np.mean([x[1]for x in dd.values()]))
#         json.dump(dd,open(f'dd_22{size}.json','w'))
#
#
# if __name__ == '__main__':
#     main()
#
