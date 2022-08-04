import os
import pickle

import numpy as np
import torch
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args,multiply
from dpipe.dataset.wrappers import cache_methods,apply
from dpipe.im.shape_utils import prepend_dims
from dpipe.io import load
from dpipe.predict import add_extract_dims, divisible_shape
from dpipe.torch import inference_step, weighted_cross_entropy_with_logits, sequence_to_var
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from spottunet.batch_iter import get_random_slice, slicewise, SPATIAL_DIMS

from spottunet.paths import DATA_PATH, st_splits_dir

from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri

from spottunet.torch.utils import load_model_state_fold_wise

from spottunet.torch.module.unet import UNet2D
from trainer import get_random_patch_2d
device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
data_path = DATA_PATH
base_model = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to(device)
my_model = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to(device)

ts_size =2
target = 1
print('1')
load_model_state_fold_wise(architecture=base_model, baseline_exp_path='/home/dsi/shaya/spottune_results/ts_size_2/source_0_target_1/posttrain_adam/model.pth')
load_model_state_fold_wise(architecture=my_model, baseline_exp_path='/home/dsi/shaya/spottune_results/ts_size_2/source_0_target_1/gradual_tl_adam/model.pth')
print('2')
voxel_spacing = (1, 0.95, 0.95)
test_ids = load(os.path.join(os.path.join(st_splits_dir,f'ts_{ts_size}',f'target_{target}'),'test_ids.json'))
preprocessed_dataset = apply(Rescale3D(CC359(data_path), voxel_spacing), load_image=scale_mri)
dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
print('3')
x_patch_size = y_patch_size = np.array([256, 256])
batch_iter = Infinite(
    load_by_random_id(dataset.load_image, dataset.load_segm, ids=test_ids,
                weights=None, random_state=42),
    unpack_args(get_random_slice, interval=1),
    unpack_args(get_random_patch_2d, x_patch_size=x_patch_size, y_patch_size=y_patch_size),
    multiply(prepend_dims),
    multiply(np.float32),
    batch_size=1000, batches_per_epoch=100
)
chosen_params1 = np.random.choice(128,512)
chosen_params2 = np.random.choice(128,512)
chosen_params3 = np.random.choice(3,512)
chosen_params4 = np.random.choice(3,512)
all_params = (list(range(512)),chosen_params1,chosen_params2,chosen_params3,chosen_params4)
print('4')
iter1 = batch_iter()

if not os.path.exists('slices.p'):
    slices = []
    segs = []
    for slice1 in tqdm(iter1):
        slices.append(slice1[0])
        segs.append(slice1[1])
        if len(slices)  == 10: break
    slices = np.stack(slices)
    segs = np.stack(segs)
    pickle.dump(slices,open('slices.p','wb'))
    pickle.dump(segs,open('segs.p','wb'))
else:
    slices = pickle.load(open('slices.p','rb'))
    segs = pickle.load(open('segs.p','rb'))
if not os.path.exists('a.p'):
    p_to_mean = torch.zeros((2,512,10))

    for i,model_pred in enumerate([my_model,base_model]):
        for j,(batch_slice,batch_seg) in tqdm(enumerate(zip(slices,segs))):
            model_pred.zero_grad()
            batch_slice,batch_seg = sequence_to_var(*[batch_slice,batch_seg], device=model_pred)

            out = model_pred(batch_slice)
            loss1 = weighted_cross_entropy_with_logits(out,batch_seg)
            loss1.backward()

            for n,p in model_pred.named_parameters():

                if 'bottleneck.3.conv_path.1.layer.weight' in n:
                    for k,k1,k2,k3,k4 in zip(*all_params):

                        p_to_mean[i,k,j] = p.grad[k1][k2][k3][k4]


    a = torch.min(p_to_mean,dim=2).values
    b = torch.max(p_to_mean,dim=2).values
    pickle.dump(a,open('a.p','wb'))
    pickle.dump(b,open('b.p','wb'))
else:
    a = pickle.load(open('a.p','rb'))
    b= pickle.load(open('b.p','rb'))

fig = plt.figure()
rrrr = 100

all_grad = np.random.uniform(low=a[0],high=b[0],size=(rrrr,512))
all_grad[0] = np.zeros(512)
t = TSNE(n_components=2,init='random',learning_rate='auto')
grads = t.fit_transform(np.array(all_grad))
all_grad = torch.tensor(all_grad,device=device)
to_plot = []
with torch.no_grad():
    for i,model_pred in enumerate(['posttrain_adam','gradual_tl_adam']):
        lr = 0.01
        dd = {}
        curr_model = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to(device)
        all_losses = []
        for kk in tqdm(range(rrrr)):
            # if kk % 10 ==0:
            #     lr /= 3
            #     if lr < 0.00001:
            #         lr = 0.00001
            load_model_state_fold_wise(architecture=curr_model, baseline_exp_path=f'/home/dsi/shaya/spottune_results/ts_size_2/source_0_target_1/{model_pred}/model.pth')
            grad = all_grad[kk] *lr

            for n,p in curr_model.named_parameters():

                if 'bottleneck.3.conv_path.1.layer.weight' in n:
                    for k,k1,k2,k3,k4 in zip(*all_params):

                        p[k1][k2][k3][k4] += grad[k]



            losses = []
            for j,(batch_slice,batch_seg) in enumerate(zip(slices,segs)):

                batch_slice,batch_seg = sequence_to_var(*[batch_slice,batch_seg], device=curr_model)

                out = curr_model(batch_slice)
                loss1 = weighted_cross_entropy_with_logits(out,batch_seg)
                losses.append(loss1.item())

            all_losses.append(np.mean(losses))

        all_losses =np.array(all_losses)
        q = np.quantile(all_losses,0.7)
        idxs = all_losses<q
        all_losses = all_losses[idxs]
        curr_grads_to_use = grads[idxs]
        to_plot.append((curr_grads_to_use[:,0], curr_grads_to_use[:,1], all_losses))


    max_val = max(np.max(to_plot[0][2]),np.max(to_plot[1][2]))
    min_val = min(np.min(to_plot[0][2]),np.min(to_plot[1][2]))
    for i,(x,y,z) in enumerate(to_plot):
        max_val = max(np.max(to_plot[i][2]),np.max(to_plot[i][2]))
        min_val = min(np.min(to_plot[i][2]),np.min(to_plot[i][2]))
        z = z -min_val
        z/= (max_val - min_val)

        ax = fig.add_subplot(int(f'22{i+1}'), projection='3d')
        ax.scatter(x,y,z, cmap='viridis')
        # ax.set_title(model_pred)
        ax = fig.add_subplot(int(f'22{i+3}'), projection='3d')
        ax.plot_trisurf(x,y,z, cmap='viridis')
        ax.scatter(x[0],y[0],marker='P')

        # ax.set_title(model_pred)

plt.savefig('tttttt.png')



