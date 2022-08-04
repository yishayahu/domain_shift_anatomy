import os
import pickle

import numpy as np
import torch
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args,multiply
from dpipe.dataset.wrappers import cache_methods,apply
from dpipe.im.shape_utils import prepend_dims
from dpipe.io import load
from dpipe.itertools import squeeze_first, pam
from dpipe.predict import add_extract_dims, divisible_shape
from dpipe.torch import inference_step, weighted_cross_entropy_with_logits, sequence_to_var
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.patches as mpatches

from spottunet.batch_iter import get_random_slice, slicewise, SPATIAL_DIMS

from spottunet.paths import DATA_PATH, st_splits_dir

from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri

from spottunet.torch.utils import load_model_state_fold_wise

from spottunet.torch.module.unet import UNet2D
from trainer import get_random_patch_2d
device = 'cuda:3'
qqq1 = []
qqq2 = []
for posttrain in ['posttrain','gradual_tl']:#,'posttrain_continue_optimizer']:
    print(f'working on {posttrain}')
    app = 'tttttttt'
    # posttrain = ''

    device =  device if torch.cuda.is_available() else 'cpu'
    import matplotlib as mpl
    data_path = DATA_PATH
    # define and load model
    def colorFader(c1='green',c2='red',mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    class SDE:
        def __init__(self):
            self.path = '/home/dsi/shaya/data_splits/sources/source_1/'
            self.name = 'base'
    ckpts = [x for x in os.scandir(f'/home/dsi/shaya/spottune_results/ts_size_2/source_1_target_5/{posttrain}/checkpoints/')]
    ckpts = sorted(ckpts,key=lambda x:60 if 'best' in x.name else int(x.name.split('_')[1]))
    ckpts[0] = SDE()
    fix,axes = plt.subplots(2,3)
    axes = axes.flatten()
    voxel_spacing = (1, 0.95, 0.95)
    train_ids = load('/home/dsi/shaya/data_splits/ts_2/target_5/train_ids.json')
    test_ids = load('/home/dsi/shaya/data_splits/ts_2/target_5/test_ids.json')
    source_ids = load('/home/dsi/shaya/data_splits/sources/source_1/train_ids.json')
    preprocessed_dataset = apply(Rescale3D(CC359(data_path), voxel_spacing), load_image=scale_mri)
    dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    x_patch_size = y_patch_size = np.array([256, 256])
    class loadByOrder:
        def __init__(self,*loaders):
            self.loaders = loaders
            self.id = None
        def gen(self):
            while True:
                if self.id is not None:
                    yield squeeze_first(tuple(pam(self.loaders, id1)))
                else:
                    yield self.id


    def dist_func(x1,x2):
        return abs(x1[0]-x2[0]) + abs(x1[1]-x2[1])

    class getSlice:
        def __init__(self):
            self.current_slice = 120
        def __call__(self,*arrays):

            to_ret = tuple(array[..., self.current_slice] for array in arrays)
            self.current_slice+=1
            if self.current_slice == 180:
                self.current_slice = 120
            return to_ret
        def restart(self):
            self.current_slice = 120
    ll = loadByOrder(dataset.load_image, dataset.load_segm)
    gs = getSlice()

    for i,ckpt in enumerate(ckpts[1:7]):
        ppp_arrs = f'{ckpt.name}_{posttrain}_arrs.p'
        ppp_losses = f'{ckpt.name}_{posttrain}_losses.p'
        if not os.path.exists(ppp_arrs):

            model = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16,get_bottleneck=True).to(device)
            print(ckpt.name)
            load_model_state_fold_wise(architecture=model, baseline_exp_path=os.path.join(ckpt.path,'model.pth'))

            # create data iterator


            id_to_params = {}
            id_to_losses = {}
            for id1 in tqdm(train_ids+test_ids+source_ids):
                id_to_params[id1] = []
                id_to_losses[id1] = []
                ll.id = id1
                gs.restart()
                batch_iter = Infinite(
                    ll.gen(),
                    unpack_args(gs),
                    unpack_args(get_random_patch_2d, x_patch_size=x_patch_size, y_patch_size=y_patch_size),
                    multiply(prepend_dims),
                    multiply(np.float32),
                    batch_size=15, batches_per_epoch=4 # change batch-size if needed
                )

                with batch_iter as iterator:

                    with torch.no_grad():
                        for batch_slices in iterator():
                            model.zero_grad()
                            batch_inp,batch_seg = sequence_to_var(*[batch_slices[0],batch_slices[1]], device=model)
                            out,out_bottleneck = model(batch_inp)
                            loss1 = weighted_cross_entropy_with_logits(out,batch_seg)
                            out_bottleneck= out_bottleneck.flatten(2)
                            out_bottleneck = torch.mean(out_bottleneck,dim=1)
                            id_to_params[id1].append(out_bottleneck.cpu())
                            id_to_losses[id1].append(loss1.cpu().item())
                    id_to_params[id1] = torch.mean(torch.cat(id_to_params[id1],dim=0),dim=0).to('cpu')
                    id_to_losses[id1] = np.mean(id_to_losses[id1])



            losses= np.array(list(id_to_losses.values()))




            t = TSNE(n_components=2,learning_rate='auto',init='pca',perplexity=10)

            arrs = t.fit_transform(np.stack(list(id_to_params.values())))
            pickle.dump(arrs,open(ppp_arrs,'wb'))
            pickle.dump(losses,open(ppp_losses,'wb'))
        else:
            arrs = pickle.load(open(ppp_arrs,'rb'))
            losses = pickle.load(open(ppp_losses,'rb'))

        ax = axes[i]
        # losses = (losses - min(losses)) / (max(losses) - min(losses))
        ax.set_ylim(-50,50)
        ax.set_xlim(-50,50)
        cc= colorFader
        c = []
        for j in range(losses.shape[0]):
            c.append(cc(mix=losses[j]))

        m = ['.','.','.']
        c = ['green','red','blue']
        sizes = [32,4,4]
        dists = [min(dist_func(x,arrs[0]),dist_func(x,arrs[1])) for x in arrs[len(train_ids):len(train_ids)+len(test_ids)]]
        llll1 = []
        llll2 = []
        for qwq in range(len(dists)):
            if dists[qwq] > 20:
                llll1.append(losses[qwq])
            else:
                llll2.append(losses[qwq])
        print('asfafsafsa')
        print(np.mean(llll1))
        print(np.mean(llll2))
        qqq1.append(np.mean(llll1))
        qqq2.append(np.mean(llll2))
        epoch = int(ckpt.name.split('_')[1])
        for tete,(s,e) in enumerate(zip([0,len(train_ids),len(test_ids)+len(train_ids)],[len(train_ids),len(test_ids)+len(train_ids),len(losses)])):
            if epoch ==50:
                green_patch = mpatches.Patch(color='green', label='target site - train')
                red_patch = mpatches.Patch(color='red', label='target site - test')
                blue_patch = mpatches.Patch(color='blue', label='source site')

                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
                          fancybox=True, shadow=True, ncol=5,handles=[green_patch,red_patch,blue_patch])
            ax.scatter(arrs[s:e,0],arrs[s:e,1],c=c.pop(0),marker=m.pop(0),s=sizes.pop(0))
            # if c and c[0]== 'blue':
            #     mm = []#np.zeros((len(dists),2))
            #     for rere in range(len(dists)):
            #         if dists[rere]  > 24:
            #             mm.append([arrs[rere,0],arrs[rere,1]])
            #     mm = np.array(mm)
            # else:
            #     mm =  arrs[s:e]
            # ax.add_patch(plt.Circle((float(np.mean(mm[:,0])), float(np.mean(mm[:,1]))), 12, color='black', fill=False))


        ax.set_title(f'epoch {epoch}')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # green_patch = mpatches.Patch(color='green', label='target site - train')
    # red_patch = mpatches.Patch(color='red', label='target site - test')
    # blue_patch = mpatches.Patch(color='blue', label='source site')
    # plt.legend(handles=[green_patch,red_patch,blue_patch],loc='upper left')
    plt.savefig(f'{app}_{posttrain}.png')
    plt.show()



