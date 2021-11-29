import argparse
import os
import pickle
from functools import partial

import numpy
import numpy as np
import skimage
import torch
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.io import load
from dpipe.torch import load_model_state
from matplotlib import cm


from spottunet.torch.module.unet import UNet2D

from spottunet.batch_iter import sample_center_uniformly, extract_patch
from tqdm import tqdm
import matplotlib.pyplot as plt

from spottunet.paths import DATA_PATH

from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri
from torch import nn

def get_random_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size):
    sp_dims_2d = (-2, -1)
    center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    x, y = extract_patch((image_slc, segm_slc, center), x_patch_size, y_patch_size, spatial_dims=sp_dims_2d)
    return x, y


from tsnecuda import TSNE as TSNECUDA
from sklearn.manifold import TSNE
from sklearn.cluster  import DBSCAN


def get_model_embeddings(ds,model_runner,ids,slices_indexes):
    X = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_runner = model_runner.to(device)
    with torch.no_grad():
        for _id in tqdm(ids):
            current_img = ds.load_image(_id)
            current_seg = ds.load_segm(_id)
            slices = []
            for slice_idx in slices_indexes:
                img_slice,_ = get_random_patch_2d(current_img[...,slice_idx],current_seg[...,slice_idx],256,256)
                img_slice = np.expand_dims(img_slice, axis=0)

                slices.append(img_slice)
            img_slice = model_runner(torch.tensor(np.stack(slices,axis=0),device=device))
            img_slice = nn.MaxPool2d(2)(img_slice)
            X.append(img_slice.cpu().numpy())
    return  X



def get_embeddings(ids,slices_indexes,model_runner,fig_name):
    pickle_file_name = f'{fig_name}.p'
    if os.path.exists(pickle_file_name):
        return pickle.load(open(pickle_file_name,'rb'))
    voxel_spacing = (1, 0.95, 0.95)
    preprocessed_dataset = apply(Rescale3D(CC359(DATA_PATH), voxel_spacing), load_image=scale_mri)
    dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)

    X = get_model_embeddings(ds=dataset,model_runner=model_runner,ids=ids,slices_indexes=slices_indexes)
    X = np.concatenate(X,axis=0)
    pickle.dump(X,open(pickle_file_name,'wb'))
    return X

def reduce_dim(X,fig_name):
    pickle_file_name = f'{fig_name}_reduced.p'
    if os.path.exists(pickle_file_name):
        return pickle.load(open(pickle_file_name,'rb'))
    X_reduced = []
    for i in tqdm(range(16),desc='running on i'):
        for j in range(16):
            if torch.cuda.is_available():
                t = TSNECUDA(n_components=2)
            else:
                t = TSNE(n_components=2)
            X_reduced.append(t.fit_transform(X[:,:,i,j]))
    X_reduced = np.concatenate(X_reduced,axis=1)
    pickle.dump(X_reduced,open(pickle_file_name,'wb'))
    return X_reduced

def cluster_embeddings(X):
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(X)
    return labels

def plot_embeddings(X,labels,train_size,fig_name):
    print('5')
    colors = cm.rainbow(np.linspace(0, 1, max(labels)+2))
    labels = [colors[l] for l in labels]
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    labels_train = labels[:train_size]
    labels_test = labels[train_size:]
    plt.scatter(X_train[:,0], X_train[:,1], c=labels_train, marker='^')
    plt.scatter(X_test[:,0], X_test[:,1], c=labels_test, marker='o')
    plt.savefig(f'{fig_name}.png')

def create_model_runner(state_dict_path):
    if state_dict_path:
        model = UNet2D(1,1,16,get_bottleneck=True)
        load_model_state(model, state_dict_path)
        model.eval()

        return model
    return None
def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--model_path", default='/home/dsi/shaya/spottune_results/ts_size_2/source_0_target_2/base/checkpoints/checkpoint_59/model.pth')
    cli.add_argument("--fig_name", default='no_fig_name')
    opts = cli.parse_args()
    model_runner = create_model_runner(opts.model_path)
    train_ids = load('/home/dsi/shaya/data_splits/ts_2/target_2/train_ids.json')
    test_ids = load('/home/dsi/shaya/data_splits/ts_2/target_2/test_ids.json')
    slices_indexes = np.random.permutation(np.arange(150))[:25]
    print('1')
    X = get_embeddings(ids=train_ids+test_ids,slices_indexes=slices_indexes,model_runner=model_runner,fig_name=opts.fig_name)
    print('1.1')
    X = reduce_dim(X,opts.fig_name)
    print('1.2')
    X = TSNE(n_components=2).fit_transform(X)
    print('2')
    labels = cluster_embeddings(X)
    print('2.1')
    print('3')
    plot_embeddings(X,labels,train_size=len(train_ids)*len(slices_indexes),fig_name=opts.fig_name)

if __name__ == '__main__':
    main()

