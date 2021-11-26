import numpy as np
import skimage
import torch
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.io import load
from tqdm import tqdm
import matplotlib.pyplot as plt

from spottunet.paths import DATA_PATH

from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri
from torch import nn

from trainer import get_random_patch_2d

if torch.cuda.is_available():
    from cuml.manifold import TSNE
    from cuml.cluster  import DBSCAN
else:
    from sklearn.manifold import TSNE
    from sklearn.cluster  import DBSCAN


colors = ['b','g','r','c','m','y']
def get_model_embeddings(load_fn,model,ids,slices_indexes):
    pass

def get_seg_map_embeddings(ds,ids,slices_indexes):
    X = []
    for _id in tqdm(ids):
        current_img = ds.load_image(_id)
        current_seg = ds.load_segm(_id)

        for slice_idx in slices_indexes:
            _,seg_slice = get_random_patch_2d(current_img[...,slice_idx],current_seg[...,slice_idx],256,256)
            seg_slice = skimage.measure.block_reduce(seg_slice, (4,4), np.max)
            X.append(seg_slice)


    return  X


def get_embeddings(ids,slices_indexes,model=None):
    voxel_spacing = (1, 0.95, 0.95)
    preprocessed_dataset = apply(Rescale3D(CC359(DATA_PATH), voxel_spacing), load_image=scale_mri)
    dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    if model is None:
        X = get_seg_map_embeddings(ds=dataset,ids=ids,slices_indexes=slices_indexes)
    else:
        X = get_model_embeddings(load_fn=dataset.load_image,model=model,ids=ids,slices_indexes=slices_indexes)
    X = np.stack(X,axis=0)
    X = X.reshape((X.shape[0],-1))
    return X

def reduce_dimenstion(X):
    print('4')
    return TSNE(n_components=2,init='pca',learning_rate='auto').fit_transform(X)

def cluster_embeddings(X):
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(X)
    return labels

def plot_embeddings(X,labels,train_size):
    print('5')
    labels = [colors[l] for l in labels]
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    labels_train = labels[:train_size]
    labels_test = labels[train_size:]
    plt.scatter(X_train[:,0], X_train[:,1], c=labels_train, marker='^')
    plt.scatter(X_test[:,0], X_test[:,1], c=labels_test, marker='o')
    plt.savefig('ttt.png')
def main():
    train_ids = load('/home/dsi/shaya/data_splits/ts_2/target_2/train_ids.json')
    test_ids = load('/home/dsi/shaya/data_splits/ts_2/target_2/test_ids.json')
    slices_indexes = np.random.permutation(np.arange(150))[:3]
    print('1')
    X = get_embeddings(ids=train_ids+test_ids,slices_indexes=slices_indexes)
    print('2')
    labels = cluster_embeddings(X)
    print('3')
    plot_embeddings(reduce_dimenstion(X),labels,train_size=len(train_ids)*len(slices_indexes))

if __name__ == '__main__':
    main()

