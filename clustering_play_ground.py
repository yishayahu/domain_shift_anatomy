import json
import pickle
import random
from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.im import crop_to_box
from dpipe.im.box import get_centered_box
from sklearn.cluster import KMeans, AgglomerativeClustering, FeatureAgglomeration, MeanShift
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, NMF
from sklearn.manifold import Isomap, TSNE
from tqdm import tqdm

from spottunet.utils import sdice

from spottunet.batch_iter import sample_center_uniformly, SPATIAL_DIMS

from spottunet.paths import DATA_PATH
from spottunet.torch.model import get_best_match
from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
def extract_patch(inputs, y_patch_size, spatial_dims=SPATIAL_DIMS):
    y, center = inputs
    y_patch_size = np.array(y_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)
    y_patch = crop_to_box(y, box=y_spatial_box, padding_values=0, axis=spatial_dims)
    return y_patch
def get_random_patch_2d(segm_slc, y_patch_size):
    sp_dims_2d = (-2, -1)
    center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    y = extract_patch((segm_slc, center), y_patch_size, spatial_dims=sp_dims_2d)
    return y
def prepare():
    y_patch_size = np.array([256, 256])
    voxel_spacing = (1, 0.95, 0.95)
    slice_to_feature_source = pickle.load(open('slc_to_feat_source.p','rb'))
    slice_to_feature_target = pickle.load(open('slc_to_feat_target.p','rb'))
    preprocessed_dataset = apply(Rescale3D(CC359(DATA_PATH), voxel_spacing), load_image=scale_mri)
    dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    slc_to_seg = {}
    for slc_id in tqdm(list(slice_to_feature_source.keys())):
        id1,slc_num = slc_id.split('_')
        id1 = 'CC0' + id1
        slc_num = int(slc_num)
        slc_seg = dataset.load_segm(id1)[...,slc_num]
        slc_seg = get_random_patch_2d(slc_seg,y_patch_size)
        slc_to_seg[slc_id] = slc_seg
    for slc_id in tqdm(list(slice_to_feature_target.keys())):
        id1,slc_num = slc_id.split('_')
        id1 = 'CC0' + id1
        slc_num = int(slc_num)
        slc_seg = dataset.load_segm(id1)[...,slc_num]
        slc_seg = get_random_patch_2d(slc_seg,y_patch_size)
        slc_to_seg[slc_id] = slc_seg

    points = np.stack(list(slice_to_feature_source.values()) + list(slice_to_feature_target.values()))
    points = points.reshape(points.shape[0],-1)
    dim_to_red_1 = [20,40,60,80]
    dim_to_red_2 = [2,3]
    n_cluster = [6,12,20,40,55,80]
    dim_recud1 = alg_to_partial(PCA) +  alg_to_partial(TruncatedSVD) + alg_to_partial(KernelPCA,kernel='poly') + alg_to_partial(NMF)
    dim_recud2 = alg_to_partial(TSNE,learning_rate='auto',init='pca',k1='perplexity',l1=[30,45,55]) + alg_to_partial(Isomap)
    clustering_algo = [KMeans,AgglomerativeClustering,FeatureAgglomeration,MeanShift]

    all1 = list(product(dim_recud1,dim_recud2,clustering_algo,n_cluster,dim_to_red_1,dim_to_red_2))
    random.shuffle(all1)

    return points,slc_to_seg,all1,slice_to_feature_source,slice_to_feature_target
def alg_to_partial(alg,k1=None,l1=None,**kwargs):
    random_state = 42
    algs = []
    inner_kwargs = kwargs.copy()
    if k1 is not None:
        for ll in l1:
            inner_kwargs[k1] = ll
            algs.append(partial(alg,random_state = random_state,**inner_kwargs))
    else:
        algs = [partial(alg,random_state = random_state,**inner_kwargs)]
    return algs
def f(xxxx,points,slc_to_seg,slice_to_feature_source,slice_to_feature_target,voxel_spacing):
    da1,da2,clus_a, n_clus,d1,d2 = xxxx

    curr_points = points.copy()

    da1 = da1(d1)
    da2 = da2(d2)
    if 'NMF' in str(da1):
        curr_points -= np.min(curr_points)
    print(f'{da1},{da2},{clus_a}, {n_clus},{d1},{d2}')
    curr_points = da1.fit_transform(curr_points)
    curr_points = da2.fit_transform(curr_points)
    source_points,target_point = curr_points[:len(slice_to_feature_source)],curr_points[len(slice_to_feature_source):]
    clus_a_source = clus_a(n_clus)
    clus_a_target = clus_a(n_clus)
    sc = clus_a_source.fit_predict(source_points)
    tc = clus_a_target.fit_predict(target_point)
    best_matchs_indexes=get_best_match(clus_a_source.cluster_centers_,clus_a_target.cluster_centers_)
    slices_for_source_cluster = {}
    slices_for_target_cluster = {}
    items = list(slice_to_feature_source.items())
    source_clusters = []
    target_clusters = []
    for i in range(n_clus):
        source_clusters.append([])
        target_clusters.append([])
    for i in range(len(slice_to_feature_source)):
        source_clusters[sc[i]].append(slc_to_seg[items[i][0]])
    items = list(slice_to_feature_target.items())
    for i in range(len(slice_to_feature_target)):
        target_clusters[tc[i]].append(slc_to_seg[items[i][0]])
    vars_sc = []
    vars_tc = []
    sdices = []
    for i in range(n_clus):
        i_source = best_matchs_indexes[i]
        source_clusters[i_source] = np.stack(source_clusters[i_source])
        sc_mean,sc_var = (np.mean(source_clusters[i_source],axis=0),np.mean(np.std(source_clusters[i_source],axis=0)))
        target_clusters[i] = np.stack(target_clusters[i])
        tc_mean,tc_var = (np.mean(target_clusters[i],axis=0),np.mean(np.std(target_clusters[i],axis=0)))
        sdice1 = sdice(np.expand_dims(tc_mean > 0.5,axis=0),np.expand_dims(sc_mean > 0.5,axis=0),voxel_spacing,1,for_clcc=True)
        vars_sc.append(sc_var)
        vars_tc.append(tc_var)
        sdices.append(sdice1)
    print(f'{da1},{da2},{clus_a}, {n_clus},{d1},{d2}',(float(np.mean(vars_tc)),float(np.mean(vars_sc)),float(np.mean(sdices))))
    return f'{da1},{da2},{clus_a}, {n_clus},{d1},{d2}',(float(np.mean(vars_tc)),float(np.mean(vars_sc)),float(np.mean(sdices)))
def main():
    points,slc_to_seg,all1,slice_to_feature_source,slice_to_feature_target = prepare()
    f1 = partial(f,slice_to_feature_source=slice_to_feature_source,slice_to_feature_target=slice_to_feature_target,voxel_spacing = (1, 0.95, 0.95),slc_to_seg=slc_to_seg,points=points)
    resres = []
    with Pool(processes=4) as pool:
        for i in range(0,len(all1),4):
            results = pool.map_async(f1, all1[i:i+4])
            resres+=list(results.get())
            pickle.dump(resres,open('resres.json','wb'))
if __name__ == '__main__':

    main()
# resres = sorted(resres,reverse=True,key=lambda x:x[2])







