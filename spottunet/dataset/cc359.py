import os
import pickle

import numpy as np
import json

import numpy as np
import torch

from dpipe.im.shape_ops import zoom
from dpipe.dataset.segmentation import SegmentationFromCSV
from dpipe.dataset.wrappers import Proxy
from tqdm import tqdm


class CC359(SegmentationFromCSV):
    def __init__(self, data_path, modalities=('MRI',), target='brain_mask', metadata_rpath='meta.csv'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)
        self.n_domains = len(self.df['fold'].unique())

    def load_image(self, i):
        return np.float32(super().load_image(i)[0])  # 4D -> 3D

    def load_segm(self, i):
        return np.float32(super().load_segm(i))  # already 3D

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing

    def load_domain_label(self, i):
        domain_id = self.df['fold'].loc[i]
        return np.eye(self.n_domains)[domain_id]  # one-hot-encoded domain

    def load_domain_label_number(self, i):
        return self.df['fold'].loc[i]

    def load_domain_label_number_binary_setup(self, i, domains):
        """Assigns '1' to the domain of the largest index; '0' to another one
        Domains may be either (index1, index2) or (sample_scan1_id, sample_scan2_id) """

        if type(domains[0]) != int:
            # the fold numbers of the corresponding 2 samples
            doms = (self.load_domain_label_number (domains[0]), self.load_domain_label_number (domains[1]))
        else:
            doms = domains
        largest_domain = max(doms)
        domain_id = self.df['fold'].loc[i]
        if domain_id == largest_domain:
            return 1
        elif domain_id in doms:  # error otherwise
            return 0


class Change(Proxy):
    def _change(self, x, i):
        raise NotImplementedError

    def load_image(self, i):
        return self._change(self._shadowed.load_image(i), i)

    def load_segm(self, i):
        return np.float32(self._change(self._shadowed.load_segm(i), i) >= .5)


class Rescale3D(Change):
    def __init__(self, shadowed, new_voxel_spacing=1., order=3):
        super().__init__(shadowed)
        self.new_voxel_spacing = np.broadcast_to(new_voxel_spacing, 3).astype(float)
        self.order = order

    def _scale_factor(self, i):
        old_voxel_spacing = self._shadowed.load_spacing(i)
        scale_factor = old_voxel_spacing / self.new_voxel_spacing
        return np.nan_to_num(scale_factor, nan=1)

    def _change(self, x, i):
        return zoom(x, self._scale_factor(i), order=self.order)

    def load_spacing(self, i):
        old_spacing = self.load_orig_spacing(i)
        spacing = self.new_voxel_spacing.copy()
        spacing[np.isnan(spacing)] = old_spacing[np.isnan(spacing)]
        return spacing

    def load_orig_spacing(self, i):
        return self._shadowed.load_spacing(i)


def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)



class CC359Ds(torch.utils.data.Dataset):
    def __init__(self,ids,ds,start,patch_func,exp_dir,source_domain,target_domain,out_domain,split_source):
        self.image_loader = ds.load_image
        self.seg_loader = ds.load_segm
        self.domain_loader = ds.load_domain_label
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.out_domain = out_domain
        self.len_ds = 0
        self.i_to_id = []
        self.start = start # todo: use
        self.patch_func = patch_func
        path_to_i_to_id = os.path.join(exp_dir,'i_to_id.p')
        path_to_len_ds = os.path.join(exp_dir,'len_ds.p')
        path_to_target_indexes = os.path.join(exp_dir,'target_indexes.p')
        path_to_source_indexes = os.path.join(exp_dir,'source_indexes.p')
        self.target_indexes = []
        self.source_indexes = []
        self.idx_to_slice = {}
        if os.path.exists(path_to_i_to_id):
            assert os.path.exists(path_to_len_ds)
            assert not split_source
            self.i_to_id = pickle.load(open(path_to_i_to_id,'rb'))
            self.len_ds = pickle.load(open(path_to_len_ds,'rb'))
            self.source_indexes = pickle.load(open(path_to_source_indexes,'rb'))
            self.target_indexes = pickle.load(open(path_to_target_indexes,'rb'))
        else:
            for id1 in tqdm(ids,desc='calculating data_len'):
                self.i_to_id.append([self.len_ds,id1])
                domain = int(np.argmax(self.domain_loader(id1)) == self.target_domain)
                num_of_slices = self.image_loader(id1).shape[-1]
                if domain > 0:
                    new_len_ds = self.len_ds + num_of_slices
                    self.target_indexes += list(range(self.len_ds, new_len_ds))
                else:
                    if split_source:
                        new_len_ds =self.len_ds + 100
                        self.idx_to_slice[id1] = np.random.choice(range(num_of_slices),size=100,replace=False)
                    else:
                        new_len_ds = self.len_ds + num_of_slices
                    self.source_indexes += list(range(self.len_ds, new_len_ds))
                self.len_ds = new_len_ds
        pickle.dump(self.i_to_id,open(path_to_i_to_id,'wb'))
        pickle.dump(self.len_ds,open(path_to_len_ds,'wb'))
        pickle.dump(self.source_indexes,open(path_to_source_indexes,'wb'))
        pickle.dump(self.target_indexes,open(path_to_target_indexes,'wb'))



    def __getitem__(self, item):
        for (i,id1),(next_i,_) in zip(self.i_to_id,self.i_to_id[1:]+[(self.len_ds,None)]):
            if i <= item < next_i:
                img = self.image_loader(id1)
                seg = self.seg_loader(id1)
                if id1 in self.idx_to_slice:
                    slice_idx = self.idx_to_slice[id1][item-i]
                    img_slc = img[...,slice_idx]
                    seg_slc = seg[...,slice_idx]
                else:
                    img_slc = img[...,item-i]
                    seg_slc = seg[...,item-i]

                img_slc,seg_slc = self.patch_func(img_slc,seg_slc,256,256)
                img_slc,seg_slc = np.expand_dims(img_slc, axis=0),np.expand_dims(seg_slc, axis=0)
                if not self.out_domain:
                    return img_slc,seg_slc
                domain = int(np.argmax(self.domain_loader(id1)) == self.target_domain)
                if domain > 0:
                    assert int(np.argmax(self.domain_loader(id1)) == self.source_domain) == 0
                return img_slc,seg_slc,domain


    def __len__(self):
        return self.len_ds

