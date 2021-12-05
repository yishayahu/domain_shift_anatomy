import os
import pickle
import random

import numpy
import torch
from sklearn.cluster import  KMeans
from tsnecuda import TSNE

from tqdm import tqdm

from clustering.clustered_sampler import ClusteredSampler
from clustering.regular_sampler import RegularSampler
import numpy as np

class DsWrapper(torch.utils.data.Dataset):
    def __init__(self,model,dataset_creator,n_clusters,feature_layer_name,warmups,exp_name,decrease_center,exp_dir,no_loss,**kwargs):
        self.exp_dir = exp_dir
        self.no_loss = no_loss
        #####
        kwargs['out_domain'] = True
        kwargs['exp_dir'] = exp_dir
        self.ds = dataset_creator(**kwargs,start=True)
        self.dataset_creator = dataset_creator
        self.future_kwargs = kwargs
        self.future_kwargs['start'] = False
        self.future_kwargs['out_domain'] = False
        self.exp_name = exp_name
        self.decrease_center= decrease_center
        #####


        #######
        source_indexes = self.ds.source_indexes

        random.shuffle(source_indexes)
        # todo: fix
        train_indexes = source_indexes[:int(len(self.ds)*0.8)]
        clustering_indexes = source_indexes[int(len(self.ds)*0.8):]+  self.ds.target_indexes
        print(f"train indexes is {len(train_indexes)}")
        print(f"clustering_indexes is {len(clustering_indexes)}")
        print(f"source_indexes is {len(source_indexes)}")
        print(f"target_indexes is {len(self.ds.target_indexes)}")

        regular_sampler = RegularSampler(data_source=self, train_indexes=train_indexes,
                                         clustering_indexes=clustering_indexes,warmup_epochs=warmups)
        self.current_sampler = regular_sampler



        ######
        self.new_indexes = []
        self.index_to_cluster = {}
        self.n_clusters = n_clusters
        self.losses = [[] for _ in range(n_clusters)]
        self.clustering_algorithm = KMeans(n_clusters=n_clusters)
        self.arrays = []
        self.indexes = []
        self.item_to_domain = {}

        def feature_layer_hook(model, _, output):
            if model.training and type(self.current_sampler) == RegularSampler:
                assert output.shape[0] == len(self.new_indexes)
                output = torch.nn.MaxPool2d(2)(output)
                for i in range(output.shape[0]):
                    self.arrays.append(output[i].cpu().detach().numpy())
                    self.indexes.append(self.new_indexes[i])

            else:
                if not model.training:
                    assert len(self.new_indexes) == 0
        for module_name, module1 in model.named_modules():
            if module_name == feature_layer_name:
                module1.register_forward_hook(feature_layer_hook)


        #######

    def send_loss(self,loss,train_loader):
        assert loss.shape[0] == len(self.new_indexes)
        if self.current_sampler.get_clustering_flag() == "clustering":
            self.new_indexes = []
            loss[loss!=0] = 0
            if len(self.arrays) > len(self.ds):
                self.arrays = self.arrays[:len(self.ds)]
                self.indexes = self.indexes[:len(self.ds)]
            if len(self.arrays) == len(self.ds):
                X = []
                print(f'before tsne len array is {len(self.arrays)}')
                self.arrays = numpy.stack(self.arrays,axis=0)

                pickle.dump(self.arrays,open(os.path.join(self.exp_dir,'array_before_tsne.p'),'wb'))
                pickle.dump(self.item_to_domain,open(os.path.join(self.exp_dir,'item_to_domain.p'),'wb'))
                pickle.dump(self.indexes,open(os.path.join(self.exp_dir,'indexes.p'),'wb'))
                for i in tqdm(range(16),desc='running on i'):
                    for j in tqdm(range(16),desc='running on j'):
                        t = TSNE(n_components=2)
                        X.append(t.fit_transform(self.arrays[:,:,i,j]))
                self.arrays = np.concatenate(X,axis=1)
                pickle.dump(self.arrays,open(os.path.join(self.exp_dir,'array_after_tsne.p'),'wb'))
                labels = self.clustering_algorithm.fit_predict(self.arrays)
                for (index, label) in zip(self.indexes, labels):
                    self.index_to_cluster[index] = label
                self.arrays = []
                self.indexes = []
        elif self.current_sampler.get_clustering_flag() == "get_loss":

            assert len(self.index_to_cluster) == len(self.ds)
            if len(self.indexes) <= len(self.ds):
                for i, index in enumerate(self.new_indexes):
                    self.losses[self.index_to_cluster[index]].append(torch.mean(loss[i]).item())
            self.new_indexes = []
            loss[loss!=0] = 0

        elif self.current_sampler.get_clustering_flag() == "done":
            self.new_indexes = []
            loss[loss!=0] = 0
            self.ds = self.dataset_creator(**self.future_kwargs)
            self.current_sampler = ClusteredSampler(data_source=self.ds, index_to_cluster=self.index_to_cluster,
                                                    n_cluster=self.n_clusters, losses=self.losses,item_to_domain=self.item_to_domain,decrease_center=self.decrease_center)
            train_loader.recreate(dataset=self.ds,sampler=self.current_sampler)
            self.new_indexes = []

        else:
            self.new_indexes = []
        if self.no_loss:
            loss[loss!=0] = 0
        return torch.mean(loss)


    def __getitem__(self, item):
        if type(self.current_sampler) == RegularSampler:
            self.new_indexes.append(item)
        t = self.ds[item]
        if len(t) == 2:
            inp,lbl = t
        else:
            inp,lbl,domain = t
            self.item_to_domain[item] = domain
        return inp,lbl
    def __len__(self):
        return len(self.ds)



