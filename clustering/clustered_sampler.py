
import numpy as np
import torch
import random

from tqdm import tqdm


class ClusteredSampler(torch.utils.data.Sampler):
    def __init__(self, data_source,index_to_cluster,n_cluster,losses,item_to_domain,decrease_center = 1):
        self.step = 0
        self.ds = data_source
        self.index_to_cluster =index_to_cluster
        self.hierarchy = []
        self.decrease_center = decrease_center
        self.n_cluster = n_cluster
        self.center = self.n_cluster + 1
        if losses:
            self.create_distribiouns(losses,item_to_domain)

    def create_distribiouns(self, losses,item_to_domain):
        assert self.center > 0
        losses_mean = 0
        for cluster_index in range(len(losses)):
            if losses[cluster_index]:
                losses[cluster_index] = np.mean(losses[cluster_index])
                losses_mean+= losses[cluster_index]

        losses_mean = losses_mean / len(losses)
        print(f"losses mean is {losses_mean}")
        print(f"losses is {losses}")
        amounts = np.zeros(self.n_cluster)
        for image_index,cluster in self.index_to_cluster.items():
            amounts[cluster]+=1
        print(f" amounts is {amounts}")
        new_losses = np.zeros(self.n_cluster)
        for cluster_index, cluster_loss in enumerate(losses):
            if cluster_loss:
                new_losses[cluster_index] = losses[cluster_index]
            else:
                new_losses[cluster_index] = losses_mean

        while len(self.hierarchy) < self.n_cluster:
            max_idx = np.argmax(new_losses)
            self.hierarchy.append(max_idx)
            new_losses[max_idx] = -1

        cluster_to_domain_dict = [[0,0] for _ in self.hierarchy]
        sum_target = 0
        for i in tqdm(range(len(self.ds)),desc='calculating distribuitns'):
            clus = self.index_to_cluster[i]
            domain = item_to_domain[i]
            if domain > 0:
                sum_target+=1
            cluster_to_domain_dict[clus][domain] +=1
        for i,l1 in enumerate(cluster_to_domain_dict):
            print(f'for clus {i} source percenteage is {l1[0] / sum(l1)}')
            print(f'for clus {i} target percenteage is from sum_target is {l1[1] / sum_target}')
        self.hierarchy = [-1] + self.hierarchy
        print(f"hierarchy is {self.hierarchy}")
        assert len(self.hierarchy) == self.n_cluster + 1
        # indexes = list(range(len(self.ds)))
        # random.shuffle(indexes)
        # for i in range(200):
        #     clus = self.index_to_cluster[indexes[i]]
        #     img ,label = self.ds[indexes[i]]
        #
        #     images[clus].append(img)

        # def imsaves(img,stt):
        #     npimg = img.numpy()
        #     plt.imsave(stt,np.transpose(npimg, (1, 2, 0)))
        # for i,clus in enumerate(images):
        #     if clus:
        #         print(i)
        #         print(self.hierarchy.index(i))
        #         new_img = torchvision.utils.make_grid(clus,normalize=True)
        #
        #         imsaves(new_img,f"{exp_name}/clus_{i}_place_in_hier{self.hierarchy.index(i)}.png")




    def __iter__(self):
        indexes = list(range(len(self.ds)))
        random.shuffle(indexes)
        print(f"self.center is {self.center}")
        print(f"steps done is {self.step}")
        curr_hierarchy = {}
        self.center -= self.decrease_center
        for i in range(len(self.hierarchy)):
            curr_hierarchy[self.hierarchy[i]] = np.exp(-0.2 * abs(self.center - i)) if i < self.center else 1
        diffs = {}
        for i in range(len(self.hierarchy)):
            diffs[i] = []
        for idx in indexes:
            if self.center >= 0:
                cluster = self.index_to_cluster[idx]
                assert cluster in curr_hierarchy
                randi = random.random()
                if randi < curr_hierarchy[cluster]:
                    self.step +=1
                    yield idx
            else:
                self.step += 1
                yield idx
    def __len__(self):
        return len(self.ds)
    def state_dict(self):
        return {"center":self.center,"index_to_cluster":self.index_to_cluster,"hierarchy":self.hierarchy}

    def load_state_dict(self, state_dict):
        self.center = state_dict["center"]
        self.index_to_cluster = state_dict["index_to_cluster"]
        self.hierarchy = state_dict["hierarchy"]