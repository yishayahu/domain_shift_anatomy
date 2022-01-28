import numpy as np
import torch
class DataLoaderWrapper(torch.utils.data.DataLoader):
    def __init__(self,source, *transformers,
                 batch_size , batches_per_epoch: int):

        self.source  = source
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.transformers = transformers
    def __call__(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def __iter__(self):
        for b in range(self.batches_per_epoch):
            currs = []
            for i in range(5):
                currs.append([])
            for i in range(self.batch_size):
                id1 = next(self.source)
                for t in self.transformers:
                    id1 = t(id1)
                for j in range(len(id1)):
                    currs[j].append(id1[j])

            for i in range(5):
                currs[i] = np.stack(currs[i])
            yield currs
