import json

import numpy as np
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.io import load
from spottunet.dataset.multiSiteMri import MultiSiteMri
from tqdm import tqdm

from spottunet.paths import DATA_PATH
from matplotlib import pyplot as plt
from spottunet.batch_iter import sample_center_uniformly, SPATIAL_DIMS, extract_patch
from spottunet.dataset.cc359 import Rescale3D, CC359, scale_mri

def st():
    voxel_spacing = (1, 0.95, 0.95)
    preprocessed_dataset = apply(Rescale3D(CC359(DATA_PATH), voxel_spacing), load_image=scale_mri)
    dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    id_to_num_slices = {}
    for i in range(0,6):
        curr_ids = load(f'/home/dsi/shaya/unsup_splits/site_{i}/train_ids.json')
        for id1 in tqdm(curr_ids):
            # img = dataset.load_image(id1)
            seg = dataset.load_segm(id1)
            id_to_num_slices[dataset.load_id(id1)] =seg.shape[-1]
        json.dump(id_to_num_slices,open('/home/dsi/shaya/id_to_num_slices.json','w'))


def msm():
    id_to_num_slices = {}
    for i in tqdm(range(0,6)):
        curr_ids = load(f'/home/dsi/shaya/unsup_splits_msm/site_{i}/train_ids.json')
        dataset = MultiSiteMri([])
        for id1 in curr_ids:
            image, _ = dataset.parse_fn(id1)

            id_to_num_slices[dataset.load_id(id1)] =image.shape[-1]
    json.dump(id_to_num_slices,open('/home/dsi/shaya/id_to_num_slices_msm.json','w'))
print('st')
st()
print('msm')
msm()
