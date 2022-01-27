import os
import random
from pathlib import Path

import numpy as np
import torch
import surface_distance.metrics as surf_dc

from dpipe.io import PathLike


def get_pred(x, threshold=0.5):
    return x > threshold


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_root(*paths: PathLike) -> Path:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError('No appropriate root found.')


def sdice(a, b, spacing, tolerance,for_clcc=False):
    if for_clcc:
        a_zeros  =np.count_nonzero(a)
        b_zeros  =np.count_nonzero(b)
        if  a_zeros ==0  and b_zeros == 0:
            return 1
        elif a_zeros == 0 or b_zeros == 0:
            a[0][0] = True
            b[0][0] = True
    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)


def skip_predict(output_path):
    print(f'>>> Passing the step of saving predictions into `{output_path}`', flush=True)
    os.makedirs(output_path)
