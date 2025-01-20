"""
This code is partially adapted from:
- https://github.com/sjtu-xai-lab/aog/blob/main/src/tools/utils.py
"""

import os
import os.path as osp

import random
import numpy as np
import tensorflow as tf
import torch
from itertools import combinations


def makedirs(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


def set_seed(seed=0):
    print(f"Set SEED: {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def get_all_nonempty_subsets(input_list):
    subsets = []
    for r in range(len(input_list) + 1):
        subsets.extend(combinations(input_list, r))
    return [list(subset) for subset in subsets if list(subset)]


def get_all_subsets(input_list):
    subsets = []
    for r in range(len(input_list) + 1):
        subsets.extend(combinations(input_list, r))
    return [list(subset) for subset in subsets]