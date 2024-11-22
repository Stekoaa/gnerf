import os
import random
import sys

import numpy as np
import torch

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]
MIPNERF360_UNBOUNDED_SCENES = [
    "garden",
    "bicycle",
    "bonsai",
    "counter",
    "kitchen",
    "room",
    "stump",
]
TANKS_TEMPLE_SCENES = [
    "Barn",
    "Caterpillar",
    "Family",
    "Ignatius",
    "Truck",
]

def append_sys_path():
    home_dir = os.path.expanduser('~')
    project_root = os.path.join(home_dir, 'gnerf')
    sys.path.append(project_root)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
