import math

import torch
import torch.nn.functional as F

def calculate_loss_warmup(step, max_steps):
    return min(4 * step / max_steps, math.exp(-4 * step / max_steps))

def calculate_lod_sigma_loss(resolution, stds):
    sigma_diff = F.relu(stds - 2 / resolution)
    return torch.mean(sigma_diff)

def calculate_smooth_l1_loss(preds, targets):
    return F.smooth_l1_loss(preds, targets)