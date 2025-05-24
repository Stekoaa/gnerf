import torch

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-2
    """Learning rate for the optimizer."""
    gaussian_factor: float = 0.1
    """Gaussian factor for the optimizer."""
    weight_decay: Optional[float] = None
    """Weight decay for the optimizer."""
    eps: float = 1e-15
    """Epsilon for the optimizer."""
    

def initialize_optimizer(config, radiance_field, weight_decay):
    params_dict = { name : param for name, param in radiance_field.named_parameters()}
    
    gau_params, codebook_params, rest_params = [], [], []
    for name in params_dict:
        if ("means" in name) or ("stds" in name):
            gau_params.append(params_dict[name])
        elif "feats" in name:
            codebook_params.append(params_dict[name])
        else:
            rest_params.append(params_dict[name])

    gau_lr = config.optimizer.learning_rate * config.optimizer.gaussian_factor
    params = [
        {"params": gau_params, "lr": gau_lr, "eps": config.optimizer.eps, "weight_decay": 0.0},
        {"params": codebook_params, "lr": config.optimizer.learning_rate, "eps": config.optimizer.eps, "weight_decay": weight_decay},
        {"params": rest_params, "lr": config.optimizer.learning_rate, "eps": config.optimizer.eps, "weight_decay": weight_decay}
    ]
    
    return torch.optim.Adam(params)