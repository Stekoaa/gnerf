import logging

from typing import Callable, Optional, List
from arrgh import arrgh

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import SplashEncoding
from .network import Network

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

log = logging.getLogger(__name__)

class NetworkwithSplashEncoding(nn.Module):
    def __init__(
        self,
        n_features_per_gauss: int = 3,
        n_neighbours: int = 5,
        n_gausses: int = 10000,  
        fixed_std: bool = False,
        decay_factor: int = 1,
        output_dim: int = 3,  # The number of output tensor channels.
        net_depth: int = 2,  # The depth of the MLP.
        net_width: int = 64,  # The width of the MLP.
        hidden_activation: str = "ReLU",
        output_activation: str = "None",
    ):
        super().__init__()
        
        self.encoding = SplashEncoding(fixed_std=fixed_std, 
                                       decay_factor=decay_factor, n_neighbours=n_neighbours,
                                       n_features_per_gauss=n_features_per_gauss, n_gausses=n_gausses)
        
        input_dim = n_features_per_gauss
        self.mlp = tcnn.Network(
                n_input_dims=input_dim,
                n_output_dims=output_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": hidden_activation,
                    "output_activation": output_activation,
                    "n_neurons": net_width,
                    "n_hidden_layers": net_depth,
                },
            )


    def forward(self, coords):
        encoding, gmm = self.encoding(coords)
        output = self.mlp(encoding)
        return output, gmm