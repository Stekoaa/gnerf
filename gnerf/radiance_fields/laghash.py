"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import torch
import lagrangian_hash
import numpy as np

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from utils.config_utils import InstantiateConfig
from dataclasses import dataclass, field
from typing import Type, Union

from gnerf.lagrangian_hash.knn.knn_algorithms import BaseKNNConfig


try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    # ord: Union[str, int] = 2,
    ord: Union[float, int] = float("inf"),
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x
    

@dataclass 
class LagHashRadianceFieldConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: LagHashRadianceField)
    """Configuration for the LagHashRadianceField."""
    aabb: list = field(default_factory=lambda: [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
    """Axis-Aligned Bounding Box (AABB) of the scene."""
    num_dim: int = 3
    """Number of dimensions for the input coordinates."""
    geo_feat_dim: int = 15
    """Number of dimensions for the geometry features."""
    grid_resolution: int = 128
    """Resolution of the occupancy grid."""
    grid_nlvl: int = 1
    """Number of levels in the occupancy grid."""
    unbounded: bool = False
    """Whether to use unbounded coordinates."""
    use_viewdirs: bool = True
    """Whether to use view directions."""
    n_features_per_gauss: int = 10
    """Number of features per Gaussian."""
    fixed_std: bool = False
    """Whether to use fixed standard deviation."""
    load_model_path: str = ""
    """Path to the model to load."""
    n_gausses: int = 40000
    """Number of Gaussians in the model."""
    knn_algorithm: BaseKNNConfig = field(default_factory=BaseKNNConfig)
    """KNN algorithm to use for nearest neighbour search."""


class LagHashRadianceField(torch.nn.Module):
    """Lagrangian Hashes Radiance Field"""

    def __init__(self, config: LagHashRadianceFieldConfig, std_decay_factor, device = "cpu"):
        self.config: LagHashRadianceFieldConfig = config
        super().__init__()

        if not isinstance(self.config.aabb, torch.Tensor):
            self.config.aabb = torch.tensor(self.config.aabb, dtype=torch.float32, device=device)

        # Turns out rectangle aabb will leads to uneven collision so bad performance.
        # We enforce a cube aabb here.
        center = (self.config.aabb[..., :self.config.num_dim] + self.config.aabb[..., self.config.num_dim:]) / 2.0
        size = (self.config.aabb[..., self.config.num_dim:] - self.config.aabb[..., :self.config.num_dim]).max()
        self.config.aabb = torch.cat([center - size / 2.0, center + size / 2.0], dim=-1)

        self.register_buffer("aabb", self.config.aabb)

        if self.config.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=self.config.num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                    ],
                },
            )

        self.knn_algorithm = self.config.knn_algorithm.setup()
        self.mlp_base = lagrangian_hash.NetworkwithSplashEncoding(
            fixed_std = self.config.fixed_std,
            decay_factor=std_decay_factor,
            n_features_per_gauss=self.config.n_features_per_gauss,
            n_gausses=self.config.n_gausses,
            output_dim=1 + self.config.geo_feat_dim,
            net_depth=1,
            net_width=64,
            knn_algorithm=self.knn_algorithm
        )

        if self.config.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    (
                        self.direction_encoding.n_output_dims
                        if self.config.use_viewdirs
                        else 0
                    )
                    + self.config.geo_feat_dim
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )


    def query_density(self, x, return_feat: bool = False, return_squared_gausses_distance: bool = False):
        if self.config.unbounded:
            x = contract_to_unisphere(x, self.config.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.config.aabb, self.config.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        out, squared_gausses_distance = self.mlp_base(x.view(-1, self.config.num_dim))
        x = (
            out.view(list(x.shape[:-1]) + [1 + self.config.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.config.geo_feat_dim], dim=-1
        )
        density_activation = lambda x: trunc_exp(x - 1)
        density = (
            density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            if return_squared_gausses_distance:
                return density, base_mlp_out, squared_gausses_distance
            else:
                return density, base_mlp_out
        else:
            if return_squared_gausses_distance:
                return density, squared_gausses_distance
            else:
                return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.config.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.config.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.config.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.config.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"

            if positions.shape[0] == 0:
                density = torch.zeros(0, device=positions.device)
                rgb = torch.zeros(0, 3, device=positions.device)
                squared_gausses_distance = torch.zeros(0, device=positions.device)
            else:
                density, embedding, squared_gausses_distance = self.query_density(positions, return_feat=True, return_squared_gausses_distance=True)
                rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density, squared_gausses_distance  # type: ignore
