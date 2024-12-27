"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import logging

from typing import Callable, List, Union

import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import lagrangian_hash

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


class LagHashRadianceField(torch.nn.Module):
    """Lagrangian Hashes Radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 15,
        # xd
        # base_resolution: int = 16,
        # max_resolution: int = 1024,
        # n_levels: int = 16,
        # log2_hashmap_size: int = 17,
        num_splashes: int = 4,
        splits: List[float] = [0.875, 0.9375],
        std_init_factor: float = 1.0,
        fixed_std: bool = False,
        decay_factor: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)

        # Turns out rectangle aabb will leads to uneven collision so bad performance.
        # We enforce a cube aabb here.
        center = (aabb[..., :num_dim] + aabb[..., num_dim:]) / 2.0
        size = (aabb[..., num_dim:] - aabb[..., :num_dim]).max()
        aabb = torch.cat([center - size / 2.0, center + size / 2.0], dim=-1)

        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.geo_feat_dim = geo_feat_dim
        # xd
        # self.base_resolution = base_resolution
        # self.max_resolution = max_resolution
        # self.n_levels = n_levels
        # self.log2_hashmap_size = log2_hashmap_size
        self.std_init_factor = std_init_factor
        self.fixed_std = fixed_std
        self.decay_factor = decay_factor
        self.splits = splits

        # xd
        # per_level_scale = np.exp(
        #     (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        # ).tolist()

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )

        self.mlp_base = lagrangian_hash.NetworkwithSplashEncoding(
            # xd
            # n_levels = n_levels,
            num_splashes=num_splashes,
            # xd
            # n_features_per_level = 2,
            # log2_hashmap_size = log2_hashmap_size,
            splits=splits,
            std_init_factor = std_init_factor,
            fixed_std = fixed_std,
            decay_factor=decay_factor,
            # xd
            # base_resolution = base_resolution,
            # per_level_scale = per_level_scale,
            output_dim=1 + self.geo_feat_dim,
            net_depth=1,
            net_width=64,
        )

        # self.mlp_base = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=num_dim,
        #     n_output_dims=1 + self.geo_feat_dim,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": n_levels,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": log2_hashmap_size,
        #         "base_resolution": base_resolution,
        #         "per_level_scale": per_level_scale,
        #     },
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     },
        # )
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    (
                        self.direction_encoding.n_output_dims
                        if self.use_viewdirs
                        else 0
                    )
                    + self.geo_feat_dim
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

    def query_density(self, x, return_feat: bool = False, return_gmm: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        out, gmm = self.mlp_base(x.view(-1, self.num_dim))
        x = (
            out.view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            if return_gmm:
                return density, base_mlp_out, gmm
            else:
                return density, base_mlp_out
        else:
            if return_gmm:
                return density, gmm
            else:
                return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
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
        log.info(f'Radiance field positions: {positions}')
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"

            if positions.shape[0] == 0:
                density = torch.zeros(0, device=positions.device)
                rgb = torch.zeros(0, 3, device=positions.device)
                gmm = torch.zeros(0, 2, device=positions.device)
            else:
                density, embedding, gmm = self.query_density(positions, return_feat=True, return_gmm=True)
                rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density, gmm  # type: ignore
