import math
import logging
import numpy as np
import faiss
import faiss.contrib.torch_utils

import time
import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

from typing import Optional
from utils.general_utils import append_sys_path
from gnerf.lagrangian_hash.knn.knn_algorithms import BaseKNN


append_sys_path()

log = logging.getLogger(__name__)

class SplashEncoding(nn.Module):
    def __init__(
        self,
        fixed_std: bool = False,
        decay_factor: int = 1,
        n_gausses: int = 10000,
        n_features_per_gauss: int = 3,
        knn_algorithm: Optional[BaseKNN] = None
    ):
        """
        """
        super().__init__()
        assert knn_algorithm is not None, "KNN algorithm must be provided"
        
        self.decay_factor = decay_factor
        self.n_features_per_gauss = n_features_per_gauss

        r = 0.125
        self.total_gaus = n_gausses # fixed number of gauss for now
        self.feats = (torch.randn(self.total_gaus, self.n_features_per_gauss) * 1e-2).to(device='cuda')
        self.feats = nn.Parameter(self.feats)
        self.init_mean()
        self.means = nn.Parameter(self.means)
        self.gaussian_constant = torch.sqrt(torch.tensor(2 * torch.pi, device='cuda'))
        if not fixed_std:
            self.stds = nn.Parameter(torch.normal(r, 2e-2, size=(self.total_gaus, 1), device='cuda'))
        self.knn = knn_algorithm
    
    def init_mean(self):
        N = self.total_gaus
        log.info(f'Total number of gauss: {self.total_gaus}')
        pts = np.random.randn(N, 3)
        r = np.sqrt(np.random.rand(N, 1))
        pts = pts / np.linalg.norm(pts, axis=1)[:, None] * r
        pts = pts * 0.5 + 0.5 # [0.25 ... 0.75]
        
        self.means = torch.tensor(pts, dtype=torch.float32, device='cuda')


    def update_factor(self):
        self.stds = self.stds * self.decay_factor


    def get_means(self):
        return self.means
    

    def set_means(self, means):
        means = means
        self.means = nn.Parameter(self.means)


    def get_stds(self):
        return self.stds
    

    def _calculate(self, coords, nearest_gausses_indicies, sq_dists, batch_size=1000):
        num_coords = coords.shape[0]
        feature_dim = self.feats.shape[1]

        feature_vector = torch.zeros((num_coords, feature_dim), device=coords.device)

        for i in range(0, num_coords, batch_size):
            batch_indices = nearest_gausses_indicies[i : i + batch_size]  # [batch_size, num_nearest]
            sq_dist = sq_dists[i : i + batch_size, :, None]  # [batch_size, num_nearest]

            nearest_features = self.feats[batch_indices]  # [batch_size, num_nearest, feature_dim]

            stds = torch.square(self.stds[batch_indices])  # [batch_size, num_nearest]
            gau_weights = torch.exp(-sq_dist / (2 * stds)) / (self.gaussian_constant * torch.sqrt(stds) + 1e-7)  # [batch_size, num_nearest, 1]

            weighted_features = nearest_features * gau_weights  # [batch_size, num_nearest, feature_dim]
            batch_feature_vector = torch.sum(weighted_features, dim=1)  # [batch_size, feature_dim]

            feature_vector[i : i + batch_size] = batch_feature_vector

        return feature_vector
        

    def forward(self, coords, lod_idx=None):
        batch_size = 5000000

        nearest_gausses_indicies = self.knn.get_nearest_neighbours(coords, self.means)

        # Calculate squared distance between each coord and its nearest mean
        nearest_means = self.means[nearest_gausses_indicies]
        squared_gausses_distance = torch.sum((coords[:, None, :] - nearest_means) ** 2, dim=-1)

        start_time = time.time()
        feats = self._calculate(coords, nearest_gausses_indicies, squared_gausses_distance, batch_size=batch_size)
        print(f"Features: {time.time() - start_time:.4f} seconds")

        return feats, squared_gausses_distance[:, 0]
    