from typing import List
import math
import logging
import numpy as np
import faiss
import faiss.contrib.torch_utils

import torch
import torch.nn as nn

# import laghash.ops.grid as grid_ops
from utils.general_utils import append_sys_path
import time
import torch.autograd.profiler as profiler

append_sys_path()

log = logging.getLogger(__name__)

class SplashEncoding(nn.Module):
    def __init__(
        self,
        fixed_std: bool = False,
        decay_factor: int = 1,
        n_neighbours: int = 5,
        n_gausses: int = 10000,
        n_features_per_gauss: int = 3,
    ):
        """
        """
        super().__init__()
        
        self.decay_factor = decay_factor
        self.n_features_per_gauss = n_features_per_gauss

        r = 0.125
        self.total_gaus = n_gausses # fixed number of gauss for now
        self.feats = (torch.randn(self.total_gaus, self.n_features_per_gauss) * 1e-2).to(device='cuda')
        self.feats = nn.Parameter(self.feats)
        self.init_mean()
        self.means = nn.Parameter(self.means)
        if not fixed_std:
            self.stds = nn.Parameter(torch.normal(r, 2e-2, size=(self.total_gaus, 1), device='cuda'))
        self.n_neighbours = n_neighbours
    
    def init_mean(self):
        N = self.total_gaus
        log.info(f'Total number of gauss: {self.total_gaus}')
        pts = np.random.randn(N, 3)
        r = np.sqrt(np.random.rand(N, 1))
        pts = pts / np.linalg.norm(pts, axis=1)[:, None] * r
        pts = pts * 0.25 + 0.5 # [0.25 ... 0.75]
        
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
    

    def _get_nearest_gausses_indicies(self, coords, batch_size=1000):

        n_coords = coords.shape[0]

        # print(f"coords shape: {coords.shape}")
        # print(f"means shape: {self.means.shape}")
        
        nearest_indices = torch.empty((n_coords, self.n_neighbours), device=coords.device, dtype=int)
        
        start_time = time.time()
        for i in range(0, n_coords, batch_size):
            batch_coords = coords[i:i+batch_size]
            distances = torch.cdist(batch_coords, self.means).to(device='cuda')
            _, batch_nearest_indices = torch.topk(distances, self.n_neighbours, largest=False, sorted=False)
            nearest_indices[i:i+batch_size] = batch_nearest_indices

        torch.cuda.synchronize()
        print(f"KNN: {time.time() - start_time:.4f} seconds")
        
        return nearest_indices


    def get_nearest_gaussians_indices_faiss(self, coords: torch.Tensor):
        """
        FAISS KNN using full GPU path and torch.cuda.FloatTensor inputs.

        Parameters:
        - coords: (N, D) torch.cuda.FloatTensor

        Returns:
        - indices: (N, n_neighbors) torch.LongTensor
        """
        assert coords.shape[1] == self.means.shape[1], "Dimension mismatch"
        assert coords.is_cuda and self.means.is_cuda, "Inputs must be on CUDA"

        N, D = coords.shape

        # Create CPU index and move to GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexFlatL2(res, D)

        # Add means directly
        gpu_index.add(torch.tensor(self.means, device=coords.device))

        # Search
        distances, nearest_indices = gpu_index.search(coords, self.n_neighbours)

        return nearest_indices


    def get_nearest_gaussians_indices_faiss_ivf(self, coords: torch.Tensor, nlist: int = 100):
        """
        Efficient FAISS KNN using IVF index and optional GPU.

        Parameters:
        - coords: (N, D) torch tensor (on CPU or CUDA)
        - nlist: number of Voronoi cells/clusters for IVF (adjust for speed/accuracy tradeoff)

        Returns:
        - nearest_indices: (N, n_neighbors) torch tensor
        """
        assert coords.shape[1] == self.means.shape[1], "Dimension mismatch"

        N, D = coords.shape
        M = self.means.shape[0]

        # Create IVF index
        res = faiss.StandardGpuResources()
        quantizer = faiss.GpuIndexFlatL2(res, D)  # the base index for coarse quantizer
        index_ivf = faiss.GpuIndexIVFFlat(res, quantizer, D, nlist, faiss.METRIC_L2)

        # Train IVF index on means
        train_sample = self.means[:min(10000, M)]
        index_ivf.train(train_sample)
        index_ivf.add(torch.tensor(self.means, device=coords.device))

        # Set nprobe (number of cells to search over, higher is more accurate/slower)
        index_ivf.nprobe = min(10, nlist)
        distances, nearest_indices = index_ivf.search(coords, self.n_neighbours)

        return nearest_indices
    

    def _calculate(self, coords, nearest_gausses_indicies, batch_size=1000):
        num_coords = coords.shape[0]
        feature_dim = self.feats.shape[1]

        feature_vector = torch.zeros((num_coords, feature_dim), device=coords.device)

        for i in range(0, num_coords, batch_size):
            batch_coords = coords[i : i + batch_size]  # [batch_size, 3]
            batch_indices = nearest_gausses_indicies[i : i + batch_size]  # [batch_size, num_nearest]

            nearest_features = self.feats[batch_indices]  # [batch_size, num_nearest, feature_dim]

            diff = batch_coords[:, None, :] - self.means[batch_indices]  # [batch_size, num_nearest, 3]
            sq_dist = torch.sum(diff ** 2, dim=-1, keepdim=True)  # [batch_size, num_nearest, 1]

            stds = torch.abs(self.stds[batch_indices])  # [batch_size, num_nearest]
            gaussian_constant = torch.sqrt(torch.tensor(2 * torch.pi, device=coords.device))
            gau_weights = torch.exp(-sq_dist / (2 * stds ** 2)) / (gaussian_constant * stds + 1e-7)  # [batch_size, num_nearest, 1]

            weighted_features = nearest_features * gau_weights  # [batch_size, num_nearest, feature_dim]
            batch_feature_vector = torch.sum(weighted_features, dim=1)  # [batch_size, feature_dim]

            feature_vector[i : i + batch_size] = batch_feature_vector

        return feature_vector
        

    def forward(self, coords, lod_idx=None):
        batch_size = 20000

        start_time = time.time()
        # nearest_gausses_indicies = self._get_nearest_gausses_indicies(coords, batch_size=batch_size)
        # nearest_gausses_indicies = self.get_nearest_gaussians_indices_faiss(coords)
        nearest_gausses_indicies = self.get_nearest_gaussians_indices_faiss_ivf(coords)
        feats = self._calculate(coords, nearest_gausses_indicies, batch_size=batch_size)
        print(f"Features: {time.time() - start_time:.4f} seconds")
        gmm=None
        return feats, gmm
    