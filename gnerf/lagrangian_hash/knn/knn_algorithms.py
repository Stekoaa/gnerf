from __future__ import annotations

import torch
import faiss
import faiss.contrib.torch_utils
import optix_knn

from dataclasses import dataclass, field
from utils.config_utils import InstantiateConfig
from typing import Type


@dataclass
class BaseKNNConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: BaseKNN)
    """Base class for KNN configuration."""
    n_neighbours: int = 16
    """Number of nearest neighbours to consider."""
    device: str = 'cuda'
    """Device to run the KNN algorithm on."""


class BaseKNN:
    """Base class for KNN algorithms."""

    def __init__(self, config: BaseKNNConfig):
        super().__init__()
        self.config: BaseKNNConfig = config

    def get_nearest_neighbours(self, query: torch.Tensor, points: torch.Tensor):
        """Get indices of nearest gaussians."""
        raise NotImplementedError("This method should be implemented by subclasses.")


@dataclass
class TorchKNNConfig(BaseKNNConfig):

    _target: Type = field(default_factory=lambda: TorchKNN)
    """Configuration for Torch KNN algorithm."""
    batch_size: int = 1024
    """Batch size for processing."""

class TorchKNN(BaseKNN):
    """KNN algorithm using PyTorch."""

    def __init__(self, config: TorchKNNConfig):
        super().__init__(config)

    def get_nearest_neighbours(self, query: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Get indices of nearest neighbours using PyTorch."""

        n_coords = query.shape[0]
        nearest_indices = torch.empty((n_coords, self.config.n_neighbours), device=self.config.device, dtype=int)
        
        for i in range(0, n_coords, self.config.batch_size):
            batch_coords = query[i:i+self.config.batch_size]
            distances = torch.cdist(batch_coords, points).to(device=self.config.device)
            _, batch_nearest_indices = torch.topk(distances, self.config.n_neighbours, largest=False, sorted=False)
            nearest_indices[i:i+self.config.batch_size] = batch_nearest_indices
        
        return nearest_indices


@dataclass
class FaissKNNConfig(BaseKNNConfig):

    _target: Type = field(default_factory=lambda: FaissKNN)
    """Configuration for FAISS KNN algorithm."""

class FaissKNN(BaseKNN):
    """KNN algorithm using FAISS."""

    def __init__(self, config: FaissKNNConfig):
        super().__init__(config)

    def get_nearest_neighbours(self, query: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        FAISS KNN using full GPU path and torch.cuda.FloatTensor inputs.

        Parameters:
        - coords: (N, D) torch.cuda.FloatTensor

        Returns:
        - indices: (N, n_neighbors) torch.LongTensor
        """
        assert query.shape[1] == points.shape[1], "Dimension mismatch"
        assert query.is_cuda and points.is_cuda, "Inputs must be on CUDA"

        N, D = query.shape

        # Prepare FAISS
        res = faiss.StandardGpuResources()

        # Create CPU index and move to GPU
        gpu_index = faiss.GpuIndexFlatL2(res, D)

        # Add means directly
        gpu_index.add(torch.tensor(points, device=self.config.device))

        # Search
        distances, nearest_indices = gpu_index.search(query, self.config.n_neighbours)

        return nearest_indices
    

@dataclass
class FaissIVFKNNConfig(BaseKNNConfig):

    _target: Type = field(default_factory=lambda: FaissIVFKNN)
    """Configuration for FAISS IVF KNN algorithm."""
    nlist: int = 100
    """Number of Voronoi cells/clusters for IVF (adjust for speed/accuracy tradeoff)."""
    
class FaissIVFKNN(BaseKNN):
    """KNN algorithm using FAISS IVF (Inverted File Index)."""

    def __init__(self, config: FaissIVFKNNConfig):
        super().__init__(config)

    def get_nearest_neighbours(self, query: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Efficient FAISS KNN using IVF index and optional GPU.

        Parameters:
        - coords: (N, D) torch tensor (on CPU or CUDA)
        - nlist: number of Voronoi cells/clusters for IVF (adjust for speed/accuracy tradeoff)

        Returns:
        - nearest_indices: (N, n_neighbors) torch tensor
        """
        assert query.shape[1] == points.shape[1], "Dimension mismatch"
        
        N, D = query.shape
        M = points.shape[0]

        # Prepare FAISS
        res = faiss.StandardGpuResources()

        # Create IVF index
        quantizer = faiss.GpuIndexFlatL2(res, D)  # the base index for coarse quantizer
        index_ivf = faiss.GpuIndexIVFFlat(res, quantizer, D, self.config.nlist, faiss.METRIC_L2)

        # Train IVF index on means
        train_sample = points[:min(10000, M)]
        index_ivf.train(train_sample)
        index_ivf.add(torch.tensor(points, device=query.device))

        # Set nprobe (number of cells to search over, higher is more accurate/slower)
        index_ivf.nprobe = min(10, self.config.nlist)
        
        distances, nearest_indices = index_ivf.search(query, self.config.n_neighbours)

        return nearest_indices
    

@dataclass
class OptixKNNConfig(BaseKNNConfig):

    _target: Type = field(default_factory=lambda: OptixKNN)
    """Configuration for OptiX KNN algorithm."""

class OptixKNN(BaseKNN):
    """KNN algorithm using OptiX."""

    def __init__(self, config: OptixKNNConfig):
        super().__init__(config)

        self.cknn = optix_knn.S_CUDA_KNN()
        optix_knn.CUDA_KNN_Init(11.3449, self.cknn)

    def get_nearest_neighbours(self, query: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Efficient KNN using OptiX.

        Parameters:
        - query: (N, D) torch tensor (on CUDA)
        - points: (M, D) torch tensor (on CUDA)

        Returns:
        - nearest_indices: (N, n_neighbors) torch tensor
        """
        assert query.shape[1] == points.shape[1], "Dimension mismatch"

        # Expand points to (N, 4) by appending a column of 0.01
        if points.shape[1] == 3:
            pad = torch.full((points.shape[0], 1), 0.01, device=points.device, dtype=points.dtype)
            pad_points = torch.cat([points, pad], dim=1)
        if query.shape[1] == 3:
            pad = torch.full((query.shape[0], 1), 0.0, device=query.device, dtype=query.dtype)
            pad_query = torch.cat([query, pad], dim=1)
        
        optix_knn.CUDA_KNN_Fit(pad_points, pad_points.shape[0], self.cknn)

        distances = torch.empty((self.config.n_neighbours, pad_query.shape[0]), dtype=torch.float32, device='cuda')
        indices = torch.empty((self.config.n_neighbours, pad_query.shape[0]), dtype=torch.int32, device='cuda')

        optix_knn.CUDA_KNN_KNeighbors(pad_query, self.config.n_neighbours, distances, indices, self.cknn)

        distances = distances.T
        indices = indices.T

        return indices