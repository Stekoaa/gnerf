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

def points_to_grid_coords(points, aabb):
    xyz_min, xyz_max = aabb[:3], aabb[3:]
    # Normalize to [0, 1]
    normalized = (points - xyz_min) / (xyz_max - xyz_min)
    # Convert to [-1, 1] for grid_sample
    return normalized * 2 - 1

def trilinear_interpolation(distance_field, points, aabb):
    """
    Args:
        distance_field: (D, H, W) tensor on same device as points.
        points: (N, 3) tensor in world coordinates.
        aabb: (6,) tensor [min_x, min_y, min_z, max_x, max_y, max_z].

    Returns:
        distances: (N,) interpolated values at points.
    """
    D, H, W = distance_field.shape
    device = points.device
    dtype = points.dtype

    xyz_min, xyz_max = aabb[:3], aabb[3:]
    grid_size = torch.tensor([W, H, D], device=device, dtype=dtype)
    voxel_size = (xyz_max - xyz_min) / grid_size

    # Normalize to grid space
    grid_coords = (points - xyz_min) / voxel_size  # shape (N, 3)

    # Clamp to avoid indexing outside
    min_val = torch.zeros(3, device=device, dtype=dtype)
    max_val = grid_size - 1 - 1e-6
    grid_coords = torch.clamp(grid_coords, min_val, max_val)

    # Get integer and fractional parts
    idx0 = grid_coords.floor().long()  # (N, 3)
    d = grid_coords - idx0.float()     # (N, 3)
    idx1 = idx0 + 1

    x0, y0, z0 = idx0.unbind(dim=1)
    x1, y1, z1 = idx1.unbind(dim=1)
    dx, dy, dz = d.unbind(dim=1)

    def get_vals(x, y, z):
        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)
        z = torch.clamp(z, 0, D - 1)
        return distance_field[z, y, x]

    c000 = get_vals(x0, y0, z0)
    c100 = get_vals(x1, y0, z0)
    c010 = get_vals(x0, y1, z0)
    c110 = get_vals(x1, y1, z0)
    c001 = get_vals(x0, y0, z1)
    c101 = get_vals(x1, y0, z1)
    c011 = get_vals(x0, y1, z1)
    c111 = get_vals(x1, y1, z1)

    # Trilinear interpolation
    c00 = c000 * (1 - dx) + c100 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c11 = c011 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy

    c = c0 * (1 - dz) + c1 * dz  # (N,)

    return c

def denormalize_points(points: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    num_dim = points.shape[-1]
    aabb_min, aabb_max = torch.split(aabb, num_dim)
    return points * (aabb_max - aabb_min) + aabb_min