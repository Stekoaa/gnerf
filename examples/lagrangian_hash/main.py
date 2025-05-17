import ctypes
import numpy as np
import random
import math
import ctypes
import time
import torch
import faiss
import faiss.contrib.torch_utils # this needs to be imported for faiss to work with torch tensors


class SGaussianComponent(ctypes.Structure):
    _fields_ = [
        ('mX', ctypes.c_float),
        ('mY', ctypes.c_float),
        ('mZ', ctypes.c_float),
        
        ('qr', ctypes.c_float),
        ('qi', ctypes.c_float),
        ('qj', ctypes.c_float),
        ('qk', ctypes.c_float),
        
        ('sX', ctypes.c_float),
        ('sY', ctypes.c_float),
        ('sZ', ctypes.c_float)
    ]

class float3(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float)
    ]

def random_spherical_gaussian():
    # Generate random spherical coordinates
    r = random.random() * 10
    theta = math.acos(random.uniform(-1, 1))
    phi = random.uniform(0, 2 * math.pi)
    
    # Convert spherical coordinates to Cartesian coordinates
    mX = r * math.sin(theta) * math.cos(phi)
    mY = r * math.sin(theta) * math.sin(phi)
    mZ = r * math.cos(theta)
    
    # Return an instance of SGaussianComponent with random values
    rand_s = random.random()
    component = SGaussianComponent(
        mX=mX, mY=mY, mZ=mZ,
        qr=1, qi=0, qj=0, qk=0,
        sX=rand_s, sY=rand_s, sZ=rand_s,
    )
    return component


def get_nearest_gausses_indicies(coords, means, batch_size=1000, n_neighbours=10):

    n_coords = coords.shape[0]
    
    nearest_indices = torch.empty((n_coords, n_neighbours), device=coords.device, dtype=int)
    
    start_time = time.time()
    for i in range(0, n_coords, batch_size):
        batch_coords = coords[i:i+batch_size]
        distances = torch.cdist(batch_coords, means).to(device='cuda')
        _, batch_nearest_indices = torch.topk(distances, n_neighbours, largest=False, sorted=False)
        nearest_indices[i:i+batch_size] = batch_nearest_indices

    torch.cuda.synchronize()
    print(f"KNN: {time.time() - start_time:.4f} seconds")
    
    return nearest_indices


def get_nearest_gaussians_indices_faiss(coords, means, n_neighbors=10):
    """
    FAISS KNN using full GPU path and torch.cuda.FloatTensor inputs.

    Parameters:
    - coords: (N, D) torch.cuda.FloatTensor
    - means: (M, D) torch.cuda.FloatTensor
    - n_neighbors: int

    Returns:
    - indices: (N, n_neighbors) torch.LongTensor
    """
    assert coords.shape[1] == means.shape[1], "Dimension mismatch"
    assert coords.is_cuda and means.is_cuda, "Inputs must be on CUDA"

    N, D = coords.shape

    # Create CPU index and move to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexFlatL2(res, D)

    # Add means directly
    gpu_index.add(means)

    # Search
    start = time.time()
    distances, indices = gpu_index.search(coords, n_neighbors)
    torch.cuda.synchronize()
    print(f"FAISS GPU KNN: {time.time() - start:.4f} s")

    return indices


def get_nearest_gaussians_indices_faiss_ivf(coords, means, n_neighbors=10, nlist=100):
    """
    Efficient FAISS KNN using IVF index and optional GPU.

    Parameters:
    - coords: (N, D) torch tensor (on CPU or CUDA)
    - means: (M, D) torch tensor (on CPU or CUDA)
    - n_neighbors: how many nearest neighbors to find
    - use_gpu: use FAISS GPU backend
    - nlist: number of Voronoi cells/clusters for IVF (adjust for speed/accuracy tradeoff)

    Returns:
    - nearest_indices: (N, n_neighbors) torch tensor
    """
    start_time = time.time()

    assert coords.shape[1] == means.shape[1], "Dimension mismatch"

    N, D = coords.shape
    M = means.shape[0]

    # Create IVF index
    res = faiss.StandardGpuResources()
    quantizer = faiss.GpuIndexFlatL2(res, D)  # the base index for coarse quantizer
    index_ivf = faiss.GpuIndexIVFFlat(res, quantizer, D, nlist, faiss.METRIC_L2)

    # Train IVF index on means
    print(f"Training IVF index with {min(10000, M)} vectors...")
    train_sample = means[:min(10000, M)]
    index_ivf.train(train_sample)
    print(f"Adding {M} means to IVF index...")
    index_ivf.add(means)

    # Set nprobe (number of cells to search over, higher is more accurate/slower)
    index_ivf.nprobe = min(10, nlist)

    inference_start_time = time.time()
    _, nearest_indices = index_ivf.search(coords, n_neighbors)
    if coords.is_cuda:
        torch.cuda.synchronize()
    print(f"Inference FAISS IVF KNN: {time.time() - inference_start_time:.4f} seconds")
    print(f"Whole FAISS IVF KNN: {time.time() - start_time:.4f} seconds")

    return nearest_indices


if __name__ == "__main__":

    # Define constants
    NUMBER_OF_GAUSSIANS = 40000
    NUMBER_OF_POINTS = 2097152

    # Create an array of SGaussianComponent structs
    GC_Array = SGaussianComponent * NUMBER_OF_GAUSSIANS
    GC = GC_Array()

    float3_Array = float3 * NUMBER_OF_POINTS
    coords = float3_Array()

    # Initialize the array with random values
    for i in range(NUMBER_OF_GAUSSIANS):
        GC[i] = random_spherical_gaussian()

    for i in range(NUMBER_OF_POINTS):
        # Generate random coordinates
        coords[i].x = random.uniform(-10, 10)
        coords[i].y = random.uniform(-10, 10)
        coords[i].z = random.uniform(-10, 10)

    print("First Gaussian Component:")
    print("mX:", GC[0].mX, "mY:", GC[0].mY, "mZ:", GC[0].mZ)
    print("sX:", GC[0].sX, "sY:", GC[0].sY, "sZ:", GC[0].sZ)
    print("qr:", GC[0].qr, "qi:", GC[0].qi, "qj:", GC[0].qj, "qk:", GC[0].qk)

    # Load the shared library
    lib_knn = ctypes.CDLL('/workspace/gnerf/examples/lagrangian_hash/knn/lib_kernel.so')

    # Prepare output arrays for distances and gauss_indices
    distances = (ctypes.c_float * NUMBER_OF_POINTS)()
    gauss_indices = (ctypes.c_int * NUMBER_OF_POINTS)()

    n_neighbours = 50

    # # Find the closest gaussian center to coords[0]
    # min_dist = float('inf')
    # min_idx = -1
    # x0, y0, z0 = coords[0].x, coords[0].y, coords[0].z
    # for i in range(NUMBER_OF_GAUSSIANS):
    #     dx = GC[i].mX - x0
    #     dy = GC[i].mY - y0
    #     dz = GC[i].mZ - z0
    #     dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    #     if dist < min_dist:
    #         min_dist = dist
    #         min_idx = i
    # print(f"Closest gaussian to coords[0] is at index {min_idx} with distance {min_dist}")

    # start_time = time.time()
    # # Call function
    # lib_knn.fit(
    #     ctypes.byref(GC),
    #     NUMBER_OF_GAUSSIANS,
    #     ctypes.byref(coords),
    #     NUMBER_OF_POINTS,
    #     ctypes.byref(distances),
    #     ctypes.byref(gauss_indices)
    # )
    # end_time = time.time()
    # print(f"Execution time for nearest neighbor loop: {end_time - start_time:.4f} seconds")

    # # Print some results
    # print("First distances and gauss_indices:")
    # for i in range(10):
    #     print(f"Distance {i}: {distances[i]}, Gauss Index: {gauss_indices[i]}")
    # print("...")
    # for i in range(NUMBER_OF_POINTS - 10, NUMBER_OF_POINTS):
    #     print(f"Distance {i}: {distances[i]}, Gauss Index: {gauss_indices[i]}")

    # # Old method
    coords = torch.tensor([[coords[i].x, coords[i].y, coords[i].z] for i in range(NUMBER_OF_POINTS)], device='cuda')
    means = torch.tensor([[GC[i].mX, GC[i].mY, GC[i].mZ] for i in range(NUMBER_OF_GAUSSIANS)], device='cuda')
    nearest_indices = get_nearest_gausses_indicies(coords, means, batch_size=1000, n_neighbours=n_neighbours)

    # # Print some results
    # print("First distances and gauss_indices:")
    # for i in range(10):
    #     print(f"Gauss Index: {nearest_indices[i].item()}")
    # print("...")
    # for i in range(NUMBER_OF_POINTS - 10, NUMBER_OF_POINTS):
    #     print(f"Gauss Index: {nearest_indices[i].item()}")

    # # Check percentage of matches
    # matches = 0
    # for i in range(NUMBER_OF_POINTS):
    #     if gauss_indices[i] == nearest_indices[i].item():
    #         matches += 1
    # print(f"Percentage of matches: {matches / NUMBER_OF_POINTS * 100:.2f}%")

    get_nearest_gaussians_indices_faiss(
        coords,
        means,
        n_neighbors=n_neighbours
    )

    get_nearest_gaussians_indices_faiss_ivf(
        coords,
        means,
        n_neighbors=n_neighbours,
        nlist=100
    )