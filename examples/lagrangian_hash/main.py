import ctypes
import numpy as np
import random
import math
import ctypes
import time
import torch


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
    print(f"KNN: {time.time() - start_time:.4f} seconds")
    
    return nearest_indices


if __name__ == "__main__":

    # Define constants
    NUMBER_OF_GAUSSIANS = 40000
    NUMBER_OF_POINTS = 495759 # 2097152

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

    # Find the closest gaussian center to coords[0]
    min_dist = float('inf')
    min_idx = -1
    x0, y0, z0 = coords[0].x, coords[0].y, coords[0].z
    for i in range(NUMBER_OF_GAUSSIANS):
        dx = GC[i].mX - x0
        dy = GC[i].mY - y0
        dz = GC[i].mZ - z0
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    print(f"Closest gaussian to coords[0] is at index {min_idx} with distance {min_dist}")

    start_time = time.time()
    # Call function
    lib_knn.fit(
        ctypes.byref(GC),
        NUMBER_OF_GAUSSIANS,
        ctypes.byref(coords),
        NUMBER_OF_POINTS,
        ctypes.byref(distances),
        ctypes.byref(gauss_indices)
    )
    end_time = time.time()
    print(f"Execution time for nearest neighbor loop: {end_time - start_time:.4f} seconds")

    # Print some results
    print("First distances and gauss_indices:")
    for i in range(10):
        print(f"Distance {i}: {distances[i]}, Gauss Index: {gauss_indices[i]}")
    print("...")
    for i in range(NUMBER_OF_POINTS - 10, NUMBER_OF_POINTS):
        print(f"Distance {i}: {distances[i]}, Gauss Index: {gauss_indices[i]}")

    # Old method
    coords = torch.tensor([[coords[i].x, coords[i].y, coords[i].z] for i in range(NUMBER_OF_POINTS)], device='cuda')
    means = torch.tensor([[GC[i].mX, GC[i].mY, GC[i].mZ] for i in range(NUMBER_OF_GAUSSIANS)], device='cuda')
    nearest_indices = get_nearest_gausses_indicies(coords, means, batch_size=20000, n_neighbours=1)

    # Print some results
    print("First distances and gauss_indices:")
    for i in range(10):
        print(f"Gauss Index: {nearest_indices[i].item()}")
    print("...")
    for i in range(NUMBER_OF_POINTS - 10, NUMBER_OF_POINTS):
        print(f"Gauss Index: {nearest_indices[i].item()}")

    # Check percentage of matches
    matches = 0
    for i in range(NUMBER_OF_POINTS):
        if gauss_indices[i] == nearest_indices[i].item():
            matches += 1
    print(f"Percentage of matches: {matches / NUMBER_OF_POINTS * 100:.2f}%")