import ctypes
import time
import random
import math
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

if __name__ == "__main__":

    # Define constants
    NUMBER_OF_GAUSSIANS = 1024 * 1024 * 2
    NUMBER_OF_POINTS = 1000

    # Load the shared library
    lib_knn = ctypes.CDLL('/workspace/gnerf/gnerf/lagrangian_hash/knn_2/lib_kernel.so')
    # Define the return type and argument types for create_cknn
    lib_knn.create_cknn.restype = ctypes.c_void_p

    # Create cknn as a pointer
    cknn = lib_knn.create_cknn()

    # Set argument types for CUDA_KNN_Init if needed
    lib_knn.CUDA_KNN_Init.argtypes = [ctypes.c_float, ctypes.c_void_p]
    lib_knn.CUDA_KNN_Init.restype = ctypes.c_int  # or appropriate return type

    # Pass cknn directly as a void pointer
    status = bool(lib_knn.CUDA_KNN_Init(ctypes.c_float(11.3449), cknn))
    print(f"CUDA_KNN_Init created: {status}")

    # Define float4 structure
    class float4(ctypes.Structure):
        _fields_ = [
            ('x', ctypes.c_float),
            ('y', ctypes.c_float),
            ('z', ctypes.c_float),
            ('w', ctypes.c_float)
        ]

    # Create 1000 means on GPU with PyTorch
    means_torch = (torch.rand((NUMBER_OF_GAUSSIANS, 4), dtype=torch.float32, device='cuda').contiguous() * 2) - 1
    means_torch[:, 3] = 0.01

    # Print first ten means
    print("First 10 means:")
    print(means_torch[:10].cpu().numpy())

    # Get pointer to the tensor's data (as void*)
    means_ptr = means_torch.data_ptr()

    # Set argument types for CUDA_KNN_Fit if needed
    lib_knn.CUDA_KNN_Fit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib_knn.CUDA_KNN_Fit.restype = ctypes.c_int  # or appropriate return type

    # Pass pointer to tensor data
    status = bool(lib_knn.CUDA_KNN_Fit(
        ctypes.c_void_p(means_ptr),
        ctypes.c_int(NUMBER_OF_GAUSSIANS),
        cknn
    ))
    print(f"CUDA_KNN_Fit: {status}")

    # Prepare queried points as a torch tensor on CUDA
    num_queried_points = NUMBER_OF_POINTS
    queried_points_torch = (torch.rand((num_queried_points, 4), dtype=torch.float32, device='cuda').contiguous() * 2) - 1
    queried_points_torch[:, 3] = 0.0
    queried_points_ptr = queried_points_torch.data_ptr()

    # Prepare output arrays for distances and indices (on CPU, as ctypes arrays)
    K = 10
    # Allocate distances and indices as torch tensors on GPU
    distances_torch = torch.empty((num_queried_points, K), dtype=torch.float32, device='cuda').contiguous()
    indices_torch = torch.empty((num_queried_points, K), dtype=torch.int32, device='cuda').contiguous()

    distances_ptr = distances_torch.data_ptr()
    indices_ptr = indices_torch.data_ptr()

    # Set argument types and restype for CUDA_KNN_KNeighbors
    lib_knn.CUDA_KNN_KNeighbors.argtypes = [
        ctypes.c_void_p,  # float4* queried_points
        ctypes.c_int,     # int number_of_queried_points
        ctypes.c_int,     # int K
        ctypes.c_void_p,  # float* distances (GPU pointer)
        ctypes.c_void_p,  # int* indices (GPU pointer)
        ctypes.c_void_p   # S_CUDA_KNN* knn
    ]
    lib_knn.CUDA_KNN_KNeighbors.restype = ctypes.c_bool

    # print("######################################################################################")
    # # Call the function
    # start_time = time.time()
    # status = lib_knn.CUDA_KNN_KNeighbors(
    #     ctypes.c_void_p(queried_points_ptr),
    #     ctypes.c_int(num_queried_points),
    #     ctypes.c_int(K),
    #     distances_ptr,
    #     indices_ptr,
    #     cknn
    # )
    # torch.cuda.synchronize()
    # print(f"CUDA_KNN_KNeighbors computation time {time.time() - start_time:.4f} seconds")

    # # Print first ten distances and indices for the first queried point
    # print("First 10 distances (CUDA):", distances_torch[0].cpu().numpy())
    # print("First 10 distances (CUDA):", distances_torch[1].cpu().numpy())
    # print("First 10 distances (CUDA):", distances_torch[2].cpu().numpy())
    # print("First 10 distances (CUDA):", distances_torch[3].cpu().numpy())
    # print("First 10 distances (CUDA):", distances_torch[4].cpu().numpy())
    # print("First 10 indices (CUDA):", indices_torch[0].cpu().numpy())
    # print("First 10 indices (CUDA):", indices_torch[1].cpu().numpy())
    # print("First 10 indices (CUDA):", indices_torch[2].cpu().numpy())
    # print("First 10 indices (CUDA):", indices_torch[3].cpu().numpy())
    # print("First 10 indices (CUDA):", indices_torch[4].cpu().numpy())
    # print("######################################################################################")

    # Compute ground truth using torch.cdist and topk
    start_time = time.time()
    dists = torch.cdist(queried_points_torch[:, :3], means_torch[:, :3])  # Only use x, y, z for distance
    gt_dists, gt_indices = torch.topk(dists, K, largest=False, dim=1)
    torch.cuda.synchronize()
    print(f"Ground truth computation took {time.time() - start_time:.4f} seconds")

    # Print first ten distances and indices for the ground truth
    print("Ground truth first 10 distances:", gt_dists[0].cpu().numpy())
    print("Ground truth first 10 distances:", gt_dists[1].cpu().numpy())
    print("Ground truth first 10 distances:", gt_dists[2].cpu().numpy())
    print("Ground truth first 10 distances:", gt_dists[3].cpu().numpy())
    print("Ground truth first 10 distances:", gt_dists[4].cpu().numpy())
    print("Ground truth first 10 indices:", gt_indices[0].cpu().numpy())
    print("Ground truth first 10 indices:", gt_indices[1].cpu().numpy())
    print("Ground truth first 10 indices:", gt_indices[2].cpu().numpy())
    print("Ground truth first 10 indices:", gt_indices[3].cpu().numpy())
    print("Ground truth first 10 indices:", gt_indices[4].cpu().numpy())
    print("######################################################################################")

    import knn_bindings

    cknn = knn_bindings.S_CUDA_KNN()
    print(f"cknn created: {cknn}")

    success = knn_bindings.CUDA_KNN_Init(11.3449, cknn)

    print(f"CUDA_KNN_Init created: {success}")

    success = knn_bindings.CUDA_KNN_Fit(means_torch, NUMBER_OF_GAUSSIANS, cknn)

    print(f"CUDA_KNN_Fit: {success}")

    distances = torch.empty((NUMBER_OF_POINTS, K), dtype=torch.float32, device='cuda')
    indices = torch.empty((NUMBER_OF_POINTS, K), dtype=torch.int32, device='cuda')

    success = knn_bindings.CUDA_KNN_KNeighbors(queried_points_torch, K, distances, indices, cknn)

    print(f"CUDA_KNN_KNeighbors: {success}")

    print("First 10 distances (CUDA):", distances[0].cpu().numpy())
    print("First 10 indices (CUDA):", indices[0].cpu().numpy())