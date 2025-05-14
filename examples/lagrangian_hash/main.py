import ctypes
import numpy as np
import random
import math
import ctypes


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
    NUMBER_OF_GAUSSIANS = 1000
    NUMBER_OF_POINTS = 1000

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

    # Call function
    lib_knn.fit(
        ctypes.byref(GC),
        NUMBER_OF_GAUSSIANS,
        ctypes.byref(coords),
        NUMBER_OF_POINTS,
        ctypes.byref(distances),
        ctypes.byref(gauss_indices)
    )

    # Print some results
    print("First 10 distances and gauss_indices:")
    for i in range(100):
        print(f"Distance {i}: {distances[i]}, Gauss Index: {gauss_indices[i]}")