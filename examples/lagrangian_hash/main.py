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

def random_spherical_gaussian():
    # Generate random spherical coordinates
    r = random.random()
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

    # Create an array of 100 SGaussianComponent structs
    GC_Array = SGaussianComponent * NUMBER_OF_GAUSSIANS
    GC = GC_Array()

    # Initialize the array with random values
    for i in range(NUMBER_OF_GAUSSIANS):
        GC[i] = random_spherical_gaussian()

    print("First Gaussian Component:")
    print("mX:", GC[0].mX, "mY:", GC[0].mY, "mZ:", GC[0].mZ)
    print("sX:", GC[0].sX, "sY:", GC[0].sY, "sZ:", GC[0].sZ)
    print("qr:", GC[0].qr, "qi:", GC[0].qi, "qj:", GC[0].qj, "qk:", GC[0].qk)

    # Load the shared library
    lib_knn = ctypes.CDLL('/workspace/otk-pyoptix/knn/lib_kernel.so')

    # Call function
    lib_knn.fit(ctypes.byref(GC), NUMBER_OF_GAUSSIANS)