#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "optix.h"
#include "optix_stubs.h"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <stdio.h>

// *** *** *** *** ***

struct S_CUDA_KNN {
	OptixDeviceContext optixContext;
	OptixPipeline pipeline;
	OptixShaderBindingTable *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;
	float chi_square_squared_radius; // !!! !!! !!!
	float3 *gaussian_as_polygon_vertices;
	int3 *gaussian_as_polygon_indices;
	float4 *means; // !!! !!! !!!
	int number_of_means; // !!! !!! !!!
	float3 *gaussians_as_polygons_vertices;
	int3 *gaussians_as_polygons_indices;
	OptixTraversableHandle asHandle;
	void *asBuffer;
};

// *** *** *** *** ***

struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *** *** *** *** ***

struct SLaunchParams {
	OptixTraversableHandle traversable;
	float4 *means;
	float4 *queried_points;
	float *distances;
	int *indices;
	float chi_square_squared_radius;
	int K;
	float max_R;
};

// *** *** *** *** ***

extern "C" bool CUDA_KNN_Init(float chi_square_squared_radius, S_CUDA_KNN* knn);
extern "C" bool CUDA_KNN_Fit(float4 *means, int number_of_means, S_CUDA_KNN* knn);
extern "C" bool CUDA_KNN_KNeighbors(
	float4 *queried_points,
	int number_of_queried_points,
	int K,
	float *distances,
	int *indices,
	S_CUDA_KNN* knn
);