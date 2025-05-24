#include "Header.cuh"

// !!! !!! !!!
#include "optix_function_table_definition.h"
// !!! !!! !!!

// *************************************************************************************************
// CUDA_KNN_Init                                                                                   *
// *************************************************************************************************

extern "C" bool CUDA_KNN_Init(float chi_square_squared_radius, S_CUDA_KNN* knn) {

	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	CUresult error_CUDA_Driver_API;

	S_CUDA_KNN cknn = *knn;

	error_CUDA = cudaSetDevice(0);
	if (error_CUDA != cudaSuccess) printf("An error occurred... .");

	// *********************************************************************************************

	error_OptiX = optixInit();
	if (error_OptiX != OPTIX_SUCCESS) return false;

	CUcontext cudaContext;
	error_CUDA_Driver_API = cuCtxGetCurrent(&cudaContext);
	if (error_CUDA_Driver_API != CUDA_SUCCESS) return false;

	error_OptiX = optixDeviceContextCreate(cudaContext, 0, &cknn.optixContext);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	FILE *f = fopen("/workspace/gnerf/gnerf/lagrangian_hash/knn_2/shaders.cu.ptx", "rb");
	fseek(f, 0, SEEK_END);
	int ptxCodeSize = ftell(f);
	fclose(f);

	char *ptxCode = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	char *buffer = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	ptxCode[0] = 0; // !!! !!! !!!

	f = fopen("/workspace/gnerf/gnerf/lagrangian_hash/knn_2/shaders.cu.ptx", "rb");
	fgets(buffer, ptxCodeSize + 1, f);
	while (!feof(f)) {
		ptxCode = strcat(ptxCode, buffer);
		fgets(buffer, ptxCodeSize + 1, f);
	}
	fclose(f);
	free(buffer);

	// *********************************************************************************************

	OptixModuleCompileOptions moduleCompileOptions = {};
	OptixPipelineCompileOptions pipelineCompileOptions = {};

	moduleCompileOptions.maxRegisterCount = 40;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 0;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	OptixModule module;
	error_OptiX = optixModuleCreateFromPTX(
		cknn.optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptxCode,
		strlen(ptxCode),
		NULL, NULL,
		&module
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	free(ptxCode);

	// *********************************************************************************************

	OptixProgramGroupOptions pgOptions = {};

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_raygen = {};
	pgDesc_raygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc_raygen.raygen.module = module;           
	pgDesc_raygen.raygen.entryFunctionName = "__raygen__";

	OptixProgramGroup raygenPG;
	error_OptiX = optixProgramGroupCreate(
		cknn.optixContext,
		&pgDesc_raygen,
		1,
		&pgOptions,
		NULL, NULL,
		&raygenPG
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_miss = {};
	pgDesc_miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

	OptixProgramGroup missPG;
	error_OptiX = optixProgramGroupCreate(
		cknn.optixContext,
		&pgDesc_miss,
		1, 
		&pgOptions,
		NULL, NULL,
		&missPG
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_hitgroup = {};
	pgDesc_hitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	// !!! !!! !!! TRIANGLES !!! !!! !!!
	pgDesc_hitgroup.hitgroup.moduleAH            = module;
	pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__";

	OptixProgramGroup hitgroupPG;
	error_OptiX = optixProgramGroupCreate(
		cknn.optixContext,
		&pgDesc_hitgroup,
		1, 
		&pgOptions,
		NULL, NULL,
		&hitgroupPG
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 0;

	OptixProgramGroup program_groups[] = { raygenPG, missPG, hitgroupPG };

	error_OptiX = optixPipelineCreate(
		cknn.optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		program_groups,
		3,
		NULL, NULL,
		&cknn.pipeline
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_OptiX = optixPipelineSetStackSize(
		cknn.pipeline, 
		0,
		0,
		2 * 1024 * 8, // !!! !!! !!! SOME NASTY CONSTANT !!! !!! !!!
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	cknn.sbt = new OptixShaderBindingTable();

	// *********************************************************************************************

	SbtRecord rec_raygen;
	error_OptiX = optixSbtRecordPackHeader(raygenPG, &rec_raygen);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&cknn.raygenRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.raygenRecordsBuffer, &rec_raygen, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	cknn.sbt->raygenRecord = (CUdeviceptr)cknn.raygenRecordsBuffer;

	// *********************************************************************************************

	SbtRecord rec_miss;
	error_OptiX = optixSbtRecordPackHeader(missPG, &rec_miss);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&cknn.missRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.missRecordsBuffer, &rec_miss, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	cknn.sbt->missRecordBase = (CUdeviceptr)cknn.missRecordsBuffer;
	cknn.sbt->missRecordStrideInBytes = sizeof(SbtRecord);
	cknn.sbt->missRecordCount = 1;

	// *********************************************************************************************

	SbtRecord rec_hitgroup;
	error_OptiX = optixSbtRecordPackHeader(hitgroupPG, &rec_hitgroup);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&cknn.hitgroupRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.hitgroupRecordsBuffer, &rec_hitgroup, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	cknn.sbt->hitgroupRecordBase          = (CUdeviceptr)cknn.hitgroupRecordsBuffer;
	cknn.sbt->hitgroupRecordStrideInBytes = sizeof(SbtRecord);
	cknn.sbt->hitgroupRecordCount         = 1;

	// *********************************************************************************************

	// !!! !!! !!!
	cknn.chi_square_squared_radius = chi_square_squared_radius;
	// !!! !!! !!!

	// *********************************************************************************************

	float3 *gaussian_as_polygon_vertices = (float3 *)malloc(sizeof(float3) * 12);
	int3 *gaussian_as_polygon_indices = (int3 *)malloc(sizeof(int3) * 20);

	float phi = (1.0f + sqrt(5.0f)) / 2.0f;
	float scale = sqrt(3.0f * chi_square_squared_radius) / (phi * phi); // !!! !!! !!!
	
	// Vertices
	gaussian_as_polygon_vertices[0] = make_float3(-1.0f * scale, phi * scale, 0.0f * scale);
	gaussian_as_polygon_vertices[1] = make_float3(1.0f * scale, phi * scale, 0.0f * scale);
	gaussian_as_polygon_vertices[2] = make_float3(-1.0f * scale, -phi * scale, 0.0f * scale);
	gaussian_as_polygon_vertices[3] = make_float3(1.0f * scale, -phi * scale, 0.0f * scale);

	gaussian_as_polygon_vertices[4] = make_float3(0.0f * scale, -1.0f * scale, phi * scale);
	gaussian_as_polygon_vertices[5] = make_float3(0.0f * scale, 1.0f * scale, phi * scale);
	gaussian_as_polygon_vertices[6] = make_float3(0.0f * scale, -1.0f * scale, -phi * scale);
	gaussian_as_polygon_vertices[7] = make_float3(0.0f * scale, 1.0f * scale, -phi * scale);

	gaussian_as_polygon_vertices[8] = make_float3(phi * scale, 0.0f * scale, -1.0f * scale);
	gaussian_as_polygon_vertices[9] = make_float3(phi * scale, 0.0f * scale, 1.0f * scale);
	gaussian_as_polygon_vertices[10] = make_float3(-phi * scale, 0.0f * scale, -1.0f * scale);
	gaussian_as_polygon_vertices[11] = make_float3(-phi * scale, 0.0f * scale, 1.0f * scale);

	// Indices
	gaussian_as_polygon_indices[0] = make_int3(0, 11, 5);
	gaussian_as_polygon_indices[1] = make_int3(0, 5, 1);
	gaussian_as_polygon_indices[2] = make_int3(0, 1, 7);
	gaussian_as_polygon_indices[3] = make_int3(0, 7, 10);
	gaussian_as_polygon_indices[4] = make_int3(0, 10, 11);

	gaussian_as_polygon_indices[5] = make_int3(1, 5, 9);
	gaussian_as_polygon_indices[6] = make_int3(5, 11, 4);
	gaussian_as_polygon_indices[7] = make_int3(11, 10, 2);
	gaussian_as_polygon_indices[8] = make_int3(10, 7, 6);
	gaussian_as_polygon_indices[9] = make_int3(7, 1, 8);

	gaussian_as_polygon_indices[10] = make_int3(3, 9, 4);
	gaussian_as_polygon_indices[11] = make_int3(3, 4, 2);
	gaussian_as_polygon_indices[12] = make_int3(3, 2, 6);
	gaussian_as_polygon_indices[13] = make_int3(3, 6, 8);
	gaussian_as_polygon_indices[14] = make_int3(3, 8, 9);

	gaussian_as_polygon_indices[15] = make_int3(4, 9, 5);
	gaussian_as_polygon_indices[16] = make_int3(2, 4, 11);
	gaussian_as_polygon_indices[17] = make_int3(6, 2, 10);
	gaussian_as_polygon_indices[18] = make_int3(8, 6, 7);
	gaussian_as_polygon_indices[19] = make_int3(9, 8, 1);

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&cknn.gaussian_as_polygon_vertices, sizeof(float3) * 12);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.gaussian_as_polygon_vertices, gaussian_as_polygon_vertices, sizeof(float3) * 12, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&cknn.gaussian_as_polygon_indices, sizeof(int3) * 20);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.gaussian_as_polygon_indices, gaussian_as_polygon_indices, sizeof(int3) * 20, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	cknn.gaussians_as_polygons_vertices = NULL; // !!! !!! !!!
	cknn.gaussians_as_polygons_indices = NULL; // !!! !!! !!! 

	// *********************************************************************************************

	free(gaussian_as_polygon_vertices);
	free(gaussian_as_polygon_indices);

	// *********************************************************************************************

	cknn.asBuffer = NULL; // !!! !!! !!!

	// *********************************************************************************************

	*knn = cknn;

	return true;
}

// *************************************************************************************************
// CUDA_KNN_Fit                                                                                    *
// *************************************************************************************************

__global__ void UpdateGaussiansPoligonsVertices(S_CUDA_KNN cknn) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < cknn.number_of_means * 12) {
		int gauss_ind = tid / 12;
		float4 mean = cknn.means[gauss_ind];
		float3 vertex2D = cknn.gaussian_as_polygon_vertices[tid % 12];
		cknn.gaussians_as_polygons_vertices[tid] = make_float3(
			mean.x + (mean.w * vertex2D.x),
			mean.y + (mean.w * vertex2D.y),
			mean.z + (mean.w * vertex2D.z)
		);
	}
}

// *************************************************************************************************

__global__ void UpdateGaussiansPoligonsIndices(S_CUDA_KNN cknn) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < cknn.number_of_means * 20) {
		int gauss_ind = tid / 20;
		int3 Gauss_as_polygon_indices = cknn.gaussian_as_polygon_indices[tid % 20];
		cknn.gaussians_as_polygons_indices[tid] = make_int3(
			Gauss_as_polygon_indices.x + (gauss_ind * 12),
			Gauss_as_polygon_indices.y + (gauss_ind * 12),
			Gauss_as_polygon_indices.z + (gauss_ind * 12)
		);
	}
}

// *************************************************************************************************

/*
mean[i].x = X coordinate of the mean
mean[i].y = Y coordinate of the mean
mean[i].z = Z coordinate of the mean
mean[i].w = Scale of the Gaussian. Note that since the covariance matrix is radial, only one scale
            parameter is needed and there's no point using the quaternions parameters to describe
			the rotation of the Gaussian.
*/
extern "C" bool CUDA_KNN_Fit(float4 *means, int number_of_means, S_CUDA_KNN* knn) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	S_CUDA_KNN cknn = *knn;

	// *********************************************************************************************

	// !!! !!! !!!
	cknn.means = means;
	cknn.number_of_means = number_of_means;
	// !!! !!! !!!

	// *********************************************************************************************

	if (cknn.gaussians_as_polygons_vertices != NULL) {
		error_CUDA = cudaFree(cknn.gaussians_as_polygons_vertices);
		if (error_CUDA != cudaSuccess) return false;
	}

	error_CUDA = cudaMalloc(&cknn.gaussians_as_polygons_vertices, sizeof(float3) * number_of_means * 12);
	if (error_CUDA != cudaSuccess) return false;

	UpdateGaussiansPoligonsVertices<<<((number_of_means * 12) + 63) >> 6, 64>>>(cknn);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	if (cknn.gaussians_as_polygons_indices != NULL) {
		error_CUDA = cudaFree(cknn.gaussians_as_polygons_indices);
		if (error_CUDA != cudaSuccess) return false;
	}

	error_CUDA = cudaMalloc(&cknn.gaussians_as_polygons_indices, sizeof(int3) * number_of_means * 20); // !!! !!! !!!
	if (error_CUDA != cudaSuccess) return false;

	UpdateGaussiansPoligonsIndices<<<((number_of_means * 20) + 63) >> 6, 64>>>(cknn);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	OptixBuildInput tri_input = {};
	tri_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	tri_input.triangleArray.vertexBuffers = (CUdeviceptr *)&cknn.gaussians_as_polygons_vertices;
	tri_input.triangleArray.numVertices = cknn.number_of_means * 12;
	tri_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	tri_input.triangleArray.indexBuffer = (CUdeviceptr)cknn.gaussians_as_polygons_indices;
	tri_input.triangleArray.numIndexTriplets = cknn.number_of_means * 20;
	tri_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

	int input_tri_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
	tri_input.triangleArray.flags = (const unsigned int *)input_tri_flags;
	tri_input.triangleArray.numSbtRecords = 1;
	
	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		cknn.optixContext,
		&accel_options,
		&tri_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	unsigned long long *compactedSizeBuffer;
	error_CUDA = cudaMalloc(&compactedSizeBuffer, sizeof(unsigned long long) * 1);
	if (error_CUDA != cudaSuccess) return false;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)compactedSizeBuffer;

	void *tempBuffer;
	
	error_CUDA = cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes);
	if (error_CUDA != cudaSuccess) return false;

	void *outputBuffer;
	
	error_CUDA = cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		cknn.optixContext,
		0,
		&accel_options,
		&tri_input,
		1,  
		(CUdeviceptr)tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&cknn.asHandle,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) return false;

	unsigned long long compactedSize;

	error_CUDA = cudaMemcpy(&compactedSize, compactedSizeBuffer, sizeof(unsigned long long) * 1, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) return false;

	if (cknn.asBuffer != NULL) {
		error_CUDA = cudaFree(cknn.asBuffer);
		if (error_CUDA != cudaSuccess) return false;
	}

	error_CUDA = cudaMalloc(&cknn.asBuffer, compactedSize);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixAccelCompact(
		cknn.optixContext,
		0,
		cknn.asHandle,
		(CUdeviceptr)cknn.asBuffer,
		compactedSize,
		&cknn.asHandle
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(compactedSizeBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(tempBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(outputBuffer);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	*knn = cknn;

	return true;
}

// *************************************************************************************************
// CUDA_KNN_KNeighbors                                                                             *
// *************************************************************************************************

struct SReductionOperator_float4 {
	__device__ float4 operator()(const float4 &a, const float4 &b) const {
		return make_float4(
			0.0f,
			0.0f,
			0.0f,
			(a.w <= b.w) ? b.w : a.w
		);
	}
};

// *************************************************************************************************

#include <chrono>
#include <iostream>

extern "C" bool CUDA_KNN_KNeighbors(
	float4 *queried_points,
	int number_of_queried_points,
	int K,
	float *distances,
	int *indices,
	S_CUDA_KNN* knn
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	// Start timing
	auto start = std::chrono::high_resolution_clock::now();

	float4 max_R;

	S_CUDA_KNN cknn = *knn;

	try {
		max_R = thrust::reduce(
			thrust::device_pointer_cast(cknn.means),
			thrust::device_pointer_cast(cknn.means) + cknn.number_of_means,
			make_float4(0.0f, 0.0f, 0.0f, -INFINITY),
			SReductionOperator_float4()
		);
	} catch (...) {
		return false;
	}

	SLaunchParams launchParams;

	launchParams.traversable = cknn.asHandle;
	launchParams.means = cknn.means;
	launchParams.queried_points = queried_points;
	launchParams.distances = distances;
	launchParams.indices = indices;
	launchParams.chi_square_squared_radius = cknn.chi_square_squared_radius;
	launchParams.K = K;
	launchParams.max_R = max_R.w * sqrtf(cknn.chi_square_squared_radius);

	void *launchParamsBuffer;

	error_CUDA = cudaMalloc(&launchParamsBuffer, sizeof(SLaunchParams) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(launchParamsBuffer, &launchParams, sizeof(SLaunchParams) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixLaunch(
		cknn.pipeline,
		0,
		(CUdeviceptr)launchParamsBuffer,
		sizeof(SLaunchParams) * 1,
		cknn.sbt,
		number_of_queried_points,
		1,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(launchParamsBuffer);
	if (error_CUDA != cudaSuccess) return false;

	// End timing
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "CUDA_KNN_KNeighbors execution time: " << elapsed.count() << " s" << std::endl;

	*knn = cknn;

	return true;
}


extern "C" S_CUDA_KNN* create_cknn() {
    return new S_CUDA_KNN;
}