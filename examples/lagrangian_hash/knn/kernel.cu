#include "Header.cuh"

#include "optix_stubs.h"
#include "optix_function_table_definition.h"

// !!! !!! !!!
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
// !!! !!! !!!

// *************************************************************************************************

// !!! !!! !!! EXPERIMENTAL !!! !!! !!!
struct AABB { float a; float b; float c; float d; float e; float f; };
int *needsToBeRemoved_host;
__device__ int *needsToBeRemoved;
int *scatterBuffer;
// !!! !!! !!! EXPERIMENTAL !!! !!! !!!

// *************************************************************************************************

struct SAuxiliaryValues {
	uint3 scene_lower_bound = make_uint3(0xFF800000, 0xFF800000, 0xFF800000);
	uint3 scene_upper_bound = make_uint3(0x007FFFFF, 0x007FFFFF, 0x007FFFFF);
} initial_values;

__device__ struct {
	uint3 scene_lower_bound;
	uint3 scene_upper_bound;
} auxiliary_values;

__constant__ float scene_extent;

// *************************************************************************************************

int densification_end_epoch_host;
float min_s_coefficients_clipping_threshold_host;
float max_s_coefficients_clipping_threshold_host;
float chi_square_squared_radius_host; 
int max_Gaussians_per_model_host;

__constant__ int densification_end_epoch;
__constant__ float min_s_coefficients_clipping_threshold;
__constant__ float max_s_coefficients_clipping_threshold;
__constant__ float chi_square_squared_radius; 
__constant__ int max_Gaussians_per_model;

// *************************************************************************************************

struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *************************************************************************************************

static void LoadFromFile(const char *fPath, int epochNum, const char *fExtension, void *buf, int size) {
	FILE *f;

	char fName[256];
	sprintf(fName, "%s/%d.%s", fPath, epochNum, fExtension);

	f = fopen(fName, "rb");
	fread(buf, size, 1, f);
	fclose(f);
}

// *************************************************************************************************

unsigned Float2SortableUint(float value) {
	unsigned tmp = *((unsigned *)&value);
	return tmp ^ ((tmp >= 0x80000000) ? 0xFFFFFFFF : 0x80000000);
}

// *************************************************************************************************

float SortableUint2Float(unsigned value) {
	unsigned tmp = value ^ ((value < 0x80000000) ? 0xFFFFFFFF : 0x80000000);
	return *((float *)&tmp);
}

// *************************************************************************************************

__global__ void UpdateGaussiansPoligonsVertices(SOptiXRenderParams params_OptiX) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < params_OptiX.numberOfGaussians * NUMBER_OF_VERTICES) {
		int Gauss_ind = tid / NUMBER_OF_VERTICES;
		
		float4 GC_2 = params_OptiX.GC_part_2_1[Gauss_ind];
		float4 GC_3 = params_OptiX.GC_part_3_1[Gauss_ind];
		
		float sX = expf(GC_2.w);
		float sY = expf(GC_3.x);
		float sZ = expf(GC_3.y);

		// KNN
		float maxS = fmaxf(sX, fmaxf(sY, sZ));
		float3 vertex2D = params_OptiX.Gaussian_as_polygon_vertices[tid % NUMBER_OF_VERTICES];
		params_OptiX.Gaussians_as_polygon_vertices[tid] = make_float3(
			GC_2.x + (maxS * vertex2D.x),
			GC_2.y + (maxS * vertex2D.y),
			GC_2.z + (maxS * vertex2D.z)
		);
	}
}

// *************************************************************************************************

__global__ void UpdateGaussiansPoligonsIndices(SOptiXRenderParams params_OptiX) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < params_OptiX.numberOfGaussians * NUMBER_OF_FACES) {
		int Gauss_ind = tid / NUMBER_OF_FACES;
		int3 Gauss_as_polygon_indices = params_OptiX.Gaussian_as_polygon_indices[tid % NUMBER_OF_FACES];
		params_OptiX.Gaussians_as_polygon_indices[tid] = make_int3(
			Gauss_as_polygon_indices.x + (Gauss_ind * NUMBER_OF_VERTICES),
			Gauss_as_polygon_indices.y + (Gauss_ind * NUMBER_OF_VERTICES),
			Gauss_as_polygon_indices.z + (Gauss_ind * NUMBER_OF_VERTICES)
		);
	}
}

//**************************************************************************************************
//* InitializeOptiXRenderer                                                                        *
//**************************************************************************************************

extern "C" bool InitializeOptiXRenderer(
	SRenderParams &params,
	SOptiXRenderParams &params_OptiX,
	bool loadFromFile = false,
	int epoch = 0
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	CUresult error_CUDA_Driver_API;

	error_CUDA = cudaFree(0);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixInit();
	if (error_OptiX != OPTIX_SUCCESS) return false;

	CUcontext cudaContext;
	error_CUDA_Driver_API = cuCtxGetCurrent(&cudaContext);
	if(error_CUDA_Driver_API != CUDA_SUCCESS) return false;

	error_OptiX = optixDeviceContextCreate(cudaContext, 0, &params_OptiX.optixContext);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	FILE *f = fopen("/workspace/gnerf/examples/lagrangian_hash/knn/shaders.cu.ptx", "rb");
	fseek(f, 0, SEEK_END);
	int ptxCodeSize = ftell(f);
	fclose(f);

	char *ptxCode = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	char *buffer = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	ptxCode[0] = 0; // !!! !!! !!!

	f = fopen("/workspace/gnerf/examples/lagrangian_hash/knn/shaders.cu.ptx", "rb");
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

	moduleCompileOptions.maxRegisterCount = 40; // 50
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
		pipelineCompileOptions.numPayloadValues = 3; // 3 // !!! !!! !!!
		pipelineCompileOptions.numAttributeValues = 2;
	#else
		pipelineCompileOptions.numPayloadValues = 4;
		pipelineCompileOptions.numAttributeValues = 4;
	#endif
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	OptixModule module;
	error_OptiX = optixModuleCreateFromPTX(
		params_OptiX.optixContext,
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
	pgDesc_raygen.raygen.entryFunctionName = "__raygen__renderFrame";

	OptixProgramGroup raygenPG;
	error_OptiX = optixProgramGroupCreate(
		params_OptiX.optixContext,
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
		params_OptiX.optixContext,
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
	/*//pgDesc_hitgroup.hitgroup.moduleAH            = module; // !!! !!! !!!
	//pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__radiance"; // !!! !!! !!!
	pgDesc_hitgroup.hitgroup.moduleCH            = module;           
	pgDesc_hitgroup.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc_hitgroup.hitgroup.moduleIS            = module;
	pgDesc_hitgroup.hitgroup.entryFunctionNameIS = "__intersection__is";*/
	
	// !!! !!! !!! TRIANGLES !!! !!! !!!
	pgDesc_hitgroup.hitgroup.moduleCH            = module;           
	pgDesc_hitgroup.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc_hitgroup.hitgroup.moduleAH            = module; // !!! !!! !!!
	pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__radiance"; // !!! !!! !!!
	// !!! !!! !!! TRIANGLES !!! !!! !!!

	OptixProgramGroup hitgroupPG;
	error_OptiX = optixProgramGroupCreate(
		params_OptiX.optixContext,
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
		params_OptiX.optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		program_groups,
		3,
		NULL, NULL,
		&params_OptiX.pipeline
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_OptiX = optixPipelineSetStackSize(
		params_OptiX.pipeline, 
		0,
		0,
		2*1024 * 8, // !!! !!! !!!
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	params_OptiX.sbt = new OptixShaderBindingTable();

	// *********************************************************************************************

	SbtRecord rec_raygen;
	error_OptiX = optixSbtRecordPackHeader(raygenPG, &rec_raygen);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&params_OptiX.raygenRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.raygenRecordsBuffer, &rec_raygen, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	params_OptiX.sbt->raygenRecord = (CUdeviceptr)params_OptiX.raygenRecordsBuffer;

	// *********************************************************************************************

	SbtRecord rec_miss;
	error_OptiX = optixSbtRecordPackHeader(missPG, &rec_miss);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&params_OptiX.missRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.missRecordsBuffer, &rec_miss, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	params_OptiX.sbt->missRecordBase = (CUdeviceptr)params_OptiX.missRecordsBuffer;
	params_OptiX.sbt->missRecordStrideInBytes = sizeof(SbtRecord);
	params_OptiX.sbt->missRecordCount = 1;

	// *********************************************************************************************

	SbtRecord rec_hitgroup;
	error_OptiX = optixSbtRecordPackHeader(hitgroupPG, &rec_hitgroup);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&params_OptiX.hitgroupRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.hitgroupRecordsBuffer, &rec_hitgroup, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	params_OptiX.sbt->hitgroupRecordBase          = (CUdeviceptr)params_OptiX.hitgroupRecordsBuffer;
	params_OptiX.sbt->hitgroupRecordStrideInBytes = sizeof(SbtRecord);
	params_OptiX.sbt->hitgroupRecordCount         = 1;

	// *********************************************************************************************

	if (!loadFromFile) {
		params_OptiX.numberOfGaussians = params.numberOfGaussians; // !!! !!! !!!
		if ((epoch + 1 <= densification_end_epoch_host) && (params_OptiX.numberOfGaussians <= max_Gaussians_per_model_host)) { // !!! !!! !!!
			params_OptiX.scatterBufferSize = 1; // !!! !!! !!!
			params_OptiX.maxNumberOfGaussians1 = params_OptiX.numberOfGaussians * 1.125f; // !!! !!! !!!
			params_OptiX.maxNumberOfGaussians = params_OptiX.numberOfGaussians * REALLOC_MULTIPLIER2; // !!! !!! !!!
		} else {
			params_OptiX.scatterBufferSize = 1; // !!! !!! !!!
			params_OptiX.maxNumberOfGaussians1 = params_OptiX.numberOfGaussians; // !!! !!! !!!
			params_OptiX.maxNumberOfGaussians = params_OptiX.numberOfGaussians;
		}
	} else {
		FILE *f;

		char fName[256];
		sprintf(fName, "dump/save/%d.GC1", epoch);

		f = fopen(fName, "rb");
		fseek(f, 0, SEEK_END);
		params_OptiX.numberOfGaussians = ftell(f) / sizeof(float4); // !!! !!! !!!
		fclose(f);

		if ((epoch + 1 <= densification_end_epoch_host) && (params_OptiX.numberOfGaussians <= max_Gaussians_per_model_host)) { // !!! !!! !!!
			params_OptiX.scatterBufferSize = 1; // !!! !!! !!!
			params_OptiX.maxNumberOfGaussians1 = params_OptiX.numberOfGaussians * 1.125f; // !!! !!! !!!
			params_OptiX.maxNumberOfGaussians = params_OptiX.numberOfGaussians * REALLOC_MULTIPLIER2; // !!! !!! !!!
		} else {
			params_OptiX.scatterBufferSize = 1; // !!! !!! !!!
			params_OptiX.maxNumberOfGaussians1 = params_OptiX.numberOfGaussians; // !!! !!! !!!
			params_OptiX.maxNumberOfGaussians = params_OptiX.numberOfGaussians;
		}
	}

	// *********************************************************************************************

	float4 *GC_part_1 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float4 *GC_part_2 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float4 *GC_part_3 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float2 *GC_part_4 = (float2 *)malloc(sizeof(float2) * params_OptiX.numberOfGaussians);

	if (!loadFromFile) {
		for (int i = 0; i < params_OptiX.numberOfGaussians; ++i) {
			GC_part_2[i].x = params.GC[i].mX;
			GC_part_2[i].y = params.GC[i].mY;
			GC_part_2[i].z = params.GC[i].mZ;
			GC_part_2[i].w = params.GC[i].sX;

			GC_part_3[i].x = params.GC[i].sY;
			GC_part_3[i].y = params.GC[i].sZ;
			GC_part_3[i].z = params.GC[i].qr;
			GC_part_3[i].w = params.GC[i].qi;

			GC_part_4[i].x = params.GC[i].qj;
			GC_part_4[i].y = params.GC[i].qk;
		}
	} else {
		LoadFromFile("dump/save", epoch, "GC1", GC_part_1, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile("dump/save", epoch, "GC2", GC_part_2, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile("dump/save", epoch, "GC3", GC_part_3, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile("dump/save", epoch, "GC4", GC_part_4, sizeof(float2) * params_OptiX.numberOfGaussians);
	}

	// *********************************************************************************************

	// !!! !!! !!! TRIANGLES !!! !!! !!!
	// Polygon
	/*float3 *Gaussian_as_polygon_vertices = (float3 *)malloc(sizeof(float3) * 1 * NUMBER_OF_VERTICES);
	int3 *Gaussian_as_polygon_indices = (int3 *)malloc(sizeof(int3) * 1 * NUMBER_OF_FACES);
	
	for (int i = 0; i < NUMBER_OF_VERTICES; ++i)
		Gaussian_as_polygon_vertices[i] = make_float3(
			0.0f,
			cosf(i * ((2.0f * M_PI) / NUMBER_OF_VERTICES)) * sqrtf(chi_square_squared_radius_host),
			sinf(i * ((2.0f * M_PI) / NUMBER_OF_VERTICES)) * sqrtf(chi_square_squared_radius_host)
		);
	for (int i = 0; i < NUMBER_OF_FACES; ++i)
		Gaussian_as_polygon_indices[i] = make_int3(0, i + 1, i + 2);*/

	// *** *** *** *** ***

	// Icosahedron
	float3 *Gaussian_as_polygon_vertices = (float3 *)malloc(sizeof(float3) * 1 * 12);
	int3 *Gaussian_as_polygon_indices = (int3 *)malloc(sizeof(int3) * 1 * 20);

	float phi = (1.0f + sqrt(5.0f)) / 2.0f;
	float scale = sqrt(3.0f * chi_square_squared_radius_host) / (phi * phi); // !!! !!! !!!

	// vertices
	Gaussian_as_polygon_vertices[0] = make_float3(-1.0f * scale, phi * scale, 0.0f * scale);
	Gaussian_as_polygon_vertices[1] = make_float3(1.0f * scale, phi * scale, 0.0f * scale);
	Gaussian_as_polygon_vertices[2] = make_float3(-1.0f * scale, -phi * scale, 0.0f * scale);
	Gaussian_as_polygon_vertices[3] = make_float3(1.0f * scale, -phi * scale, 0.0f * scale);
		
	Gaussian_as_polygon_vertices[4] = make_float3(0.0f * scale, -1.0f * scale, phi * scale);
	Gaussian_as_polygon_vertices[5] = make_float3(0.0f * scale, 1.0f * scale, phi * scale);
	Gaussian_as_polygon_vertices[6] = make_float3(0.0f * scale, -1.0f * scale, -phi * scale);
	Gaussian_as_polygon_vertices[7] = make_float3(0.0f * scale, 1.0f * scale, -phi * scale);
	
	Gaussian_as_polygon_vertices[8] = make_float3(phi * scale, 0.0f * scale, -1.0f * scale);
	Gaussian_as_polygon_vertices[9] = make_float3(phi * scale, 0.0f * scale, 1.0f * scale);
	Gaussian_as_polygon_vertices[10] = make_float3(-phi * scale, 0.0f * scale, -1.0f * scale);
	Gaussian_as_polygon_vertices[11] = make_float3(-phi * scale, 0.0f * scale, 1.0f * scale);

	// indices
	Gaussian_as_polygon_indices[0] = make_int3(0, 11, 5);
	Gaussian_as_polygon_indices[1] = make_int3(0, 5, 1);
	Gaussian_as_polygon_indices[2] = make_int3(0, 1, 7);
	Gaussian_as_polygon_indices[3] = make_int3(0, 7, 10);
	Gaussian_as_polygon_indices[4] = make_int3(0, 10, 11);
		
	Gaussian_as_polygon_indices[5] = make_int3(1, 5, 9);
	Gaussian_as_polygon_indices[6] = make_int3(5, 11, 4);
	Gaussian_as_polygon_indices[7] = make_int3(11, 10, 2);
	Gaussian_as_polygon_indices[8] = make_int3(10, 7, 6);
	Gaussian_as_polygon_indices[9] = make_int3(7, 1, 8);
		
	Gaussian_as_polygon_indices[10] = make_int3(3, 9, 4);
	Gaussian_as_polygon_indices[11] = make_int3(3, 4, 2);
	Gaussian_as_polygon_indices[12] = make_int3(3, 2, 6);
	Gaussian_as_polygon_indices[13] = make_int3(3, 6, 8);
	Gaussian_as_polygon_indices[14] = make_int3(3, 8, 9);
		
	Gaussian_as_polygon_indices[15] = make_int3(4, 9, 5);
	Gaussian_as_polygon_indices[16] = make_int3(2, 4, 11);
	Gaussian_as_polygon_indices[17] = make_int3(6, 2, 10);
	Gaussian_as_polygon_indices[18] = make_int3(8, 6, 7);
	Gaussian_as_polygon_indices[19] = make_int3(9, 8, 1);
	// !!! !!! !!! TRIANGLES !!! !!! !!!

	// *********************************************************************************************

	SAuxiliaryValues auxiliary_values_local;
	auxiliary_values_local.scene_lower_bound = initial_values.scene_lower_bound;
	auxiliary_values_local.scene_upper_bound = initial_values.scene_upper_bound;

	float scene_extent_local;

	float max_R = -INFINITY;

	for (int i = 0; i < params_OptiX.numberOfGaussians; ++i) {
		float aa = GC_part_3[i].z * GC_part_3[i].z;
		float bb = GC_part_3[i].w * GC_part_3[i].w;
		float cc = GC_part_4[i].x * GC_part_4[i].x;
		float dd = GC_part_4[i].y * GC_part_4[i].y;
		float s = 2.0f / (aa + bb + cc + dd);

		float bs = GC_part_3[i].w * s;  float cs = GC_part_4[i].x * s;  float ds = GC_part_4[i].y * s;
		float ab = GC_part_3[i].z * bs; float ac = GC_part_3[i].z * cs; float ad = GC_part_3[i].z * ds;
		bb = bb * s;                    float bc = GC_part_3[i].w * cs; float bd = GC_part_3[i].w * ds;
		cc = cc * s;                    float cd = GC_part_4[i].x * ds;       dd = dd * s;

		float Q11 = 1.0f - cc - dd;
		float Q12 = bc - ad;
		float Q13 = bd + ac;

		float Q21 = bc + ad;
		float Q22 = 1.0f - bb - dd;
		float Q23 = cd - ab;

		float Q31 = bd - ac;
		float Q32 = cd + ab;
		float Q33 = 1.0f - bb - cc;

		// OLD INVERSE SIGMOID ACTIVATION FUNCTION FOR SCALE PARAMETERS
		/*float sX = 1.0f / (1.0f + expf(-GC_part_2[i].w));
		float sY = 1.0f / (1.0f + expf(-GC_part_3[i].x));
		float sZ = 1.0f / (1.0f + expf(-GC_part_3[i].y));*/

		// NEW EXPONENTIAL ACTIVATION FUNCTION FOR SCALE PARAMETERS
		float sX = expf(GC_part_2[i].w);
		float sY = expf(GC_part_3[i].x);
		float sZ = expf(GC_part_3[i].y);
		
		// KNN
		float R = -INFINITY;
		if (sX > R) R = sX;
		if (sY > R) R = sY;
		if (sZ > R) R = sZ;
		R = R * sqrtf(chi_square_squared_radius_host);
		if (R > max_R) max_R = R;

		// v2
		float tmpX = sqrtf(chi_square_squared_radius_host * ((sX * sX * Q11 * Q11) + (sY * sY * Q12 * Q12) + (sZ * sZ * Q13 * Q13)));
		float tmpY = sqrtf(chi_square_squared_radius_host * ((sX * sX * Q21 * Q21) + (sY * sY * Q22 * Q22) + (sZ * sZ * Q23 * Q23)));
		float tmpZ = sqrtf(chi_square_squared_radius_host * ((sX * sX * Q31 * Q31) + (sY * sY * Q32 * Q32) + (sZ * sZ * Q33 * Q33)));

		float lB = GC_part_2[i].x - tmpX; // !!! !!! !!!
		float rB = GC_part_2[i].x + tmpX; // !!! !!! !!!

		float uB = GC_part_2[i].y - tmpY; // !!! !!! !!!
		float dB = GC_part_2[i].y + tmpY; // !!! !!! !!!

		float bB = GC_part_2[i].z - tmpZ; // !!! !!! !!!
		float fB = GC_part_2[i].z + tmpZ; // !!! !!! !!!

		auxiliary_values_local.scene_lower_bound.x = (
			(Float2SortableUint(lB) < auxiliary_values_local.scene_lower_bound.x) ?
			Float2SortableUint(lB) :
			auxiliary_values_local.scene_lower_bound.x
		);
		auxiliary_values_local.scene_lower_bound.y = (
			(Float2SortableUint(uB) < auxiliary_values_local.scene_lower_bound.y) ?
			Float2SortableUint(uB) :
			auxiliary_values_local.scene_lower_bound.y
		);
		auxiliary_values_local.scene_lower_bound.z = (
			(Float2SortableUint(bB) < auxiliary_values_local.scene_lower_bound.z) ?
			Float2SortableUint(bB) :
			auxiliary_values_local.scene_lower_bound.z
		);
	
		auxiliary_values_local.scene_upper_bound.x = (
			(Float2SortableUint(rB) > auxiliary_values_local.scene_upper_bound.x) ?
			Float2SortableUint(rB) :
			auxiliary_values_local.scene_upper_bound.x
		);
		auxiliary_values_local.scene_upper_bound.y = (
			(Float2SortableUint(dB) > auxiliary_values_local.scene_upper_bound.y) ?
			Float2SortableUint(dB) :
			auxiliary_values_local.scene_upper_bound.y
		);
		auxiliary_values_local.scene_upper_bound.z = (
			(Float2SortableUint(fB) > auxiliary_values_local.scene_upper_bound.z) ?
			Float2SortableUint(fB) :
			auxiliary_values_local.scene_upper_bound.z
		);
	}

	// *** *** *** *** ***

	// KNN
	float lB = SortableUint2Float(auxiliary_values_local.scene_lower_bound.x); // !!! !!! !!!
	float rB = SortableUint2Float(auxiliary_values_local.scene_upper_bound.x); // !!! !!! !!!

	float uB = SortableUint2Float(auxiliary_values_local.scene_lower_bound.y); // !!! !!! !!!
	float dB = SortableUint2Float(auxiliary_values_local.scene_upper_bound.y); // !!! !!! !!!

	float bB = SortableUint2Float(auxiliary_values_local.scene_lower_bound.z); // !!! !!! !!!
	float fB = SortableUint2Float(auxiliary_values_local.scene_upper_bound.z); // !!! !!! !!!

	float max_t = -INFINITY;

	params_OptiX.max_t = max_t;
	params_OptiX.max_R = max_R;

	params_OptiX.distances_host = (float *)malloc(sizeof(float) * params_OptiX.batch_size);
	params_OptiX.gauss_indices_host = (int *)malloc(sizeof(int) * params_OptiX.batch_size);

	error_CUDA = cudaMalloc(&params_OptiX.distances, sizeof(float) * params_OptiX.batch_size);
	if (error_CUDA != cudaSuccess) return false;
	error_CUDA = cudaMalloc(&params_OptiX.gauss_indices, sizeof(int) * params_OptiX.batch_size);
	if (error_CUDA != cudaSuccess) return false;

	// *** *** *** *** ***

	float dX = SortableUint2Float(auxiliary_values_local.scene_upper_bound.x) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.x);
	float dY = SortableUint2Float(auxiliary_values_local.scene_upper_bound.y) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.y);
	float dZ = SortableUint2Float(auxiliary_values_local.scene_upper_bound.z) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.z);
	
	scene_extent_local = sqrtf((dX * dX) + (dY * dY) + (dZ * dZ));

	
	for (int i = 0; i < params_OptiX.numberOfGaussians; ++i) {
		// OLD INVERSE SIGMOID ACTIVATION FUNCTION FOR SCALE PARAMETERS
		/*float sX = 1.0f / (1.0f + expf(-GC_part_2[i].w));
		float sY = 1.0f / (1.0f + expf(-GC_part_3[i].x));
		float sZ = 1.0f / (1.0f + expf(-GC_part_3[i].y));*/

		// NEW EXPONENTIAL ACTIVATION FUNCTION FOR SCALE PARAMETERS
		float sX = expf(GC_part_2[i].w);
		float sY = expf(GC_part_3[i].x);
		float sZ = expf(GC_part_3[i].y);

		sX = ((sX < scene_extent_local * min_s_coefficients_clipping_threshold_host) ? scene_extent_local * min_s_coefficients_clipping_threshold_host : sX); // !!! !!! !!!
		sY = ((sY < scene_extent_local * min_s_coefficients_clipping_threshold_host) ? scene_extent_local * min_s_coefficients_clipping_threshold_host : sY);
		sZ = ((sZ < scene_extent_local * min_s_coefficients_clipping_threshold_host) ? scene_extent_local * min_s_coefficients_clipping_threshold_host : sZ);

		sX = ((sX > scene_extent_local * max_s_coefficients_clipping_threshold_host) ? scene_extent_local * max_s_coefficients_clipping_threshold_host : sX); // !!! !!! !!!
		sY = ((sY > scene_extent_local * max_s_coefficients_clipping_threshold_host) ? scene_extent_local * max_s_coefficients_clipping_threshold_host : sY);
		sZ = ((sZ > scene_extent_local * max_s_coefficients_clipping_threshold_host) ? scene_extent_local * max_s_coefficients_clipping_threshold_host : sZ);

		// OLD INVERSE SIGMOID ACTIVATION FUNCTION FOR SCALE PARAMETERS
		/*GC_part_2[i].w = -logf((1.0f / sX) - 1.0f);
		GC_part_3[i].x = -logf((1.0f / sY) - 1.0f);
		GC_part_3[i].y = -logf((1.0f / sZ) - 1.0f);*/

		// NEW EXPONENTIAL ACTIVATION FUNCTION FOR SCALE PARAMETERS
		GC_part_2[i].w = logf(sX);
		GC_part_3[i].x = logf(sY);
		GC_part_3[i].y = logf(sZ);
	}

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&needsToBeRemoved_host, sizeof(int) * params_OptiX.maxNumberOfGaussians);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpyToSymbol(needsToBeRemoved, &needsToBeRemoved_host, sizeof(int *));
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&scatterBuffer, sizeof(float) * 4 * params_OptiX.scatterBufferSize);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.maxNumberOfGaussians);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.maxNumberOfGaussians);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.maxNumberOfGaussians);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.maxNumberOfGaussians);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_1_1, GC_part_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_2_1, GC_part_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_3_1, GC_part_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_4_1, GC_part_4, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	free(GC_part_1);
	free(GC_part_2);
	free(GC_part_3);
	free(GC_part_4);

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.Gaussian_as_polygon_vertices, sizeof(float3) * 1 * NUMBER_OF_VERTICES);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.Gaussian_as_polygon_vertices, Gaussian_as_polygon_vertices, sizeof(float3) * 1 * NUMBER_OF_VERTICES, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&params_OptiX.Gaussian_as_polygon_indices, sizeof(int3) * 1 * NUMBER_OF_FACES);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(params_OptiX.Gaussian_as_polygon_indices, Gaussian_as_polygon_indices, sizeof(int3) * 1 * NUMBER_OF_FACES, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&params_OptiX.Gaussians_as_polygon_vertices, sizeof(float3) * params_OptiX.maxNumberOfGaussians1 * NUMBER_OF_VERTICES); // !!! !!! !!!
	if (error_CUDA != cudaSuccess) return false;

	UpdateGaussiansPoligonsVertices<<<((params_OptiX.numberOfGaussians * NUMBER_OF_VERTICES) + 63) >> 6, 64>>>(params_OptiX);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&params_OptiX.Gaussians_as_polygon_indices, sizeof(int3) * params_OptiX.maxNumberOfGaussians1 * NUMBER_OF_FACES); // !!! !!! !!!
	if (error_CUDA != cudaSuccess) return false;

	UpdateGaussiansPoligonsIndices<<<((params_OptiX.numberOfGaussians * NUMBER_OF_FACES) + 63) >> 6, 64>>>(params_OptiX);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpyToSymbol(auxiliary_values, &auxiliary_values_local, sizeof(SAuxiliaryValues) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpyToSymbol(scene_extent, &scene_extent_local, sizeof(float) * 1);
	if (error_CUDA != cudaSuccess) return false;

	free(Gaussian_as_polygon_vertices);
	free(Gaussian_as_polygon_indices);

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	// !!! !!! !!! TRIANGLES !!! !!! !!!
	OptixBuildInput aabb_input = {};
	aabb_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	aabb_input.triangleArray.vertexBuffers = (CUdeviceptr *)&params_OptiX.Gaussians_as_polygon_vertices;
	aabb_input.triangleArray.numVertices = params_OptiX.numberOfGaussians * NUMBER_OF_VERTICES;
	aabb_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	aabb_input.triangleArray.indexBuffer = (CUdeviceptr)params_OptiX.Gaussians_as_polygon_indices;
	aabb_input.triangleArray.numIndexTriplets = params_OptiX.numberOfGaussians * NUMBER_OF_FACES;
	aabb_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

	int input_tri_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
	aabb_input.triangleArray.flags = (const unsigned int *)input_tri_flags;
	aabb_input.triangleArray.numSbtRecords = 1;
	// !!! !!! !!! TRIANGLES !!! !!! !!!
	
	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		params_OptiX.optixContext,
		&accel_options,
		&aabb_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.compactedSizeBuffer, 8);
	if (error_CUDA != cudaSuccess) return false;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)params_OptiX.compactedSizeBuffer;

	params_OptiX.tempBufferSize = blasBufferSizes.tempSizeInBytes * 2; // !!! !!! !!!
	error_CUDA = cudaMalloc(&params_OptiX.tempBuffer, params_OptiX.tempBufferSize);
	if (error_CUDA != cudaSuccess) return false;

	params_OptiX.outputBufferSize = blasBufferSizes.outputSizeInBytes * 2; // !!! !!! !!!
	error_CUDA = cudaMalloc(&params_OptiX.outputBuffer, params_OptiX.outputBufferSize);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		params_OptiX.optixContext,
		0,
		&accel_options,
		&aabb_input,
		1,  
		(CUdeviceptr)params_OptiX.tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)params_OptiX.outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&params_OptiX.asHandle,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	unsigned long long compactedSize;
	error_CUDA = cudaMemcpy(&compactedSize, params_OptiX.compactedSizeBuffer, 8, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) return false;

	params_OptiX.asBufferSize = compactedSize * 2; // !!! !!! !!! 
	error_CUDA = cudaMalloc(&params_OptiX.asBuffer, params_OptiX.asBufferSize);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixAccelCompact(
		params_OptiX.optixContext,
		0,
		params_OptiX.asHandle,
		(CUdeviceptr)params_OptiX.asBuffer,
		compactedSize,
		&params_OptiX.asHandle
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	params_OptiX.width = params.w; // !!! !!! !!!
	params_OptiX.height = params.h; // !!! !!! !!!

	// *********************************************************************************************

	return true;
}

//**************************************************************************************************
//* RenderOptiX                                                                                    *
//**************************************************************************************************

extern "C" bool RenderOptiX(SOptiXRenderParams& params_OptiX) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	
	// *********************************************************************************************

	LaunchParams launchParams;
	launchParams.width = params_OptiX.width;
	launchParams.height = params_OptiX.height;
	launchParams.traversable = params_OptiX.asHandle;
	launchParams.GC_part_1 = params_OptiX.GC_part_1_1;
	launchParams.GC_part_2 = params_OptiX.GC_part_2_1;
	launchParams.GC_part_3 = params_OptiX.GC_part_3_1;
	launchParams.GC_part_4 = params_OptiX.GC_part_4_1;
	launchParams.chi_square_squared_radius = chi_square_squared_radius_host;
	// KNN
	launchParams.max_t = params_OptiX.max_t;
	launchParams.max_R = params_OptiX.max_R;
	launchParams.distances = params_OptiX.distances;
	launchParams.gauss_indices = params_OptiX.gauss_indices;
	launchParams.coords = params_OptiX.coords;
	launchParams.batch_size = params_OptiX.batch_size;

	void *launchParamsBuffer;
	error_CUDA = cudaMalloc(&launchParamsBuffer, sizeof(LaunchParams) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(launchParamsBuffer, &launchParams, sizeof(LaunchParams) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixLaunch(
		params_OptiX.pipeline,
		0,
		(CUdeviceptr)launchParamsBuffer,
		sizeof(LaunchParams) * 1,
		params_OptiX.sbt,
		params_OptiX.width,
		params_OptiX.height,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(
		params_OptiX.distances_host,
		params_OptiX.distances,
		sizeof(float) * params_OptiX.batch_size,
		cudaMemcpyDeviceToHost
	);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(
		params_OptiX.gauss_indices_host,
		params_OptiX.gauss_indices,
		sizeof(int) * params_OptiX.batch_size,
		cudaMemcpyDeviceToHost
	);

	return true;
}

extern "C" void fit(SGaussianComponent* GC, int numberOfGaussians, float3* coords, int batchSize, float* distances, int* gaussIndices) {

	SRenderParams params;
	params.GC = GC;
	params.numberOfGaussians = numberOfGaussians;
	params.w = 20;
	params.h = 20;
	params.double_tan_half_fov_x = 1.0f;
	params.double_tan_half_fov_y = 1.0f;

	// Initialize the OptiX parameters
	densification_end_epoch_host = 100;
	min_s_coefficients_clipping_threshold_host = 0.01f;
	max_s_coefficients_clipping_threshold_host = 10.0f;
	chi_square_squared_radius_host = 0.95f;
	max_Gaussians_per_model_host = 1024;
	
	SOptiXRenderParams params_OptiX;

	float3* d_coords;
	cudaMalloc(&d_coords, sizeof(float3) * batchSize);
	cudaMemcpy(d_coords, coords, sizeof(float3) * batchSize, cudaMemcpyHostToDevice);

	params_OptiX.coords = d_coords;
	params_OptiX.batch_size = batchSize;

	bool success = InitializeOptiXRenderer(params, params_OptiX, false, 0);
	printf("OptiX Renderer initialized: %s\n", success ? "true" : "false");

	success = RenderOptiX(params_OptiX);
	printf("OptiX Rendered: %s\n", success ? "true" : "false");
	
	// Copy data into memory provided by Python
	for (int i = 0; i < params_OptiX.batch_size; ++i) {
		distances[i] = params_OptiX.distances_host[i];
		gaussIndices[i] = params_OptiX.gauss_indices_host[i];
	}

	cudaFree(d_coords);

}