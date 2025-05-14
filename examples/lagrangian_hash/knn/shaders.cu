#include <optix_device.h>

#include "Header.cuh"

// *************************************************************************************************

struct LaunchParams {
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	float chi_square_squared_radius;

	float max_t;
	float max_R;
	float *distances;
	int *gauss_indices;
};

// *************************************************************************************************

extern "C" __constant__ LaunchParams optixLaunchParams;

// *************************************************************************************************

struct SRayPayload {
	// KNN
	float min_distance;
	float dist_array[1024];
	int gauss_ind[1024];
	int neighbors_num;
	float max_dist_so_far;
};

extern "C" __global__ void __raygen__renderFrame() {
	// KNN
	int x = optixGetLaunchIndex().x;
	int y = optixGetLaunchIndex().y;

	REAL3_R d = make_REAL3_R(
		(((REAL_R)-0.5) + ((x + ((REAL_R)0.5)) / optixLaunchParams.width)) * optixLaunchParams.double_tan_half_fov_x,
		(((REAL_R)-0.5) + ((y + ((REAL_R)0.5)) / optixLaunchParams.height)) * optixLaunchParams.double_tan_half_fov_y,
		1
	);
	REAL3_R v = make_REAL3_R(
		MAD_R(optixLaunchParams.R.x, d.x, MAD_R(optixLaunchParams.D.x, d.y, optixLaunchParams.F.x * d.z)),
		MAD_R(optixLaunchParams.R.y, d.x, MAD_R(optixLaunchParams.D.y, d.y, optixLaunchParams.F.y * d.z)),
		MAD_R(optixLaunchParams.R.z, d.x, MAD_R(optixLaunchParams.D.z, d.y, optixLaunchParams.F.z * d.z))
	);

	// *** *** *** *** ***

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *** *** *** *** ***

	for (int i = 0; i < NUMBER_OF_SAMPLES; ++i) {
		float t = ((i + 0.5f) / NUMBER_OF_SAMPLES) * optixLaunchParams.max_t;
		float3 O_prim = make_float3(
			optixLaunchParams.O.x + (v.x * t),
			optixLaunchParams.O.y + (v.y * t),
			optixLaunchParams.O.z + (v.z * t)
		);

		rp.neighbors_num = 0;
		rp.max_dist_so_far = -INFINITY;

		optixTrace(
			optixLaunchParams.traversable,
			O_prim,
			v,
			0.0f,
			INFINITY,
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES, //OPTIX_RAY_FLAG_NONE
			0,
			1,
			0,

			rp_addr_lo,
			rp_addr_hi
		);

		for (int j = 1; j < rp.neighbors_num; ++j) {
			float dist1 = rp.dist_array[j];
			float ind1  = rp.gauss_ind[j];

			int k;
			for (k = j; k > 0; --k) {
				float dist2 = rp.dist_array[k - 1];

				if (dist1 < dist2) {
					rp.dist_array[k] = dist2;
					rp.gauss_ind[k] = rp.gauss_ind[k - 1];
				} else {
					break;
				}
			}

			if (k < j) {
				rp.dist_array[k] = dist1;
				rp.gauss_ind[k] = ind1;
			}
		}

		if ((x == 10) && (y == 10)) {
			if (rp.neighbors_num >= 1){
				optixLaunchParams.distances[i] = rp.dist_array[0];
				optixLaunchParams.gauss_indices[i] = rp.gauss_ind[0];
			}
			else{
				optixLaunchParams.distances[i] = INFINITY;
				optixLaunchParams.gauss_indices[i] = -1;
			}
		}
	}
}

// *************************************************************************************************

extern "C" __global__ void __anyhit__radiance() {
	// KNN
	unsigned Gauss_ind = optixGetPrimitiveIndex();
	Gauss_ind /= NUMBER_OF_FACES;

	float4 GC_2 = optixLaunchParams.GC_part_2[Gauss_ind];
	float4 GC_3 = optixLaunchParams.GC_part_3[Gauss_ind];
	
	// *** *** *** *** ***

	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *** *** *** *** ***

	float tMin = optixGetRayTmax();
	float3 O = optixGetObjectRayOrigin();
	float3 d = make_float3(GC_2.x - O.x, GC_2.y - O.y, GC_2.z - O.z);
	float max_distance = fmaxf(expf(GC_2.w), fmaxf(expf(GC_3.x), expf(GC_3.y))) * sqrtf(11.34487f);
	float distance = sqrtf((d.x * d.x) + (d.y * d.y) + (d.z * d.z));
	if (distance < max_distance) {
		if (rp->neighbors_num < 16) {
			if (distance > rp->max_dist_so_far)
				rp->max_dist_so_far = distance;
			rp->dist_array[rp->neighbors_num] = distance;
			rp->gauss_ind[rp->neighbors_num] = (float)Gauss_ind;
			++rp->neighbors_num;
		} else {
			if (distance < rp->max_dist_so_far) {
				if (rp->neighbors_num < 1024) {
					rp->dist_array[rp->neighbors_num] = distance;
					rp->gauss_ind[rp->neighbors_num] = (float)Gauss_ind;
					++rp->neighbors_num;
				}
			}
		}
	}
	if (rp->neighbors_num < 16) {
		if (tMin <= 2.0f * optixLaunchParams.max_R)
			optixIgnoreIntersection();
	} else {
		if (tMin <= (2.0f * optixLaunchParams.max_R) - rp->max_dist_so_far)
			optixIgnoreIntersection();
	}
}

// *************************************************************************************************

extern "C" __global__ void __closesthit__radiance() {
}
