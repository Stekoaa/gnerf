#include <optix_device.h>

#include "Header.cuh"

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

	REAL3_R v = make_REAL3_R(1.0f, 0.0f, 0.0f);

	// *** *** *** *** ***

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *** *** *** *** ***

	rp.neighbors_num = 0;
	rp.max_dist_so_far = -INFINITY;

	optixTrace(
		optixLaunchParams.traversable,
		optixLaunchParams.coords[x],
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

	if (rp.neighbors_num >= 1){
		optixLaunchParams.distances[x] = rp.dist_array[0];
		optixLaunchParams.gauss_indices[x] = rp.gauss_ind[0];
	}
	else{
		optixLaunchParams.distances[x] = INFINITY;
		optixLaunchParams.gauss_indices[x] = -1;
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
