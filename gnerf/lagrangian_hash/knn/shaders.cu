#include "Header.cuh"

// *************************************************************************************************

extern "C" __constant__ SLaunchParams optixLaunchParams;

// *************************************************************************************************

struct SRayPayload {
	float dist_array[1024];
	int gauss_ind[1024];
	int neighbors_num;
	float max_dist_so_far;
};

// *************************************************************************************************

extern "C" __global__ void __raygen__() {
	int x = optixGetLaunchIndex().x;
	float4 queried_point = optixLaunchParams.queried_points[x];
	int number_of_queried_points = optixGetLaunchDimensions().x;
	float3 v = make_float3(1.0f, 0.0f, 0.0f);

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
		make_float3(queried_point.x, queried_point.y, queried_point.z),
		v,
		0.0f,
		INFINITY,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
		0,
		1,
		0,

		rp_addr_lo,
		rp_addr_hi
	);

	for (int i = 1; i < rp.neighbors_num; ++i) {
		float dist1 = rp.dist_array[i];
		float ind1  = rp.gauss_ind[i];

		int j;
		for (j = i; j > 0; --j) {
			float dist2 = rp.dist_array[j - 1];
			int ind2 = rp.gauss_ind[j - 1];

			if (dist1 < dist2) {
				rp.dist_array[j] = dist2;
				rp.gauss_ind[j] = ind2;
			} else {
				break;
			}
		}

		if (j < i) {
			rp.dist_array[j] = dist1;
			rp.gauss_ind[j] = ind1;
		}
	}

	for (int i = 0; i < optixLaunchParams.K; ++i) {
		if (i < rp.neighbors_num) {
			optixLaunchParams.distances[(i * number_of_queried_points) + x] = rp.dist_array[i];
			optixLaunchParams.indices[(i * number_of_queried_points) + x] = rp.gauss_ind[i];
		} else {
			optixLaunchParams.distances[(i * number_of_queried_points) + x] = -INFINITY;
			optixLaunchParams.indices[(i * number_of_queried_points) + x] = -1;
		}
	}
}

// *************************************************************************************************

extern "C" __global__ void __anyhit__() {
	unsigned gauss_ind = optixGetPrimitiveIndex();
	gauss_ind /= 20;

	float4 mean = optixLaunchParams.means[gauss_ind];
	
	// *** *** *** *** ***

	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *** *** *** *** ***

	float tMin = optixGetRayTmax();
	float3 O = optixGetObjectRayOrigin();
	float3 d = make_float3(mean.x - O.x, mean.y - O.y, mean.z - O.z);
	float max_distance = mean.w * sqrtf(optixLaunchParams.chi_square_squared_radius);
	float distance = sqrtf((d.x * d.x) + (d.y * d.y) + (d.z * d.z));
	if (distance < max_distance) {
		if (rp->neighbors_num < optixLaunchParams.K) {
			if (distance > rp->max_dist_so_far)
				rp->max_dist_so_far = distance;
			rp->dist_array[rp->neighbors_num] = distance;
			rp->gauss_ind[rp->neighbors_num] = gauss_ind;
			++rp->neighbors_num;
		} else {
			if (distance < rp->max_dist_so_far) {
				if (rp->neighbors_num < 1024) {
					rp->dist_array[rp->neighbors_num] = distance;
					rp->gauss_ind[rp->neighbors_num] = gauss_ind;
					++rp->neighbors_num;
				}
			}
		}
	}
	if (rp->neighbors_num < optixLaunchParams.K) {
		if (tMin <= 2.0f * optixLaunchParams.max_R)
			optixIgnoreIntersection();
	} else {
		if (tMin <= (2.0f * optixLaunchParams.max_R) - rp->max_dist_so_far)
			optixIgnoreIntersection();
	}
}