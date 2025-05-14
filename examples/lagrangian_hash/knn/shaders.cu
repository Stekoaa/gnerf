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
};

// *************************************************************************************************

extern "C" __constant__ LaunchParams optixLaunchParams;

// *************************************************************************************************

struct SRayPayload {
	// v2
	/*float T_approx;
	int Gauss_num;
	bool threshold_exceeded;
	float max_t_value_encountered_so_far;

	float t_array[1024];
	int Gauss_ind_array[1024];
	float alpha_array[1024];*/

	// KNN
	float min_distance;
	float dist_array[1024];
	int neighbors_num;
	float max_dist_so_far;
};

extern "C" __global__ void __raygen__renderFrame() {
	// v1
	/*int x = optixGetLaunchIndex().x;
	int y = optixGetLaunchIndex().y;
	int pixel_ind = (y * optixLaunchParams.width) + x;
	
	REAL3_R d = make_REAL3_R(
		(((REAL_R)-0.5) + ((x + ((REAL_R)0.5)) / optixLaunchParams.width)) * optixLaunchParams.double_tan_half_fov_x,
		(((REAL_R)-0.5) + ((y + ((REAL_R)0.5)) / optixLaunchParams.height)) * optixLaunchParams.double_tan_half_fov_y,
		1,
	);
	REAL3_R v = make_REAL3_R(
		MAD_R(optixLaunchParams.R.x, d.x, MAD_R(optixLaunchParams.D.x, d.y, optixLaunchParams.F.x * d.z)),
		MAD_R(optixLaunchParams.R.y, d.x, MAD_R(optixLaunchParams.D.y, d.y, optixLaunchParams.F.y * d.z)),
		MAD_R(optixLaunchParams.R.z, d.x, MAD_R(optixLaunchParams.D.z, d.y, optixLaunchParams.F.z * d.z))
	);

	float tMin = 0.0f;

	REAL_R R = 0;
	REAL_R G = 0;
	REAL_R B = 0;

	REAL_R T = 1;
	REAL_R TLastGrad = 1;

	bool belowThreshold = false;

	// !!! !!! !!!
	//int i = 0;
	//for (i = 0; i < optixLaunchParams.max_Gaussians_per_ray; ++i) {
	// !!! !!! !!!

	for (int i = 0; i < optixLaunchParams.max_Gaussians_per_ray; ++i) {
		unsigned Gauss_ind = -1;
		unsigned t_as_uint;
				
		optixTrace(
			optixLaunchParams.traversable,
			optixLaunchParams.O,

			#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
				v,
			#else
				make_float3(v.x, v.y, v.z),
			#endif

			tMin,
			INFINITY,
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			0,
			1,
			0,

			Gauss_ind,
			t_as_uint
		);
		
		optixLaunchParams.Gaussians_indices[(i * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = Gauss_ind;
		if (Gauss_ind != -1) {
			float4 GC_1 = optixLaunchParams.GC_part_1[Gauss_ind];
			float4 GC_2 = optixLaunchParams.GC_part_2[Gauss_ind];
			float4 GC_3 = optixLaunchParams.GC_part_3[Gauss_ind];
			float2 GC_4 = optixLaunchParams.GC_part_4[Gauss_ind];

			// *************************************************************************************

			REAL_R aa = ((REAL_R)GC_3.z) * GC_3.z;
			REAL_R bb = ((REAL_R)GC_3.w) * GC_3.w;
			REAL_R cc = ((REAL_R)GC_4.x) * GC_4.x;
			REAL_R dd = ((REAL_R)GC_4.y) * GC_4.y;
			REAL_R s = ((REAL_R)0.5) * (aa + bb + cc + dd);

			REAL_R ab = GC_3.z * GC_3.w; REAL_R ac = GC_3.z * GC_4.x; REAL_R ad = GC_3.z * GC_4.y;
										 REAL_R bc = GC_3.w * GC_4.x; REAL_R bd = GC_3.w * GC_4.y;
																	  REAL_R cd = GC_4.x * GC_4.y;

			REAL_R R11 = s - cc - dd;
			REAL_R R12 = bc - ad;
			REAL_R R13 = bd + ac;

			REAL_R R21 = bc + ad;
			REAL_R R22 = s - bb - dd;
			REAL_R R23 = cd - ab;

			REAL_R R31 = bd - ac;
			REAL_R R32 = cd + ab;
			REAL_R R33 = s - bb - cc;

			// *************************************************************************************

			REAL3_R O = make_REAL3_R(
				optixLaunchParams.O.x - GC_2.x,
				optixLaunchParams.O.y - GC_2.y,
				optixLaunchParams.O.z - GC_2.z
			);

			REAL3_R O_prim;
			REAL3_R v_prim;

			REAL_R sXInv = EXP_R(-GC_2.w);
			O_prim.x = MAD_R(R11, O.x, MAD_R(R21, O.y, R31 * O.z)) * sXInv;
			v_prim.x = MAD_R(R11, v.x, MAD_R(R21, v.y, R31 * v.z)) * sXInv;

			REAL_R sYInv = EXP_R(-GC_3.x);
			O_prim.y = MAD_R(R12, O.x, MAD_R(R22, O.y, R32 * O.z)) * sYInv;
			v_prim.y = MAD_R(R12, v.x, MAD_R(R22, v.y, R32 * v.z)) * sYInv;

			REAL_R sZInv = EXP_R(-GC_3.y);
			O_prim.z = MAD_R(R13, O.x, MAD_R(R23, O.y, R33 * O.z)) * sZInv;
			v_prim.z = MAD_R(R13, v.x, MAD_R(R23, v.y, R33 * v.z)) * sZInv;

			// *************************************************************************************
			
			REAL_R tHit = __uint_as_float(t_as_uint);
			REAL_R PHitX = MAD_R(v_prim.x, tHit, O_prim.x); // !!! !!! !!!
			REAL_R PHitY = MAD_R(v_prim.y, tHit, O_prim.y); // !!! !!! !!!
			REAL_R PHitZ = MAD_R(v_prim.z, tHit, O_prim.z); // !!! !!! !!!
			REAL_R alpha = EXP_R(-((REAL_R)0.5) * (MAD_R(PHitX, PHitX, MAD_R(PHitY, PHitY, PHitZ * PHitZ)) / (s * s)));
			#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
				alpha = __saturatef(alpha) / (((REAL_R)1.0) + EXP_R(-GC_1.w));
			#else
				alpha = (alpha < 0) ? 0 : alpha;
				alpha = (alpha > 1) ? 1 : alpha;
				alpha = alpha / (((REAL_R)1.0) + EXP_R(-GC_1.w));
			#endif
			
			REAL_R tmp = T * alpha;

			R = R + (GC_1.x * tmp);
			G = G + (GC_1.y * tmp);
			B = B + (GC_1.z * tmp);
			T = T - tmp;

			if (T < ((REAL_R)optixLaunchParams.ray_termination_T_threshold)) {
				if (!belowThreshold) belowThreshold = true;
				else
					TLastGrad = TLastGrad * (1 - alpha);
			}
			if (TLastGrad >= ((REAL_R)optixLaunchParams.last_significant_Gauss_alpha_gradient_precision)) {					
				float t = __uint_as_float(t_as_uint);
				tMin = nextafter(t, INFINITY);
			} else {
				if (i < optixLaunchParams.max_Gaussians_per_ray - 1)
					optixLaunchParams.Gaussians_indices[((i + 1) * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = -1;
				break;
			}
		} else
			break;
	}

	// !!! !!! !!!
	//R = ((REAL_R)i) / optixLaunchParams.max_Gaussians_per_ray;
	//G = ((REAL_R)i) / optixLaunchParams.max_Gaussians_per_ray;
	//B = ((REAL_R)i) / optixLaunchParams.max_Gaussians_per_ray;
	// !!! !!! !!!

	#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
		R = __saturatef(R);
		G = __saturatef(G);
		B = __saturatef(B);
	#else
		// Cannot use the code below due to the error in OptiX:
		//component = MIN_R(MAX_R(((REAL_R)component), ((REAL_R)0)), ((REAL_R)1));
		
		R = (R < 0.0) ? 0.0 : R;
		R = (R > 1.0) ? 1.0 : R;
		
		G = (G < 0.0) ? 0.0 : G;
		G = (G > 1.0) ? 1.0 : G;
		
		B = (B < 0.0) ? 0.0 : B;
		B = (B > 1.0) ? 1.0 : B;
	#endif
	int Ri = RINT_R(R * 255);
	int Gi = RINT_R(G * 255);
	int Bi = RINT_R(B * 255);

	optixLaunchParams.bitmap[pixel_ind] = (Ri << 16) + (Gi << 8) + Bi;
	optixLaunchParams.bitmap_out_R[(y * (optixLaunchParams.width + 11 - 1)) + x] = R; // !!! !!! !!!
	optixLaunchParams.bitmap_out_G[(y * (optixLaunchParams.width + 11 - 1)) + x] = G; // !!! !!! !!!
	optixLaunchParams.bitmap_out_B[(y * (optixLaunchParams.width + 11 - 1)) + x] = B; // !!! !!! !!!*/

	// *** *** *** *** ***

	// v2
	/*int x = optixGetLaunchIndex().x;
	int y = optixGetLaunchIndex().y;
	int pixel_ind = (y * optixLaunchParams.width) + x;

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

	rp.Gauss_num = 0;
	rp.T_approx = 1.0f;
	rp.max_t_value_encountered_so_far = -INFINITY;
	rp.threshold_exceeded = false;

	optixTrace(
		optixLaunchParams.traversable,
		optixLaunchParams.O,
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

	// *** *** *** *** ***

	for (int i = 1; i < rp.Gauss_num; ++i) {
		float t1 = rp.t_array[i];
		int ind1 = rp.Gauss_ind_array[i];
		float alpha1 = rp.alpha_array[i];

		int j;
		for (j = i; j > 0; --j) {
			float t2 = rp.t_array[j - 1];
			int ind2 = rp.Gauss_ind_array[j - 1];
			float alpha2 = rp.alpha_array[j - 1];

			if (t1 < t2) {
				rp.t_array[j] = t2;
				rp.Gauss_ind_array[j] = ind2;
				rp.alpha_array[j] = alpha2;
			} else
				break;
		}
		if (j < i) {
			rp.t_array[j] = t1;
			rp.Gauss_ind_array[j] = ind1;
			rp.alpha_array[j] = alpha1;
		}
	}
	if (rp.Gauss_num > optixLaunchParams.max_Gaussians_per_ray)
		rp.Gauss_num = optixLaunchParams.max_Gaussians_per_ray;

	// *** *** *** *** ***

	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;
	float T = 1.0f;

	int i;
	for (i = 0; i < rp.Gauss_num; ++i) {
		int ind = rp.Gauss_ind_array[i];
		float alpha = rp.alpha_array[i];
		float4 GC_1 = optixLaunchParams.GC_part_1[ind];
		optixLaunchParams.Gaussians_indices[(i * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = ind;

		float tmp = T * alpha;
		R = R + (GC_1.x * tmp);
		G = G + (GC_1.y * tmp);
		B = B + (GC_1.z * tmp);
		T = T * (1.0f - alpha);

		if (T < optixLaunchParams.ray_termination_T_threshold) break;
	}
	if (i < rp.Gauss_num) {
		if (i < optixLaunchParams.max_Gaussians_per_ray - 1)
			optixLaunchParams.Gaussians_indices[((i + 1) * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = -1;
	} else {
		if (rp.Gauss_num < optixLaunchParams.max_Gaussians_per_ray)
			optixLaunchParams.Gaussians_indices[(rp.Gauss_num * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = -1;
	}

	R = __saturatef(R);
	G = __saturatef(G);
	B = __saturatef(B);

	int Ri = RINT_R(R * 255);
	int Gi = RINT_R(G * 255);
	int Bi = RINT_R(B * 255);

	optixLaunchParams.bitmap[pixel_ind] = (Ri << 16) + (Gi << 8) + Bi;
	optixLaunchParams.bitmap_out_R[(y * (optixLaunchParams.width + 11 - 1)) + x] = R; // !!! !!! !!!
	optixLaunchParams.bitmap_out_G[(y * (optixLaunchParams.width + 11 - 1)) + x] = G; // !!! !!! !!!
	optixLaunchParams.bitmap_out_B[(y * (optixLaunchParams.width + 11 - 1)) + x] = B; // !!! !!! !!!*/

	// *** *** *** *** ***

	// KNN
	int x = optixGetLaunchIndex().x;
	int y = optixGetLaunchIndex().y;
	int pixel_ind = (y * optixLaunchParams.width) + x;

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
			
			int k;
			for (k = j; k > 0; --k) {
				float dist2 = rp.dist_array[k - 1];
				
				if (dist1 < dist2)
					rp.dist_array[k] = dist2;
				else
					break;
			}
			if (k < j)
				rp.dist_array[k] = dist1;
		}

		if ((x == 400) && (y == 400)) {
			if (rp.neighbors_num >= 1)
				optixLaunchParams.distances[i] = rp.dist_array[0];
			else
				optixLaunchParams.distances[i] = INFINITY;
		}
	}
}

// *************************************************************************************************

extern "C" __global__ void __anyhit__radiance() {
	// v2
	/*unsigned Gauss_ind = optixGetPrimitiveIndex();
	Gauss_ind /= NUMBER_OF_FACES; // !!! !!! !!!

	float4 GC_1 = optixLaunchParams.GC_part_1[Gauss_ind];
	float4 GC_2 = optixLaunchParams.GC_part_2[Gauss_ind];
	float4 GC_3 = optixLaunchParams.GC_part_3[Gauss_ind];
	float2 GC_4 = optixLaunchParams.GC_part_4[Gauss_ind];

	float3 v = optixGetObjectRayDirection();

	// *** *** *** *** ***

	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *** *** *** *** ***

	float tMin = optixGetRayTmax();
	float T_approx = rp->T_approx;
	int Gauss_num = rp->Gauss_num;

	if ((!rp->threshold_exceeded) && (tMin >= rp->max_t_value_encountered_so_far))
		rp->max_t_value_encountered_so_far = tMin;

	// *** *** *** *** ***

	float aa = GC_3.z * GC_3.z;
	float bb = GC_3.w * GC_3.w;
	float cc = GC_4.x * GC_4.x;
	float dd = GC_4.y * GC_4.y;
	float s = 0.5f * (aa + bb + cc + dd);

	float ab = GC_3.z * GC_3.w; float ac = GC_3.z * GC_4.x; float ad = GC_3.z * GC_4.y;
								float bc = GC_3.w * GC_4.x; float bd = GC_3.w * GC_4.y;
															float cd = GC_4.x * GC_4.y;

	float R11 = s - cc - dd;
	float R12 = bc - ad;
	float R13 = bd + ac;

	float R21 = bc + ad;
	float R22 = s - bb - dd;
	float R23 = cd - ab;

	float R31 = bd - ac;
	float R32 = cd + ab;
	float R33 = s - bb - cc;

	// *** *** *** *** ***

	float3 O = make_float3(
		optixLaunchParams.O.x - GC_2.x,
		optixLaunchParams.O.y - GC_2.y,
		optixLaunchParams.O.z - GC_2.z
	);
	
	float3 O_prim;
	float3 v_prim;

	float sXInv = EXP_R(-GC_2.w);
	O_prim.x = __fmaf_rn(R11, O.x, __fmaf_rn(R21, O.y, R31 * O.z)) * sXInv;
	v_prim.x = __fmaf_rn(R11, v.x, __fmaf_rn(R21, v.y, R31 * v.z)) * sXInv;

	float sYInv = EXP_R(-GC_3.x);
	O_prim.y = __fmaf_rn(R12, O.x, __fmaf_rn(R22, O.y, R32 * O.z)) * sYInv;
	v_prim.y = __fmaf_rn(R12, v.x, __fmaf_rn(R22, v.y, R32 * v.z)) * sYInv;

	float sZInv = EXP_R(-GC_3.y);
	O_prim.z = __fmaf_rn(R13, O.x, __fmaf_rn(R23, O.y, R33 * O.z)) * sZInv;
	v_prim.z = __fmaf_rn(R13, v.x, __fmaf_rn(R23, v.y, R33 * v.z)) * sZInv;

	// *** *** *** *** ***

	float v_dot_v = __fmaf_rn(v_prim.x, v_prim.x, __fmaf_rn(v_prim.y, v_prim.y, v_prim.z * v_prim.z));
	float O_dot_v = __fmaf_rn(O_prim.x, v_prim.x, __fmaf_rn(O_prim.y, v_prim.y, O_prim.z * v_prim.z));
	
	float tmp = 1.0f / v_dot_v;
	float t = -O_dot_v * tmp;

	float PHitX = __fmaf_rn(v_prim.x, t, O_prim.x); // !!! !!! !!!
	float PHitY = __fmaf_rn(v_prim.y, t, O_prim.y); // !!! !!! !!!
	float PHitZ = __fmaf_rn(v_prim.z, t, O_prim.z); // !!! !!! !!!

	float alpha = EXP_R(-0.5f * (__fmaf_rn(PHitX, PHitX, __fmaf_rn(PHitY, PHitY, PHitZ * PHitZ)) / (s * s)));
	alpha = alpha / (1.0f + EXP_R(-GC_1.w));
	T_approx = T_approx * (1 - alpha);

	if ((T_approx < optixLaunchParams.ray_termination_T_threshold) || (Gauss_num >= optixLaunchParams.max_Gaussians_per_ray))
		rp->threshold_exceeded = true;

	// *** *** *** *** ***

	if (
		(Gauss_num < 1024) && (
			(!rp->threshold_exceeded) ||
			(tMin <= rp->max_t_value_encountered_so_far)
		)
	) {
		rp->T_approx = T_approx;
		rp->Gauss_num = Gauss_num + 1;

		rp->t_array[Gauss_num] = tMin;
		rp->Gauss_ind_array[Gauss_num] = Gauss_ind;
		rp->alpha_array[Gauss_num] = alpha;

		optixIgnoreIntersection();
	}*/

	// *** *** *** *** ***

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
			++rp->neighbors_num;
		} else {
			if (distance < rp->max_dist_so_far) {
				if (rp->neighbors_num < 1024) {
					rp->dist_array[rp->neighbors_num] = distance;
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
	/*unsigned Gauss_ind = optixGetPrimitiveIndex();
	Gauss_ind /= (NUMBER_OF_SIDES - 2);
	optixSetPayload_0(Gauss_ind);
	optixSetPayload_1(__float_as_uint(optixGetRayTmax()));*/
}
