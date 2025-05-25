#include "Header.cuh"

// *** *** *** *** ***

__device__ float RandomFloat(unsigned n) {
	const unsigned a = 1664525;
	const unsigned c = 1013904223;

	unsigned tmp1 = 1;
	unsigned tmp2 = a;
	unsigned tmp3 = 0;
	while (n != 0) {
		if ((n & 1) != 0) tmp3 = (tmp2 * tmp3) + tmp1;
		tmp1 = (tmp2 * tmp1) + tmp1;
		tmp2 = tmp2 * tmp2;
		n >>= 1;
	}
	float result = __uint_as_float(1065353216 | ((tmp3 * c) & 8388607)) - 1.0f;
	return result;
}

// *** *** *** *** ***

__global__ void SampleBoxUniform(unsigned n, unsigned nStart, float4 lower_bound, float4 upper_bound, float4 *samples) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < n) {
		float U1 = RandomFloat(nStart + (tid << 2));
		float U2 = RandomFloat(nStart + (tid << 2) + 1);
		float U3 = RandomFloat(nStart + (tid << 2) + 2);
		float U4 = RandomFloat(nStart + (tid << 2) + 3);
		float4 sample = make_float4(
			lower_bound.x + ((upper_bound.x - lower_bound.x) * U1),
			lower_bound.y + ((upper_bound.y - lower_bound.y) * U2),
			lower_bound.z + ((upper_bound.z - lower_bound.z) * U3),
			lower_bound.w + ((upper_bound.w - lower_bound.w) * U4)
		);
		samples[tid] = sample;
	}
}

// *** *** *** *** ***

const int NUMBER_OF_MEANS = 1024 * 256;
const int NUMBER_OF_QUERIED_POINTS = 1024 * 1024 * 32;
const int K = 16;

const float4 lower_bound_means = make_float4(-1.0f, -1.0f, -1.0f, 0.01f);
const float4 upper_bound_means = make_float4(1.0f, 1.0f, 1.0f, 0.01f);

const float4 lower_bound_queried_points = make_float4(-1.0f, -1.0f, -1.0f, 0.0f);
const float4 upper_bound_queried_points = make_float4(1.0f, 1.0f, 1.0f, 0.0f);

const float chi_square_squared_radius = 11.3449f;

// *** *** *** *** ***

int main() {
	cudaError_t error_CUDA;

	error_CUDA = cudaSetDevice(0);
	if (error_CUDA != cudaSuccess) printf("An error occurred... .");

	// *** *** *** *** ***

	S_CUDA_KNN* cknn = new S_CUDA_KNN;
	bool result;

	result = CUDA_KNN_Init(chi_square_squared_radius, cknn);
	if (!result) {
		printf("CUDA_KNN_Init failed... .\n");
		printf("An error occurred... .");
	} else
		printf("CUDA_KNN_Init succeeded... .\n");

	// *** *** *** *** ***

	unsigned n = 0;
	int number_of_passes = 1;

	while (true) {
		float4 *means;
		
		error_CUDA = cudaMalloc(&means, sizeof(float4) * NUMBER_OF_MEANS);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		SampleBoxUniform<<<(NUMBER_OF_MEANS + 63) >> 6, 64>>>(NUMBER_OF_MEANS, n, lower_bound_means, upper_bound_means, means);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		float *means_host = (float *)malloc(sizeof(float4) * NUMBER_OF_MEANS);
		error_CUDA = cudaMemcpy(means_host, means, sizeof(float4) * NUMBER_OF_MEANS, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		for (int i = 0; i < 10 && i < NUMBER_OF_MEANS; ++i) {
			float4* m = ((float4*)means_host) + i;
			printf("mean[%d]: (%f, %f, %f, %f)\n", i, m->x, m->y, m->z, m->w);
		}

		free(means_host);

		n += (NUMBER_OF_MEANS << 2); // !!! !!! !!!

		// *** *** *** *** ***

		result = CUDA_KNN_Fit(means, NUMBER_OF_MEANS, cknn);
		if (!result) {
			printf("CUDA_KNN_Fit failed... .\n");
			printf("An error occurred... .");
		} else
			printf("CUDA_KNN_Fit succeeded... .\n");

		// *** *** *** *** ***

		float4 *queried_points;

		error_CUDA = cudaMalloc(&queried_points, sizeof(float4) * NUMBER_OF_QUERIED_POINTS);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		SampleBoxUniform<<<(NUMBER_OF_QUERIED_POINTS + 63) >> 6, 64>>>(
			NUMBER_OF_QUERIED_POINTS,
			n,
			lower_bound_queried_points,
			upper_bound_queried_points,
			queried_points
		);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		float *queried_points_host = (float *)malloc(sizeof(float4) * NUMBER_OF_QUERIED_POINTS);
		error_CUDA = cudaMemcpy(queried_points_host, queried_points, sizeof(float4) * NUMBER_OF_QUERIED_POINTS, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		for (int i = 0; i < 10 && i < NUMBER_OF_QUERIED_POINTS; ++i) {
			float4* qp = ((float4*)queried_points_host) + i;
			printf("queried_point[%d]: (%f, %f, %f, %f)\n", i, qp->x, qp->y, qp->z, qp->w);
		}

		free(queried_points_host);

		n += (NUMBER_OF_QUERIED_POINTS << 2); // !!! !!! !!!

		// *** *** *** *** ***

		float *distances;
		int *indices;

		error_CUDA = cudaMalloc(&distances, sizeof(float) * NUMBER_OF_QUERIED_POINTS * K);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		error_CUDA = cudaMalloc(&indices, sizeof(int) * NUMBER_OF_QUERIED_POINTS * K);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		result = CUDA_KNN_KNeighbors(
			queried_points,
			NUMBER_OF_QUERIED_POINTS,
			K,
			distances,
			indices,
			cknn
		);
		if (!result) {
			printf("CUDA_KNN_KNeighbors failed... .\n");
			printf("An error occurred... .");
		} else
			printf("CUDA_KNN_KNeighbors succeeded... .\n");

		// *** *** *** *** ***

		// TEST (massive slowdown due to the Device->Host memory copy)
		/*float *distances_host = (float *)malloc(sizeof(float) * NUMBER_OF_QUERIED_POINTS * K);
		int *indices_host = (int *)malloc(sizeof(int) * NUMBER_OF_QUERIED_POINTS * K);

		error_CUDA = cudaMemcpy(distances_host, distances, sizeof(float) * NUMBER_OF_QUERIED_POINTS * K, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");
		
		error_CUDA = cudaMemcpy(indices_host, indices, sizeof(int) * NUMBER_OF_QUERIED_POINTS * K, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");
		
		for (int j = 0; j < K; ++j) {
			float distance = distances_host[(j * NUMBER_OF_QUERIED_POINTS) + 0];
			int index = indices_host[(j * NUMBER_OF_QUERIED_POINTS) + 0];
			printf("POINT: %d; RANK %d; DISTANCE: %f; INDEX: %d;\n", 0 + 1, j + 1, distance, index);
		}
		free(distances_host);
		free(indices_host);*/

		// *** *** *** *** ***

		error_CUDA = cudaFree(means);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		error_CUDA = cudaFree(queried_points);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		error_CUDA = cudaFree(distances);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		error_CUDA = cudaFree(indices);
		if (error_CUDA != cudaSuccess) printf("An error occurred... .");

		// *** *** *** *** ***

		printf("NUMBER OF PASSES: %d\n", number_of_passes);
		++number_of_passes;
	}
}