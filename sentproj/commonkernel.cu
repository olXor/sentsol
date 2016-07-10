#include "commonkernel.cuh"

#ifdef MAX_WEIGHT_CHANGE
__device__ float boundChange(float change) {
	if (change > MAX_WEIGHT_CHANGE)
		change = MAX_WEIGHT_CHANGE;
	else if (change < -MAX_WEIGHT_CHANGE)
		change = -MAX_WEIGHT_CHANGE;
	return change;
}
#endif

__device__ bool isNan(float num) {
	return !isfinite(num);
}

__device__ void sumVector(float* vec, size_t size, size_t threadNum, size_t numThreads) {
	size_t stride = 1;
	while (stride < size) {
		for (size_t j = 2 * stride*threadNum; j + stride < size; j += 2 * stride*numThreads) {
			vec[j] += vec[j + stride];
		}
		stride *= 2;
		__syncthreads();
	}
}
