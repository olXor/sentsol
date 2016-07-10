#ifndef COMMONKERNEL_HEADER
#define COMMONKERNEL_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "params.h"

#ifdef MAX_WEIGHT_CHANGE
__device__ float boundChange(float change);
#endif

__device__ bool isNan(float num);
__device__ void sumVector(float* vec, size_t size, size_t threadNum, size_t numThreads);

#endif