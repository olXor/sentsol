#ifndef THOUGHTKERNEL_HEADER
#define THOUGHTKERNEL_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "params.h"

#include "thoughtnet.cuh"
#include "commonkernel.cuh"

//forward declarations
struct ThoughtMatrices;
struct ThoughtParameters;

__host__ __device__ float thoughtTransferFunction(float in);
__host__ __device__ float thoughtTransferDerivative(float in);

__global__ void computeThoughtLayer(ThoughtMatrices* tm, ThoughtParameters* tp, bool turn1front);
__global__ void backPropagateThoughtLayer(ThoughtMatrices* tm, ThoughtParameters* tp, bool turn1front);
__global__ void copyOutputToHost(ThoughtMatrices* tm, ThoughtParameters* tp, float* hostoutput, bool turn1front);

size_t getThoughtComputeSharedSize(ThoughtParameters* tp);
size_t getThoughtBackPropSharedSize(ThoughtParameters* tp);

#endif