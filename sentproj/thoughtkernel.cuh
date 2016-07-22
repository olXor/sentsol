#ifndef THOUGHTKERNEL_HEADER
#define THOUGHTKERNEL_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "params.h"
#include <curand_kernel.h>

#include "thoughtnet.cuh"

//forward declarations
struct ThoughtMatrices;
struct ThoughtParameters;

__host__ __device__ float thoughtTransferFunction(float in);
__host__ __device__ float thoughtTransferDerivative(float in);

__global__ void computeThoughtLayer(ThoughtMatrices* tm, ThoughtParameters* tp, size_t bpTurn, size_t prevTurn);
__global__ void backPropagateThoughtLayer(ThoughtMatrices* tm, ThoughtParameters* tp, size_t bpTurn);
__global__ void copyThoughtKernelOutputToHost(ThoughtMatrices* tm, ThoughtParameters* tp, float* hostoutput, size_t bpTurn);
__global__ void initRandomStates(ThoughtMatrices* tm, ThoughtParameters* tp, size_t seed, size_t sequenceStart);

size_t getThoughtComputeSharedSize(ThoughtParameters* tp);
size_t getThoughtBackPropSharedSize(ThoughtParameters* tp);

#endif