#ifndef VALUEKERNEL_HEADER
#define VALUEKERNEL_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "params.h"

#include "valuenet.cuh"

//forward declarations
struct ValueMatrices;
struct ValueParameters;

__host__ __device__ float valueTransferFunction(float in);
__host__ __device__ float valueTransferDerivative(float in);

__global__ void computeValueLayer(ValueMatrices* vm, ValueParameters* vp, bool turn1front);
__global__ void backPropagateValueToValue(ValueMatrices* vm, ValueParameters* vp, float posErrorFact, float negErrorFact);
__global__ void backPropagateValueToValueFirstLayer(ValueMatrices* vm, ValueParameters* vp, float posErrorFact, float negErrorFact);
__global__ void backPropagateValueToThought(ValueMatrices* vm, ValueParameters* vp, bool turn1front, float posErrorFact, float negErrorFact);
__global__ void updateValueWeights(ValueMatrices* vm, ValueParameters* vp, float pleasurePain);
__global__ void setErrorFactors(ValueMatrices* vm, ValueParameters* vp, float* posErrorFact, float* negErrorFact);

size_t getValueComputeSharedSize(ValueParameters* vp);
size_t getValueBackPropValueToValueSharedSize(ValueParameters* vp);
size_t getValueBackPropValueToValueFirstLayerSharedSize(ValueParameters* vp);
size_t getValueBackPropValueToThoughtSharedSize(ValueParameters* vp);
size_t getValueUpdateSharedSize(ValueParameters* vp);

#endif