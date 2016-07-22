#include "thoughtkernel.cuh"

__host__ __device__ float thoughtTransferFunction(float in) {
#if THOUGHT_TRANSFER == RECTIFIER
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return in;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return NEGATIVE_TRANSFER_FACTOR*in;
	return TRANSFER_WIDTH*(log(1.0f + exp(in / TRANSFER_WIDTH)) - NEGATIVE_TRANSFER_FACTOR*log(1.0f + exp(-in / TRANSFER_WIDTH)));
#elif THOUGHT_TRANSFER == SIGMOID
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return 1.0f;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return 0.0f;
	return 1.0f / (1.0f + exp(-in / TRANSFER_WIDTH));
#else
	return 0.0f;
#endif
}

__host__ __device__ float thoughtTransferDerivative(float in) {
#if THOUGHT_TRANSFER == RECTIFIER
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return 1.0f;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return NEGATIVE_TRANSFER_FACTOR;
	return 1.0f / (1.0f + exp(-in / TRANSFER_WIDTH)) + NEGATIVE_TRANSFER_FACTOR / (1.0f + exp(in / TRANSFER_WIDTH));
#elif THOUGHT_TRANSFER == SIGMOID
	float tf = thoughtTransferFunction(in);
	return tf*(1-tf);
#else
	return 0.0f;
#endif
}

#ifdef MAX_WEIGHT_CHANGE
__device__ float thoughtBoundChange(float change) {
	if (change > MAX_WEIGHT_CHANGE)
		change = MAX_WEIGHT_CHANGE;
	else if (change < -MAX_WEIGHT_CHANGE)
		change = -MAX_WEIGHT_CHANGE;
	return change;
}
#endif

__device__ bool thoughtIsNan(float num) {
	return !isfinite(num);
}

__device__ void thoughtSumVector(float* vec, size_t size, size_t threadNum, size_t numThreads) {
	size_t stride = 1;
	while (stride < size) {
		for (size_t j = 2 * stride*threadNum; j + stride < size; j += 2 * stride*numThreads) {
			vec[j] += vec[j + stride];
		}
		stride *= 2;
		__syncthreads();
	}
}

__global__ void computeThoughtLayer(ThoughtMatrices* tm, ThoughtParameters* tp, bool turn1front) {
	size_t outNeuron = blockIdx.x;
	size_t clusterStart = outNeuron - outNeuron%CLUSTER_SIZE;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = tp->forBlockX;

	size_t backCon = tp->backwardConnectivity;
	size_t sideCon = tp->sideConnectivity;
	size_t totalCon = backCon + sideCon;

	size_t numInputs = tp->numInputs;
	size_t numOutputs = tp->numOutputs;

	float* inlayer;
	float* outlayer;
	float* prevoutlayer;
	
	if (turn1front) {
		inlayer = tm->inlayer2;
		outlayer = tm->outlayer1;
		prevoutlayer = tm->outlayer2;
	}
	else {
		inlayer = tm->inlayer1;
		outlayer = tm->outlayer2;
		prevoutlayer = tm->outlayer1;
	}

	extern __shared__ float outputs[];

	for (size_t i = inConnection; i < totalCon; i += numInThreads) {
		if (i < backCon)
			outputs[i] = tm->weights[i + totalCon*outNeuron] * inlayer[(clusterStart + i) % numInputs];
		else
			outputs[i] = tm->weights[i + totalCon*outNeuron] * prevoutlayer[(clusterStart + i - backCon) % numOutputs];
	}

	__syncthreads();

	thoughtSumVector(outputs, totalCon, inConnection, numInThreads);

	__shared__ float randFact;
	if (threadIdx.x == 0) {
		randFact = THOUGHT_RAND_WIDTH * curand_normal(&tm->randStates[outNeuron]);
	}
	
	__syncthreads();

	if (threadIdx.x == 0) {
		outlayer[outNeuron] = thoughtTransferFunction(outputs[0] - tm->thresholds[outNeuron] + randFact);
	}
	else if (threadIdx.x == 1 % numInThreads) {
		tm->outTDs[outNeuron] = thoughtTransferDerivative(outputs[0] - tm->thresholds[outNeuron] + randFact);
	}
}

size_t getThoughtComputeSharedSize(ThoughtParameters* tp) {
	size_t size = 0;
	size += tp->backwardConnectivity + tp->sideConnectivity;
	size *= sizeof(float);
	return size;
}

__global__ void backPropagateThoughtLayer(ThoughtMatrices* tm, ThoughtParameters* tp, bool turn1front, float* valueResult) {
	size_t outNeuron = blockIdx.x;
	size_t clusterStart = outNeuron - outNeuron%CLUSTER_SIZE;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = tp->backBlockX;

	size_t backCon = tp->backwardConnectivity;
	size_t sideCon = tp->sideConnectivity;
	size_t totalCon = backCon + sideCon;

	size_t numInputs = tp->numInputs;
	size_t numOutputs = tp->numOutputs;

	float* inlayer;
	//float* outlayer;
	float* prevoutlayer;
	
	if (turn1front) {
		inlayer = tm->inlayer2;
		//outlayer = tm->outlayer1;
		prevoutlayer = tm->outlayer2;
	}
	else {
		inlayer = tm->inlayer1;
		//outlayer = tm->outlayer2;
		prevoutlayer = tm->outlayer1;
	}

	//temporary testing of whether this works without error propagation from the value net
	//note this REQUIRES that the thought net has sigmoid transfer function
	float outErrorTD = tm->errors[outNeuron] * tm->outTDs[outNeuron];
	/*
	float relError;
	float outTF = outlayer[outNeuron];
	if (outTF > 0.5f)
		relError = outTF - 1.0f;
	else
		relError = outTF + 1.0f;
	float outErrorTD = valueResult[0] * relError * fabs(tm->errors[outNeuron]) * tm->outTDs[outNeuron];
	*/

	if (inConnection == 0) {
#ifdef MAX_WEIGHT_CHANGE
		float change = thoughtBoundChange(outErrorTD);
#else
		float change = outErrorTD;
#endif
		tm->thresholds[outNeuron] += change;
	}

	for (size_t i = inConnection; i < totalCon; i += numInThreads) {
		float change;
		if (i < backCon)
			change = outErrorTD * inlayer[(clusterStart + i) % numInputs];
		else
			change = outErrorTD * prevoutlayer[(clusterStart + i - backCon) % numOutputs];
#ifdef MAX_WEIGHT_CHANGE
		change = thoughtBoundChange(change);
#endif
		tm->weights[i + totalCon*outNeuron] -= change;
	}
}

size_t getThoughtBackPropSharedSize(ThoughtParameters* tp) {
	return 0;
}

__global__ void copyThoughtKernelOutputToHost(ThoughtMatrices* tm, ThoughtParameters* tp, float* hostoutput, bool turn1front) {
	size_t outNeuron = threadIdx.x;
	float* currentoutput;
	if (turn1front)
		currentoutput = tm->outlayer1;
	else
		currentoutput = tm->outlayer2;
	hostoutput[outNeuron] = currentoutput[outNeuron];
}

//if we expand to > 1024 neurons per layer we'll have to break this up into blocks
__global__ void initRandomStates(ThoughtMatrices* tm, ThoughtParameters* tp, size_t seed, size_t sequenceStart) {
	size_t outNeuron = threadIdx.x;
	size_t seq = sequenceStart + outNeuron;
	curand_init(seed, seq, 0, &tm->randStates[outNeuron]);
}

__global__ void kernelResetThoughts(ThoughtMatrices* tm, ThoughtParameters* tp) {
	size_t outNeuron = threadIdx.x;
	size_t layerNum = blockIdx.x;
	if (layerNum == 0)
		tm->outlayer1[outNeuron] = 0.0f;
	else if (layerNum == 1)
		tm->outlayer2[outNeuron] = 0.0f;
}