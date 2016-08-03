#include "thoughtkernel.cuh"

__host__ __device__ float thoughtTransferFunction(float in) {
#if THOUGHT_TRANSFER == RECTIFIER
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return in;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return THOUGHT_NEGATIVE_TRANSFER_FACTOR*in;
	return TRANSFER_WIDTH*(log(1.0f + exp(in / TRANSFER_WIDTH)) - THOUGHT_NEGATIVE_TRANSFER_FACTOR*log(1.0f + exp(-in / TRANSFER_WIDTH)));
#elif THOUGHT_TRANSFER == SIGMOID
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return 1.0f + THOUGHT_NEGATIVE_TRANSFER_FACTOR*in + THOUGHT_BASELINE;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return THOUGHT_NEGATIVE_TRANSFER_FACTOR*in + THOUGHT_BASELINE;
	return 1.0f / (1.0f + exp(-in / TRANSFER_WIDTH)) + THOUGHT_NEGATIVE_TRANSFER_FACTOR*in + THOUGHT_BASELINE;
#else
	return 0.0f;
#endif
}

__host__ __device__ float thoughtTransferDerivative(float in) {
#if THOUGHT_TRANSFER == RECTIFIER
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return 1.0f;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return THOUGHT_NEGATIVE_TRANSFER_FACTOR;
	return 1.0f / (1.0f + exp(-in / TRANSFER_WIDTH)) + THOUGHT_NEGATIVE_TRANSFER_FACTOR / (1.0f + exp(in / TRANSFER_WIDTH));
#elif THOUGHT_TRANSFER == SIGMOID
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return THOUGHT_NEGATIVE_TRANSFER_FACTOR;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return THOUGHT_NEGATIVE_TRANSFER_FACTOR;
	float expin = exp(-in / TRANSFER_WIDTH);
	float denom = (expin + 1)*(expin + 1);
	return expin/denom/TRANSFER_WIDTH + THOUGHT_NEGATIVE_TRANSFER_FACTOR;
	//float tf = thoughtTransferFunction(in) - THOUGHT_NEGATIVE_TRANSFER_FACTOR*in;
	//return tf*(1-tf) + THOUGHT_NEGATIVE_TRANSFER_FACTOR;
#else
	return 0.0f;
#endif
}

#ifdef THOUGHT_MAX_WEIGHT_CHANGE
__device__ float thoughtBoundChange(float change) {
	if (change > THOUGHT_MAX_WEIGHT_CHANGE)
		change = THOUGHT_MAX_WEIGHT_CHANGE;
	else if (change < -THOUGHT_MAX_WEIGHT_CHANGE)
		change = -THOUGHT_MAX_WEIGHT_CHANGE;
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

__global__ void computeThoughtLayer(ThoughtMatrices* tm, ThoughtParameters* tp, size_t bpTurn, size_t prevTurn) {
	size_t outNeuron = blockIdx.x;
	size_t clusterStart = outNeuron - outNeuron%CLUSTER_SIZE;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = tp->forBlockX;

	size_t backCon = tp->backwardConnectivity;
	size_t sideCon = tp->sideConnectivity;
	size_t totalCon = backCon + sideCon;

	size_t numInputs = tp->numInputs;
	size_t numOutputs = tp->numOutputs;

	float* inlayer = &tm->inlayer[prevTurn*numInputs];
	float* outlayer = &tm->outlayer[bpTurn*numOutputs];
	float* prevoutlayer = &tm->outlayer[prevTurn*numOutputs];
	float* outTDs = &tm->outTDs[bpTurn*numOutputs];
	float* outNew = &tm->outNew[bpTurn*numOutputs];

	extern __shared__ float outputs[];

	if (bpTurn != prevTurn) {
		for (size_t i = inConnection; i < totalCon; i += numInThreads) {
			if (i < backCon)
				outputs[i] = tm->weights[i + totalCon*outNeuron] * inlayer[(clusterStart + i) % numInputs];
			else
				outputs[i] = tm->weights[i + totalCon*outNeuron] * prevoutlayer[(clusterStart + i - backCon) % numOutputs];
		}

		__syncthreads();

		thoughtSumVector(outputs, totalCon, inConnection, numInThreads);
	}
	else {
		outputs[0] = 0;
	}

	__shared__ float randFact;
	if (threadIdx.x == 0) {
		randFact = THOUGHT_RAND_WIDTH * curand_normal(&tm->randStates[outNeuron]);
	}
	
	__syncthreads();

	if (threadIdx.x == 0) {
		float retain = tm->retains[outNeuron];
		outlayer[outNeuron] = (1-retain)*thoughtTransferFunction(outputs[0] - tm->thresholds[outNeuron] + randFact) + retain * prevoutlayer[outNeuron];
	}
	else if (threadIdx.x == 1 % numInThreads) {
		float retain = tm->retains[outNeuron];
		outTDs[outNeuron] = (1-retain)*thoughtTransferDerivative(outputs[0] - tm->thresholds[outNeuron] + randFact);
	}
	else if (threadIdx.x == 2 % numInThreads) {
		outNew[outNeuron] = thoughtTransferFunction(outputs[0] - tm->thresholds[outNeuron] + randFact);
	}
}

size_t getThoughtComputeSharedSize(ThoughtParameters* tp) {
	size_t size = 0;
	size += tp->backwardConnectivity + tp->sideConnectivity;
	size *= sizeof(float);
	return size;
}

__global__ void backPropagateThoughtLayer(ThoughtMatrices* tm, ThoughtParameters* tp, size_t bpTurn) {
	size_t outNeuron = blockIdx.x;
	size_t clusterStart = outNeuron - outNeuron%CLUSTER_SIZE;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = tp->backBlockX;

	size_t backCon = tp->backwardConnectivity;
	size_t sideCon = tp->sideConnectivity;
	size_t totalCon = backCon + sideCon;

	size_t numInputs = tp->numInputs;
	size_t numOutputs = tp->numOutputs;
	
	float* inlayer = &tm->inlayer[(bpTurn-1)*numInputs];
	float* prevoutlayer = &tm->outlayer[(bpTurn-1)*numOutputs];
	float* outTDs = &tm->outTDs[bpTurn*numOutputs];
	float* outNew = &tm->outNew[bpTurn*numOutputs];
	float* errors = &tm->errors[bpTurn*numOutputs];
	float* inerrors = tm->inerrors;
	if (inerrors != NULL)
		inerrors = &inerrors[(bpTurn-1)*numInputs];
	float* preverrors = &tm->errors[(bpTurn - 1)*numOutputs];

	float outError = errors[outNeuron];
	float outErrorTD = outError * outTDs[outNeuron];
	float retain = tm->retains[outNeuron];

	for (size_t i = inConnection; i < totalCon; i += numInThreads) {
		float change;
		size_t weightNum = i + totalCon*outNeuron;
		float weight = tm->weights[weightNum];
		if (i < backCon) {
			size_t inNeuron = (clusterStart + i) % numInputs;
			change = outErrorTD * inlayer[inNeuron];
			if (inerrors != NULL)
				inerrors[inNeuron] += outErrorTD * weight;
		}
		else {
			size_t prevNeuron = (clusterStart + i - backCon) % numOutputs;
			change = outErrorTD * prevoutlayer[prevNeuron];
			float preverror = outErrorTD * weight;
			if (prevNeuron == outNeuron)
				preverror += outError*retain;
			preverrors[prevNeuron] += preverror;
		}
		tm->weightChanges[weightNum] -= change;	
	}

	if (inConnection == 0) {
		tm->thresholdChanges[outNeuron] += outErrorTD; 
	}
	if (inConnection == 1 % numInThreads) {
		tm->retainChanges[outNeuron] -= outError*(prevoutlayer[outNeuron] - outNew[outNeuron]);
	}
}

size_t getThoughtBackPropSharedSize(ThoughtParameters* tp) {
	return 0;
}

__global__ void updateThoughtWeights(ThoughtMatrices* tm, ThoughtParameters* tp) {
	size_t outNeuron = blockIdx.x;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = blockDim.x; //totalCon + 2

	size_t backCon = tp->backwardConnectivity;
	size_t sideCon = tp->sideConnectivity;
	size_t totalCon = backCon + sideCon;

	for (size_t i = inConnection; i < totalCon + 2; i += numInThreads) {
		if (i == totalCon + 1) {
			float change = RETAIN_STEP_MULT*tm->retainChanges[outNeuron];
#ifdef THOUGHT_MAX_WEIGHT_CHANGE
			change = thoughtBoundChange(change);
#endif
			float newretain = tm->retains[outNeuron] + change;
			if (newretain > 1.0f) newretain = 1.0f;
			if (newretain < 0.0f) newretain = 0.0f;
			tm->retains[outNeuron] = newretain;
			tm->retainChanges[outNeuron] = 0;
		}
		else if (i == totalCon) {
			float change = tm->thresholdChanges[outNeuron];
#ifdef THOUGHT_MAX_WEIGHT_CHANGE
			change = thoughtBoundChange(change);
#endif
			tm->thresholds[outNeuron] += change;
			tm->thresholdChanges[outNeuron] = 0;
		}
		else {
			size_t weightNum = i + outNeuron*totalCon;
			float change = tm->weightChanges[weightNum];
#ifdef THOUGHT_MAX_WEIGHT_CHANGE
			change = thoughtBoundChange(change);
#endif
			tm->weights[weightNum] += change;
			tm->weightChanges[weightNum] = 0;
		}
	}
}

size_t getThoughtUpdateSharedSize(ThoughtParameters* tp) {
	return 0;
}

__global__ void copyThoughtKernelOutputToHost(ThoughtMatrices* tm, ThoughtParameters* tp, float* hostoutput, size_t bpTurn) {
	size_t outNeuron = threadIdx.x;
	float* currentoutput = &tm->outlayer[bpTurn*tp->numOutputs];
	hostoutput[outNeuron] = currentoutput[outNeuron];
}

//if we expand to > 1024 neurons per layer we'll have to break this up into blocks
__global__ void initRandomStates(ThoughtMatrices* tm, ThoughtParameters* tp, size_t seed, size_t sequenceStart) {
	size_t outNeuron = threadIdx.x;
	size_t seq = sequenceStart + outNeuron;
	curand_init(seed, seq, 0, &tm->randStates[outNeuron]);
}

__global__ void normalizeOutputs(ThoughtMatrices* tm, ThoughtParameters* tp, size_t bpTurn) {
	size_t x = threadIdx.x;
	size_t numThreads = blockDim.x;
	size_t numOutputs = tp->numOutputs;

	float* outlayer = &tm->outlayer[bpTurn*numOutputs];
	extern __shared__ float mins[];
	float* maxes = &mins[numOutputs];
	for (size_t i = x; i < numOutputs; i += numThreads) {
		mins[i] = outlayer[i];
		maxes[i] = outlayer[i];
	}

	__syncthreads();

	size_t stride = 1;
	while (stride < numOutputs) {
		for (size_t i = 2 * stride*x; i + stride < numOutputs; i += 2 * stride*numThreads) {
			if (mins[i + stride] < mins[i])
				mins[i] = mins[i + stride];
		}
		for (size_t i = 2 * stride*(numOutputs - x - 1); i + stride < numOutputs; i += 2 * stride*numThreads) {
			if (maxes[i + stride] > maxes[i])
				maxes[i] = maxes[i + stride];
		}

		stride *= 2;
		__syncthreads();
	}

	float min = mins[0];
	float max = maxes[0];
	float spread = max - min;
	for (size_t i = x; i < numOutputs; i += numThreads) {
		if (spread == 0)
			outlayer[i] = 0;
		else {
			outlayer[i] = 2 * (outlayer[i] - min) / spread - 1.0f;
		}
	}
}