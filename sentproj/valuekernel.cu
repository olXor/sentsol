#include "valuekernel.cuh"

__host__ __device__ float valueTransferFunction(float in) {
#if VALUE_TRANSFER == RECTIFIER
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return in;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return NEGATIVE_TRANSFER_FACTOR*in;
	return TRANSFER_WIDTH*(log(1.0f + exp(in / TRANSFER_WIDTH)) - NEGATIVE_TRANSFER_FACTOR*log(1.0f + exp(-in / TRANSFER_WIDTH)));
#elif VALUE_TRANSFER == SIGMOID
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return 1.0f;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return 0.0f;
	return 1.0f / (1.0f + exp(-in / TRANSFER_WIDTH));
#else
	return 0.0f;
#endif
}

__host__ __device__ float valueTransferDerivative(float in) {
#if VALUE_TRANSFER == RECTIFIER
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return 1.0f;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return NEGATIVE_TRANSFER_FACTOR;
	return 1.0f / (1.0f + exp(-in / TRANSFER_WIDTH)) + NEGATIVE_TRANSFER_FACTOR / (1.0f + exp(in / TRANSFER_WIDTH));
#elif VALUE_TRANSFER == SIGMOID
	float tf = valueTransferFunction(in);
	return tf*(1-tf);
#else
	return 0.0f;
#endif
}

//apparently we need to have these generic helper functions in every compilation unit, aka in every file. Annoying!
#ifdef MAX_WEIGHT_CHANGE
__device__ float valueBoundChange(float change) {
	if (change > MAX_WEIGHT_CHANGE)
		change = MAX_WEIGHT_CHANGE;
	else if (change < -MAX_WEIGHT_CHANGE)
		change = -MAX_WEIGHT_CHANGE;
	return change;
}
#endif

__device__ bool valueIsNan(float num) {
	return !isfinite(num);
}

__device__ void valueSumVector(float* vec, size_t size, size_t threadNum, size_t numThreads) {
	size_t stride = 1;
	while (stride < size) {
		for (size_t j = 2 * stride*threadNum; j + stride < size; j += 2 * stride*numThreads) {
			vec[j] += vec[j + stride];
		}
		stride *= 2;
		__syncthreads();
	}
}

__global__ void computeValueLayer(ValueMatrices* vm, ValueParameters* vp, bool turn1front) {
	size_t outNeuron = blockIdx.x;
	size_t clusterStart = outNeuron - outNeuron%CLUSTER_SIZE;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = vp->forBlockX;

	size_t backCon = vp->backwardConnectivity;
	size_t sideCon = vp->thoughtConnectivity;
	size_t totalCon = backCon + sideCon;

	size_t numInputs = vp->numInputs;
	size_t numThoughtInputs = vp->numThoughtInputs;

	float* thoughtlayer;
	
	if (turn1front) {
		thoughtlayer = vm->thoughtlayer1;
	}
	else {
		thoughtlayer = vm->thoughtlayer2;
	}

	extern __shared__ float outputs[];		//totalCon

	for (size_t i = inConnection; i < totalCon; i += numInThreads) {
		if (i < backCon)
			outputs[i] = vm->weights[i + totalCon*outNeuron] * vm->inlayer[(clusterStart + i) % numInputs];
		else
			outputs[i] = vm->weights[i + totalCon*outNeuron] * thoughtlayer[(clusterStart + i - backCon) % numThoughtInputs];
	}

	__syncthreads();

	valueSumVector(outputs, totalCon, inConnection, numInThreads);

	if (threadIdx.x == 0) {
		vm->outlayer[outNeuron] = valueTransferFunction(outputs[0] - vm->thresholds[outNeuron]);
	}
	else if (threadIdx.x == 1 % numInThreads) {
		vm->outTDs[outNeuron] = valueTransferDerivative(outputs[0] - vm->thresholds[outNeuron]);
	}
}

size_t getValueComputeSharedSize(ValueParameters* vp) {
	size_t size = 0;
	size += vp->backwardConnectivity + vp->thoughtConnectivity;
	size *= sizeof(float);
	return size;
}

__global__ void computeValueLayerLast(ValueMatrices* vm, ValueParameters* vp, bool turn1front) {
	size_t outNeuron = blockIdx.x;
	size_t clusterStart = outNeuron - outNeuron%CLUSTER_SIZE;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = vp->forBlockX;

	size_t backCon = vp->backwardConnectivity;

	size_t numInputs = vp->numInputs;

	extern __shared__ float outputs[];		//totalCon

	for (size_t i = inConnection; i < backCon; i += numInThreads) {
		outputs[i] = vm->weights[i + backCon*outNeuron] * vm->inlayer[(clusterStart + i) % numInputs];
	}

	__syncthreads();

	valueSumVector(outputs, backCon, inConnection, numInThreads);

	if (threadIdx.x == 0) {
		vm->outlayer[outNeuron] = outputs[0] - vm->thresholds[outNeuron];
	}
	else if (threadIdx.x == 1 % numInThreads) {
		vm->outTDs[outNeuron] = 1.0f;
	}
}

//note: this only works for layers other than the first, and assumes that numInputs >= numOutputs
__global__ void backPropagateValueToValue(ValueMatrices* vm, ValueParameters* vp, float* posErrorFact, float* negErrorFact) {
	size_t inNeuron = blockIdx.x;
	size_t outNeuronPosInCluster = threadIdx.x;		//0 to CLUSTER_SIZE-1 unless numOutputs < CLUSTER_SIZE
	size_t numOutNeuronThreads = vp->backValueBlockX;
	size_t outClusterOffset = threadIdx.y;			//0 to backCon/CLUSTER_SIZE-1 rounded up
	size_t numOffsetThreads = vp->backValueBlockY;

	size_t backCon = vp->backwardConnectivity;
	size_t thoughtCon = vp->thoughtConnectivity;
	size_t totalCon = backCon + thoughtCon;

	size_t numInputs = vp->numInputs;
	size_t numOutputs = vp->numOutputs;
	//size_t numOutputClusters = (numOutputs%CLUSTER_SIZE == 0 ? numOutputs / CLUSTER_SIZE : numOutputs / CLUSTER_SIZE + 1);

	size_t inputClusterStart = inNeuron - inNeuron % CLUSTER_SIZE;
	size_t numClusterOffsets = (backCon % CLUSTER_SIZE == 0 ? backCon / CLUSTER_SIZE : backCon / CLUSTER_SIZE + 1);
	size_t numClusterPositions = ((int)numOutputs < CLUSTER_SIZE ? numOutputs : CLUSTER_SIZE);

	extern __shared__ float errors[];	//numClusterOffsets*numClusterPositions

	for (size_t i = outClusterOffset; i < numClusterOffsets; i += numOffsetThreads) {
		int outClusterStart = (inputClusterStart - i*CLUSTER_SIZE) % numOutputs;
		if (outClusterStart < 0)
			outClusterStart += numOutputs;
		for (size_t j = outNeuronPosInCluster; j < numClusterPositions; j += numOutNeuronThreads) {
			size_t outNeuron = (size_t)outClusterStart + j;
			size_t errorIndex = i*numClusterPositions + j;
			size_t conNum = (numInputs + inNeuron - outClusterStart) % numInputs;
			size_t weightPos = conNum + outNeuron*totalCon;
			if (outNeuron >= numOutputs || conNum >= backCon)
				errors[errorIndex] = 0;
			else {
				float outErrorTD = vm->outerrors[outNeuron] * vm->outTDs[outNeuron];
				errors[errorIndex] = outErrorTD * vm->weights[weightPos];

				float change = outErrorTD * vm->inlayer[inNeuron];
#ifdef MAX_WEIGHT_CHANGE
				//change = valueBoundChange(change);
#endif
				vm->posWeightChanges[weightPos] = vm->posWeightChanges[weightPos]*VALUE_DECAY_FACTOR - change * posErrorFact[0];
				vm->negWeightChanges[weightPos] = vm->negWeightChanges[weightPos]*VALUE_DECAY_FACTOR - change * negErrorFact[0];
				
				//threshold
#ifdef MAX_WEIGHT_CHANGE
				//outErrorTD = valueBoundChange(outErrorTD);
#endif
				if (inNeuron == outClusterStart) {
					vm->posThresholdChanges[outNeuron] = vm->posThresholdChanges[outNeuron] * VALUE_DECAY_FACTOR + outErrorTD*posErrorFact[0];
					vm->negThresholdChanges[outNeuron] = vm->negThresholdChanges[outNeuron] * VALUE_DECAY_FACTOR + outErrorTD*negErrorFact[0];
				}
			}
		}
	}

	__syncthreads();

	//now finalize the error propagation
	valueSumVector(errors, numClusterOffsets*numClusterPositions, threadIdx.x + numOutNeuronThreads*threadIdx.y, numOutNeuronThreads*numOffsetThreads);

	vm->inerrors[inNeuron] = errors[0];
}

size_t getValueBackPropValueToValueSharedSize(ValueParameters* vp) {
	size_t backCon = vp->backwardConnectivity;
	size_t numClusterOffsets = (backCon % CLUSTER_SIZE == 0 ? backCon / CLUSTER_SIZE : backCon / CLUSTER_SIZE + 1);
	size_t numClusterPositions = std::min((int)vp->numOutputs, CLUSTER_SIZE);

	size_t size = 0;
	size += numClusterOffsets*numClusterPositions;
	size *= sizeof(float);
	return size;
}

__global__ void backPropagateValueToValueFirstLayer(ValueMatrices* vm, ValueParameters* vp, float* posErrorFact, float* negErrorFact) {
	size_t outNeuron = blockIdx.x;
	size_t clusterStart = outNeuron - outNeuron%CLUSTER_SIZE;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = vp->backValueFirstBlockX;

	size_t backCon = vp->backwardConnectivity;
	size_t thoughtCon = vp->thoughtConnectivity;
	size_t totalCon = backCon + thoughtCon;

	size_t numInputs = vp->numInputs;

	float outErrorTD = vm->outerrors[outNeuron] * vm->outTDs[outNeuron];
	if (inConnection == 0) {
		//float threshchange = valueBoundChange(outErrorTD);
		float threshchange = outErrorTD;
		vm->posThresholdChanges[outNeuron] = vm->posThresholdChanges[outNeuron] * VALUE_DECAY_FACTOR + threshchange*posErrorFact[0];
		vm->negThresholdChanges[outNeuron] = vm->negThresholdChanges[outNeuron] * VALUE_DECAY_FACTOR + threshchange*negErrorFact[0];
	}

	for (size_t i = inConnection; i < backCon; i += numInThreads) {
		float change = outErrorTD * vm->inlayer[(clusterStart + i) % numInputs];

#ifdef MAX_WEIGHT_CHANGE
		//change = valueBoundChange(change);
#endif
		size_t weightPos = i + totalCon*outNeuron;
		vm->posWeightChanges[weightPos] = vm->posWeightChanges[weightPos] * VALUE_DECAY_FACTOR - change * posErrorFact[0];
		vm->negWeightChanges[weightPos] = vm->negWeightChanges[weightPos] * VALUE_DECAY_FACTOR - change * negErrorFact[0];
	}
}

size_t getValueBackPropValueToValueFirstLayerSharedSize(ValueParameters* vp) {
	return 0;
}

//assumes numThoughts >= numOutputs
__global__ void backPropagateValueToThought(ValueMatrices* vm, ValueParameters* vp, bool turn1front, float* posErrorFact, float* negErrorFact) {
	size_t thoughtNeuron = blockIdx.x;
	size_t thoughtClusterStart = thoughtNeuron - thoughtNeuron%CLUSTER_SIZE;
	size_t outNeuronPosInCluster = threadIdx.x;		//0 to CLUSTER_SIZE-1 unless numOutputs < CLUSTER_SIZE
	size_t numOutNeuronThreads = vp->backThoughtBlockX;
	size_t outClusterOffset = threadIdx.y;			//0 to thoughtCon/CLUSTER_SIZE-1 rounded up
	size_t numOffsetThreads = vp->backThoughtBlockY;

	size_t backCon = vp->backwardConnectivity;
	size_t thoughtCon = vp->thoughtConnectivity;
	size_t totalCon = backCon + thoughtCon;

	size_t numThoughts = vp->numThoughtInputs;
	size_t numOutputs = vp->numOutputs;
	//size_t numOutputClusters = (numOutputs%CLUSTER_SIZE == 0 ? numOutputs / CLUSTER_SIZE : numOutputs / CLUSTER_SIZE + 1);

	size_t numClusterOffsets = (thoughtCon % CLUSTER_SIZE == 0 ? thoughtCon / CLUSTER_SIZE : thoughtCon / CLUSTER_SIZE + 1);
	size_t numClusterPositions = ((int)numOutputs < CLUSTER_SIZE ? numOutputs : CLUSTER_SIZE);

	float* thoughtlayer;
	if (turn1front)
		thoughtlayer = vm->thoughtlayer1;
	else
		thoughtlayer = vm->thoughtlayer2;

	extern __shared__ float errors[];	//numClusterOffsets*numClusterPositions

	for (size_t i = outClusterOffset; i < numClusterOffsets; i += numOffsetThreads) {
		int outClusterStart = (thoughtClusterStart - i*CLUSTER_SIZE) % numOutputs;
		if (outClusterStart < 0)
			outClusterStart += numOutputs;
		for (size_t j = outNeuronPosInCluster; j < numClusterPositions; j += numOutNeuronThreads) {
			size_t outNeuron = (size_t)outClusterStart + j;
			size_t errorIndex = i*numClusterPositions + j;
			size_t conNum = (numThoughts + thoughtNeuron - outClusterStart) % numThoughts;
			size_t weightPos = backCon + conNum + outNeuron*totalCon;
			if (outNeuron >= numOutputs || conNum + backCon >= totalCon)
				errors[errorIndex] = 0;
			else {
				float outErrorTD = vm->outerrors[outNeuron] * vm->outTDs[outNeuron];
				errors[errorIndex] = outErrorTD * vm->weights[weightPos];

				float change = outErrorTD * thoughtlayer[thoughtNeuron];
#ifdef MAX_WEIGHT_CHANGE
				//change = valueBoundChange(change);
#endif
				vm->posWeightChanges[weightPos] = vm->posWeightChanges[weightPos] * VALUE_DECAY_FACTOR - change * posErrorFact[0];
				vm->negWeightChanges[weightPos] = vm->negWeightChanges[weightPos] * VALUE_DECAY_FACTOR - change * negErrorFact[0];
			}
		}
	}

	__syncthreads();

	//now finalize the error propagation
	valueSumVector(errors, numClusterOffsets*numClusterPositions, threadIdx.x + numOutNeuronThreads*threadIdx.y, numOutNeuronThreads*numOffsetThreads);

	vm->thoughterrors[thoughtNeuron] = posErrorFact[0] * errors[0];
}

size_t getValueBackPropValueToThoughtSharedSize(ValueParameters* vp) {
	size_t thoughtCon = vp->thoughtConnectivity;
	size_t numClusterOffsets = (thoughtCon % CLUSTER_SIZE == 0 ? thoughtCon / CLUSTER_SIZE : thoughtCon / CLUSTER_SIZE + 1);
	size_t numClusterPositions = std::min((int)vp->numOutputs, CLUSTER_SIZE);

	size_t size = 0;
	size += numClusterOffsets*numClusterPositions;
	size *= sizeof(float);
	return size;
}

__global__ void updateValueWeights(ValueMatrices* vm, ValueParameters* vp, float pleasurePain) {
	size_t outNeuron = blockIdx.x;
	size_t inConnection = threadIdx.x;
	size_t numInThreads = vp->updateBlockX;	//totalCon

	size_t backCon = vp->backwardConnectivity;
	size_t thoughtCon = vp->thoughtConnectivity;
	size_t totalCon = backCon + thoughtCon;

	for (size_t i = inConnection; i < totalCon + 1; i += numInThreads) {
		if (i == totalCon) {
			float change;
			if (pleasurePain > 0)
				change = pleasurePain*vm->posThresholdChanges[outNeuron];
			else
				change = -pleasurePain*vm->negThresholdChanges[outNeuron];
#ifdef MAX_WEIGHT_CHANGE
			change = valueBoundChange(change);
#endif
			vm->thresholds[outNeuron] += change;

#ifdef CLEAR_VALUE_WEIGHTS_AFTER_UPDATE
			vm->posThresholdChanges[outNeuron] = 0;
			vm->negThresholdChanges[outNeuron] = 0;
#endif
		}
		else {
			float change;
			size_t weightNum = i + outNeuron*totalCon;
			if (pleasurePain > 0)
				change = pleasurePain*vm->posWeightChanges[weightNum];
			else
				change = -pleasurePain*vm->negWeightChanges[weightNum];
#ifdef MAX_WEIGHT_CHANGE
			change = valueBoundChange(change);
#endif
			vm->weights[weightNum] += change;

#ifdef CLEAR_VALUE_WEIGHTS_AFTER_UPDATE
			vm->posWeightChanges[weightNum] = 0;
			vm->negWeightChanges[weightNum] = 0;
#endif
		}
	}
}

size_t getValueUpdateSharedSize(ValueParameters* vp) {
	return 0;
}

__global__ void setErrorFactors(ValueMatrices* vm, ValueParameters* vp, float* posErrorFact, float* negErrorFact) {
	if (threadIdx.x == 0) {
		float err = (vm->outlayer[0] - POS_VALUE_GOAL);
#ifdef ERROR_FACTOR_MAX
		if (err > ERROR_FACTOR_MAX)
			err = ERROR_FACTOR_MAX;
		if (err < -ERROR_FACTOR_MAX)
			err = -ERROR_FACTOR_MAX;
#endif
		posErrorFact[0] = err;
	}
	else if (threadIdx.x == 1) {
		float err = (vm->outlayer[0] - NEG_VALUE_GOAL);
#ifdef ERROR_FACTOR_MAX
		if (err > ERROR_FACTOR_MAX)
			err = ERROR_FACTOR_MAX;
		if (err < -ERROR_FACTOR_MAX)
			err = -ERROR_FACTOR_MAX;
#endif
		negErrorFact[0] = err;
	}
}