#include "sentbot.cuh"

SentBot::SentBot(size_t nInputs, size_t nOutputs, size_t nLayers, size_t nClusters) {
	thoughtNet = new ThoughtNet(nInputs, nOutputs, nLayers, nClusters);
	valueNet = new ValueNet(thoughtNet);
	numInputs = nInputs;
	numOutputs = nOutputs;
	h_inputs = new float[numInputs];
	checkCudaErrors(cudaHostAlloc(&h_outputs, numOutputs*sizeof(float), cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer(&d_outputs, h_outputs, 0));
}

SentBot::~SentBot() {
	delete thoughtNet;
	delete valueNet;
	delete[] h_inputs;
	delete[] h_outputs;
}

void SentBot::takeTurn() {
	thoughtNet->incrementTurn();
	valueNet->incrementTurn();
	float* d_inputs = thoughtNet->getDeviceInputLayer();
	//use the below if we're not using mapped memory for the output
	//float* d_outputs = thoughtNet->getDeviceOutputLayer();

	cudaEvent_t calcDone;
	checkCudaErrors(cudaEventCreate(&calcDone));

	checkCudaErrors(cudaMemcpyAsync(d_inputs, h_inputs, thoughtNet->getNumInputs()*sizeof(float), cudaMemcpyHostToDevice));

	thoughtNet->compute();

	copyOutputToHost << <1, 1, 0 >> >(thoughtNet->getLastLevelMatrices(), thoughtNet->getLastLevelParameters(), d_outputs, thoughtNet->turn1Front());
	checkCudaErrors(cudaPeekAtLastError());

	checkCudaErrors(cudaEventRecord(calcDone));

	valueNet->compute();

	valueNet->backPropagate();

	thoughtNet->backPropagate();

	checkCudaErrors(cudaEventSynchronize(calcDone));
}

void SentBot::givePleasurePain(float pleasurePain) {
	valueNet->updateWeights(pleasurePain);
}