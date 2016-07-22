#include "sentbot.cuh"
#include <sstream>

SentBot::SentBot(size_t nInputs, size_t nOutputs, size_t nLayers, size_t nClusters) {
	thoughtNet = new ThoughtNet(nInputs, nOutputs, nLayers, nClusters);
	valueNet = new ValueNet(thoughtNet);
	numInputs = nInputs;
	numOutputs = nOutputs;
	h_inputs = new float[numInputs];
	checkCudaErrors(cudaHostAlloc(&h_outputs, numOutputs*sizeof(float), cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer(&d_outputs, h_outputs, 0));
	checkCudaErrors(cudaEventCreate(&calcDone));
}

SentBot::~SentBot() {
	delete thoughtNet;
	delete valueNet;
	delete[] h_inputs;
	delete[] h_outputs;
}

void SentBot::takeTurn() {
	thoughtNet->incrementTurn();	//automatically backprops thoughtNet here
	valueNet->incrementTurn();
	float* d_inputs = thoughtNet->getDeviceInputLayer();
	//use the below if we're not using mapped memory for the output
	//float* d_outputs = thoughtNet->getDeviceOutputLayer();

	checkCudaErrors(cudaMemcpyAsync(d_inputs, h_inputs, thoughtNet->getNumInputs()*sizeof(float), cudaMemcpyHostToDevice));

	thoughtNet->compute();

	thoughtNet->copyOutputToHost(d_outputs);
	checkCudaErrors(cudaPeekAtLastError());

	checkCudaErrors(cudaEventRecord(calcDone));

	valueNet->compute();

	valueNet->backPropagate();

	checkCudaErrors(cudaEventSynchronize(calcDone));
}

void SentBot::givePleasurePain(float pleasurePain) {
	valueNet->updateWeights(pleasurePain);
}

void SentBot::saveWeights(std::string fname) {
	std::stringstream base;
	base << "saveweights/" << fname;

	std::stringstream tss;
	tss << base.str() << "thought";
	thoughtNet->saveWeights(tss.str());

	std::stringstream vss;
	vss << base.str() << "value";
	valueNet->saveWeights(vss.str());
}

void SentBot::loadWeights(std::string fname) {
	std::stringstream base;
	base << "saveweights/" << fname;

	std::stringstream tss;
	tss << base.str() << "thought";
	thoughtNet->loadWeights(tss.str());

	std::stringstream vss;
	vss << base.str() << "value";
	valueNet->loadWeights(vss.str());
}

void SentBot::resetThoughts() {
	thoughtNet->resetThoughts();
}