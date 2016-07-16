#ifndef SENTBOT_HEADER
#define SENTBOT_HEADER

#include "thoughtnet.cuh"
#include "valuenet.cuh"

class SentBot {
public:
	SentBot(size_t nInputs, size_t nOutputs, size_t nLayers, size_t nClusters);
	~SentBot();
	float* h_inputs;
	float* h_outputs;

	void takeTurn();
	void givePleasurePain(float pleasurePain);

	void saveWeights(std::string fname);

private:
	ThoughtNet* thoughtNet;
	ValueNet* valueNet;
	size_t numInputs;
	size_t numOutputs;
	float* d_outputs;
};

#endif