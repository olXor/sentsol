#ifndef THOUGHT_HEADER
#define THOUGHT_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <vector>
#include "params.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "thoughtkernel.cuh"

struct ThoughtMatrices {
	float* inlayer1;
	float* inlayer2;
	float* outlayer1;
	float* outlayer2;

	float* weights;
	float* thresholds;
	float* outTDs;
	float* errors;

	size_t forwardSharedMem;
	size_t backwardSharedMem;
};

struct ThoughtParameters {
	size_t forNBlockX;
	size_t forBlockX;

	size_t backNBlockX;
	size_t backBlockX;

	size_t numInputs;
	size_t numOutputs;
	size_t backwardConnectivity;
	size_t sideConnectivity;
};

struct ThoughtCollection {
	std::vector<ThoughtMatrices> thoughtMats;
	std::vector<ThoughtParameters> thoughtPars;

	std::vector<ThoughtMatrices*> d_thoughtMats;
	std::vector<ThoughtParameters*> d_thoughtPars;

	size_t numThoughtLayers;
};

class ThoughtNet {
public:
	ThoughtNet(size_t nInputs, size_t nOutputs, size_t nLayers, size_t nClusters);
	~ThoughtNet();
	ThoughtCollection* getThoughtCollection();
	void incrementTurn();
	bool turn1Front();

	size_t getNumInputs();
	size_t getNumOutputs();
	size_t getNumLayers();
	size_t getNumClusters();

	float* getDeviceInputLayer();
	float* getDeviceOutputLayer();

	void compute();
	void backPropagate();

	ThoughtMatrices* getLastLevelMatrices();
	ThoughtParameters* getLastLevelParameters();

private:
	size_t turn = 0;
	ThoughtCollection thoughtCollection;
	size_t numInputs;
	size_t numOutputs;
	size_t numLayers;
	size_t numClusters;

	ThoughtCollection createThoughtCollection();
};

void instantiateThoughtMatrices(ThoughtMatrices* tm, ThoughtParameters* tp);
void linkThoughtLayers(ThoughtCollection* tc);
void copyThoughtLayersToDevice(ThoughtCollection* tc);

#endif