#ifndef THOUGHT_HEADER
#define THOUGHT_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <curand_kernel.h>
#include <vector>
#include "params.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

struct ThoughtMatrices {
	float* inlayer;	//THOUGHT_BP_DEPTH*numInputs
	float* outlayer;
	float* outTDs;
	float* outNew;	//the new TF, before retain is applied

	float* weights;
	float* weightChanges;
	float* thresholds;
	float* thresholdChanges;
	float* retains;
	float* retainChanges;

	float* errors;
	float* inerrors;

	curandState* randStates;

	size_t forwardSharedMem;
	size_t backwardSharedMem;
	size_t updateSharedMem;
};

struct ThoughtParameters {
	size_t forNBlockX;
	size_t forBlockX;

	size_t backNBlockX;
	size_t backBlockX;

	size_t updateNBlockX;
	size_t updateBlockX;

	size_t numInputs;
	size_t numOutputs;
	size_t backwardConnectivity;
	size_t sideConnectivity;

	bool zeroRetains;

	size_t layer;
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

	size_t getNumInputs();
	size_t getNumOutputs();
	size_t getNumLayers();
	size_t getNumClusters();
	size_t getBPTurn();

	void compute();
	void backPropagate();
	void copyOutputToHost(float* d_outputs);
	void resetThoughts();

	float* getDeviceInputLayer();

	ThoughtMatrices* getLastLevelMatrices();
	ThoughtParameters* getLastLevelParameters();

	void saveWeights(std::string fname);
	void loadWeights(std::string fname);

	void setValueResultPointer(float* vr);

private:
	size_t bpTurn = 0;
	size_t prevTurn = 0;		//only used for compute when the arrays wrap around
	ThoughtCollection thoughtCollection;
	size_t numInputs;
	size_t numOutputs;
	size_t numLayers;
	size_t numClusters;

	float* valueResult;	//one element, on device

	ThoughtCollection createThoughtCollection();
};

void instantiateThoughtMatrices(ThoughtMatrices* tm, ThoughtParameters* tp);
void linkThoughtLayers(ThoughtCollection* tc);
void copyThoughtLayersToDevice(ThoughtCollection* tc);
void initializeRandomStates(ThoughtCollection* tc);

#endif