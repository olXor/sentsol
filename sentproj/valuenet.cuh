#ifndef VALUE_HEADER
#define VALUE_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <vector>
#include "params.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "thoughtnet.cuh"

struct ValueMatrices {
	float* inlayer;
	float* outlayer;
	float* thoughtlayer;

	float* weights;
	float* posWeightChanges;
	float* negWeightChanges;
	float* thresholds;
	float* posThresholdChanges;
	float* negThresholdChanges;
	float* outTDs;

	float* inerrors;
	float* outerrors;
	float* thoughterrors;

	size_t forwardSharedMem;
	size_t bpValueSharedMem;
	size_t bpValueFirstSharedMem;
	size_t bpThoughtSharedMem;
	size_t updateSharedMem;
};

struct ValueParameters {
	size_t forNBlockX;
	size_t forBlockX;

	size_t backValueNBlockX;
	size_t backValueBlockX;
	size_t backValueBlockY;

	size_t backValueFirstNBlockX;
	size_t backValueFirstBlockX;

	size_t backThoughtNBlockX;
	size_t backThoughtBlockX;
	size_t backThoughtBlockY;

	size_t updateNBlockX;
	size_t updateBlockX;

	size_t numInputs;
	size_t numOutputs;
	size_t numThoughtInputs;

	size_t backwardConnectivity;
	size_t thoughtConnectivity;
};

struct ValueCollection {
	std::vector<ValueMatrices> valueMats;
	std::vector<ValueParameters> valuePars;

	std::vector<ValueMatrices*> d_valueMats;
	std::vector<ValueParameters*> d_valuePars;

	size_t numValueLayers;
};

class ValueNet {
public:
	ValueNet(ThoughtNet* tn);
	~ValueNet();
	ValueCollection getValueCollection();
	void incrementTurn();
	bool turn1Front();

	void compute();
	void backPropagate();
	void updateWeights(float pleasurePain);

	void saveWeights(std::string fname);
	void loadWeights(std::string fname);

private:
	size_t turn = 0;
	ValueCollection valueCollection;
	size_t numInputs;
	size_t numLayers;
	size_t numThoughtNetOutputs;
	size_t numClusters;
	ThoughtNet* thoughtNet;

	float* posErrorFact; //one element, on device
	float* negErrorFact;

	ValueCollection createValueCollection(ThoughtNet* tn);
};

void instantiateValueMatrices(ValueMatrices* vm, ValueParameters* vp);
void linkValueLayers(ValueCollection* vc, ThoughtNet* tn);
void copyValueLayersToDevice(ValueCollection* vc);

#endif