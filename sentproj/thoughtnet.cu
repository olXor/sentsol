#include "thoughtnet.cuh"

ThoughtNet::ThoughtNet(size_t nInputs, size_t nOutputs, size_t nLayers, size_t nClusters) {
	numInputs = nInputs;
	numOutputs = nOutputs;
	numLayers = nLayers;
	numClusters = nClusters;
	thoughtCollection = createThoughtCollection();
}

ThoughtCollection ThoughtNet::createThoughtCollection() {
	ThoughtCollection tc;

	tc.numThoughtLayers = numLayers;
	tc.thoughtMats.resize(numLayers);
	tc.thoughtPars.resize(numLayers);

	//set up ThoughtParameters
	for (size_t i = 0; i < numLayers; i++) {
		ThoughtParameters* tp = &tc.thoughtPars[i];
		if (i == 0)
			tp->numInputs = numInputs;
		else
			tp->numInputs = numClusters*CLUSTER_SIZE;

		if (i == numLayers - 1)
			tp->numOutputs = numOutputs;
		else
			tp->numOutputs = numClusters*CLUSTER_SIZE;

		tp->backwardConnectivity = std::min(2 * CLUSTER_SIZE, (int)tp->numInputs);
		tp->sideConnectivity = std::min(CLUSTER_SIZE, (int)tp->numOutputs);

		//---thread shaping
		tp->forNBlockX = tp->numOutputs;
		tp->forBlockX = tp->backwardConnectivity + tp->sideConnectivity;

		tp->backNBlockX = tp->numOutputs;
		tp->backBlockX = tp->backwardConnectivity + tp->sideConnectivity;
		//-----------------

		instantiateThoughtMatrices(&tc.thoughtMats[i], &tc.thoughtPars[i]);
	}
	linkThoughtLayers(&tc);
	copyThoughtLayersToDevice(&tc);
}

ThoughtCollection ThoughtNet::getThoughtCollection() {
	return thoughtCollection;
}

ThoughtMatrices instantiateThoughtMatrices(ThoughtMatrices* tm, ThoughtParameters* tp) {
	checkCudaErrors(cudaMalloc(&tm->inlayer1, tp->numInputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&tm->inlayer2, tp->numInputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&tm->outlayer1, tp->numOutputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&tm->outlayer2, tp->numOutputs*sizeof(float)));

	size_t totalConnectivity = tp->backwardConnectivity + tp->sideConnectivity;
	size_t numWeights = tp->numOutputs*totalConnectivity;

	float* h_weights = new float[numWeights];
	for (size_t i = 0; i < numWeights; i++) {
		h_weights[i] = (rand() % 21 - 10.0f) / 10.0f / (totalConnectivity + 1);
	}
	checkCudaErrors(cudaMalloc(&tm->weights, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemcpy(tm->weights, h_weights, numWeights*sizeof(float), cudaMemcpyHostToDevice));
	delete [] h_weights;

	size_t numThresholds = tp->numOutputs;
	float* h_thresholds = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++) {
		h_thresholds[i] = (rand() % 21 - 10.0f) / 10.0f / (totalConnectivity + 1);
	}
	checkCudaErrors(cudaMalloc(&tm->thresholds, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(tm->thresholds, h_thresholds, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	delete [] h_thresholds;

	checkCudaErrors(cudaMalloc(&tm->outTDs, tp->numOutputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&tm->errors, tp->numOutputs*sizeof(float)));

	tm->forwardSharedMem = getThoughtComputeSharedSize(tp);
	tm->backwardSharedMem = getThoughtBackPropSharedSize(tp);
}

void linkThoughtLayers(ThoughtCollection* tc) {
	for (size_t i = 1; i < tc->numThoughtLayers; i++) {
		ThoughtMatrices* mat = &tc->thoughtMats[i];
		ThoughtMatrices* prevMat = &tc->thoughtMats[i-1];
		ThoughtParameters* par = &tc->thoughtPars[i];
		ThoughtParameters* prevPar = &tc->thoughtPars[i-1];

		if (prevPar->numOutputs != par->numInputs) {
			std::cout << "Layer sizes didn't match during link step" << std::endl;
			throw new std::runtime_error("Layer sizes didn't match during link step");
		}

		checkCudaErrors(cudaFree(mat->inlayer1));
		mat->inlayer1 = prevMat->outlayer1;
		checkCudaErrors(cudaFree(mat->inlayer2));
		mat->inlayer2 = prevMat->outlayer2;
	}
}

void copyThoughtLayersToDevice(ThoughtCollection* tc) {
	for (size_t i = 0; i < tc->thoughtMats.size(); i++){
		ThoughtMatrices* d_thoughtMat;
		checkCudaErrors(cudaMalloc(&d_thoughtMat, sizeof(ThoughtMatrices)));
		checkCudaErrors(cudaMemcpy(d_thoughtMat, &tc->thoughtMats[i], sizeof(ThoughtMatrices), cudaMemcpyHostToDevice));
		tc->d_thoughtMats.push_back(d_thoughtMat);
	}

	for (size_t i = 0; i < tc->thoughtPars.size(); i++){
		ThoughtParameters* d_thoughtPar;
		checkCudaErrors(cudaMalloc(&d_thoughtPar, sizeof(ThoughtParameters)));
		checkCudaErrors(cudaMemcpy(d_thoughtPar, &tc->thoughtPars[i], sizeof(ThoughtParameters), cudaMemcpyHostToDevice));
		tc->d_thoughtPars.push_back(d_thoughtPar);
	}
}

ThoughtNet::~ThoughtNet() {
	for (size_t i = 0; i < thoughtCollection.numThoughtLayers; i++) {
		ThoughtMatrices tm = thoughtCollection.thoughtMats[i];
		checkCudaErrors(cudaFree(tm.inlayer1));
		checkCudaErrors(cudaFree(tm.inlayer2));
		checkCudaErrors(cudaFree(tm.outlayer1));
		checkCudaErrors(cudaFree(tm.outlayer2));
		checkCudaErrors(cudaFree(tm.weights));
		checkCudaErrors(cudaFree(tm.thresholds));
		checkCudaErrors(cudaFree(tm.outTDs));
		checkCudaErrors(cudaFree(tm.errors));

		checkCudaErrors(cudaFree(thoughtCollection.d_thoughtMats[i]));
		checkCudaErrors(cudaFree(thoughtCollection.d_thoughtPars[i]));
	}
}

void ThoughtNet::incrementTurn() {
	turn++;
}

bool ThoughtNet::turn1Front() {
	return turn % 2 == 0;
}

size_t ThoughtNet::getNumInputs() {
	return numInputs;
}

size_t ThoughtNet::getNumOutputs() {
	return numOutputs;
}

size_t ThoughtNet::getNumLayers() {
	return numLayers;
}

size_t ThoughtNet::getNumClusters() {
	return numClusters;
}

void ThoughtNet::compute() {
	bool turn1front = turn1Front();
	for (size_t i = 0; i < thoughtCollection.numThoughtLayers; i++) {
		ThoughtMatrices* tm = &thoughtCollection.thoughtMats[i];
		ThoughtParameters* tp = &thoughtCollection.thoughtPars[i];
		dim3 nBlocks(tp->forNBlockX);
		dim3 shape(tp->forBlockX);
		size_t shared = tm->forwardSharedMem;
		computeThoughtLayer<<<nBlocks, shape, shared>>>(tm, tp, turn1front);
		checkCudaErrors(cudaPeekAtLastError());
	}
}

//note that the order of execution shouldn't matter here
void ThoughtNet::backPropagate() {
	bool turn1front = turn1Front();
	for (size_t i = 0; i < thoughtCollection.numThoughtLayers; i++) {
		ThoughtMatrices* tm = &thoughtCollection.thoughtMats[i];
		ThoughtParameters* tp = &thoughtCollection.thoughtPars[i];
		dim3 nBlocks(tp->backNBlockX);
		dim3 shape(tp->backBlockX);
		size_t shared = tm->backwardSharedMem;
		backPropagateThoughtLayer<<<nBlocks, shape, shared>>>(tm, tp, turn1front);
		checkCudaErrors(cudaPeekAtLastError());
	}
}

//note: returns the BACK input layer, so that one can immediately call calculate afterwards
float* ThoughtNet::getDeviceInputLayer() {
	bool turn1front = turn1Front();
	if (turn1front)
		return thoughtCollection.thoughtMats[0].inlayer2;
	else
		return thoughtCollection.thoughtMats[0].inlayer1;
}

//note: returns the FRONT output layer, so that one can immediately call this after calculating
float* ThoughtNet::getDeviceOutputLayer() {
	bool turn1front = turn1Front();
	if (turn1front)
		return thoughtCollection.thoughtMats[thoughtCollection.numThoughtLayers - 1].outlayer1;
	else
		return thoughtCollection.thoughtMats[thoughtCollection.numThoughtLayers - 1].outlayer2;
}

ThoughtMatrices* ThoughtNet::getLastLevelMatrices() {
	return &thoughtCollection.thoughtMats[numLayers - 1];
}

ThoughtParameters* ThoughtNet::getLastLevelParameters() {
	return &thoughtCollection.thoughtPars[numLayers - 1];
}