#include "thoughtnet.cuh"
#include "thoughtkernel.cuh"

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
		tp->layer = i;
		if (i == 0)
			tp->numInputs = numInputs;
		else
			tp->numInputs = numClusters*CLUSTER_SIZE;

		if (i == numLayers - 1)
			tp->numOutputs = numOutputs;
		else
			tp->numOutputs = numClusters*CLUSTER_SIZE;

		if (i == numLayers - 1)
			tp->backwardConnectivity = tp->numInputs;
		else
			tp->backwardConnectivity = std::min(2 * CLUSTER_SIZE, (int)tp->numInputs);
		tp->sideConnectivity = std::min(CLUSTER_SIZE, (int)tp->numOutputs);

		//---thread shaping
		tp->forNBlockX = tp->numOutputs;
		tp->forBlockX = tp->backwardConnectivity + tp->sideConnectivity;

		tp->backNBlockX = tp->numOutputs;
		tp->backBlockX = tp->backwardConnectivity + tp->sideConnectivity;

		tp->updateNBlockX = tp->numOutputs;
		tp->updateBlockX = tp->backwardConnectivity + tp->sideConnectivity + 2;
		//-----------------

		if (i == numLayers - 1)
			tp->zeroRetains = true;
		else
			tp->zeroRetains = false;

		instantiateThoughtMatrices(&tc.thoughtMats[i], &tc.thoughtPars[i]);
	}
	linkThoughtLayers(&tc);
	copyThoughtLayersToDevice(&tc);

	initializeRandomStates(&tc);

	return tc;
}

ThoughtCollection* ThoughtNet::getThoughtCollection() {
	return &thoughtCollection;
}

void instantiateThoughtMatrices(ThoughtMatrices* tm, ThoughtParameters* tp) {
	checkCudaErrors(cudaMalloc(&tm->inlayer, tp->numInputs*THOUGHT_BP_DEPTH*sizeof(float)));
	checkCudaErrors(cudaMalloc(&tm->outlayer, tp->numOutputs*THOUGHT_BP_DEPTH*sizeof(float)));
	checkCudaErrors(cudaMalloc(&tm->outNew, tp->numOutputs*THOUGHT_BP_DEPTH*sizeof(float)));

	size_t totalConnectivity = tp->backwardConnectivity + tp->sideConnectivity;
	size_t numWeights = tp->numOutputs*totalConnectivity;

	float* h_weights = new float[numWeights];
	for (size_t i = 0; i < numWeights; i++) {
		h_weights[i] = (rand() % 21 - 10.0f) / 10.0f / (totalConnectivity + 1);
	}
	checkCudaErrors(cudaMalloc(&tm->weights, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemcpy(tm->weights, h_weights, numWeights*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_weights;

	float* h_weightChanges = new float[numWeights];
	for (size_t i = 0; i < numWeights; i++) {
		h_weightChanges[i] = 0;
	}
	checkCudaErrors(cudaMalloc(&tm->weightChanges, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemcpy(tm->weightChanges, h_weightChanges, numWeights*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_weightChanges;

	size_t numThresholds = tp->numOutputs;
	float* h_thresholds = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++) {
		h_thresholds[i] = (rand() % 21 - 10.0f) / 10.0f / (totalConnectivity + 1);
	}
	checkCudaErrors(cudaMalloc(&tm->thresholds, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(tm->thresholds, h_thresholds, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_thresholds;

	float* h_thresholdChanges = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++) {
		h_thresholdChanges[i] = 0;
	}
	checkCudaErrors(cudaMalloc(&tm->thresholdChanges, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(tm->thresholdChanges, h_thresholdChanges, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_thresholdChanges;

	float* h_retains = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++) {
		if (!tp->zeroRetains)
			h_retains[i] = 1.0f*(rand() % 10) / 10.0f;
		else
			h_retains[i] = 0;
	}
	checkCudaErrors(cudaMalloc(&tm->retains, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(tm->retains, h_retains, numThresholds*sizeof(float), cudaMemcpyHostToDevice));

	float* h_retainChanges = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++) {
		h_retainChanges[i] = 0;
	}
	checkCudaErrors(cudaMalloc(&tm->retainChanges, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(tm->retainChanges, h_retainChanges, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_retainChanges;

	checkCudaErrors(cudaMalloc(&tm->outTDs, tp->numOutputs*THOUGHT_BP_DEPTH*sizeof(float)));
	checkCudaErrors(cudaMalloc(&tm->errors, tp->numOutputs*THOUGHT_BP_DEPTH*sizeof(float)));

	tm->forwardSharedMem = getThoughtComputeSharedSize(tp);
	tm->backwardSharedMem = getThoughtBackPropSharedSize(tp);
	tm->updateSharedMem = getThoughtUpdateSharedSize(tp);

	checkCudaErrors(cudaMalloc(&tm->randStates, tp->numOutputs*sizeof(curandState)));
}

void linkThoughtLayers(ThoughtCollection* tc) {
	(&tc->thoughtMats[0])->inerrors = NULL;

	for (size_t i = 1; i < tc->numThoughtLayers; i++) {
		ThoughtMatrices* mat = &tc->thoughtMats[i];
		ThoughtMatrices* prevMat = &tc->thoughtMats[i-1];
		ThoughtParameters* par = &tc->thoughtPars[i];
		ThoughtParameters* prevPar = &tc->thoughtPars[i-1];

		if (prevPar->numOutputs != par->numInputs) {
			std::cout << "Layer sizes didn't match during link step" << std::endl;
			throw new std::runtime_error("Layer sizes didn't match during link step");
		}

		checkCudaErrors(cudaFree(mat->inlayer));
		mat->inlayer = prevMat->outlayer;
		mat->inerrors = prevMat->errors;
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

void initializeRandomStates(ThoughtCollection* tc) {
	size_t seed = rand();
	size_t sequenceStart = 0;
	for (size_t i = 0; i < tc->numThoughtLayers; i++) {
		ThoughtMatrices* tm = &tc->thoughtMats[i];
		ThoughtParameters* tp = &tc->thoughtPars[i];
		ThoughtMatrices* d_tm = tc->d_thoughtMats[i];
		ThoughtParameters* d_tp = tc->d_thoughtPars[i];
		initRandomStates << <1, tp->numOutputs >> >(d_tm, d_tp, seed, sequenceStart);
		checkCudaErrors(cudaPeekAtLastError());
		sequenceStart += tp->numOutputs;
	}
}

ThoughtNet::~ThoughtNet() {
	for (size_t i = 0; i < thoughtCollection.numThoughtLayers; i++) {
		ThoughtMatrices tm = thoughtCollection.thoughtMats[i];
		checkCudaErrors(cudaFree(tm.inlayer));
		checkCudaErrors(cudaFree(tm.outlayer));
		checkCudaErrors(cudaFree(tm.weights));
		checkCudaErrors(cudaFree(tm.thresholds));
		checkCudaErrors(cudaFree(tm.outTDs));
		checkCudaErrors(cudaFree(tm.errors));

		checkCudaErrors(cudaFree(thoughtCollection.d_thoughtMats[i]));
		checkCudaErrors(cudaFree(thoughtCollection.d_thoughtPars[i]));
	}
}

void ThoughtNet::incrementTurn() {
	bpTurn++;
	if (bpTurn >= THOUGHT_BP_DEPTH) {
		bpTurn = THOUGHT_BP_DEPTH - 1;
		backPropagate();
	}
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
	size_t pTurn;
	if (bpTurn == 0)
		pTurn = prevTurn;
	else
		pTurn = bpTurn - 1;

	for (size_t i = 0; i < thoughtCollection.numThoughtLayers; i++) {
		ThoughtMatrices* tm = &thoughtCollection.thoughtMats[i];
		ThoughtParameters* tp = &thoughtCollection.thoughtPars[i];
		ThoughtMatrices* d_tm = thoughtCollection.d_thoughtMats[i];
		ThoughtParameters* d_tp = thoughtCollection.d_thoughtPars[i];
		dim3 nBlocks(tp->forNBlockX);
		dim3 shape(tp->forBlockX);
		size_t shared = tm->forwardSharedMem;
		computeThoughtLayer<<<nBlocks, shape, shared>>>(d_tm, d_tp, bpTurn, pTurn);
		checkCudaErrors(cudaPeekAtLastError());
	}

#ifdef NORMALIZE_THOUGHT_OUTPUTS
	ThoughtMatrices* tm = &thoughtCollection.thoughtMats[thoughtCollection.numThoughtLayers - 1];
	ThoughtParameters* tp = &thoughtCollection.thoughtPars[thoughtCollection.numThoughtLayers - 1];
	ThoughtMatrices* d_tm = thoughtCollection.d_thoughtMats[thoughtCollection.numThoughtLayers - 1];
	ThoughtParameters* d_tp = thoughtCollection.d_thoughtPars[thoughtCollection.numThoughtLayers - 1];
	dim3 nBlocks(1);
	dim3 shape(tp->numOutputs);
	size_t shared = 2 * tp->numOutputs;
	normalizeOutputs << <nBlocks, shape, shared >> >(d_tm, d_tp, bpTurn);
	checkCudaErrors(cudaPeekAtLastError());
#endif
}

//note that the order of execution shouldn't matter here
void ThoughtNet::backPropagate() {
	prevTurn = bpTurn;
	while (bpTurn > 0) {
		for (size_t i = 0; i < thoughtCollection.numThoughtLayers; i++) {
			ThoughtMatrices* tm = &thoughtCollection.thoughtMats[i];
			ThoughtParameters* tp = &thoughtCollection.thoughtPars[i];
			ThoughtMatrices* d_tm = thoughtCollection.d_thoughtMats[i];
			ThoughtParameters* d_tp = thoughtCollection.d_thoughtPars[i];
			dim3 nBlocks(tp->backNBlockX);
			dim3 shape(tp->backBlockX);
			size_t shared = tm->backwardSharedMem;
			backPropagateThoughtLayer << <nBlocks, shape, shared >> >(d_tm, d_tp, bpTurn);
			checkCudaErrors(cudaPeekAtLastError());
		}
		bpTurn--;
	}

	for (size_t i = 0; i < thoughtCollection.numThoughtLayers; i++) {
		ThoughtMatrices* tm = &thoughtCollection.thoughtMats[i];
		ThoughtParameters* tp = &thoughtCollection.thoughtPars[i];
		ThoughtMatrices* d_tm = thoughtCollection.d_thoughtMats[i];
		ThoughtParameters* d_tp = thoughtCollection.d_thoughtPars[i];
		dim3 nBlocks(tp->updateNBlockX);
		dim3 shape(tp->updateBlockX);
		size_t shared = tm->updateSharedMem;
		updateThoughtWeights << <nBlocks, shape, shared >> >(d_tm, d_tp);
		checkCudaErrors(cudaPeekAtLastError());
	}
}

void ThoughtNet::resetThoughts() {
	backPropagate();
	prevTurn = 0;
}

ThoughtMatrices* ThoughtNet::getLastLevelMatrices() {
	return thoughtCollection.d_thoughtMats[numLayers - 1];
}

ThoughtParameters* ThoughtNet::getLastLevelParameters() {
	return thoughtCollection.d_thoughtPars[numLayers - 1];
}

void ThoughtNet::copyOutputToHost(float* d_outputs) {
	copyThoughtKernelOutputToHost << <1, numOutputs, 0 >> >(getLastLevelMatrices(), getLastLevelParameters(), d_outputs, bpTurn);
}

void ThoughtNet::saveWeights(std::string fname) {
	std::ofstream outfile(fname.c_str());

	for (size_t i = 0; i < numLayers; i++) {
		ThoughtMatrices* tm = &thoughtCollection.thoughtMats[i];
		ThoughtParameters* tp = &thoughtCollection.thoughtPars[i];
		size_t totalCon = tp->backwardConnectivity + tp->sideConnectivity;

		size_t numWeights = tp->numOutputs*totalCon;
		size_t numThresholds = tp->numOutputs;
		float* h_weights = new float[numWeights];
		float* h_thresholds = new float[numThresholds];
		float* h_retains = new float[numThresholds];
		checkCudaErrors(cudaMemcpy(h_weights, tm->weights, numWeights*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_thresholds, tm->thresholds, numThresholds*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_retains, tm->retains, numThresholds*sizeof(float), cudaMemcpyDeviceToHost));

		outfile << "Thought Layer " << i << ": " << std::endl;
		for (size_t j = 0; j < numThresholds; j++) {
			outfile << h_retains[j] << "| " << h_thresholds[j] << "| " << std::endl;
			for (size_t k = 0; k < totalCon; k++) {
				outfile << h_weights[k + j*totalCon] << " ";
			}
			outfile << std::endl << std::endl;
		}

		delete[] h_weights;
		delete[] h_thresholds;
	}
}

void ThoughtNet::loadWeights(std::string fname) {
	std::ifstream infile(fname.c_str());

	if (!infile.is_open()) {
		return;
	}

	for (size_t i = 0; i < numLayers; i++) {
		ThoughtMatrices* tm = &thoughtCollection.thoughtMats[i];
		ThoughtParameters* tp = &thoughtCollection.thoughtPars[i];
		size_t totalCon = tp->backwardConnectivity + tp->sideConnectivity;

		size_t numWeights = tp->numOutputs*totalCon;
		size_t numThresholds = tp->numOutputs;
		float* h_weights = new float[numWeights];
		float* h_thresholds = new float[numThresholds];
		float* h_retains = new float[numThresholds];

		std::string dum;
		infile >> dum >> dum >> dum; //Thought Layer #:
		for (size_t j = 0; j < numThresholds; j++) {
			infile >> h_retains[j] >> dum >> h_thresholds[j] >> dum;
			for (size_t k = 0; k < totalCon; k++) {
				infile >> h_weights[k + j*totalCon];
			}
		}
		checkCudaErrors(cudaMemcpy(tm->weights, h_weights, numWeights*sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(tm->thresholds, h_thresholds, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(tm->retains, h_retains, numThresholds*sizeof(float), cudaMemcpyHostToDevice));

		delete[] h_weights;
		delete[] h_thresholds;
	}
}

void ThoughtNet::setValueResultPointer(float* vr) {
	valueResult = vr;
}

size_t ThoughtNet::getBPTurn() {
	return bpTurn;
}

float* ThoughtNet::getDeviceInputLayer() {
	ThoughtMatrices* tm = &thoughtCollection.thoughtMats[0];
	ThoughtParameters* tp = &thoughtCollection.thoughtPars[0];
	return &tm->inlayer[bpTurn*tp->numInputs];
}