#include "valuenet.cuh"

ValueNet::ValueNet(ThoughtNet* tn) {
	numInputs = tn->getNumInputs();
	numLayers = tn->getNumLayers() + 1;
	numClusters = tn->getNumClusters();
	numThoughtNetOutputs = tn->getNumOutputs();
	thoughtNet = tn;
	valueCollection = createValueCollection(tn);
	checkCudaErrors(cudaMalloc(&posErrorFact, sizeof(float)));
	checkCudaErrors(cudaMalloc(&negErrorFact, sizeof(float)));
}

ValueCollection ValueNet::createValueCollection(ThoughtNet* tn) {
	ValueCollection vc;

	vc.numValueLayers = numLayers;
	vc.valueMats.resize(numLayers);
	vc.valuePars.resize(numLayers);

	if (numInputs != tn->getNumInputs()) {
		std::cout << "Value net inputs don't match Thought net!" << std::endl;
		throw new std::runtime_error("Value net inputs don't match Thought net!");
	}

	//set up ValueParameters
	for (size_t i = 0; i < numLayers; i++) {
		ValueParameters* vp = &vc.valuePars[i];
		if (i == 0)
			vp->numInputs = numInputs;
		else
			vp->numInputs = numClusters*CLUSTER_SIZE;

		if (i == numLayers - 1)
			vp->numOutputs = 1;
		else if (i == numLayers - 2)
			vp->numOutputs = numThoughtNetOutputs;
		else
			vp->numOutputs = numClusters*CLUSTER_SIZE;

		if (i == numLayers - 1)
			vp->numThoughtInputs = 0;
		else
			vp->numThoughtInputs = tn->getThoughtCollection().thoughtPars[i].numOutputs;

		vp->backwardConnectivity = std::min(2 * CLUSTER_SIZE, (int)vp->numInputs);
		vp->thoughtConnectivity = std::min(CLUSTER_SIZE, (int)vp->numThoughtInputs);

		//---thread shaping
		vp->forNBlockX = vp->numOutputs;
		vp->forBlockX = vp->backwardConnectivity + vp->thoughtConnectivity;

		vp->backValueNBlockX = vp->numInputs;
		vp->backValueBlockX = std::min(CLUSTER_SIZE, (int)vp->numOutputs);
		vp->backValueBlockY = (vp->backwardConnectivity % CLUSTER_SIZE == 0 ? vp->backwardConnectivity / CLUSTER_SIZE : vp->backwardConnectivity / CLUSTER_SIZE + 1);

		vp->backValueFirstNBlockX = vp->numOutputs;
		vp->backValueFirstBlockX = vp->backwardConnectivity;

		vp->backThoughtNBlockX = vp->numThoughtInputs;
		vp->backThoughtBlockX = std::min(CLUSTER_SIZE, (int)vp->numOutputs);
		vp->backThoughtBlockY = (vp->thoughtConnectivity % CLUSTER_SIZE == 0 ? vp->thoughtConnectivity / CLUSTER_SIZE : vp->thoughtConnectivity / CLUSTER_SIZE + 1);

		vp->updateNBlockX = vp->numOutputs;
		vp->updateBlockX = vp->backwardConnectivity + vp->thoughtConnectivity;
		//-----------------

		instantiateValueMatrices(&vc.valueMats[i], &vc.valuePars[i]);
	}
	linkValueLayers(&vc, tn);
	copyValueLayersToDevice(&vc);
}

ValueCollection ValueNet::getValueCollection() {
	return valueCollection;
}

ValueMatrices instantiateValueMatrices(ValueMatrices* vm, ValueParameters* vp) {
	checkCudaErrors(cudaMalloc(&vm->inlayer, vp->numInputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&vm->outlayer, vp->numOutputs*sizeof(float)));

	size_t totalConnectivity = vp->backwardConnectivity + vp->thoughtConnectivity;
	size_t numWeights = vp->numOutputs*totalConnectivity;

	float* h_weights = new float[numWeights];
	for (size_t i = 0; i < numWeights; i++) {
		h_weights[i] = (rand() % 21 - 10.0f) / 10.0f / (totalConnectivity + 1);
	}
	checkCudaErrors(cudaMalloc(&vm->weights, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemcpy(vm->weights, h_weights, numWeights*sizeof(float), cudaMemcpyHostToDevice));
	delete [] h_weights;

	float* h_weightChanges = new float[numWeights];
	for (size_t i = 0; i < numWeights; i++) {
		h_weightChanges[i] = 0;
	}
	checkCudaErrors(cudaMalloc(&vm->posWeightChanges, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemcpy(vm->posWeightChanges, h_weightChanges, numWeights*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(&vm->negWeightChanges, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemcpy(vm->negWeightChanges, h_weightChanges, numWeights*sizeof(float), cudaMemcpyHostToDevice));
	delete [] h_weightChanges;

	size_t numThresholds = vp->numOutputs;
	float* h_thresholds = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++) {
		h_thresholds[i] = (rand() % 21 - 10.0f) / 10.0f / (totalConnectivity + 1);
	}
	checkCudaErrors(cudaMalloc(&vm->thresholds, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(vm->thresholds, h_thresholds, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	delete [] h_thresholds;

	size_t numThresholds = vp->numOutputs;
	float* h_thresholdChanges = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++) {
		h_thresholdChanges[i] = 0;
	}
	checkCudaErrors(cudaMalloc(&vm->posThresholdChanges, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(vm->posThresholdChanges, h_thresholdChanges, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(&vm->negThresholdChanges, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(vm->negThresholdChanges, h_thresholdChanges, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	delete [] h_thresholdChanges;

	checkCudaErrors(cudaMalloc(&vm->outTDs, vp->numOutputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&vm->inerrors, vp->numInputs*sizeof(float)));

	//initialize all output errors to STEPFACTOR. The last layer will never deviate from this value, and for other layers it will be immediately overwritten
	size_t numErrors = vp->numOutputs;
	float* h_outErrors = new float[numErrors];
	for (size_t i = 0; i < numErrors; i++) {
		h_outErrors[i] = STEPFACTOR;
	}
	checkCudaErrors(cudaMalloc(&vm->outerrors, vp->numOutputs*sizeof(float)));
	checkCudaErrors(cudaMemcpy(vm->outerrors, h_outErrors, numErrors*sizeof(float), cudaMemcpyHostToDevice));

	vm->forwardSharedMem = getValueComputeSharedSize(vp);
	vm->bpValueSharedMem = getValueBackPropValueToValueSharedSize(vp);
	vm->bpValueFirstSharedMem = getValueBackPropValueToValueFirstLayerSharedSize(vp);
	vm->bpThoughtSharedMem = getValueBackPropValueToThoughtSharedSize(vp);
	vm->updateSharedMem = getValueUpdateSharedSize(vp);
}

void linkValueLayers(ValueCollection* vc, ThoughtNet* tn) {
	ThoughtCollection* tc = &tn->getThoughtCollection();
	if (vc->numValueLayers != tc->numThoughtLayers) {
		std::cout << "ValueNet and ThoughtNet have different numbers of layers!";
		throw new std::runtime_error("ValueNet and ThoughtNet have different numbers of layers!");
	}
	for (size_t i = 0; i < vc->numValueLayers; i++) {
		if (i > 0) {
			if (vc->valuePars[i - 1].numOutputs != vc->valuePars[i].numInputs) {
				std::cout << "Layer sizes didn't match during link step" << std::endl;
				throw new std::runtime_error("Layer sizes didn't match during link step");
			}

			checkCudaErrors(cudaFree(vc->valueMats[i].inlayer));
			vc->valueMats[i].inlayer = vc->valueMats[i - 1].outlayer;
		}

		if (vc->valuePars[i].numThoughtInputs != tc->thoughtPars[i].numOutputs) {
			std::cout << "Failed to link value net to thought net!" << std::endl;
			throw new std::runtime_error("Failed to link value net to thought net!");
		}

		vc->valueMats[i].thoughtlayer1 = tc->thoughtMats[i].outlayer1;
		vc->valueMats[i].thoughtlayer2 = tc->thoughtMats[i].outlayer2;
	}
}

void copyValueLayersToDevice(ValueCollection* vc) {
	for (size_t i = 0; i < vc->valueMats.size(); i++){
		ValueMatrices* d_valueMat;
		checkCudaErrors(cudaMalloc(&d_valueMat, sizeof(ValueMatrices)));
		checkCudaErrors(cudaMemcpy(d_valueMat, &vc->valueMats[i], sizeof(ValueMatrices), cudaMemcpyHostToDevice));
		vc->d_valueMats.push_back(d_valueMat);
	}

	for (size_t i = 0; i < vc->valuePars.size(); i++){
		ValueParameters* d_valuePar;
		checkCudaErrors(cudaMalloc(&d_valuePar, sizeof(ValueParameters)));
		checkCudaErrors(cudaMemcpy(d_valuePar, &vc->valuePars[i], sizeof(ValueParameters), cudaMemcpyHostToDevice));
		vc->d_valuePars.push_back(d_valuePar);
	}
}

void ValueNet::incrementTurn() {
	turn++;
}

bool ValueNet::turn1Front() {
	return turn % 2 == 0;
}

//need layers in correct order for this net
void ValueNet::compute() {
	bool turn1front = turn1Front();
	for (size_t i = 0; i < valueCollection.numValueLayers; i++) {
		ValueMatrices* vm = &valueCollection.valueMats[i];
		ValueParameters* vp = &valueCollection.valuePars[i];
		dim3 nBlocks(vp->forNBlockX);
		dim3 shape(vp->forBlockX);
		size_t shared = vm->forwardSharedMem;
		computeValueLayer<<<nBlocks, shape, shared>>>(vm, vp, turn1front);
		checkCudaErrors(cudaPeekAtLastError());
	}
}

void ValueNet::backPropagate() {
	bool turn1front = turn1Front();

	size_t valueOutputLayer = valueCollection.numValueLayers - 1;
	setErrorFactors<<<1,2,0>>>(&valueCollection.valueMats[valueOutputLayer], &valueCollection.valuePars[valueOutputLayer], posErrorFact, negErrorFact);
	checkCudaErrors(cudaPeekAtLastError());

	//we must propagate the first layer separately. Also must propagate in reverse order
	for (size_t i = valueCollection.numValueLayers - 1; i > 0; i--) {
		ValueMatrices* vm = &valueCollection.valueMats[i];
		ValueParameters* vp = &valueCollection.valuePars[i];
		dim3 nBlocks(vp->backValueNBlockX);
		dim3 shape(vp->backValueBlockX, vp->backValueBlockY);
		size_t shared = vm->bpValueSharedMem;
		backPropagateValueToValue << <nBlocks, shape, shared >> >(vm, vp, posErrorFact, negErrorFact);
		checkCudaErrors(cudaPeekAtLastError());

		dim3 thoughtNBlocks(vp->backThoughtNBlockX);
		dim3 thoughtShape(vp->backThoughtBlockX, vp->backThoughtBlockY);
		size_t thoughtShared = vm->bpThoughtSharedMem;
		backPropagateValueToThought << <thoughtNBlocks, thoughtShape, thoughtShared >> >(vm, vp, posErrorFact, negErrorFact);
		checkCudaErrors(cudaPeekAtLastError());
	}

	//now first layer
	ValueMatrices* vm = &valueCollection.valueMats[0];
	ValueParameters* vp = &valueCollection.valuePars[0];
	dim3 nBlocks(vp->backValueFirstNBlockX);
	dim3 shape(vp->backValueFirstBlockX);
	size_t shared = vm->bpValueFirstSharedMem;
	backPropagateValueToValueFirstLayer << <nBlocks, shape, shared >> >(vm, vp, posErrorFact, negErrorFact);
	checkCudaErrors(cudaPeekAtLastError());

	dim3 thoughtNBlocks(vp->backThoughtNBlockX);
	dim3 thoughtShape(vp->backThoughtBlockX, vp->backThoughtBlockY);
	size_t thoughtShared = vm->bpThoughtSharedMem;
	backPropagateValueToThought << <thoughtNBlocks, thoughtShape, thoughtShared >> >(vm, vp, posErrorFact, negErrorFact);
	checkCudaErrors(cudaPeekAtLastError());
}

void ValueNet::updateWeights(float pleasurePain) {
	for (size_t i = 0; i < valueCollection.numValueLayers; i++) {
		ValueMatrices* vm = &valueCollection.valueMats[0];
		ValueParameters* vp = &valueCollection.valuePars[0];
		dim3 nBlocks(vp->updateNBlockX);
		dim3 shape(vp->updateBlockX);
		size_t shared = vm->updateSharedMem;
		updateValueWeights << <nBlocks, shape, shared >> >(vm, vp, pleasurePain);
		checkCudaErrors(cudaPeekAtLastError());
	}
}

ValueNet::~ValueNet() {
	for (size_t i = 0; i < valueCollection.numValueLayers; i++) {
		ValueMatrices vp = valueCollection.valueMats[i];
		checkCudaErrors(cudaFree(vp.inlayer));
		checkCudaErrors(cudaFree(vp.outlayer));
		checkCudaErrors(cudaFree(vp.weights));
		checkCudaErrors(cudaFree(vp.posWeightChanges));
		checkCudaErrors(cudaFree(vp.negWeightChanges));
		checkCudaErrors(cudaFree(vp.thresholds));
		checkCudaErrors(cudaFree(vp.posThresholdChanges));
		checkCudaErrors(cudaFree(vp.negThresholdChanges));
		checkCudaErrors(cudaFree(vp.outTDs));
		checkCudaErrors(cudaFree(vp.inerrors));
		checkCudaErrors(cudaFree(vp.outerrors));
		checkCudaErrors(cudaFree(vp.thoughterrors));

		checkCudaErrors(cudaFree(valueCollection.d_valueMats[i]));
		checkCudaErrors(cudaFree(valueCollection.d_valuePars[i]));
	}
}