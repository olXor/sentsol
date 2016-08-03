#ifndef PARAMS_HEADER
#define PARAMS_HEADER

#define CLUSTER_SIZE 32

#define TRANSFER_FUNCTION_LIMIT 50.0f
#define TRANSFER_WIDTH 1.0f
#define THOUGHT_NEGATIVE_TRANSFER_FACTOR 0.0f
#define VALUE_NEGATIVE_TRANSFER_FACTOR 0.1f

#define THOUGHT_MAX_WEIGHT_CHANGE 1.0f
#define VALUE_MAX_WEIGHT_CHANGE 1.0f

#define STEPFACTOR 1e-6f
#define THOUGHT_STEP_MULT 1e+2f
#define RETAIN_STEP_MULT 1e+1f		//in addition to the thought step mult

#define VALUE_DECAY_FACTOR 0.999f

#define THOUGHT_BP_DEPTH 50000

#define POS_VALUE_GOAL 10.0f
#define NEG_VALUE_GOAL -10.0f

#define RECTIFIER 0
#define SIGMOID 1

#define THOUGHT_TRANSFER SIGMOID
#define THOUGHT_BASELINE -0.5f	//added to every transfer function
#define VALUE_TRANSFER RECTIFIER

#define THOUGHT_RAND_WIDTH 0.1f

#define ERROR_FACTOR_MAX (POS_VALUE_GOAL - NEG_VALUE_GOAL)

#define CLEAR_VALUE_WEIGHTS_AFTER_UPDATE

//#define NORMALIZE_THOUGHT_OUTPUTS

#endif