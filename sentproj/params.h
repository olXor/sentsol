#ifndef PARAMS_HEADER
#define PARAMS_HEADER

#define CLUSTER_SIZE 32

#define TRANSFER_FUNCTION_LIMIT 50.0f
#define TRANSFER_WIDTH 1.0f
#define NEGATIVE_TRANSFER_FACTOR 0.1f

#define MAX_WEIGHT_CHANGE 0.1f

#define STEPFACTOR 1e-5f

#define VALUE_DECAY_FACTOR 1.0

#define THOUGHT_BP_DEPTH 1000

#define POS_VALUE_GOAL 10.0f
#define NEG_VALUE_GOAL -10.0f

#define RECTIFIER 0
#define SIGMOID 1

#define THOUGHT_TRANSFER SIGMOID
#define VALUE_TRANSFER RECTIFIER

#define THOUGHT_RAND_WIDTH 0.1f

#define ERROR_FACTOR_MAX 2.0f

#define CLEAR_VALUE_WEIGHTS_AFTER_UPDATE

#endif