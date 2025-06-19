
#ifndef _MLP_H_
#define _MLP_H_


#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include "ap_fixed.h"
#include "math.h"
#include "ap_int.h"


//Neural Architecture
#define NUM_INPUTS 1024
#define NUM_LAYERS 1
#define NEURONS_PER_LAYER 1024
#define SCAN_LENGTH 62
#define LOOP_ITERATIONS 64512
#define NUM_OUTPUTS 10
#define NNZ 32768
#define NUM_TEST_SAMPLES 10000

#define CATEGORY_COUNT 10

#define TEST_DATA_FILE "..//..//..//..//..//X_test.dat"
#define LABEL_FILE "..//..//..//..//..//Y_test.dat"


using namespace std;

// data types for weights, biases, activations etc.
typedef ap_int<4> weight_t;
typedef ap_int<8> bias_t;
typedef ap_uint<4> act_t;
typedef ap_uint<5> idx_t;
typedef ap_uint<10> idxfull_t;
typedef ap_uint<64> map_t;
typedef ap_fixed<10,5,AP_RND > actf_t;
typedef ap_fixed<10,1,AP_RND > scale_t;

//top level function
void mlp_sparse(act_t Inputs[NUM_INPUTS], int *output);

void process_Layer(act_t* inp, \
                   const idx_t* idx, \
                   const map_t* map, \
                   const weight_t* val, \
                   act_t* activations, \
                   const bias_t* bias, \
                   scale_t fused_all_scales);

void func_out(act_t* inp, \
              const idxfull_t* idx, \
              const weight_t* val, \
              scale_t fused_input_weight_scale, \
              const bias_t* bias, \
              int* out );

#endif
