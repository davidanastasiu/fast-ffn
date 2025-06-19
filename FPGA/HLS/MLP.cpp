
    #include "mlp.h"
    #include "weights.h"

    void mlp_sparse(
            act_t Inputs[NUM_INPUTS],
            int *output
            )

    {

    #pragma HLS INTERFACE mode=ap_ctrl_chain port=return

    

act_t Act1[NEURONS_PER_LAYER];
#pragma HLS stream type=pipo variable=Act1 depth=2

act_t Act2[NEURONS_PER_LAYER];
#pragma HLS stream type=pipo variable=Act2 depth=2

#pragma HLS bind_storage variable=val1 type=rom_1p impl=BRAM
#pragma HLS bind_storage variable=idx1 type=rom_1p impl=BRAM
#pragma HLS bind_storage variable=map1 type=rom_1p impl=BRAM
#pragma HLS bind_storage variable=bias1 type=rom_1p impl=LUTRAM

#pragma HLS bind_storage variable=val2 type=rom_1p impl=BRAM
#pragma HLS bind_storage variable=idx2 type=rom_1p impl=BRAM
#pragma HLS bind_storage variable=map2 type=rom_1p impl=BRAM
#pragma HLS bind_storage variable=bias2 type=rom_1p impl=LUTRAM

#pragma HLS bind_storage variable=val3 type=rom_1p impl=BRAM
#pragma HLS bind_storage variable=idx3 type=rom_1p impl=BRAM
#pragma HLS bind_storage variable=bias3 type=rom_1p impl=LUTRAM

#pragma HLS dataflow

process_Layer(Inputs, idx1, map1, val1, Act1, bias1, 0.05433153643371236);
process_Layer(Act1, idx2, map2, val2, Act2, bias2,0.039182770441338215);

func_out(Act2, idx3, val3,0.00631765928119421, bias3,output);

}

void process_Layer( act_t* inp, \
                    const idx_t* idx, \
                    const map_t* map, \
                    const weight_t* val, \
                    act_t* activations, \
                    const bias_t* bias, \
                    scale_t fused_all_scales)
{
ap_int<7> bits = 62;
ap_uint<64> bitmask = 0;
ap_uint<10> neuron = 0;
ap_uint<10> currentbase = 0;
ap_uint<16> index  = 0;
ap_uint<10> addr = 0;
ap_uint<6> num_params = 0;
ap_uint<5> ac = 0;
int accumi = 0;
actf_t accumf = 0.0;

for(int k = 0; k < LOOP_ITERATIONS; k++){
#pragma HLS pipeline II = 4

if(bits == SCAN_LENGTH){
    bitmask = map[neuron];
    }

if(num_params < 32 && bitmask[bits] == 0){
        addr = currentbase+idx[index];
        accumi += inp[addr]*val[index];
        num_params = num_params + 1;
        index=index+1;
}

    if(bitmask.test(bits)){
        currentbase += 32;
    }

    bits--;

    if(bits == -1){
        //Dequantize
        //accumf = (accumi * fused_input_weight_scale + bias[neuron] * bias_scale) * activation_scale; 
        //accumf = (accumi + bias[neuron]) * fused_input_weight_scale * activation_scale;
        accumf = (accumi + bias[neuron]) * fused_all_scales;        
        //apply Relu activation
        if (accumf < 0) {accumf = 0;}
        ac = (int)accumf;        
        if(ac > 15){ac = 15;} //clipping
        activations[neuron] = ac;
        bits = SCAN_LENGTH;
        neuron++;
        currentbase = 0;
        num_params = 0;
        accumi = 0;
    }

}
}


void func_out(act_t* inp, \
              const idxfull_t* idx, \
              const weight_t* val, \
              scale_t fused_input_weight_scale, \
              const bias_t* bias, \
              int* out )
    {
    float act = 0.0;
    int act_int = 0;
    float max = -100.0;
    int cnt = 0;
    int j = 0;
    int cls = 0;
    for(int i=0; i < 10240; i++){
        #pragma HLS pipeline II = 2
        act_int += inp[idx[i]]*val[i];
        cnt = cnt + 1;
        if(cnt == 1024){
            //Dequantize
            //act = act_int * fused_input_weight_scale + bias[j] * bias_scale;
            act = (act_int + bias[j]) * fused_input_weight_scale;            
            if(act > max){
                max = act;
                cls = j;                
                }
            j = j+1;
            cnt = 0;
            act_int = 0;
    }   
    }
    *out = cls;
}
