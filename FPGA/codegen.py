#this program reads tsv files and converts them to CSR format. restricts the idx value to 0-32. uses pre calculated bases to compute actual idx value
import csv
import random
import itertools
import json
from pathlib import Path
import os

def pad_binary(b):
     #return bin(int(b,2)<<2)
     return   str(b) + "".join(itertools.repeat("0", (67-len(str(b)))))  #bring up the total chars to 67 . We need this to maintain a bitfield of exactly 64 bits.

def verify_bitmap(idx_orig, idx, bitmap_lst):
    cnt = 0
    
    for i in range(len(bitmap_lst)):
         print("Processing neuron", i)
         j = 0
         currentbase = 0
         bitmap_lst[i] = bitmap_lst[i][3:]  #take out '0b1' prefix
         #print(bitmap_lst[i], pad_binary(bitmap_lst[i]) )
         parameter_count = 0
         for j in range (len(bitmap_lst[i])):
             if (parameter_count == 32):
                 continue
             #print(bitmap_lst[i][j])
             if (bitmap_lst[i][j] == '0'):
                 print (idx_orig[cnt], int(idx[cnt])+currentbase)
                 cnt = cnt + 1
                 parameter_count = parameter_count+1
             if (bitmap_lst[i][j] == '1'):  
                 currentbase = currentbase + 32  

def verify_intmap(idx_orig, idx, bitmap_lst):
    cnt = 0 
    for i in range(len(bitmap_lst)):
         print("Processing neuron", i)
         j = 0
         currentbase = 0
         #print(bitmap_lst[i])
         bitmap_lst[i] = bin(int(bitmap_lst[i]))[3:] + '0' #take out '0b1' prefix. (ignore the leading '1')
         #print(bitmap_lst[i])
         parameter_count = 0
         for j in range (len(bitmap_lst[i])):
             if (parameter_count == 32):
                 continue
             #print(bitmap_lst[i][j])
             if (bitmap_lst[i][j] == '0'):
                 print (idx_orig[cnt], int(idx[cnt])+currentbase)
                 cnt = cnt + 1
                 parameter_count = parameter_count + 1
             if (bitmap_lst[i][j] == '1'):  
                 currentbase = currentbase + 32


def gen_address(idx_orig, idx, intmap_lst):
    bits = 0
    bitmask = ''
    neuron = 0
    currentbase = 0
    index = 0
    addr = 0
    num_params = 0
    #print (len(intmap_lst))
    for i in range(len(intmap_lst)*64): 

        if(bits == 0):
            bitmask = bin(intmap_lst[neuron])[3:] + '0' # skip the leading '0b1'. Replace the last zero dropped earlier.
            #print(neuron, len(bitmask))

        if(bitmask[bits] == '0' and num_params < 32):
            addr = currentbase + idx[index]
            print(idx_orig[index], addr)
            index = index + 1
            num_params = num_params + 1
        if(bitmask[bits] == '1' and num_params < 32):
            currentbase = currentbase + 32

        bits = bits + 1

        if (bits == 64 ):
            print("Processed Neuron", neuron)
            print("i = ", i)
            bits = 0  
            neuron = neuron + 1 
            currentbase = 0
            num_params = 0            
                 

def get_submap(v, currentbase):
    submap = ''
    while(v-currentbase >= 32):
        submap = submap + '1'
        currentbase = currentbase + 32
    return submap , currentbase
     
def gen_hls_code(num_layers,input_scales, weight_scales, bias_scales, activation_scales,neuron_len = 32):

    #num_layers = 2
    #neuron_len = 32

    subdir = 'HLS'

    hfile = os.path.join(os.pardir, subdir, 'weights.h')
    cfile = os.path.join(os.pardir, subdir, 'MLP.cpp')

    if os.path.exists(hfile):
        os.remove(hfile)
    if os.path.exists(cfile):
        os.remove(cfile)

    PEfile = 'PE.txt'
    output_layer_file = 'tmp/output_layer.txt'

    with open(hfile, "a") as f:
        f.write("#include \"mlp.h\" ")
        f.write("\n\n")


    for i in range(1, num_layers+1):
        idx_orig = []
        idx = []
        val = []
        idxptr = [0]
        neuron = []
        nnz = 0
        v = 0
        bitmap='0b1'
        currentbase = 0
        submap=''
        parametercount = 0
        neuroncount = 0
        bitmap_lst = []
        intmap_lst = []
        indices = []
        subindices = []

        #build an array of random indices
        for n in range(1024):
            for p in range(32):
                subindices.append(random.randint(0,1023))
            subindices.sort()
            indices.extend(subindices)
            subindices = []
        linecount = 0

        with open('tmp/l' + str(i) + '.tsv') as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                parametercount = parametercount+1
                v = int(line[0])-1 #use indices from .tsv
                #v = indices[linecount] # use random indices
                linecount = linecount+1
                submap, currentbase = get_submap(v, currentbase)
                bitmap=bitmap + submap + '0'
                #print(v)
                idx_orig.append(v)
                idx.append(v-currentbase)
                val.append(line[2])
                #val.append(f"{float(line[2]):3.3f}")
                #val.append(f"{random.random():4.4f}") #populate with unique weights. test data has all weights equal
                neuron.append(int(line[1])-1)
                nnz = nnz + 1
                if(parametercount == 32):
                    neuroncount = neuroncount+1
                    parametercount = 0
                    bitmap = pad_binary(bitmap)
                    #print(neuroncount, bitmap)
                    bitmap_lst.append(bitmap)
                    intmap_lst.append(int(bitmap[0:-1],2)) #ignore last digit from bitmap. will replace later. will always be zero
                    #print(len(bin(int(bitmap[0:-1],2))))
                    bitmap = '0b1'
                    currentbase = 0
                #print(line)
            #write arrays
            lst_idx = ','.join(str(x) for x in idx) 
            lst_val = ','.join(str(x) for x in val) 
            intmap_str = ','.join(str(x) for x in intmap_lst)
            #lst_neuron = ','.join(str(x) for x in neuron) 
            
            
            with open(hfile, "a") as f:
                    print("Layer = " + str(i) )
                    f.write("const idx_t idx" + str(i) + "[] = {" + lst_idx + "};")
                    f.write('\n')
                    print("idx count = " + str(len(idx)) )
                    f.write("const map_t map" + str(i) + "[] = {" + intmap_str + "};")
                    f.write('\n')
                    print("map count = " + str(len(intmap_lst)) )
                    f.write("const weight_t val" + str(i) + "[] = {" + lst_val + "};")
                    f.write('\n')
            
            #verify
            #verify_bitmap(idx_orig, idx, bitmap_lst)
            #verify_intmap(idx_orig, idx, intmap_lst)
            #gen_address(idx_orig, idx, intmap_lst)

    #######################################################################
    ####                     Write Biases to header file               ####
    #######################################################################

    fname = 'json_models_no_BN/model_weights_float.json'
    #bias_scales = [0.006536283530294895,0.00631765928119421,0.008686156012117863]
    f1 = open(fname)
    data1 = json.load(f1)
    bias = 0
    for  key, value in data1.items():
        if ("bias" in key):
            #print(key)
            bias = bias+1
            biases = []
            for vector in value:
                biases.append(int(vector/bias_scales[bias-1]))
            #write to file
            lst_biases = ','.join(str(x) for x in biases) 
            with open(hfile, "a") as f:
                f.write("const bias_t bias" + str(bias) + "[] = {" + lst_biases + "};")
                f.write('\n')
    # Closing file
    f1.close()

    #write weights of output layer 
    with open(output_layer_file, 'r') as input_file, open(hfile, 'a') as output_file:
        content = input_file.read()
        output_file.write(content)


    #######################################################################################
    ####                                Write Accelerator Code                         ####
    #######################################################################################

    #write file preamble

    preamble = """
    #include "mlp.h"
    #include "weights.h"

    void mlp_sparse(
            act_t Inputs[NUM_INPUTS],
            int *output
            )

    {

    #pragma HLS INTERFACE mode=ap_ctrl_chain port=return

    """

    with open(cfile, "a") as f:
        f.write(preamble)
        f.write('\n\n')

    #write channels
    for channels in range(num_layers):
        with open(cfile, "a") as f:
            #weight_t Act1[NEURONS_PER_LAYER];
            f.write( "act_t Act" + str(channels+1) + "[NEURONS_PER_LAYER];")
            f.write('\n')
            f.write( "#pragma HLS stream type=pipo variable=Act" + str(channels+1) + " depth=2")
            f.write('\n\n')
            

    #write pragmas
    for pragma in range(num_layers):
        with open(cfile, "a") as f:
            #pragma HLS bind_storage variable=val1 type=rom_1p impl=BRAM
            f.write( "#pragma HLS bind_storage variable=val" + str(pragma+1) + " type=rom_1p impl=BRAM")
            f.write('\n')
            f.write( "#pragma HLS bind_storage variable=idx" + str(pragma+1) + " type=rom_1p impl=BRAM")
            f.write('\n')
            f.write( "#pragma HLS bind_storage variable=map" + str(pragma+1) + " type=rom_1p impl=BRAM")
            f.write('\n')
            f.write( "#pragma HLS bind_storage variable=bias" + str(pragma+1) + " type=rom_1p impl=LUTRAM")
            f.write('\n')
            f.write('\n')

    #pragmas for output layer

    with open(cfile, "a") as f:
        #pragma HLS bind_storage variable=val1 type=rom_1p impl=BRAM
        f.write( "#pragma HLS bind_storage variable=val" + str(pragma+2) + " type=rom_1p impl=BRAM")
        f.write('\n')
        f.write( "#pragma HLS bind_storage variable=idx" + str(pragma+2) + " type=rom_1p impl=BRAM")
        f.write('\n')
        f.write( "#pragma HLS bind_storage variable=bias" + str(pragma+2) + " type=rom_1p impl=LUTRAM")
        f.write('\n')
        f.write('\n')

    #write function calls

    #input layer
    with open(cfile, "a") as f:
        f.write("#pragma HLS dataflow")
        f.write("\n\n")
        fused_scales = float(bias_scales[0]) * 1/float(activation_scales[0])
        #process_layer1(inp, idx1, map1, val1, Act1, bias1, fused_scales);
        f.write("process_Layer(Inputs, idx1, map1, val1, Act1, bias1, " + str(fused_scales) + ");")
        f.write('\n')

    #hidden layers
    for funcs in range(1,num_layers):
        with open(cfile, "a") as f:
            fused_scales = float(bias_scales[funcs]) * 1/float(activation_scales[funcs])
            #process_layer1(Act1, idx2, map2, val2, Act2, bias2, fused_scales);
            f.write( "process_Layer(Act" + str(funcs) + ", idx" + str(funcs+1) + ", map" + str(funcs+1) + ", val" + str(funcs+1) + ", Act" + str(funcs+1) + ", bias" + str(funcs+1) + "," + str(fused_scales) + ");" )
            f.write('\n\n')

    #output layer
    with open(cfile, "a") as f:
        fused_input_weight_scales = float(bias_scales[funcs])
        #func_out(Act2,idx3,map3,val3,fused_input_weight_scales,bias3,output)
        f.write( "func_out(Act" + str(funcs+1) + ", idx" + str(funcs+2) +  ", val" + str(funcs+2) + "," + str(fused_input_weight_scales) + ", bias" + str(funcs+2) + ",output);" )
        f.write('\n\n')

    with open(cfile, "a") as f:
        f.write('}\n\n')

    #write function definitions
    with open(PEfile, 'r') as input_file, open(cfile, 'a') as output_file:
        content = input_file.read()
        output_file.write(content)