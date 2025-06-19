import json
import os


def json_model_to_python_array(bias_scales):

    fname = 'json_models_no_BN/model_weights_float.json'
    fname_quant = 'json_models_no_BN/model_weights_quant.json'

    #bias_scales = [0.006536283530294895,0.00631765928119421,0.008686156012117863]

    f = open(fname)
    f_quant = open(fname_quant)
    idx = []
    val = []
    idxptr = []
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    data_quant = json.load(f_quant)
    zz = 0
    tzz = 0 
    nnz = 0
    # Iterating through the json
    # list
    params = 0
    bias = 0

    if os.path.exists('tmp/csrmat.py'):
        os.remove('tmp/csrmat.py')
    if os.path.exists('tmp/output_layer.txt'):
        os.remove('tmp/output_layer.txt')

    for  key, value in data.items():
        if ("fc" in key and "bias" not in key):
            #print(key)
            params = params+1
            idx = []
            val = []
            idxptr = []
            vecnum = -1
            for vector in value:
                vecnum = vecnum + 1
                #print("vector #" + str(value.index(vector)))
                #scan the vector
                i = 0
                zz = 0
                vnum = -1
                for v in vector:
                    vnum = vnum + 1
                    if (v == 0):
                        zz += 1
                    else:
                        # add index, val to lists  
                        idx.append(zz+i)
                        #val.append(v)
                        #append quant value
                        val.append(data_quant[key][vecnum][vnum])
                        i = i+1  
                #print("zeros = " + str(zz))
                tzz += zz
                nnz = nnz + i
                idxptr.append(nnz)
            #write to file
            lst_idx = ','.join(str(x) for x in idx) 
            lst_val = ','.join(str(x) for x in val) 
            lst_ptr = ','.join(str(x) for x in idxptr) 
            with open("tmp/csrmat.py", "a") as f:
                f.write("idx" + str(params) + " = [" + lst_idx + "]")
                f.write('\n')
                f.write("val" + str(params) + " = [" + lst_val + "]")
                f.write('\n')
                #f.write("ptr" + str(params) + " = [" + lst_ptr + "]")
                #f.write('\n')          
            tzz = 0        
            nnz = 0
    # Closing file
    f.close()

    #write indices and weights of output layer
    with open("tmp/output_layer.txt", "a") as f:
        f.write("const idxfull_t idx" + str(params) + "[] = {" + lst_idx + "};")
        f.write('\n')
        f.write("const weight_t val" + str(params) + "[] = {" + lst_val + "};")
        f.write('\n')

        
    f1 = open(fname)
    data1 = json.load(f1)
    for  key, value in data1.items():
        if ("bias" in key):
            #print(key)
            bias = bias+1
            biases = []
            for vector in value:
                biases.append(int(vector/bias_scales[bias-1]))
            #write to file
            lst_biases = ','.join(str(x) for x in biases) 
            with open("tmp/csrmat.py", "a") as f:
                f.write("bias" + str(bias) + " = [" + lst_biases + "]")
                f.write('\n')
    # Closing file
    f1.close()