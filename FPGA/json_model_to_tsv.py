import json
import os

def json_model_to_tsv():
    f = open('json_models_no_BN/model_weights_float.json')
    fquant = open('json_models_no_BN/model_weights_quant.json')

    #os.remove('tmp/*.tsv')
    
    idx = []
    val = []
    idxptr = []
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    data_quant = json.load(fquant)
    zz = 0
    tzz = 0 
    nnz = 0
    k = 0
    # Iterating through the json
    # list
    for  key, value in data.items():
        if ("fc" in key and "bias" not in key):
            k = k+1
            #print(k, key)
            #delete old tsv file
            if os.path.exists('tmp/l' + str(k) +'.tsv'):
                os.remove('tmp/l' + str(k) +'.tsv')
            #create new tsv file
            with open('tmp/l' + str(k) +'.tsv', 'a') as fp:
                idx = []
                val = []
                idxptr = []
                vecnum = -1
                for vector in value:
                    #print("vector #" + str(value.index(vector)))
                    #scan the vector
                    i = 0
                    zz = 0
                    vecnum = vecnum + 1
                    dim = 0
                    vnum = -1
                    for v in vector:
                        vnum = vnum + 1
                        dim = dim + 1
                        if (v == 0):
                            zz += 1
                        else:
                            # add index, val to lists  
                            idx.append(zz+i)
                            val.append(v)
                            i = i+1 
                            #write line to tsv file
                            fp.write(str(dim) + '\t' + str(vecnum+1) + '\t' + str(data_quant[key][vecnum][vnum]))
                            fp.write('\n')
                    #print("zeros = " + str(zz))
                    tzz += zz
                    nnz = nnz + i
                    idxptr.append(nnz)
            fp.close()
            tzz = 0        
            nnz = 0
    # Closing file
    f.close()