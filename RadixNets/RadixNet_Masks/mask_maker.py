import csv
import random
import torch
import numpy as np
import itertools


class get_mask:

    def __init__(self,n_layers, neuron_len):
        self.n_layers = n_layers
        self.neuron_len = neuron_len
        #create standard mask of n_layers + output layer 
        standard_mask = {}
        for i in range(n_layers):
             standard_mask["fc"+str(i+1)] = torch.zeros(1024, 1024)
        standard_mask["fc"+str(i+2)] = torch.ones(10, 1024)
        self.standard_mask = standard_mask
    
    def __call__(self):

        for i in range(self.n_layers):
            idx_orig = []
            val = []
            idxptr = [0]
            neuron = []
            nnz = 0
            v = 0
            parametercount = 0
            neuroncount = 0
            linecount = 0
            with open('../RadixNets/RadixNet_Masks/tsvs/tsv_' + str(self.n_layers) + '/l' + str(i+1) + '.tsv') as tsv:
                for line in csv.reader(tsv, dialect="excel-tab"):
                    parametercount = parametercount+1
                    v = int(line[0])-1 #use indices from .tsv
                    #print(linecount)
                    self.standard_mask["fc" + str(i+1)][linecount][v] = 1 
                    #print(v)
                    idx_orig.append(v)
                    #val.append(f"{float(line[2]):3.3f}")
                    val.append(f"{random.random():4.4f}") #populate with unique weights. test data has all weights equal
                    neuron.append(int(line[1])-1)
                    nnz = nnz + 1
                    if(parametercount == self.neuron_len):
                            neuroncount = neuroncount+1
                            linecount = linecount+1
                            parametercount = 0
                    #print(line)

        return self.standard_mask
        
#np.savetxt('masks.txt', get_mask()["fc1"])
