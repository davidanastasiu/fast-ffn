import torch
import pandas as pd
from torch.utils.data import Dataset
import h5pickle as h5py
import io
import os
import numpy as np
import gzip
import pickle
import pandas as pd
from sklearn import preprocessing

class MnistDataset(Dataset):
    """MNIST 32X32 dataset"""

    def __init__(self, dataPath, yamlConfig, normalize=True, filenames=None):
        data_path = dataPath 
        nb_classes = 10
        self.labels_list = ['0','1','2','3','4','5','6','7','8','9']
        if os.path.isfile(data_path):
            print("Single data file found!")
            print(data_path)
            #load file to np array
            xy = np.loadtxt(str(data_path), delimiter=",", dtype=int)
            #print(xy.shape)
            features_arr = xy[:, 1:]
            features_arr_reshaped = features_arr.reshape(features_arr.shape[0], 28,28)
            # 32X32 images 
            features_arr_padded = np.pad(features_arr_reshaped, ((0,0),(2,2),(2,2)), 'constant')
            features_arr_input = features_arr_padded.reshape(features_arr_padded.shape[0],features_arr_padded.shape[1]*features_arr_padded.shape[2] )
            print(features_arr_input.shape)
            features_arr_normalized = features_arr_input/(np.max(features_arr_input)-np.min(features_arr_input))
            labels_arr = xy[:, 0]
            one_hot_targets = np.zeros((labels_arr.size, labels_arr.max() + 1))
            one_hot_targets[np.arange(labels_arr.size), labels_arr] = 1
            #print(features_arr_input.shape)
            #print(labels_arr.shape)
            self.x = torch.from_numpy(features_arr_normalized)
            self.y = torch.from_numpy(one_hot_targets)
            self.n_samples = self.x.shape[0]

        else:
            print("Error! path specified is a special file (socket, FIFO, device file), or isn't valid")
            print("Given Path: {}".format(data_path))
        
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        #return self.features_val[index], self.labels_val[index]

    def __len__(self):
        return self.n_samples
        #return len(self.features_val)

    def close(self):
        return
        #self.h5File.close()