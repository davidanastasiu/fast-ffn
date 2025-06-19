import torch
#import numpy as np
#from torch.utils.data import Dataset
import sys
import os

def create_test_vectors(config,test_dataset,input_scale):

    device = "cpu"

    if os.path.exists("../HLS/X_test.dat"):
        os.remove("../HLS/X_test.dat")
    if os.path.exists("../HLS/Y_test.dat"):
        os.remove("../HLS/Y_test.dat")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(os.path.abspath(parent_dir)) 
    parent_dir = os.path.dirname(parent_dir)
    sys.path.append(os.path.abspath(parent_dir))

    from Datasets.mnist_dataset import MnistDataset
    from Training.tools.parse_yaml_config import parse_config

    yamlConfig = parse_config(config)

    test_dataset = MnistDataset(test_dataset, yamlConfig)

    test_size = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size, shuffle=False, pin_memory=False)

    for i, data in enumerate(test_loader):
        #print ("i =", i)
        local_batch, local_labels = data
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    #QuantIdentity
    local_batch = local_batch / input_scale 
    local_batch = local_batch.int()

    #print(local_batch[0])
    #print(local_labels[0])

    for i in range(test_size):
        with open("../HLS/X_test.dat", "a") as f:
            row_str = " ".join(map(str, local_batch[i].tolist()))
            f.write(row_str + "\n")
            
            
    for i in range(test_size):
        with open("../HLS/Y_test.dat", "a") as g:
            row_str = " ".join(map(str, local_labels[i].tolist()))
            g.write(row_str + "\n")

    print("{} test vectors created".format(test_size))