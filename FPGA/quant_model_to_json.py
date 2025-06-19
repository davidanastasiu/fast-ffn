import torch
import os
import sys
import numpy as np
import json
import json
import importlib


def quant_model_to_json(layers, bit_width, test, config):
    
    device = "cpu"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(os.path.abspath(parent_dir))

    parent_dir = os.path.dirname(parent_dir)
    sys.path.append(os.path.abspath(parent_dir))

    from RadixNets.RadixNet_Masks.mask_maker import get_mask
    from Training.mnist_dataset import MnistDataset
    from Training.tools.parse_yaml_config import parse_config

    mask = get_mask(layers,32, '../../RadixNets/RadixNet_Masks/tsvs/tsv_')
    standard_mask = mask()
    
    #import the specified model
    module_name = "models_mlp_" + str(layers) + "_RadixNet.py"
    directory_path = parent_dir + '/RadixNets/RadixNet_Models/' + module_name  
    spec = importlib.util.spec_from_file_location("module.name", directory_path)
    foo = importlib.util.module_from_spec(spec)
    print("loaded model", foo)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    model = foo.model_bv_masked(standard_mask, bit_width)
    model.load_state_dict(torch.load('../../RadixNets/RadixNet_Trained_Models/' + str(layers) +'L/models/4b/4b_0pruned.pth', map_location=torch.device('cpu')))
    state_dict = model.state_dict()
    print("loaded state dict to model")
    model.eval()

    json_serializable_dict = {k: v.tolist() for k, v in state_dict.items()}

    json_file = {}

    json_file["quant_inp.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"] = json_serializable_dict["quant_inp.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]
    
    for i in range(1, int(layers)+2):
        layername = 'fc' + str(i)
        for name, module in model.named_modules():
            if (name == layername):
                #print(name) 
                W = module.quant_weight().int().cpu().numpy().astype(np.int64)
                param_list = W.tolist()
                json_file[layername + ".weight"] = param_list
    
    for i in range(1, int(layers)+1):
        json_file["act" + str(i) + ".act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"] = json_serializable_dict["act" + str(i) + ".act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]
   

    with open('json_models_no_BN/model_weights_float.json', 'w') as f:
        json.dump(json_serializable_dict, f)


    with open('json_models_no_BN/model_weights_quant.json', 'w') as f:
        json.dump(json_file, f)


