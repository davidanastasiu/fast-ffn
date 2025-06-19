import netron
import time
from IPython.display import IFrame
import brevitas.nn as qnn
import torch
from brevitas.export import export_onnx_qcdq
import sys
import os
import importlib
from optparse import OptionParser


IN_CH = 1024
OUT_CH = 128
BATCH_SIZE = 1

# set seed
torch.manual_seed(0)

# helpers
def assert_with_message(condition):
    assert condition
    print(condition)

def show_netron(model_path, port):
    time.sleep(3.)
    netron.start(model_path, address=("localhost", port), browse=False)
    return IFrame(src=f"http://localhost:{port}/", width="100%", height=400)


'''
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = qnn.QuantLinear(IN_CH, OUT_CH, bias=True, weight_bit_width=3)
        self.act = qnn.QuantReLU(bit_width=4)

    def forward(self, inp):
        inp = self.linear(inp)
        inp = self.act(inp)
        return inp
'''

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-L','--layers', action='store', type='int', dest='layers', default=2, help='Number of layers')
    parser.add_option('-w','--width'   ,action='store',type='string',dest='width'   ,default=4, help='bit width')
    parser.add_option('-t','--test'   ,action='store',type='string',dest='test'   ,default='', help='Location of test data set')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='../../Training/configs/train_config_threelayer.yml', help='tree name')

    (options,args) = parser.parse_args()

    device = "cpu"
    inp = torch.randn(BATCH_SIZE, IN_CH)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(os.path.abspath(parent_dir))

    from RadixNets.RadixNet_Masks.mask_maker import get_mask

    mask = get_mask(options.layers,32, '../RadixNets/RadixNet_Masks/tsvs/tsv_')
    standard_mask = mask()

    #import the specified model
    module_name = "models_mlp_" + str(options.layers) + "_RadixNet.py"
    directory_path = parent_dir + '/RadixNets/RadixNet_Models/' + module_name  
    spec = importlib.util.spec_from_file_location("module.name", directory_path)
    foo = importlib.util.module_from_spec(spec)
    print(foo)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    model = foo.model_bv_masked(standard_mask, options.width)

    model.load_state_dict(torch.load('../RadixNets/RadixNet_Trained_Models/' + str(str(options.layers)) + 'L/models/4b/4b_0pruned.pth'))
    model.to(device)
    model.eval()


    #inp = torch.randn(BATCH_SIZE, IN_CH)

    path = 'onnx_model.onnx'

    exported_model = export_onnx_qcdq(model, args=inp, export_path=path, opset_version=13)

    show_netron(path, 8082)