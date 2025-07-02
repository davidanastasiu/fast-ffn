import codegen
import quant_model_to_json
import json_model_to_python_array
import json_model_to_tsv
import extract_test_data
from optparse import OptionParser
import csv


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-L','--layers', action='store', type='int', dest='layers', default=2, help='Number of layers')
    parser.add_option('-w','--width' ,action='store',type='string',dest='width',default=4, help='bit width')
    parser.add_option('-t','--test'   ,action='store',type='string',dest='test'   ,default='../../Datasets/mnist_data/test/mnist_test.csv', help='Location of test data set')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='../../Training/configs/train_config_threelayer.yml', help='location of config file')
   
    (options,args) = parser.parse_args()

    #read in scale factors
    with open('scales.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index == 0:
                input_scales = [float(x) for x in row]
                #print("input_scales", input_scales)
            if index == 1:
                weight_scales = [float(x) for x in row]
                #print("weight_scales", weight_scales)
            if index == 2:
                bias_scales = [float(x) for x in row]
                #print("bias_scales", bias_scales)
            if index == 3:
                activation_scales = [float(x) for x in row]
                #print("activation_scales", activation_scales)
          
    #bias_scales = [0.006536283530294895,0.00631765928119421,0.008686156012117863]

    print("Extracting quant model paramteres in json format...")
    quant_model_to_json.quant_model_to_json(options.layers,options.width,options.test,options.config)
    print("Done. \n")

    print("Transforming to python arrays...")
    json_model_to_python_array.json_model_to_python_array(bias_scales)
    print("Done. \n")

    print("Transforming to tsv format...")
    json_model_to_tsv.json_model_to_tsv()
    print("Done. \n")

    print("Generating accelerator code...")
    codegen.gen_hls_code(options.layers,input_scales,weight_scales,bias_scales,activation_scales)
    print("Done. \n")

    print("Extracting test vectors for accelerator.")
    extract_test_data.create_test_vectors(options.config, options.test,float(input_scales[0]))
    print("Done. \n")
