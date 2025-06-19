# fast-ffn
We present FAST_FFN, a highly efficient way of deploying large Feed Forward Networks(FFNs) on FPGAs. We detail our work in our paper, "Efficient Deployment of Very Wide and Very Deep Hypersparse FFNs on FPGA" which will be presented at ISVLSI 2025. If you make use of our code or data, please cite our paper.

```bibtex
@inproceedings{singh-SinghA2025,
   author    = {Paramdeep Singh and David C. Anastasiu},
   title     = {Efficient Deployment of Very Wide and Very Deep Hypersparse FFNs on FPGA},
   booktitle = {2025 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)},
   year      = {2025},
}
```
## Preliminaries

We configured and trained candidate FFNs in an Anaconda environment with Python 3.8.20. We also created a code generator to create the accelerator code for the trained FFNs. Our code generator uses High Level Synthesis (HLS) to generate the accelerator. The following will create an Anaconda environment and install the requisite packages for training the FFNs and generating the FPGA accelerators for them. 

```bash
conda create --name fastffn python=3.8
conda activate fastffn
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scikit-learn
conda install pandas
conda install matplotlib
conda install seaborn
pip install h5pickle

```
We completed the FPGA design flow (C-Simulation->C-Synthesis->C/RTL Cos-simulation->Implementation) using the accelerator code in the Vitis HLS (2024.2) environment.

## Files organization

This project is divided into 4 sub-modules.These modules are listed below in the order they need to be executed.

The 4 Modules:

1. "Radixnets"
2. "Training"
3. "Utils"
4. "FPGA"


1. The Radixnet module hosts pre-pruned FFN architectures. RadixNet pre-pruning is employed to generate these FFNs. Our code includes RadixNet architectures in the Radixnets/RadixNet_Masks folder. These FFNs are stored using a tab separated value (.tsv) format. We used the repository available at (https://github.com/Graphegon/pygraphblas/blob/main/demo/RadiX-Net-with-pygraphblas.ipynb) to generate RadixNet architectures. In addition to hosting RadixNet architectures, this module also serves as the repository for trained FFN models, and the code to generate the masks required to train such models.  

2. The Training module contains the training code for training the RadixNet pre-pruned FFNs. FFNs can be trained using the command shown below, where -m indicates the bitwidth of model weights and -L indicates the number of layers in the FFN.
   ```bash
   python Train_Radixnets.py -m 4 -L 2
   ```
The trained moodels are moved to Radixnets/RadixNet_Trained_Models folder upon training completion.
   
3. The Utils module allows extraction of scale factors from trained quantized models, which are required to implement the forward pass. Our work uses 4-bit quantization. We use the Brevitas library to train models with 4-bit weights. A reliable way to extract scale factors from Brevitas trained models is to export the trained model to the Open Neural Network Exchange (ONNX) format and read the scale factors from the ONNX representation of the quantized model. Trained models can be exported to ONNX format as shown below -L indicates the number of layers in the FFN.
  ```bash
   python onnx_export -L 2
   ```

5. Use "TestResult_Merging.ipynb" to combine the 3 parts in "./test" to generate the final prediction file.

## Inference mode

This can be easily achieved by hiding the train_loop() code line in the 3 Jupyter Notebooks, once the model has been trained. The other steps are the same as during training since we still need to follow the same steps to preprocess the original data and get the GMM Indicators. Fortunately, this step only needs to be run one time. After that, "generate_test()" can be executed repeately with different test timestamps.
