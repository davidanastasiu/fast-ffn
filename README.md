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
