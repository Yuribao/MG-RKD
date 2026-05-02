<!-- #region -->
# Multi-Granularity Reverse Knowledge Distillation (MG-RKD)



## Overview
### Distillation framework
<p align="center">
  <br />
  <img src="Imgs/kuang.pdf" width="800">
  <br />
</p>




## Getting Started

### Setup Environment

We use conda for environment setup. You can use

`bash ./prepare_env.sh`

which will create a conda environment named `glnn` and install relevant requirements (from `requirements.txt`).   For simplicity, we use CPU-based `torch` and `dgl` versions in this guide, as specified in requirements.  To run experiments with CUDA, please install `torch` and `dgl` with proper CUDA support, remove them from `requirements.txt`, and properly set the `--device` argument in the scripts. See https://pytorch.org/ and https://www.dgl.ai/pages/start.html for more installation details. 

Be sure to activate the environment with

`conda activate glnn`

before running experiments as described below.



### Preparing datasets
To run experiments for dataset used in the paper, please download from the following links and put them under `data/` (see below for instructions on organizing the datasets).

*CPF data* (`cora`, `citeseer`, `pubmed`, `a-computer`, and `a-photo`): Download the '.npz' files from [here](https://github.com/BUPT-GAMMA/CPF/tree/master/data/npz). Rename `amazon_electronics_computers.npz` and `amazon_electronics_photo.npz` to `a-computer.npz` and `a-photo.npz` respectively.

## Usage
### To run Classical Node Classification Task
To quickly train a teacher model you can run train_teacher.py by specifying the experiment setting
```
python train_teacher.py --exp_setting tran --teacher GCN --dataset cora
```

To quickly train a student model with a pretrained teacher you can run `train_student.py` by specifying the experiment setting, teacher model, student model, and dataset like the example below. Make sure you train the teacher using the train_teacher.py first and have its result stored in the correct path specified by `--out_t_path`.


### 





<!-- #endregion -->
