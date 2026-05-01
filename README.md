<!-- #region -->
# Multi-Granularity Reverse Knowledge Distillation (MG-RKD)



## Overview
### Distillation framework
<p align="center">
  <br />
  <img src="Imgs/kuang.pdf" width="00">
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


### To run Classical Node Classification Task
To quickly train a teacher model you can run train_teacher.py by specifying the experiment setting

python train_teacher.py --exp_setting tran --teacher GCN --dataset cora


## Results

GraphSAGE vs. MLP vs. GLNN under the production setting described in the paper (transductive and inductive combined). Delta_MLP (Delta_GNN) represents difference between the GLNN and the MLP (GNN). Results show classification accuracy (higher is better); Delta_GNN > 0 indicates GLNN outperforms GNN. We observe that GLNNs always improve from MLPs by large margins and achieve competitive results as GNN on 6/7 datasets. Please see Table 3 in the paper for more details.  

| Datasets   | GNN(SAGE)      | MLP          | GLNN           | Delta_MLP       | Delta_GNN         |
|------------|----------------|--------------|----------------|-----------------|-------------------|
| Cora       | **79.29**      | 58.98        | 78.28          | 19.30 (32.72\%) | -1.01 (-1.28\%)   |
| Citseer    | 68.38          | 59.81        | **69.27**      | 9.46 (15.82\%)  | 0.89 (1.30\%)     |
| Pubmed     | **74.88**      | 66.80        | 74.71          | 7.91 (11.83\%)  | -0.17 (-0.22\%)   |
| A-computer | 82.14          | 67.38        | **82.29**      | 14.90 (22.12\%) | 0.15 (0.19\%)     |
| A-photo    | 91.08          | 79.25        | **92.38**      | 13.13 (16.57\%) | 1.30 (1.42\%)     |
| Arxiv      | **70.73**      | 55.30        | 65.09          | 9.79 (17.70\%)  | -5.64 (-7.97\%)   |
| Products   | **76.60**      | 63.72        | 75.77          | 12.05 (18.91\%) | -0.83 (-1.09\%)   |




<!-- #endregion -->
