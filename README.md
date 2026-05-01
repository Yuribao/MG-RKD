<!-- #region -->
# Multi-Granularity Reverse Knowledge Distillation (MG-RKD)



## Overview
### Distillation framework
<p align="center">
  <br />
  <img src="imgs/glnn.png" width="800">
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

- *CPF data* (`cora`, `citeseer`, `pubmed`, `a-computer`, and `a-photo`): Download the '.npz' files from [here](https://github.com/BUPT-GAMMA/CPF/tree/master/data/npz). Rename `amazon_electronics_computers.npz` and `amazon_electronics_photo.npz` to `a-computer.npz` and `a-photo.npz` respectively.

- *OGB data* (`ogbn-arxiv` and `ogbn-products`): Datasets will be automatically downloaded when running the `load_data` function in `dataloader.py`. More details [here](https://ogb.stanford.edu/).

- *BGNN data* (`house_class` and `vk_class`): Follow the instructions [here](https://github.com/dmlc/dgl/tree/473d5e0a4c4e4735f1c9dc9d783e0374328cca9a/examples/pytorch/bgnn) and download dataset pre-processed in DGL format from [here](https://www.dropbox.com/s/verx1evkykzli88/datasets.zip).

- *NonHom data* (`penn94` and `pokec`): Follow the instructions [here](https://github.com/CUAI/Non-Homophily-Benchmarks) to download the `penn94` dataset and its splits. The `pokec` dataset will be automatically downloaded when running the `load_data` function in `dataloader.py`.

- Your favourite datasets: download and add to the `load_data` function in `dataloader.py`.


### Usage

To quickly train a teacher model you can run `train_teacher.py` by specifying the experiment setting, i.e. transductive (`tran`) or inductive (`ind`), teacher model, e.g. `GCN`, and dataset, e.g. `cora`, as per the example below.

```
python mlp_weight_matching.py --model_a v8 --model_b v10 --load_epoch 70 --hidden_dim 128 --dataset cora
python mlp_weight_matching.py --model_a v8 --model_b v8 --load_epoch 65
python mlp_weight_matching.py --model_a v5 --model_b v6 --load_epoch 93 

python GCN_weight_matching.py --model_a v8 --model_b v9 --load_epoch 78 
python GCN_weight_matching.py --model_a v3 --model_b v5 --load_epoch 48

融合模型知识蒸馏
#cora
python GCN_MLP_weight_matching.py --model_a v1 --model_b v2 --teacher GCN --num_layers 3 --load_epoch 46 --seed 0 --hidden_dim 128 --dataset cora
python GCN_MLP_weight_matching.py --model_a v1 --model_b v2 --teacher GCN --num_layers 3 --load_epoch 46 --seed 0 --hidden_dim 128 --dataset citeseer
python GCN_MLP_weight_matching.py --model_a v4 --model_b v6 --teacher GCN --num_layers 5 --load_epoch 108 --seed 2 --hidden_dim 128 --dataset cora
python GCN_MLP_weight_matching.py --model_a v7 --model_b v8 --teacher GCN --num_layers 2 --load_epoch 8 --seed 1 --hidden_dim 128 --dataset cora
python train_student.py --exp_setting tran --teacher GCN --student GCN --dataset cora --num_layers 2 --hidden_dim 128 --temperature 7.0 --out_t_path outputs
#cora
python train_student.py --exp_setting tran --teacher GCN --student GCN --teacher_num_layers 3 --dataset cora  --out_t_path outputs
python train_student.py --exp_setting tran --teacher GCN --student GCN --teacher_num_layers 5 --dataset cora  --out_t_path outputs 
python train_student.py --exp_setting tran --teacher GCN --student GCN --teacher_num_layers 2 --dataset cora  --out_t_path outputs

#cora
python train_teacher.py --exp_setting tran --teacher GCN --num_layers 5 --hidden_dim 128 --seed 0 --dropout_ratio 0.3 --dataset cora  
python train_teacher.py --exp_setting tran --teacher MLP --num_layers 5 --hidden_dim 128 --seed 0 --dropout_ratio 0.3 --dataset cora 
python GCN_MLP_weight_matching.py --model_a v16 --model_b v17 --teacher GCN --num_layers 5 --load_epoch 104 --seed 0 --hidden_dim 128 --dataset cora
python GCN_MLP_weight_matching.py --model_a v7 --model_b v8 --teacher GCN --num_layers 2 --load_epoch 8 --seed 1 --hidden_dim 128 --dataset cora
python GCN_MLP_weight_matching.py --model_a v1 --model_b v2 --teacher GCN --num_layers 3 --load_epoch 46 --seed 0 --hidden_dim 128 --dataset cora

知识蒸馏
python train_student.py --exp_setting tran --teacher GCN --student GCN --teacher_num_layers 3 --dataset cora  --out_t_path outputs


#citeseer
python train_teacher.py --exp_setting tran --teacher MLP --num_layers 2 --hidden_dim 128 --seed 0 --dropout_ratio 0 --dataset citeseer  
python GCN_MLP_weight_matching.py --model_a v10 --model_b v11 --teacher GCN --num_layers 2 --load_epoch 32 --seed 0 --hidden_dim 128 --dataset citeseer
python GCN_weight_matching.py --model_a v0 --model_b v1 --load_epoch 39
知识蒸馏
python train_student.py --exp_setting tran --teacher GCN --student GCN --teacher_num_layers 2 --dataset citeseer  --out_t_path outputs

#pubmed
python train_teacher.py --exp_setting tran --teacher GCN --dataset pubmed
python train_teacher.py --exp_setting tran --teacher MLP --num_layers 2 --hidden_dim 1024 --lamb 0.50 --seed 2 --dropout_ratio 0.5 --dataset pubmed  
python GCN_MLP_weight_matching.py --model_a v16 --model_b v17 --teacher GCN --num_layers 2 --load_epoch 13 --seed 2 --hidden_dim 1024 --dataset pubmed
python GCN_weight_matching.py --model_a v0 --model_b v1 --load_epoch 39
知识融合
python train_student.py --exp_setting tran --teacher GCN --student GCN --teacher_num_layers 2 --dataset pubmed --out_t_path outputs

#actor
python train_teacher.py --exp_setting tran --teacher GCN --dataset actor --ratio_train 0.6 --ratio_val 0.2 --num_layers 3 --hidden_dim 64 --seed 0 --dropout_ratio 0.3
python train_teacher.py --exp_setting tran --teacher MLP --dataset actor --ratio_train 0.6 --ratio_val 0.2 --num_layers 3 --hidden_dim 64 --seed 0 --dropout_ratio 0.5
python GCN_MLP_weight_matching.py --model_a v5 --model_b v6 --teacher GCN --num_layers 3 --load_epoch 138 --seed 0 --hidden_dim 64 --dataset actor

python GCN_MLP_weight_matching.py --model_a v5 --model_b v6 --teacher MLP --num_layers 3 --load_epoch 138 --seed 0 --hidden_dim 64 --dataset  actor
知识融合
python train_student.py --exp_setting tran --teacher MLP --student GCN --teacher_num_layers 3 --dataset actor --out_t_path outputs


#cornell
python train_teacher.py --exp_setting tran --teacher GCN --dataset cornell --ratio_train 0.6 --ratio_val 0.2 --num_layers 2 --hidden_dim 256 --seed 0 --dropout_ratio 0.5
python train_teacher.py --exp_setting tran --teacher MLP --dataset cornell --ratio_train 0.6 --ratio_val 0.2 --num_layers 2 --hidden_dim 256 --seed 2 --dropout_ratio 0.5
python GCN_MLP_weight_matching.py --model_a v7 --model_b v8 --teacher GCN --num_layers 2 --load_epoch 188 --seed 2 --hidden_dim 256 --dataset cornell

python GCN_MLP_weight_matching.py --model_a v7 --model_b v8 --teacher MLP --num_layers 2 --load_epoch 188 --seed 1 --hidden_dim 256 --dataset  cornell
知识融合
python train_student.py --exp_setting tran --teacher MLP --student GCN --teacher_num_layers 2 --dataset cornell --out_t_path outputs


#texas
python train_teacher.py --exp_setting tran --teacher GCN --dataset texas --ratio_train 0.6 --ratio_val 0.2 --num_layers 3 --hidden_dim 1024 --seed 1 --dropout_ratio 0.3
python train_teacher.py --exp_setting tran --teacher MLP --dataset texas --ratio_train 0.6 --ratio_val 0.2 --num_layers 3 --hidden_dim 1024 --seed 1 --dropout_ratio 0.5
python GCN_MLP_weight_matching.py --model_a v10 --model_b v11 --teacher GCN --num_layers 3 --load_epoch 29 --seed 1 --hidden_dim 1024 --dataset texas

python GCN_MLP_weight_matching.py --model_a v10 --model_b v11 --teacher MLP --num_layers 3 --load_epoch 30 --seed 1 --hidden_dim 1024 --dataset  texas

知识融合
python train_student.py --exp_setting tran --teacher MLP --student GCN --teacher_num_layers 3 --dataset texas --out_t_path outputs


#wisconsin
python train_teacher.py --exp_setting tran --teacher GCN --dataset wisconsin --ratio_train 0.6 --ratio_val 0.2 --num_layers 2 --hidden_dim 1024 --seed 0 --dropout_ratio 0.3
python train_teacher.py --exp_setting tran --teacher MLP --dataset wisconsin --ratio_train 0.6 --ratio_val 0.2 --num_layers 2 --hidden_dim 1024 --seed 0 --dropout_ratio 0.3
python GCN_MLP_weight_matching.py --model_a v14 --model_b v15 --teacher GCN --num_layers 2 --load_epoch 92 --seed 0 --hidden_dim 1024 --dataset wisconsin

python GCN_MLP_weight_matching.py --model_a v14 --model_b v15 --teacher MLP --num_layers 2 --load_epoch 92 --seed 0 --hidden_dim 1024 --dataset wisconsin

知识融合
python train_student.py --exp_setting tran --teacher MLP --student GCN --teacher_num_layers 2 --dataset wisconsin --out_t_path outputs



cora：
python train_teacher.py --exp_setting tran --teacher MLP --num_layers 2 --hidden_dim 128 --seed 0 --dropout_ratio 0 --dataset citeseer  
python train_teacher.py --exp_setting tran --teacher GCN --num_layers 2 --hidden_dim 128 --dropout_ratio 0 --dataset cora
python train_teacher.py --exp_setting tran --teacher MLP --num_layers 2 --hidden_dim 128 --dropout_ratio 0 --dataset cora  

python train_teacher.py --exp_setting tran --teacher SAGE --num_layers 3 --hidden_dim 1024 --dropout_ratio 0.3 --dataset cora 
python train_teacher.py --exp_setting tran --teacher GAT --num_layers 2 --hidden_dim 128 --dropout_ratio 0 --dataset cora

citeseer：
python train_teacher.py --exp_setting tran --teacher SAGE --num_layers 2 --hidden_dim 128 --dropout_ratio 0.8 --dataset citeseer 
python train_teacher.py --exp_setting tran --teacher SAGE --num_layers 5 --hidden_dim 128 --dropout_ratio 0.8 --dataset citeseer  
python train_teacher.py --exp_setting tran --teacher GCN_GCN --num_layers 10 --hidden_dim 256 --dropout_ratio 0.8 --dataset cora


```
多教师融合：
python train_teacher.py --exp_setting tran --num_teacher 2 --teacher GCN GAT --num_layers 3 --hidden_dim 128 --dropout_ratio 0.8 --dataset cora
python train_teacher.py --exp_setting tran --num_teacher 2 --teacher GCN SAGE --num_layers 5 --hidden_dim 128 --dropout_ratio 0 --dataset cora

python train_student.py --exp_setting tran --teacher GCN_SAGE --student MLP --dataset cora --num_layers 8 --hidden_dim 128 --dropout_ratio 0 --temperature 10.0 --out_t_path outputs  

To quickly train a student model with a pretrained teacher you can run `train_student.py` by specifying the experiment setting, teacher model, student model, and dataset like the example below. Make sure you train the teacher using the `train_teacher.py` first and have its result stored in the correct path specified by `--out_t_path`.

```
cora:
python train_student.py --exp_setting tran --teacher GCN_GCN --student GCN --dataset cora --num_layers 20 --hidden_dim 64 --temperature 10.0 --out_t_path outputs  
python train_student.py --exp_setting tran --teacher SAGE --student MLP --dataset cora --num_layers 3 --hidden_dim 32 --dropout_ratio 0 --temperature 10.0 --out_t_path outputs 	
python train_student.py --exp_setting tran --teacher GCN --student MLP --dataset cora --num_layers 2 --hidden_dim 128 --dropout_ratio 0 --temperature 5.0 --out_t_path outputs   

citeseer:
python train_student.py --exp_setting tran --teacher SAGE --student MLP --dataset citeseer --num_layers 3 --hidden_dim 128 --temperature 2.0 --out_t_path outputs  

```
python train_student.py --exp_setting tran --teacher GCN_GCN --student GCN --dataset cora --num_layers 20 --hidden_dim 128 --temperature 10.0 --out_t_path outputs 

python train_student.py --exp_setting tran --teacher SGC --student SGC --dataset cora --num_layers 9 --hidden_dim 256 --temperature 2.0 --out_t_path outputs 
python train_student.py --exp_setting tran --teacher SAGE --student SAGE --dataset cora --num_layers 3 --hidden_dim 128 --temperature 5.0 --out_t_path outputs

For more examples, and to reproduce results in the paper, please refer to scripts in `experiments/` as below.

```
bash experiments/sage_cpf.sh
```

To extend GLNN to your own model, you may do one of the following.
- Add your favourite model architectures to the `Model` class in `model.py`. Then follow the examples above.
- Train teacher model and store its output (log-probabilities). Then train the student by `train_student.py` with the correct `--out_t_path`.


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

<!-- 
| Datasets   | Eval | SAGE           | MLP/MLP+     | GLNN/GLNN+     | $\Delta_{MLP}$  | $\Delta_{GNN}$    |
|------------|------|----------------|--------------|----------------|-----------------|-------------------|
| Cora       | prod | **79.29**      | 58.98        | 78.28          | 19.30 (32.72\%) | -1.01 (-1.28\%)   |
|            | ind  | 81.33 ± 2.19   | 59.09 ± 2.96 | 73.82 ± 1.93   | 14.73 (24.93\%) | -7.51 (-9.23\%)   |
|            | tran | 78.78 ± 1.92   | 58.95 ± 1.66 | 79.39 ± 1.64   | 20.44 (34.66\%) | 0.61 (0.77\%)     |
| Citseer    | prod | 68.38          | 59.81        | **69.27**      | 9.46 (15.82\%)  | 0.89 (1.30\%)     |
|            | ind  | 69.75 ± 3.59   | 60.06 ± 5.00 | 69.25 ± 2.25   | 9.19 (15.30\%)  | -0.5 (-0.7\%)     |
|            | tran | 68.04 ± 3.34   | 59.75 ± 2.48 | 69.28 ± 3.12   | 9.63 (15.93\%)  | 1.24 (1.82\%)     |
| Pubmed     | prod | **74.88**      | 66.80        | 74.71          | 7.91 (11.83\%)  | -0.17 (-0.22\%)   |
|            | ind  | 75.26 ± 2.57   | 66.85 ± 2.96 | 74.30 ± 2.61   | 7.45 (11.83\%)  | -0.96 (-1.27\%)   |
|            | tran | 74.78 ± 2.22   | 66.79 ± 2.90 | 74.81 ± 2.39   | 8.02 (12.01\%)  | 0.03 (0.04\%)     |
| A-computer | prod | 82.14          | 67.38        | **82.29**      | 14.90 (22.12\%) | 0.15 (0.19\%)     |
|            | ind  | 82.08 ± 1.79   | 67.84 ± 1.78 | 80.92 ± 1.36   | 13.08 (19.28\%) | -1.16 (-1.41\%)   |
|            | tran | 82.15 ± 1.55   | 67.27 ± 1.36 | 82.63 ± 1.40   | 15.36 (22.79\%) | 0.48 (0.58\%)     |
| A-photo    | prod | 91.08          | 79.25        | **92.38**      | 13.13 (16.57\%) | 1.30 (1.42\%)     |
|            | ind  | 91.50 ± 0.79   | 79.44 ± 1.72 | 91.18 ± 0.81   | 11.74 (14.78\%) | -0.32 (-0.35\%)   |
|            | tran | 90.80 ± 0.77   | 79.20 ± 1.64 | 92.68 ± 0.56   | 13.48 (17.01\%) | 1.70 (1.87\%)     |
| Arxiv      | prod | **70.73**      | 55.30        | 65.09          | 9.79 (17.70\%)  | -5.64 (-7.97\%)   |
|            | ind  | 70.64 ± 0.67   | 55.40 ± 0.56 | 60.48 ± 0.46   | 4.3 (7.76\%)    | -10.94 (-15.49\%) |
|            | tran | 70.75 ± 0.27   | 55.28 ± 0.49 | 71.46 ± 0.33   | 11.16 (20.18\%) | -4.31 (-6.09\%)   |
| Products   | prod | **76.60**      | 63.72        | 75.77          | 12.05 (18.91\%) | -0.83 (-1.09\%)   |
|            | ind  | 76.89 ± 0.53   | 63.70 ± 0.66 | 75.16 ± 0.34   | 11.44 (17.96\%) | -1.73 (-2.25\%)   |
|            | tran | 76.53 ±0.55    | 63.73 ± 0.69 | 75.92 ± 0.61   | 12.20 (19.15\%) | -0.61 (-0.79\%)   |
 -->

## Citation

If you find our work useful, please cite the following:

```BibTeX
@inproceedings{zhang2021graphless,
      title={Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation}, 
      author={Shichang Zhang and Yozen Liu and Yizhou Sun and Neil Shah},
      booktitle={International Conference on Learning Representations}
      year={2022},
      url={https://arxiv.org/abs/2110.08727}
}
```

## Contact Us

Please open an issue or contact `shichang@cs.ucla.edu` if you have any questions.

<!-- #endregion -->
