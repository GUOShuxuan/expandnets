# ExpandNets on CIFAR-10 and CIFAR-100

This implements training of baseline SmallNets with 3 × 3 and 7 × 7 kernels and their ExpandNets on CIFAR-10 and CIFAR-100 respectively, 
producing results in Table 1 in the main paper, Tables S1 and S3 in the supplementary material.

## Requirements 

- Create a conda environment to run the code

  `conda env create -f env.yml`
  
- We use Pytorch04 and nvidia Titan X (12Gb) in these experiments. 

## Data

The script downloads the data for CIFAR-10 and CIFAR-100 automatically and saves them in the folder `data/`. 
These datasets contain 50,000 training images and 10,000 test images of 10 and 100 classes. The images are of size 32 × 32.
All networks are trained on training images and tested on test images.

## Model
SmallNets and their ExpandNets are defined in `models/cifar_tiny_3.py` with kernel size of 3 × 3 and 
`models/cifar_tiny_7.py` with kernel size of 7 × 7.

## Training
To train a baseline SmallNet and its ExpandNets with a random seed 0:

```bash
cd exp_cifar
python train_models_ks_3_cifar10.py --seed 0
python train_models_ks_7_cifar10.py --seed 0
python train_models_ks_3_cifar100.py --seed 0
python train_models_ks_7_cifar100.py --seed 0
```

To train a batch of experiments with multiple random seeds:

```bash
cd exp_cifar
sh cifar_experiments.sh
```

## Results
- models will be saved in `results/models`
- logs will be saved in `results/logs`
- evaluated results will be saved in `results/evals`

To print all results of several trails of experiments after training on CIFAR-10 and CIFAR-100:
```bash
cd exp_cifar
python print_results.py --seeds [seeds used in experiments]
# usage: python print_results.py --seeds 0 1 2 3 4

```
By this script, results of Table 1 in the main paper, Tables S1 and S3 in the supplementary material can be produced. They are:

Table A: Top-1 accuracy (%) of SmallNet with 7 × 7 kernels vs ExpandNets with r = 4 on CIFAR-10 and CIFAR-100 (without KD).

|           Model 	|   CIFAR-10   	|   CIFAR-100  	|
|----------------:	|:------------:	|:------------:	|
|        SmallNet 	| 78.63 ± 0.41 	| 46.63 ± 0.27 	|
|     FC(Arora18) 	| 78.64 ± 0.39 	| 46.59 ± 0.45 	|
|    ExpandNet-CL 	| 78.47 ± 0.20 	| 46.90 ± 0.66 	|
| ExpandNet-CL+FC 	| 79.11 ± 0.23 	| 46.66 ± 0.43 	|
|    ExpandNet-CK 	| 80.27 ± 0.24 	| 48.55 ± 0.51 	|
| ExpandNet-CK+FC 	| 80.31 ± 0.27 	| 48.62 ± 0.47 	|

Table B: Top-1 accuracy (%) of SmallNet with 3 × 3 kernels vs ExpandNets with r = 4 on CIFAR-10 and CIFAR-100 (without KD).

|           Model 	|   CIFAR-10   	|   CIFAR-100  	|
|----------------:	|:------------:	|:------------:	|
|        SmallNet 	| 73.32 ± 0.20 	| 40.40 ± 0.60 	|
|     FC(Arora18) 	| 73.78 ± 0.83 	| 40.52 ± 0.71 	|
|    ExpandNet-CL 	| 73.96 ± 0.30 	| 40.91 ± 0.47 	|
| ExpandNet-CL+FC 	| 74.45 ± 0.29 	| 41.12 ± 0.49 	|


## Model Evaluation

To test the contraction of SmallNet from a ExpandNet: 
```bash
cd exp_cifar
python test_contract.py --expand_path [Path-to-Expand-model]
```

Functions used to contract ExpandNets to SmallNets are in `utils/compute_new_weights.py` .

- `def from_expandnet_cl_to_snet`: contract ExpandNet-CL/ExpandNet-FC/ExpandNet-CL+FC to SmallNet
    
- `def from_expandnet_ck_to_snet`: contract ExpandNet-CK/ExpandNet-FC/ExpandNet-CK+FC to SmallNet
