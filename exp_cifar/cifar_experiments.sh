#!/usr/bin/env bash

# This script can be used to produce results in Table 1 in our main paper, and Table S1 and S3 in the supplementary material.

## set seed to 0 1 2 3 4 to run 5 trails of each experiment

CUDA_VISIBLE_DEVICES=0

# cifar10, SmallNet with 3*3
python train_models_ks_3_cifar10.py --seed 0 &
python train_models_ks_3_cifar10.py --seed 1 &
python train_models_ks_3_cifar10.py --seed 2 &
python train_models_ks_3_cifar10.py --seed 3 &
python train_models_ks_3_cifar10.py --seed 4


# cifar10, SmallNet with 7*7
python train_models_ks_7_cifar10.py --seed 0 &
python train_models_ks_7_cifar10.py --seed 1 &
python train_models_ks_7_cifar10.py --seed 2 &
python train_models_ks_7_cifar10.py --seed 3 &
python train_models_ks_7_cifar10.py --seed 4
#
#
## cifar100, SmallNet with 3*3
python train_models_ks_3_cifar100.py --seed 0 &
python train_models_ks_3_cifar100.py --seed 1 &
python train_models_ks_3_cifar100.py --seed 2 &
python train_models_ks_3_cifar100.py --seed 3 &
python train_models_ks_3_cifar100.py --seed 4


# cifar100, SmallNet with 7*7
python train_models_ks_7_cifar100.py --seed 0 &
python train_models_ks_7_cifar100.py --seed 1 &
python train_models_ks_7_cifar100.py --seed 2 &
python train_models_ks_7_cifar100.py --seed 3 &
python train_models_ks_7_cifar100.py --seed 4

