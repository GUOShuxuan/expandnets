#!/usr/bin/env bash

source ./setup.sh

## Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3

# MobileNet： 90 epochs
python train_mobilenets.py -a mobilenetv1 [imagenet-folder with train and val folders] \
                           --run 0 -b 256  --epochs 90 --wd 0.0001  --lr 0.1

python train_mobilenets.py -a mobilenetv1_expand [imagenet-folder with train and val folders] \
                           --run 0 -b 256  --epochs 90 --wd 0.0001  --lr 0.1


# MobileNetV2:: 90 epochs
python train_mobilenets.py -a mobilenetv2 [imagenet-folder with train and val folders] \
                           --run 0 -b 256  --epochs 90 --wd 0.0001  --lr 0.1

python train_mobilenets.py -a mobilenetv2_expand [imagenet-folder with train and val folders] \
                           --run 0 -b 256  --epochs 90 --wd 0.0001  --lr 0.1


# ShuffleNet
python train_shufflenet.py -a shufflenet_v2_x0_5 [imagenet-folder with train and val folders] \
                           --run 0 -b 1024  --epochs 250 --wd 0.00004  --lr 0.5  --seed 42 -p 100

python train_shufflenet.py -a shufflenet_v2_x0_5_expand [imagenet-folder with train and val folders] \
                           --run 0 -b 1024  --epochs 250 --wd 0.00004  --lr 0.5  --seed 42 -p 100


# Evaluation code
# Dummy evaluation to test the implementation / environment
python test_contract.py -a mobilenetv1

python test_contract.py -a mobilenetv2

python test_contract.py -a shufflenet_v2_x0_5

# Evaluation of the trained model
python test_contract.py -a mobilenetv1 -data [imagenet-folder with train and val folders] \
                        --emodel_path [Path-to-ExpandNet-Pretrained-model]

python test_contract.py -a mobilenetv2 -data [imagenet-folder with train and val folders] \
                        --emodel_path [Path-to-ExpandNet-Pretrained-model]

python test_contract.py -a shufflenet_v2_x0_5 -data [imagenet-folder with train and val folders] \
                        --emodel_path [Path-to-ExpandNet-Pretrained-model]



