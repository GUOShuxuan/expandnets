# ExpandNets: Linear Over-parameterization to Train Compact Convolutional Networks

> Code is in early release and may be subject to change. Please feel free to open an issue in case of questions.

## Overview

We propose 3 strategies to linearly expand a compact network. An expanded network can then be contracted back to the compact one algebraically.

![Framework](framework.png)

## Image Classification
Here are code for image classification experiments on CIFAR-10, CIFAR-100 and ImageNet.

Details on each experiment are listed in corresponding README.md in each folder.


## Dummy test

We provide some toy code to expand a convolutional layer with either standard or 
depthwise convolutions and contract the expanded layers back.


Code in dummy_test.py is same as it in our supplementary material, which can be run simply. 

```bash
python dummy_test.py
 ```




