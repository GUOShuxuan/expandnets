# ExpandNets on ImageNet

This implements training of compact model architectures, such as MobileNet, MobileNetV2, and ShuffleNetV2 0.5× 
and their ExpandNets on the ImageNet dataset.

## Requirements 
We provide a docker image as located in the `exp_imagenet/Dockerfile`. Please build this image as the exact training environment. For the hard-ware infrastructure, we use NVIDIA Tesla V100 and RTX 2080 Ti as GPU resources. 

In order to build and run the image, you could use the following command as a reference. It may vary depending on your own infrastructure. We use `"pytorch=1.0.0=py3.6_cuda10.0.130_cudnn7.4.1_1"` with CUDA 10.0. 

```bash
cd exp_imagenet
# buidling the docker image
docker build -t NeurIPS-submission .
# Run the docker image.
nvidia-docker run -ti -v /local/path/to/dir/:/desired_folder_in_image/NeurIPS-submission
```


## Data

Download the ImageNet dataset from http://www.image-net.org/
   - Then move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

ImageNet dataset contains 1.2 million training images of 1000 categories and 50,000 validation images. All models are trained on training set 
and we report top-1 accuracy on validation set.

## Model

Models used on ImageNet in our paper are MobileNet, MobileNetV2, and ShuffleNetV2 0.5× and their ExpandNets, which are in `/models`.


## Training

To train a model, under the `exp_imagenet` folder, run training scripts with the desired model architecture and the path to the ImageNet dataset:

- MobileNet, MobileNetV2 and their ExpandNets
```bash
cd exp_imagenet
python train_mobilenets.py -a mobilenetv1 [imagenet-folder with train and val folders] --run 0 -b 256  --epochs 90 --wd 0.0001  --lr 0.1
            # -a ['mobilenetv1', 'mobilenetv1_expand', 'mobilenetv2', 'mobilenetv2_expand']
```

- ShuffleNetV2 0.5× and its ExpandNet
```bash
cd exp_imagenet
python train_shufflenet.py -a shufflenet_v2_x0_5 [imagenet-folder with train and val folders] --run 0 -b 1024  --epochs 250 --wd 0.00004  --lr 0.5  --seed 42 -p 100
            # -a ['shufflenet_v2_x0_5', 'shufflenet_v2_x0_5_expand']
```

To run a batch of experiments:
```bash
cd exp_imagenet
sh imagenet_experiments.sh   
```
This will produce results in Table 2 in the main paper, which are:

Table A: Top-1 accuracy (%) on the ILSVRC2012 validation set (ExpandNets with $r = 4$) (without KD)

|           Model 	|   MobileNet  	|   MobileNetV2  | ShuffleNetV2	|
|----------------:	|:------------:	|:------------:	 |:------------:|
|        Original 	| 66.48 	    | 63.75 	     | 56.89        |
|    ExpandNet-CL 	| 69.40 	    | 65.62          | 57.38        |



## Model Evaluation
To test the contraction of a CompactNet from a ExpandNet (code can be found in imagenet_experiments.sh as well):

- Real test: test on real ImageNet validation dataset
    ```bash
    cd exp_imagenet
    python test_contract.py -a [architecture] -data [imagenet-folder with train and val folders] --emodel_path [Path-to-ExpandNet-Pretrained-model] 
    ```
    - contract the pre-trained ExpandNet to original compact network and test the accuracy on validation dataset.
    - the contracted compact network and the ExpandNet should get the same accuracy.
    
- Dummy test: test on random inputs
    ```bash
    cd exp_imagenet
    python test_contract.py -a [architecture] 
    ```
    - contract the ExpandNet to original compact network and test the predictions and output values on random inputs.
    - the contracted compact network and ExpandNet should get the same predictions and output values.



Functions used to contract ExpandNets to SmallNets are in `utils/compute_new_weights.py` .

- `def from_expand_cl_to_mobilenetv1`: contract ExpandNet-CL to mobilenetv1
    
- `def from_expand_cl_to_mobilenetv2`: contract ExpandNet-CL to mobilenetv2

- `def from_expand_cl_to_shufflenet`: contract ExpandNet-CL to shufflenet_v2_x0_5


### Usage
ImageNet train from scratch code usage.

```
usage: train_*.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
                  [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
                  [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
                  [--rank RANK] [--dist-url DIST_URL]
                  [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
                  [--multiprocessing-distributed]
                  DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: mobilenetv1 | mobilenetv1_expand |
                        mobilenetv2 | mobilenetv2_expand |
                        shufflenet_v2_x0_5 | shufflenet_v2_x0_5_expand
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
```
