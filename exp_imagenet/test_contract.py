"""
contract ExpandNets back to Original networks
"""

import argparse
import numpy as np
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import models
from data import ImageFolderv2, get_loader, prefetched_loader
from utils import logs, utils, compute_new_weights


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR', default=None,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv2',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--emodel_path', default=None,
                    help='path to pre-trained expandnet')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--run', default=1, type=int,
                    help='nth run of experiments')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')

best_acc1 = 0

args = parser.parse_args()


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        print("=> Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            # Init the worker seed.
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:
        def _worker_init_fn(id):
            pass

    args.worker_init_fn = _worker_init_fn

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating ExpandNet '{}'".format(args.arch + '_expand'))
    Emodel = models.get_model(args.arch + '_expand')
    if args.emodel_path is not None:
        pretrain = torch.load(args.emodel_path)
        best_acc1 = pretrain['best_acc1']
        print('Best acc1 from Expand model: {:.4f}'.format(best_acc1.item()))
        pre_dict = pretrain['state_dict']

        tmp_dict = {}
        for k, v in pre_dict.items():
            new_k = k.split('.', 1)[1]
            tmp_dict.update({new_k: v})

        pre_dict_ = Emodel.state_dict()
        tmp_dict = {k: v for k, v in tmp_dict.items() if k in pre_dict_}
        pre_dict_.update(tmp_dict)
        Emodel.load_state_dict(pre_dict_)
    #
    print("=> creating CompactNet '{}'".format(args.arch))
    model = models.get_model(args.arch)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            Emodel.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            Emodel = torch.nn.parallel.DistributedDataParallel(Emodel, device_ids=[args.gpu])
        else:
            model.cuda()
            Emodel.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            Emodel = torch.nn.parallel.DistributedDataParallel(Emodel)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        Emodel = Emodel.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            Emodel.features = torch.nn.DataParallel(Emodel.features)
            Emodel.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            Emodel = torch.nn.DataParallel(Emodel).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    cudnn.benchmark = True

    # Data loading code
    if args.emodel_path is not None:
        print("=> loading data'{}'".format(args.data))
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # 1-crop validation
        val_dataset = ImageFolderv2(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.emodel_path is None:
        print('****** Doing a dummy test on random inputs')
        dummy_test(model, Emodel)
    else:
        print('****** Doing a real test on imagenet validation set')
        real_test(val_loader, criterion, model, Emodel)


def dummy_test(model, Emodel):

    inputs = torch.randn(128, 3, 224, 224)

    print('==> Before contracting')
    Eoutput = utils.dummy_validate(inputs, Emodel, args)

    print('......Contracting............')
    new_model = compute_new_weights.expandnet_contract(model, Emodel, args)

    print('==> After contracting')
    Soutput = utils.dummy_validate(inputs, new_model, args)

    print('Output Difference between ExpandNet and Contracted CompactNet: ', (Eoutput - Soutput).sum().item())
    utils.dummy_validate_with_outputs(inputs, new_model, Emodel, args)
    print('Contract successfully!')


def real_test(val_loader, criterion, model, Emodel):

    print('==> Before contracting')
    print('......ExpandNet TEST............')
    Etest_acc = utils.validate(val_loader, Emodel, criterion, args, epoch=0)
    print('ExpandNet test_acc: {:.4f}'.format(Etest_acc))

    print('......Original ComapctNet TEST............')
    test_acc = utils.validate(val_loader, model, criterion, args, epoch=0)
    print('Original ComapctNet test_acc: {:.4f}'.format(test_acc))

    print('====> Contracting............')
    new_model = compute_new_weights.expandnet_contract(model, Emodel, args)

    print('======> After contracting')
    print('......Contracted CompactNet TEST............')
    test_acc = utils.validate(val_loader, new_model.cuda(args.gpu), criterion, args, epoch=0)
    print('Contracted CompactNet test_acc: {:.4f}'.format(test_acc))


if __name__ == '__main__':
    main()

