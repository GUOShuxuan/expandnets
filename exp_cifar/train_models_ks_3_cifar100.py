"""
Training script for SmallNet with kernel size 3*3 on cifar100
    --seed [random seed for one experiment]

"""
import os
import shutil

import torch.optim as optim
from utils.evaluation import evaluate_model
from data.cifar_dataset import cifar100_loader
from models.cifar_tiny_3 import *
from utils.nn_utils import save_model, train_model, test_model
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Training Process CIFAR100')
parser.add_argument('--seed', metavar='random seed for this trail', default=3,
                    help='seed: 0, 1, 2, 3, 4, 5')

args = parser.parse_args()
seed = str(args.seed)
print('Seed: ', seed)

learning_rates = [0.01, 0.001, 0.0001]
iters = [50, 50, 50]


# make directory
if not os.path.exists('results/models/'):
    os.makedirs('results/models/')

if not os.path.exists('results/evals/'):
    os.makedirs('results/evals/')


def train_cifar100_model(net, learning_rates=[0.001, 0.0001], iters=[50, 50],
                         output_path=None, log_name=None):
    """
    Trains a baseline (classification model)
    :param net: the network to be trained
    :param learning_rates: the learning rates to be used during the training
    :param iters: number of epochs using each of the supplied learning rates
    :param output_path: path to save the trained model
    :return:
    """

    # Load data
    train_loader, test_loader, _ = cifar100_loader(batch_size=128)

    # Define loss
    criterion = nn.CrossEntropyLoss()
    log_path = os.path.join('results/logs/', log_name)

    if os.path.exists(log_path):
        shutil.rmtree(log_path)
        if os.path.exists(log_path):
            raise Exception("Delete Error")
        else:
            print(f"save_path deleted {log_path}")
    os.makedirs(log_path)
    Logwriter = SummaryWriter(log_path)

    best_acc = 0
    best_epoch = 0
    nepochs = iters[0]+iters[1]+iters[2]
    for epoch in range(nepochs):
        if epoch < iters[0]:
            lr = learning_rates[0]
        elif epoch < iters[0]+iters[1]:
            lr = learning_rates[1]
        else:
            lr = learning_rates[2]
        print("Training with lr=%.4f for %d epoch" % (lr, epoch))
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        train_loss, train_acc = train_model(net, optimizer, criterion, train_loader)
        test_loss, test_acc = test_model(net, criterion, test_loader)
        print("Epoch: %3d, Train Loss = %.4f, acc = %.4f" % (epoch, train_loss, train_acc))
        print("            Test Loss = %.4f, acc = %.4f" % (test_loss, test_acc))
        if Logwriter is not None:
            Logwriter.add_scalar('train_loss', train_loss, epoch)
            Logwriter.add_scalar('train_acc_top1', train_acc, epoch)
            Logwriter.add_scalar('test_loss', test_loss, epoch)
            Logwriter.add_scalar('test_acc_top1', test_acc, epoch)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(net, output_file=output_path)

    Logwriter.add_scalar('best_acc', best_acc, best_epoch)


def train_cifar_models():
    """
    Train all models
    :return:
    """

    # Baseline model: SmallNet 3*3
    print('***  Baseline ***')
    net = Cifar_Tiny(num_classes=100)
    net.cuda()
    train_cifar100_model(net, learning_rates=learning_rates, iters=iters,
                         output_path='results/models/tiny_cifar100_3_'+seed+'.model',
                         log_name='tiny_cifar100_3_'+seed)

    # FC(Arora18): ExpandNet-FC with er=4
    print('***  ExpandNet-FC ***')
    net = Cifar_Tiny_ExpandNet_fc(num_classes=100)
    net.cuda()
    train_cifar100_model(net, learning_rates=learning_rates, iters=iters,
                         output_path='results/models/tiny_cifar100_3_enet_fc_' + seed + '.model',
                         log_name='tiny_cifar100_3_enet_fc_' + seed)

    # ExpandNet-CL
    print('***  ExpandNet-CL ***')
    net = Cifar_Tiny_ExpandNet_cl(num_classes=100)
    net.cuda()
    train_cifar100_model(net, learning_rates=learning_rates, iters=iters,
                        output_path='results/models/tiny_cifar100_3_enet_cl_' + seed + '.model',
                        log_name='tiny_cifar100_3_enet_cl_' + seed)

    # ExpandNet-CL+FC
    print('*** ExpandNet-CL+FC ***')
    net = Cifar_Tiny_ExpandNet_cl_fc(num_classes=100)
    net.cuda()
    train_cifar100_model(net, learning_rates=learning_rates, iters=iters,
                        output_path='results/models/tiny_cifar100_3_enet_cl_fc_' + seed + '.model',
                        log_name='tiny_cifar100_3_enet_cl_fc_' + seed)


def evaluate_cifar_models():
    """
    Evaluates the baseline and its ExpandNets and its
    :return:
    """

    print('***  Baseline ***')
    evaluate_model(net=Cifar_Tiny(num_classes=100),
                   path='results/models/tiny_cifar100_3_' + seed + '.model',
                   result_path='results/evals/tiny_cifar100_3_'+seed+'.pickle',
                   dataset_name='cifar100', dataset_loader=cifar100_loader)

    print('***  ExpandNet-FC ***')
    evaluate_model(net=Cifar_Tiny_ExpandNet_fc(num_classes=100),
                   path='results/models/tiny_cifar100_3_enet_fc_' + seed + '.model',
                   result_path='results/evals/tiny_cifar100_3_enet_fc_' + seed + '.pickle',
                   dataset_name='cifar100', dataset_loader=cifar100_loader)

    print('***  ExpandNet-CL ***')
    evaluate_model(net=Cifar_Tiny_ExpandNet_cl(num_classes=100),
                   path='results/models/tiny_cifar100_3_enet_cl_' + seed + '.model',
                   result_path='results/evals/tiny_cifar100_3_enet_cl_' + seed + '.pickle',
                   dataset_name='cifar100', dataset_loader=cifar100_loader)

    print('***  ExpandNet-CL+FC ***')
    evaluate_model(net=Cifar_Tiny_ExpandNet_cl_fc(num_classes=100),
                   path='results/models/tiny_cifar100_3_enet_cl_fc_' + seed + '.model',
                   result_path='results/evals/tiny_cifar100_3_enet_cl_fc_' + seed + '.pickle',
                   dataset_name='cifar100', dataset_loader=cifar100_loader)


if __name__ == '__main__':

    print('Training networks with 3*3 kernels on CIFAR-100')

    train_cifar_models()
    evaluate_cifar_models()



