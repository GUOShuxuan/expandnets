"""
Test script: contract ExpandNets back to SmallNets
usage: python test_contract.py --expand_path [Path-to-Expand-model]
"""

from utils.evaluation import evaluate_test_accuracy
from data.cifar_dataset import cifar10_loader, cifar100_loader
from utils.nn_utils import check_equal_contract, load_model
import argparse
from utils.compute_new_weights import *


def contract_expandnet(net, n_class, test_loader, expand='FC',
                       expand_path=None):
    """
    Small network contracted from its ExpandNets
    :param net: small/compact network
    :param n_class: the number of classes (10 for cifar10, 100 for cifar100)
    :param test_loader: test dataloader
    :param expand: expand strategies: FC, CL, CK, CL+FC, CK+FC
    :param expand_path: path to the trained expandnet
    :return:
    """
    print('** ExpandNet-%s: ' % expand)
    if expand == 'FC':
        init_net = Cifar_Tiny_ExpandNet_fc(num_classes=n_class)
        load_model(init_net, expand_path)
        evaluate_test_accuracy(init_net.cuda(), test_loader)

        net = from_expandnet_ck_to_snet(init_net, net, exp_layer_names=['fc1'])
        check_equal_contract(net, init_net, expand_layers=['fc1'])

    elif expand == 'CL':
        init_net = Cifar_Tiny_ExpandNet_cl(num_classes=n_class)
        load_model(init_net, expand_path)
        evaluate_test_accuracy(init_net.cuda(), test_loader)

        net = from_expandnet_cl_to_snet(init_net.cpu(), net.cpu(), exp_layer_names=['conv1', 'conv2', 'conv3'])
        check_equal_contract(net, init_net, expand_layers=['conv1', 'conv2', 'conv3'])

    elif expand == 'CL+FC':
        init_net = Cifar_Tiny_ExpandNet_cl_fc(num_classes=n_class)
        load_model(init_net, expand_path)
        evaluate_test_accuracy(init_net.cuda(), test_loader)

        net = from_expandnet_cl_to_snet(init_net.cpu(), net.cpu(), exp_layer_names=['conv1', 'conv2', 'conv3', 'fc1'])
        check_equal_contract(net, init_net, expand_layers=['conv1', 'conv2', 'conv3', 'fc1'])

    elif expand == 'CK':
        init_net = Cifar_Tiny_ExpandNet_ck(num_classes=n_class)
        load_model(init_net, expand_path)
        evaluate_test_accuracy(init_net.cuda(), test_loader)

        net = from_expandnet_ck_to_snet(init_net, net, exp_layer_names=['conv1', 'conv2', 'conv3'])
        check_equal_contract(net, init_net, expand_layers=['conv1', 'conv2', 'conv3'])

    elif expand == 'CK+FC':
        init_net = Cifar_Tiny_ExpandNet_ck_fc(num_classes=n_class)
        load_model(init_net, expand_path)
        evaluate_test_accuracy(init_net.cuda(), test_loader)

        net = from_expandnet_ck_to_snet(init_net, net, exp_layer_names=['conv1', 'conv2', 'conv3', 'fc1'])
        check_equal_contract(net, init_net, expand_layers=['conv1', 'conv2', 'conv3', 'fc1'])

    print('======> After contracting')
    print('** Contracted SmallNet: ')
    evaluate_test_accuracy(net.cuda(), test_loader)


def parse_path(path):
    model = path.split('/')[-1]
    m_splits = model.split('_', 4)
    dataset = m_splits[1]
    ks = int(m_splits[2])
    expand_splits = m_splits[-1].split('_')
    if len(expand_splits) == 2:
        expand = expand_splits[0].upper()
    else:
        expand = expand_splits[0].upper() + '+' + expand_splits[1].upper()

    return dataset, ks, expand


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Test Compreesion')
    parser.add_argument('--expand_path', type=str, default=None,
                        help='path to the trained expandnet')

    args = parser.parse_args()

    dataset, ks, expand = parse_path(args.expand_path)

    print('*** SmallNet with kernels of %d * %d, contracted from ExpandNet-%s on %s' % (ks, ks, expand, dataset))

    if dataset == 'cifar10':
        train_loader, test_loader, _ = cifar10_loader(batch_size=128)
        n_class = 10
    else:
        train_loader, test_loader, _ = cifar100_loader(batch_size=128)
        n_class = 100

    if ks == 3:
        from models.cifar_tiny_3 import *
    else:
        from models.cifar_tiny_7 import *

    net = Cifar_Tiny(num_classes=n_class)

    print('==> Before contracting')
    print('** Original SmallNet: ')
    evaluate_test_accuracy(net.cuda(), test_loader)

    contract_expandnet(net, n_class, test_loader,
                       expand=expand,
                       expand_path=args.expand_path)




