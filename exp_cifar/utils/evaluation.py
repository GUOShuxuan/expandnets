from tqdm import tqdm
import pickle
from data.cifar_dataset import cifar10_loader
from .nn_utils import load_model
import torch


def evaluate_model(path='', net=None, result_path='', dataset_name='cifar10', dataset_loader=cifar10_loader):
    """
    Wrapper function for the evaluation that also saves the results into the appropriate output files
    :param path: path to model
    :param net: network
    :param result_path: path to save results
    :param dataset_name: cifar10/100
    :param dataset_loader: dataloader
    :return:
    """
    # If a path is supplied load the model
    if path != '':
        net.cuda()
        load_model(net, path)

    _, test_loader, train_loader = dataset_loader(batch_size=128)
    results = {}

    acc = evaluate_test_accuracy(net=net, test_loader=test_loader)
    results['acc'] = acc

    results = {dataset_name: results}
    with open(result_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_test_accuracy(net=None, test_loader=None):
    net.eval()
    loss, correct, total = 0, 0, 0
    for (inputs, targets) in tqdm(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        # Calculate statistics
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    acc = correct.float() / total
    print("     acc = %.4f" % acc)
    return acc
