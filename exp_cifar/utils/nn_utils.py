import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


def save_model(net, output_file='model.state'):
    """
    Saves a pytorch model
    :param net:
    :param output_file:
    :return:
    """
    torch.save(net.state_dict(), output_file)


def load_model(net, input_file='model.state'):
    """
    Loads a pytorch model
    :param net:
    :param input_file:
    :return:
    """
    state_dict = torch.load(input_file)
    net.load_state_dict(state_dict)


def train_model(net, optimizer, criterion, train_loader):
    """
    Train a pytorch model
    :param net:
    :param optimizer:
    :param criterion:
    :param train_loader:
    :param epochs:
    :return:
    """

    net.train()

    train_loss, correct, total = 0, 0, 0
    for (inputs, targets) in tqdm(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Calculate statistics
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        acc = correct.float() / total
        avg_train_loss = train_loss / total

    return avg_train_loss, acc


def test_model(net, criterion, test_loader):
    """
    Test a pytorch model
    :param net:
    :param optimizer:
    :param criterion:
    :param train_loader:
    :param epochs:
    :return:
    """

    net.eval()

    test_loss, correct, total = 0, 0, 0
    for (inputs, targets) in tqdm(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # Calculate statistics
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        acc = correct.float() / total
        avg_test_loss = test_loss / total

    return avg_test_loss, acc


def test_model_distill(net, test_loader):
    """
    Trains a pytorch model
    :param net:
    :param optimizer:
    :param criterion:
    :param train_loader:
    :param epochs:
    :return:
    """

    net.eval()
    test_loss, correct, total = 0, 0, 0
    for (inputs, targets) in tqdm(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        acc = correct.float() / total
    return acc


def get_labels(test_loader):
    """
    Extracts the labels from a loader
    :return:
    """
    labels = []
    for (inputs, targets) in tqdm(test_loader):
        labels.append(targets.numpy())

    return np.concatenate(labels).reshape((-1,))


def check_equal_contract(net, enet, expand_layers):
    enet_dict = enet.state_dict()
    for k, v in net.state_dict().items():
        if k.find('bn')!=-1:
            assert((torch.eq(v, enet_dict[k])).all() == 1)
        elif k.find('fc') != -1 and not any(layer in k for layer in expand_layers):
            assert ((torch.eq(v, enet_dict[k])).all() == 1)
        elif k.find('conv') != -1 and not any(layer in k for layer in expand_layers):
            assert ((torch.eq(v, enet_dict[k])).all() == 1)



