from copy import copy

import torch
import torch.nn as nn


def compute_cl(s_1, s_2):
    """
    Compute weights from s_1 and s_2
    :param s_1: 1*1 conv layer
    :param s_2: 3*3 conv layer
    :return: new weight and bias
    """
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1) * w_s_1.size(2) * w_s_1.size(3))
    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3))

    new_weight = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_2.size(2)*w_s_2.size(3))
    for i in range(w_s_2.size(0)):
        tmp = w_s_2_tmp[i, :, :].view( w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3))
        new_weight[i, :, :] = torch.matmul(w_s_1_.t(), tmp)
    new_weight = new_weight.view(w_s_2.size(0), w_s_1.size(1),  w_s_2.size(2), w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = None

    return {'weight': new_weight, 'bias': new_bias}


def compute_cl_2(s_1, s_2):
    """
    compute weights from former computation and last 1*1 conv layer
    :param s_1: 3*3 conv layer
    :param s_2: 1*1 conv layer
    :return: new weight and bias
    """
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1),  w_s_1.size(2) * w_s_1.size(3)) # 4 1 9
    w_s_2_ = w_s_2.view(w_s_2.size(0), w_s_2.size(1) * w_s_2.size(2) * w_s_2.size(3)) # 2 4
    new_weight_ = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_1.size(2)*w_s_1.size(3)) # 2 1 9
    for i in range(w_s_1.size(1)):
        tmp = w_s_1_[:, i, :].view(w_s_1.size(0),  w_s_1.size(2) * w_s_1.size(3)) # 4 9
        new_weight_[:, i, :] = torch.matmul(w_s_2_, tmp)
    new_weight = new_weight_.view(w_s_2.size(0), w_s_1.size(1),  w_s_1.size(2), w_s_1.size(3)) # 2 2 3 3

    if b_s_1 is not None and b_s_2 is not None:
        new_bias = torch.matmul(w_s_2_, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = None

    return {'weight': new_weight, 'bias': new_bias}


def compute_cl_dw_group(s_1, s_2):
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1),  w_s_1.size(2) * w_s_1.size(3)) # 16 1 1
    w_s_2_ = w_s_2.view(w_s_2.size(0), w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3)) # 16 2 9

    new_weight_ = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_2.size(2) * w_s_2.size(3))  # 16 1 9

    num_groups = w_s_2.size(1)
    groups = w_s_1.size(0) // num_groups

    for i in range(groups):
        w_2_tmp = w_s_2_[num_groups*i:num_groups*(i+1), :, :].transpose(1,0).contiguous() # 2 2 9
        w_1_tmp = w_s_1_[num_groups*i:num_groups*(i+1), :, :].view(num_groups,  w_s_1.size(2) * w_s_1.size(3)) # 2 1 => 1 2
        new_weight_tmp = torch.matmul(w_1_tmp.t(), w_2_tmp.view(w_2_tmp.size(0), -1)).view(1, num_groups, w_s_2.size(2) * w_s_2.size(3)) # 1 2 9
        new_weight_[num_groups*i:num_groups*(i+1), :, :] = new_weight_tmp.transpose(1,0) # 2 1 9

    new_weight = new_weight_.view(w_s_2.size(0), w_s_1.size(1),  w_s_2.size(2), w_s_2.size(3)) # 16 1 3 3

    # new_bias = torch.matmul(w_s_2_, b_s_1) + b_s_2
    new_bias = b_s_2

    return {'weight': new_weight, 'bias': new_bias}


def compute_cl_dw_group_2(s_1, s_2):
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']


    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1),  w_s_1.size(2) * w_s_1.size(3)) # 16 1 9  # 16 1 1 #16 2 9
    w_s_2_ = w_s_2.view(w_s_2.size(0), w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3)) # 8 2  1 # 8 2 1 #
    num_groups = w_s_2.size(1)
    w_s_1_ = w_s_1_.view(w_s_2.size(0), num_groups, w_s_1.size(2) * w_s_1.size(3)) # 16
    new_weight = w_s_1_ * w_s_2_

    new_weight = new_weight.sum(dim=1)

    new_weight = new_weight.view(w_s_2.size(0), w_s_1.size(1),  w_s_1.size(2), w_s_1.size(3)) # 2 2 3 3 # 32 1 3 3

    # new_bias = torch.matmul(w_s_2_, b_s_1) + b_s_2
    new_bias = b_s_2

    return {'weight': new_weight, 'bias': new_bias}


# mobilenetv1
def from_expand_cl_to_mobilenetv1(Snet, Enet, expand_layers=14):
    """
    contract ExpandNet-CL back to mobilenetv1
    :param Snet: compact network
    :param Enet: expandnet
    :param expand_layers: layers which are expanded
    :return: contracted compact network
    """
    Sdict = Snet.state_dict()
    Edict = Enet.state_dict()
    select_keys = []
    for layer in range(14):
        for i in range(3):
            select_keys.append('module.model.' + str(layer) + '.' + str(i) + '.weight')

    newEdict={}
    for k, v in Edict.items():
        if k not in select_keys:
            k_slip = k.split('.')
            if 'fc' not in k_slip:
                k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2] + '.' + str(int(k_slip[3]) - 2) + '.' + k_slip[4]
            newEdict.update({k: v})

    Sdict.update(newEdict)
    Snet.load_state_dict(Sdict)

    for i in range(expand_layers):
        conv1_1 = Enet.module.model[i][0]
        conv1_2 = Enet.module.model[i][1]
        conv1_3 = Enet.module.model[i][2]
        if i == 0:
            tmp = compute_cl(conv1_1, conv1_2)
            tmp = compute_cl_2(tmp, conv1_3)
        else:
            tmp = compute_cl_dw_group(conv1_1, conv1_2)
            tmp = compute_cl_dw_group_2(tmp, conv1_3)

        conv1 = Snet.module.model[i][0]
        if conv1.bias is not None:
            conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
        else:
            conv1.weight.data = tmp['weight']

    return Snet


# mobilenetv2
def from_expand_cl_to_mobilenetv2(Snet, Enet, expand_layers=None):
    """
    contract ExpandNet-CL back to mobilenetv2
    :param Snet: compact network
    :param Enet: expandnet
    :param expand_layers: layers which are expanded
    :return: contracted compact network
    """
    Sdict = Snet.state_dict()
    Edict = Enet.state_dict()
    select_keys = []
    for layer in expand_layers:
        for i in range(3):
            if layer == 0:
                select_keys.append('module.features.' + str(layer) + '.' + str(i) + '.weight')
            elif layer == 1:
                select_keys.append('module.features.' + str(layer) + '.conv.' + str(i) + '.weight')
            else:
                select_keys.append('module.features.' + str(layer) + '.conv.' + str(i+3) + '.weight')

    newEdict = {}
    for k, v in Edict.items():
        if k not in select_keys:
            k_slip = k.split('.')
            if 'classifier' not in k_slip and '18' not in k_slip:
                if int(k_slip[2]) == 0:
                    k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2] + '.' + str(int(k_slip[3]) - 2) + '.' + k_slip[4]

                elif int(k_slip[2]) in expand_layers:
                    if int(k_slip[4]) <= 2:
                        k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2] + '.' + k_slip[3] + '.' + str(
                            int(k_slip[4])) + '.' + k_slip[5]
                    else:
                        k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2] + '.' + k_slip[3] + '.' + str(
                            int(k_slip[4]) - 2) + '.' + k_slip[5]
            newEdict.update({k: v})

    Sdict.update(newEdict)
    Snet.load_state_dict(Sdict)

    for i in expand_layers:
        if i == 0:
            conv1_1 = Enet.module.features[i][0]
            conv1_2 = Enet.module.features[i][1]
            conv1_3 = Enet.module.features[i][2]

            tmp = compute_cl(conv1_1, conv1_2)
            tmp = compute_cl_2(tmp, conv1_3)
            conv1 = Snet.module.features[i][0]

        elif i == 1:
            conv1_1 = Enet.module.features[i].conv[0]
            conv1_2 = Enet.module.features[i].conv[1]
            conv1_3 = Enet.module.features[i].conv[2]

            tmp = compute_cl_dw_group(conv1_1, conv1_2)
            tmp = compute_cl_dw_group_2(tmp, conv1_3)
            conv1 = Snet.module.features[i].conv[0]

        else:
            conv1_1 = Enet.module.features[i].conv[3]
            conv1_2 = Enet.module.features[i].conv[4]
            conv1_3 = Enet.module.features[i].conv[5]
            tmp = compute_cl_dw_group(conv1_1, conv1_2)
            tmp = compute_cl_dw_group_2(tmp, conv1_3)
            conv1 = Snet.module.features[i].conv[3]

        if conv1.bias is not None:
            conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
        else:
            conv1.weight.data = tmp['weight']

    return Snet


# shufflenet
def from_expand_cl_to_shufflenet(Snet, Enet, expand_layers=['conv1', 'stage2', 'stage3', 'stage4']):
    """
    contract ExpandNet-CL back to shufflenet
    :param Snet: compact network
    :param Enet: expandnet
    :param expand_layers: layers which are expanded
    :return: contracted compact network
    """

    Sdict = Snet.state_dict()
    Edict = Enet.state_dict()
    select_keys = []
    for stage in expand_layers:
        if stage == 'conv1':
            for i in range(3):
                select_keys.append('module.' + stage + '.' + str(i) + '.weight')
        elif stage == 'stage2' or stage == 'stage4':
            for layer in range(4):
                if layer == 0:
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch1.0' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch1.1' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch1.2' + '.weight')

                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.3' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.4' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.5' + '.weight')
                else:
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.3' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.4' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.5' + '.weight')
        elif stage == 'stage3':
            for layer in range(8):
                if layer == 0:
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch1.0' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch1.1' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch1.2' + '.weight')

                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.3' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.4' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.5' + '.weight')
                else:
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.3' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.4' + '.weight')
                    select_keys.append('module.' + stage + '.' + str(layer) + '.branch2.5' + '.weight')

    newEdict={}
    for k, v in Edict.items():
        if k not in select_keys:
            k_slip = k.split('.')
            if 'conv1' in k_slip:
                k = k_slip[0] + '.' + k_slip[1] + '.' + str(int(k_slip[2]) - 2) + '.' + k_slip[3]
            elif 'branch1' in k_slip:
                k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2] + '.' + k_slip[3] + '.' + str(int(k_slip[4]) - 2) + '.' + k_slip[5]
            elif 'branch2' in k_slip:
                if int(k_slip[4]) < 3 :
                    k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2] + '.' + k_slip[3] + '.' + str(int(k_slip[4])) + '.' + k_slip[5]
                else:
                    k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2] + '.' + k_slip[3] + '.' + str(int(k_slip[4]) - 2) + '.' + k_slip[5]
            elif 'conv5' in k_slip:
                k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2] + '.' + k_slip[3]
            elif 'fc' in k_slip:
                k = k_slip[0] + '.' + k_slip[1] + '.' + k_slip[2]

            newEdict.update({k: v})

    Sdict.update(newEdict)
    Snet.load_state_dict(Sdict)

    for stage in expand_layers:
        if stage == 'conv1':
            conv1_1 = Enet.module.conv1[0]
            conv1_2 = Enet.module.conv1[1]
            conv1_3 = Enet.module.conv1[2]
            tmp = compute_cl(conv1_1, conv1_2)
            tmp = compute_cl_2(tmp, conv1_3)
            conv1 = Snet.module.conv1[0]
            if conv1.bias is not None:
                conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
            else:
                conv1.weight.data = tmp['weight']

        elif stage == 'stage2' or stage == 'stage4':
            for layer in range(4):
                if layer == 0:
                    conv1_1 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch1')[0]
                    conv1_2 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch1')[1]
                    conv1_3 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch1')[2]
                    tmp = compute_cl_dw_group(conv1_1, conv1_2)
                    tmp = compute_cl_dw_group_2(tmp, conv1_3)
                    conv1 = getattr(getattr(getattr(Snet.module, stage), str(layer)), 'branch1')[0]
                    if conv1.bias is not None:
                        conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
                    else:
                        conv1.weight.data = tmp['weight']

                    conv1_1 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[3]
                    conv1_2 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[4]
                    conv1_3 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[5]
                    tmp = compute_cl_dw_group(conv1_1, conv1_2)
                    tmp = compute_cl_dw_group_2(tmp, conv1_3)
                    conv1 = getattr(getattr(getattr(Snet.module, stage), str(layer)), 'branch2')[3]
                    if conv1.bias is not None:
                        conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
                    else:
                        conv1.weight.data = tmp['weight']
                else:
                    conv1_1 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[3]
                    conv1_2 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[4]
                    conv1_3 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[5]
                    tmp = compute_cl_dw_group(conv1_1, conv1_2)
                    tmp = compute_cl_dw_group_2(tmp, conv1_3)
                    conv1 = getattr(getattr(getattr(Snet.module, stage), str(layer)), 'branch2')[3]
                    if conv1.bias is not None:
                        conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
                    else:
                        conv1.weight.data = tmp['weight']

        elif stage == 'stage3':
            for layer in range(8):
                if layer == 0:
                    conv1_1 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch1')[0]
                    conv1_2 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch1')[1]
                    conv1_3 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch1')[2]
                    tmp = compute_cl_dw_group(conv1_1, conv1_2)
                    tmp = compute_cl_dw_group_2(tmp, conv1_3)
                    conv1 = getattr(getattr(getattr(Snet.module, stage), str(layer)), 'branch1')[0]
                    if conv1.bias is not None:
                        conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
                    else:
                        conv1.weight.data = tmp['weight']

                    conv1_1 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[3]
                    conv1_2 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[4]
                    conv1_3 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[5]
                    tmp = compute_cl_dw_group(conv1_1, conv1_2)
                    tmp = compute_cl_dw_group_2(tmp, conv1_3)
                    conv1 = getattr(getattr(getattr(Snet.module, stage), str(layer)), 'branch2')[3]
                    if conv1.bias is not None:
                        conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
                    else:
                        conv1.weight.data = tmp['weight']
                else:
                    conv1_1 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[3]
                    conv1_2 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[4]
                    conv1_3 = getattr(getattr(getattr(Enet.module, stage), str(layer)), 'branch2')[5]
                    tmp = compute_cl_dw_group(conv1_1, conv1_2)
                    tmp = compute_cl_dw_group_2(tmp, conv1_3)
                    conv1 = getattr(getattr(getattr(Snet.module, stage), str(layer)), 'branch2')[3]
                    if conv1.bias is not None:
                        conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
                    else:
                        conv1.weight.data = tmp['weight']

    return Snet


def expandnet_contract(model, Emodel, args):

    if args.arch == 'mobilenetv1':
        new_model = from_expand_cl_to_mobilenetv1(model.cpu(), Emodel.cpu(), 14)
    elif args.arch == 'mobilenetv2':
        new_model = from_expand_cl_to_mobilenetv2(model.cpu(), Emodel.cpu(),
                                                                      [0, 2, 4, 7, 14])
    elif args.arch == 'shufflenet_v2_x0_5':
        new_model = from_expand_cl_to_shufflenet(model.cpu(), Emodel.cpu())
    else:
        raise KeyError('Network is not defined.')

    return new_model
