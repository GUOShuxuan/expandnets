'''
Code here is same as it in the supplementary, which can be simply run.
'''

import torch
import torch.nn as nn

# Expansion and contraction of a standard convolutional layer

m = 3  # input channels
n = 8  # output channels
k = 5  # kernel size
r = int(4)  # expansion rate
imgs = torch.randn((8, m, 7, 7))  # input images with batch size as 8

# original standard convolutional layer
F = nn.Conv2d(m, n, k)

# Expand-CL with r
F1 = nn.Conv2d(m, r*m, 1)
F2 = nn.Conv2d(r*m, r*n, k)
F3 = nn.Conv2d(r*n, n, 1)

# contracting
from exp_cifar.utils.compute_new_weights \
    import compute_cl, compute_cl_2
tmp = compute_cl(F1, F2)
tmp = compute_cl_2(tmp, F3)
F.weight.data, F.bias.data = tmp['weight'], tmp['bias']

# test
res_cl = F3(F2(F1(imgs)))
res_F = F(imgs)
print('Contract from Expand-CL: %.7f' % (res_cl-res_F).sum())  # <10^-5

# Expand-CK
# k = 5, l=2
F1 = nn.Conv2d(m, r*m, 3)
F2 = nn.Conv2d(r*m, n, 3)

# contracting
from exp_cifar.utils.compute_new_weights import compute_ck
tmp = compute_ck(F1, F2)
F.weight.data = tmp['weight']
F.bias.data = tmp['bias']

# test
res_ck = F2(F1(imgs))
res_F = F(imgs)
print('Contract from Expand-CK: %.7f' % (res_ck-res_F).sum())  # <10^-5


# Expansion and contraction of a depthwise convolutional layer
# for depthwise conv, input channels=out channels
m = 4  # input channels
n = 4  # output channels
k = 3  # kernel size
r = int(4)  # expansion rate
imgs = torch.randn((8, m, 7, 7))

# original depthwise convolutional layer
F = nn.Conv2d(m, n, k, groups=m, bias=False)

# Expand-CL with r
F1 = nn.Conv2d(m, r*m, 1, groups=m, bias=False)
F2 = nn.Conv2d(r*m, r*m, k, groups=m, bias=False)
F3 = nn.Conv2d(r*m, n, 1, groups=m, bias=False)

# contracting
from exp_imagenet.utils.compute_new_weights \
    import compute_cl_dw_group, compute_cl_dw_group_2

tmp = compute_cl_dw_group(F1, F2)
tmp = compute_cl_dw_group_2(tmp, F3)

F.weight.data = tmp['weight']

# test
res_depthwise_cl = F3(F2(F1(imgs)))
res_F = F(imgs)
print('Contract from depthwise Expand-CL: %.7f' % (res_depthwise_cl-res_F).sum())  # <10^-5