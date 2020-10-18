"""
SmallNet with 7*7 kernels
And its ExpandNets
"""

import torch.nn as nn
import torch.nn.functional as F


# Baseline model: SmallNet 7*7
class Cifar_Tiny(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, padding=3)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, padding=3)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=7, padding=3)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# ExpandNet-CL with er=4
class Cifar_Tiny_ExpandNet_cl(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny_ExpandNet_cl, self).__init__()
        a = int(4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, padding=3),
                nn.Conv2d(3, 8*a, kernel_size=7),
                nn.Conv2d(8*a, 8, kernel_size=1)
            )
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
                nn.Conv2d(8, 8*a, kernel_size=1, padding=3),
                nn.Conv2d(8*a, 16*a, kernel_size=7),
                nn.Conv2d(16*a, 16, kernel_size=1)
        )
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Sequential(
                nn.Conv2d(16, 16*a, kernel_size=1, padding=3),
                nn.Conv2d(16*a, 32*a, kernel_size=7),
                nn.Conv2d(32*a, 32, kernel_size=1)
            )
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# ExpandNet-CL+FC with er=4
class Cifar_Tiny_ExpandNet_cl_fc(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny_ExpandNet_cl_fc, self).__init__()
        a = int(4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, padding=3),
                nn.Conv2d(3, 8*a, kernel_size=7),
                nn.Conv2d(8*a, 8, kernel_size=1)
            )
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
                nn.Conv2d(8, 8*a, kernel_size=1, padding=3),
                nn.Conv2d(8*a, 16*a, kernel_size=7),
                nn.Conv2d(16*a, 16, kernel_size=1)
        )
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Sequential(
                nn.Conv2d(16, 16*a, kernel_size=1, padding=3),
                nn.Conv2d(16*a, 32*a, kernel_size=7),
                nn.Conv2d(32*a, 32, kernel_size=1)
            )
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Sequential(
                nn.Linear(in_features=512, out_features=512*a),
                nn.Linear(512*a, 64*a),
                nn.Linear(in_features=64*a, out_features=64)
                )
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# FC(Arora18): ExpandNet-FC with er=4
class Cifar_Tiny_ExpandNet_fc(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny_ExpandNet_fc, self).__init__()
        a = int(4)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, padding=3)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, padding=3)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=7, padding=3)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Sequential(
                nn.Linear(in_features=512, out_features=512*a),
                nn.Linear(512*a, 64*a),
                nn.Linear(in_features=64*a, out_features=64)
                )
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# ExpandNet-CK with er=4
class Cifar_Tiny_ExpandNet_ck(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny_ExpandNet_ck, self).__init__()
        a = int(4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=3),
                nn.Conv2d(3, 8*a, kernel_size=3),
                nn.Conv2d(8*a, 8, kernel_size=3)
            )
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
                nn.Conv2d(8, 8*a, kernel_size=3, padding=3),
                nn.Conv2d(8*a, 16*a, kernel_size=3),
                nn.Conv2d(16*a, 16, kernel_size=3)
        )
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Sequential(
                nn.Conv2d(16, 16*a, kernel_size=3, padding=3),
                nn.Conv2d(16*a, 32*a, kernel_size=3),
                nn.Conv2d(32*a, 32, kernel_size=3)
            )
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# ExpandNet-CK+FC with er=4
class Cifar_Tiny_ExpandNet_ck_fc(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny_ExpandNet_ck_fc, self).__init__()
        a = int(4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=3),
            nn.Conv2d(3, 8 * a, kernel_size=3),
            nn.Conv2d(8 * a, 8, kernel_size=3)
        )
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8 * a, kernel_size=3, padding=3),
            nn.Conv2d(8 * a, 16 * a, kernel_size=3),
            nn.Conv2d(16 * a, 16, kernel_size=3)
        )
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16 * a, kernel_size=3, padding=3),
            nn.Conv2d(16 * a, 32 * a, kernel_size=3),
            nn.Conv2d(32 * a, 32, kernel_size=3)
        )
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Sequential(
                nn.Linear(in_features=512, out_features=512*a),
                nn.Linear(512*a, 64*a),
                nn.Linear(in_features=64*a, out_features=64)
                )
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# Nonlinear ExpandNet-CK+FC with er=4
class Cifar_Tiny_ExpandNet_ck_fc_Nonlinear(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny_ExpandNet_ck_fc_Nonlinear, self).__init__()
        a = int(4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=3),
            nn.Conv2d(3, 8 * a, kernel_size=3),
            nn.Conv2d(8 * a, 8, kernel_size=3)
        )
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8 * a, kernel_size=3, padding=3),
            nn.Conv2d(8 * a, 16 * a, kernel_size=3),
            nn.Conv2d(16 * a, 16, kernel_size=3)
        )
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16 * a, kernel_size=3, padding=3),
            nn.Conv2d(16 * a, 32 * a, kernel_size=3),
            nn.Conv2d(32 * a, 32, kernel_size=3)
        )
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512*a),
            nn.Linear(512*a, 64*a),
            nn.Linear(in_features=64*a, out_features=64)
        )
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        out = F.relu(self.conv1_bn(self.conv1[2](F.relu(self.conv1[1](F.relu(self.conv1[0](x)))))))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2[2](F.relu(self.conv2[1](F.relu(self.conv2[0](out)))))))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3[2](F.relu(self.conv3[1](F.relu(self.conv3[0](out)))))))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1[2](F.relu(self.fc1[1](F.relu(self.fc1[0](out))))))
        out = self.fc2(out)

        return out


# Nonlinear ExpandNet-CL+FC with er=4
class Cifar_Tiny_ExpandNet_cl_fc_Nonlinear(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny_ExpandNet_cl_fc_Nonlinear, self).__init__()
        a = int(4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, padding=3),
                nn.Conv2d(3, 8*a, kernel_size=7),
                nn.Conv2d(8*a, 8, kernel_size=1)
            )
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
                nn.Conv2d(8, 8*a, kernel_size=1, padding=3),
                nn.Conv2d(8*a, 16*a, kernel_size=7),
                nn.Conv2d(16*a, 16, kernel_size=1)
        )
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Sequential(
                nn.Conv2d(16, 16*a, kernel_size=1, padding=3),
                nn.Conv2d(16*a, 32*a, kernel_size=7),
                nn.Conv2d(32*a, 32, kernel_size=1)
            )
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512*a),
            nn.Linear(512*a, 64*a),
            nn.Linear(in_features=64*a, out_features=64)
        )
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        out = F.relu(self.conv1_bn(self.conv1[2](F.relu(self.conv1[1](F.relu(self.conv1[0](x)))))))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2[2](F.relu(self.conv2[1](F.relu(self.conv2[0](out)))))))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3[2](F.relu(self.conv3[1](F.relu(self.conv3[0](out)))))))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1[2](F.relu(self.fc1[1](F.relu(self.fc1[0](out))))))
        out = self.fc2(out)

        return out


