"""
MobileNetV1 and its ExpandNet-CL with er (expansion_rate) = 4
"""
import torch.nn as nn

__all__ = ['MobileNetV1', 'mobilenetv1',
           'MobileNetV1_Expand', 'mobilenetv1_expand',
           ]

# expansion rate
er = int(4)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

# exoand conv layer
def conv_bn_expand(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp*er, 1, 1, 1, bias=False),
        nn.Conv2d(inp*er, oup*er, 3, stride, 0, bias=False),
        nn.Conv2d(oup*er, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) #,
    )


def conv_dw_expand(inp, oup, stride):
    return nn.Sequential(
        # nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.Conv2d(inp, inp * er, 1, 1, 1, groups=inp, bias=False),
        nn.Conv2d(inp * er, inp * er, 3, stride, 0, groups=inp, bias=False),
        nn.Conv2d(inp * er, inp, 1, 1, 0, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        block = conv_dw
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            block(32, 64, 1),
            block(64, 128, 2),
            block(128, 128, 1),
            block(128, 256, 2),
            block(256, 256, 1),
            block(256, 512, 2),
            block(512, 512, 1),
            block(512, 512, 1),
            block(512, 512, 1),
            block(512, 512, 1),
            block(512, 512, 1),
            block(512, 1024, 2),
            block(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class MobileNetV1_Expand(nn.Module):
    def __init__(self):
        super(MobileNetV1_Expand, self).__init__()
        block = conv_dw_expand
        self.model = nn.Sequential(
            conv_bn_expand(3, 32, 2),
            block(32, 64, 1),
            block(64, 128, 2),
            block(128, 128, 1),
            block(128, 256, 2),
            block(256, 256, 1),
            block(256, 512, 2),
            block(512, 512, 1),
            block(512, 512, 1),
            block(512, 512, 1),
            block(512, 512, 1),
            block(512, 512, 1),
            block(512, 1024, 2),
            block(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def mobilenetv1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV1(**kwargs)
    if pretrained:
        raise Exception('No pretrained url for expandnet')
    return model


def mobilenetv1_expand(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV1_Expand(**kwargs)
    if pretrained:
        raise Exception('No pretrained url for expandnet')
    return model

