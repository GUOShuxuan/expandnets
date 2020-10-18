from .mobilenetv2 import *
from .mobilenetv1 import *
from .shufflenet import *


def get_model(arch):

    # MobileNet
    if arch == 'mobilenetv1':
        model = mobilenetv1()
    elif arch == 'mobilenetv1_expand':
        model = mobilenetv1_expand()

    # MobileNetV2
    elif arch == 'mobilenetv2':
        model = mobilenetv2()
    elif arch == 'mobilenetv2_expand':
        model = mobilenetv2_expand()

    # ShuffleNet
    elif arch == 'shufflenet_v2_x0_5':
        model = shufflenet_v2_x0_5()
    elif arch == 'shufflenet_v2_x0_5_expand':
        model = shufflenet_v2_x0_5_expand()

    else:
        raise KeyError('Network is not defined')

    return model

