from torch.nn import init
import torch
import torch.nn as nn


def weights_init_normal(m):
    class_name = m.__class__.__name__
    print(class_name)
    if class_name.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.xavier_normal_(m.weights.data, gain=1)
    elif class_name.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif class_name.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_name.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_name.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):  # 正交
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif class_name.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif class_name.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)

