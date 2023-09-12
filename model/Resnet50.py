import torch
from torchvision.models import resnet50
from torch import nn
import torch.nn.functional as F


def Resnet50(out_channels):
    net = resnet50()

    net = nn.Sequential(*list(net.children())[:5])  # 解析出来切片选取层  到下采样128
    net[4][2] = nn.Sequential(*list(net[4][2].children())[:6])  # 去掉 BN ReLU
    net[4][2][4] = nn.Conv2d(64, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)  # 修改输出通道
    # net[4][2][5] = F.interpolate(out, size=(mask_img.shape[2], mask_img.shape[3]), mode='bilinear')
    net[4][2][5] = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    # print(net[4][2])

    return net


if __name__ == '__main__':
    from torchsummary import summary

    net = Resnet50(out_channels=6)
    data = torch.randn((4, 3, 224, 224))
    out = net(data)
    print(out.shape)
    summary(net, (3, 224, 224), 1, device='cpu')
