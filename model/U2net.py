import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# 定义卷积块
class ReBNConv(nn.Module):  # ReBNConv

    def __init__(self, in_ch=3, out_ch=3, dirate=1):  # dirate膨胀率
        super(ReBNConv, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout


# 定义上采样过程
def _upsample_like(src, tar, scale=2):
    # 对HW进行上采样
    # src = F.upsample(src, size=tar.shape[2:], mode='bilinear', align_corners=False)
    src = F.interpolate(src, scale_factor=scale, mode='bilinear', align_corners=False)
    return src


# RSU-7 #
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = ReBNConv(in_ch, out_ch, dirate=1)
        self.rebnconv1 = ReBNConv(out_ch, mid_ch, dirate=1)

        # down
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv2 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv3 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv4 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv5 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = ReBNConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = ReBNConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ReBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)  # [2, 64, 224, 224]
        hx1 = self.rebnconv1(hxin)  # [2, 32, 224, 224]
        hx = self.pool1(hx1)  # [2, 32, 112, 112]
        hx2 = self.rebnconv2(hx)  # [2, 32, 56, 56]

        hx = self.pool2(hx2)  # [2, 32, 56, 56]
        hx3 = self.rebnconv3(hx)  # [2, 32, 56, 56]

        hx = self.pool3(hx3)  # [2, 32, 28, 28]
        hx4 = self.rebnconv4(hx)  # [2, 32, 28, 28]

        hx = self.pool4(hx4)  # [2, 32, 14, 14]
        hx5 = self.rebnconv5(hx)  # [2, 32, 14, 14]

        hx = self.pool5(hx5)  # [2, 32, 7, 7]
        hx6 = self.rebnconv6(hx)  # [2, 32, 7, 7]

        hx7 = self.rebnconv7(hx6)  # [2, 32, 7, 7]

        # RSU7

        hx6d = self.rebnconv6d(torch.cat([hx7, hx6], dim=1))  # [2, 32, 7, 7]
        hx6dup = _upsample_like(hx6d, hx5)  # [2, 32, 14, 14]

        hx5d = self.rebnconv5d(torch.cat([hx6dup, hx5], dim=1))  # [2, 32, 14, 14]
        hx5dup = _upsample_like(hx5d, hx4)  # [2, 32, 28, 28]

        hx4d = self.rebnconv4d(torch.cat([hx5dup, hx4], dim=1))  # [2, 32, 28, 28]
        hx4dup = _upsample_like(hx4d, hx3)  # [2, 32, 56, 56]

        hx3d = self.rebnconv3d(torch.cat([hx4dup, hx3], dim=1))  # [2, 32, 56, 56]
        hx3dup = _upsample_like(hx3d, hx2)  # [2, 32, 112, 112]

        hx2d = self.rebnconv2d(torch.cat([hx3dup, hx2], dim=1))  # [2, 32, 112, 112]
        hx2dup = _upsample_like(hx2d, hx1)  # [2, 32, 224, 224]

        hx1d = self.rebnconv1d(torch.cat([hx2dup, hx1], dim=1))  # [2, 64, 224, 224]

        return hx1d + hxin


# RSU-6 #
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = ReBNConv(in_ch, out_ch, dirate=1)
        self.rebnconv1 = ReBNConv(out_ch, mid_ch, dirate=1)

        # down
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv2 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv3 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv4 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv5 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.rebnconv6 = ReBNConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv5d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ReBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)

        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx)

        # RSU6

        hx5d = self.rebnconv5d(torch.cat([hx6, hx5], dim=1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat([hx5dup, hx4], dim=1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat([hx4dup, hx3], dim=1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat([hx3dup, hx2], dim=1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat([hx2dup, hx1], dim=1))

        return hx1d + hxin


# RSU-5 #
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = ReBNConv(in_ch, out_ch, dirate=1)
        self.rebnconv1 = ReBNConv(out_ch, mid_ch, dirate=1)

        # down
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv2 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv3 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv4 = ReBNConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = ReBNConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv4d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ReBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx)
        # RSU5

        hx4d = self.rebnconv4d(torch.cat([hx5, hx4], dim=1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat([hx4dup, hx3], dim=1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat([hx3dup, hx2], dim=1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat([hx2dup, hx1], dim=1))

        return hx1d + hxin


# RSU-4 #
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = ReBNConv(in_ch, out_ch, dirate=1)
        self.rebnconv1 = ReBNConv(out_ch, mid_ch, dirate=1)

        # down
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv2 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.rebnconv3 = ReBNConv(mid_ch, mid_ch, dirate=1)

        # down
        self.rebnconv4 = ReBNConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv3d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ReBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ReBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx)

        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], dim=1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat([hx3dup, hx2], dim=1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat([hx2dup, hx1], dim=1))

        return hx1d + hxin


# RSU-4F #
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = ReBNConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ReBNConv(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = ReBNConv(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = ReBNConv(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = ReBNConv(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = ReBNConv(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = ReBNConv(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = ReBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv2(hx2)

        hx4 = self.rebnconv2(hx3)

        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], dim=1))
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], dim=1))
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], dim=1))

        return hx1d + hxin


class U2NET(nn.Module):

    def __init__(self, in_ch=3, out_ch=3):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # side 从上采样最快的开始，输入通道就是decoder的输出通道
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x
        # stage 1
        hx1 = self.stage1(hx)  # [2, 64, 224, 224]
        hx = self.pool12(hx1)  # [2, 64, 112, 112]

        # stage 2
        hx2 = self.stage2(hx)  # [2, 128, 112, 112]
        hx = self.pool23(hx2)  # [2, 128, 56, 56]
        # stage 3
        hx3 = self.stage3(hx)  # [2, 256, 56, 56]
        hx = self.pool34(hx3)  # [2, 256, 28, 28]

        # stage 4
        hx4 = self.stage4(hx)  # [2, 512, 28, 28]
        hx = self.pool45(hx4)  # [2, 512, 14, 14]

        # stage 5
        hx5 = self.stage5(hx)  # [2, 512, 14, 14]
        hx = self.pool56(hx5)  # [2, 512, 7, 7]

        # stage 6
        hx6 = self.stage6(hx)  # [2, 512, 7, 7]
        hx6up = _upsample_like(hx6, hx5)  # [2, 512, 14, 14]

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat([hx6up, hx5], dim=1))  # [2, 512, 14, 14]
        hx5dup = _upsample_like(hx5d, hx4)  # [2, 512, 28, 28]

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))  # [2, 256, 28, 28]
        hx4dup = _upsample_like(hx4d, hx3)  # [2, 256, 56, 56]

        hx3d = self.stage3d(torch.cat([hx4dup, hx3], 1))  # [2, 128, 56, 56]
        hx3dup = _upsample_like(hx3d, hx2)  # [2, 128, 112, 112]

        hx2d = self.stage2d(torch.cat([hx3dup, hx2], 1))  # [2, 128, 56, 56]
        hx2dup = _upsample_like(hx2d, hx1)  # [2, 64, 224, 224]

        hx1d = self.stage1d(torch.cat([hx2dup, hx1], dim=1))  # [2, 64, 224, 224]

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)  # [2, 1, 112, 112]
        d2 = _upsample_like(d2, d1, scale=2)  # [2, 1, 224, 224]

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1, scale=4)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1, scale=8)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1, scale=16)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1, scale=32)

        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))

        # d0 = torch.sigmoid(d0)
        # d1 = torch.sigmoid(d1)
        # d2 = torch.sigmoid(d2)
        # d3 = torch.sigmoid(d3)
        # d4 = torch.sigmoid(d4)
        # d5 = torch.sigmoid(d5)
        # d6 = torch.sigmoid(d6)

        return d0


class U2NETP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        return d0


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    net = U2NET(in_ch=3,out_ch=4)
    # print(net)
    # print(net.stage1.rebnconvin.conv_s1.weight.grad)
    y = net(x)
    print(y.shape)
