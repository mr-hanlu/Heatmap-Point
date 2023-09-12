import torch
import torch.nn as nn
from model.layers import unetConv2
from model.init_weights import init_weights

"""
网络模型的发展，由简单到难，再到简单的过程
"""


class UNet3Plus(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet3Plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.bilinear = bilinear
        self.is_batchnorm = is_batchnorm
        # 编码结构中的输出通道0-4总共5个
        filters = [64, 128, 256, 512, 1024]

        # ----------------Encoder------------------
        self.conv1 = unetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # ----------------Decoder------------------
        self.CatChannels = filters[0]  # 64  用于cat的通道数
        self.CatBlocks = 5  # 5 总共有5个层cat到一起
        self.UpChannels = self.CatChannels * self.CatBlocks  # 64*5=320

        'stage 4d'
        # h1:320*320*64->hd4:64*40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.Sequential(
            nn.MaxPool2d(8, 8, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # h2:160*160*128->hd4:64*40*40, Pooling 4 times  输入的编码器的h2
        self.h2_PT_hd4 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(filters[1], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # h3:80*80*256->hd4:64*40*40, Pooling 2 times  输入的编码器的h3
        self.h3_PT_hd4 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[2], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # h4:40*40*512->hd4:64*40*40
        self.h4_Cat_hd4_conv = nn.Sequential(
            nn.Conv2d(filters[3], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # h5d:20*20*1024->hd4:40*40*64, Upsample 2 times
        self.hd5_UT_hd4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(filters[4], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4) 联合
        self.conv4d_1 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.ReLU(inplace=True),
        )

        'stage 3d'
        # h1:320*320*64->hd3:80*80*64, Pooling 4 times
        self.h1_PT_hd3 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # h2:160*160*128->hd3:80*80*64
        self.h2_PT_hd3 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[1], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # h3:80*80*256->hd3:80*80*64
        self.h3_Cat_hd3_conv = nn.Sequential(
            nn.Conv2d(filters[2], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd4:40*40*320->hd3:80*80*64
        self.hd4_UT_hd3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd5:20*20*1024->hd4:80*80*64
        self.hd5_UT_hd3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(filters[4], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # fusion()
        self.conv3d_1 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.ReLU(inplace=True)
        )

        'stage 2d'
        # h1:320*320*64->hd2:160*160*320, Pooling 2 times
        self.h1_PT_hd2 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # h2:160*160*128->hd2:160*160*320, Concatenation
        self.h2_Cat_hd2_conv = nn.Sequential(
            nn.Conv2d(filters[1], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd3:80*80*320, hd2->160*160*320, Upsample 2 times
        self.hd3_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd4:40*40*320->hd2:160*160*320, Upsample 4 times
        self.hd4_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd5:20*20*1024->hd2:160*160*320, Upsample 8 times
        self.hd5_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(filters[4], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.ReLU(inplace=True)
        )

        'stage 1d'
        # h1:320*320*64->hd1:320*320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Sequential(
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd2:160*160*320->hd1:320*320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 14*14
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd3:80*80*320->hd1:320*320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 14*14
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd4:40*40*320->hd1:320*320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),  # 14*14
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # hd5:20*20*1024->hd1:320*320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),  # 14*14
            nn.Conv2d(filters[4], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),  # 16
            nn.BatchNorm2d(self.UpChannels),
            nn.ReLU(inplace=True)
        )

        # 多少分类输出通道就是多少
        self.outconv = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # ----------------Encoder------------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        # ----------------Decoder------------------
        h1_PT_hd4 = self.h1_PT_hd4(h1)
        h2_PT_hd4 = self.h2_PT_hd4(h2)
        h3_PT_hd4 = self.h3_PT_hd4(h3)
        h4_Cat_hd4 = self.h4_Cat_hd4_conv(h4)
        h5_UT_hd4 = self.hd5_UT_hd4(hd5)
        hd4 = self.conv4d_1(torch.cat([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, h5_UT_hd4], dim=1))  # 320, 40, 40

        h1_PT_hd3 = self.h1_PT_hd3(h1)
        h2_PT_hd3 = self.h2_PT_hd3(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3_conv(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3(hd4)
        hd5_UT_hd3 = self.hd5_UT_hd3(hd5)
        hd3 = self.conv3d_1(
            torch.cat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3], dim=1)
        )  # 320, 80, 80

        h1_PT_hd2 = self.h1_PT_hd2(h1)
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2(hd3)
        hd4_UT_hd2 = self.hd4_UT_hd2(hd4)
        hd5_UT_hd2 = self.hd5_UT_hd2(hd5)
        hd2 = self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), dim=1)
        )  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_conv(h1)
        hd2_UT_hd1 = self.hd2_UT_hd1(hd2)
        hd3_UT_hd1 = self.hd3_UT_hd1(hd3)
        hd4_UT_hd1 = self.hd4_UT_hd1(hd4)
        hd5_UT_hd1 = self.hd5_UT_hd1(hd5)
        hd1 = self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)
        )  # hd1->320*320*UpChannels
        d1 = self.outconv(hd1)
        # return torch.sigmoid(d1)
        return d1


if __name__ == '__main__':
    from torchsummary import summary
    from torchstat import stat

    # x = torch.randn(2, 3, 224, 224)
    net = UNet3Plus(n_channels=3, n_classes=4)
    # y = net(x)
    # print(y.shape)

    # 查看网络层形状、参数，运算量、内存占用情况
    stat(net.cpu(),(3, 224, 224))  # 152.0GFlops

    # summary(net, (3, 224, 224), device="cpu")
    # print(net)
