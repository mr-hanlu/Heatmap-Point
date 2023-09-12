import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNLayer(nn.Module):
    def __init__(self, C_in, C_out, C_mid=None):
        super(CNNLayer, self).__init__()
        if not C_mid:
            C_mid = C_out
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_mid, 3, 1, 1),
            nn.BatchNorm2d(C_mid),
            # nn.Dropout(0.3),
            nn.ReLU(inplace=True),

            nn.Conv2d(C_mid, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # nn.Dropout(0.4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.layer(x)
        return y


class Down(nn.Module):
    def __init__(self, C_in, C_out):
        super(Down, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            CNNLayer(C_in, C_out)
        )

    def forward(self, x):
        y = self.layer(x)
        return y


class Up(nn.Module):
    def __init__(self, C_in, C_out):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(C_in, C_in // 2, 3, 1, 1)
        )
        self.conv = CNNLayer(C_in, C_out, C_in // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.shape,x2.shape)  # torch.Size([2, 256, 56, 56]) torch.Size([2, 256, 57, 57])
        x = torch.cat((x1, x2), dim=1)
        y = self.conv(x)
        return y


class MainNet(nn.Module):  # 416
    def __init__(self, C_in, C_out):
        super(MainNet, self).__init__()
        self.c1 = CNNLayer(C_in, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        self.d3 = Down(256, 512)
        self.d4 = Down(512, 1024)
        self.u1 = Up(1024, 512)
        self.u2 = Up(512, 256)
        self.u3 = Up(256, 128)
        self.u4 = Up(128, 64)
        self.outconv = nn.Conv2d(64, C_out, 3, 1, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        out = self.outconv(x)
        # out = torch.sigmoid(out)  # 二分类交叉熵
        return out


if __name__ == '__main__':
    from torchsummary import summary

    a = torch.randn(2, 3, 416, 416)
    net = MainNet(C_in=3, C_out=4)
    print(net)
    print(net.c1.layer[0].weight.grad)
    # summary(net, (3, 416, 416),device="cpu")
    # load_params = f"./weight/unet_{356}_epoch.pth"
    # net.load_state_dict(torch.load(load_params, map_location=torch.device('cpu')))
    # model_dict = net.state_dict()
    # prete = torch.load(load_params)
    # print(prete)
    # print(model_dict)
    # out = net(a)
    # print(out.shape)  # [2, 4, 416, 416]  n,c,h,w
