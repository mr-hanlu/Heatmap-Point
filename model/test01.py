import torch
from torchvision.models import resnet50
import torchsummary
from torch import nn

net = resnet50()
imgsize = max(512, 512)
scale = 4

data = torch.rand((1, 3, imgsize, imgsize))
for i, (name, layer) in enumerate(net.named_children()):
    print("---->", i)
    # print(name, layer)
    # print(data.shape)
    data = layer(data)
    out_shape = data.shape

    # print(out_shape)
    # exit()
    #
    # # print(list(layer.named_parameters()))
    # layer_ = list(layer.named_parameters())
    # # if len(layer_)>0:
    # #     layer_shape = layer_[-1][1].shape
    #
    # layer_shape = layer_[0][1].shape if len(layer_) > 0 else [0]

    if (imgsize / (scale*2)) in out_shape[2:]:
        print(i)
        # exit()
        model = nn.Sequential(*list(net.children())[:i])  # 解析出来切片选取层  到下采样128
        break



torchsummary.summary(model, (3, 512, 512), 1, device='cpu')
