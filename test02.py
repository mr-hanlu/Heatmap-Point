from torchvision import transforms
import numpy as np
# from PIL
import torch
from torch import nn
from torchvision import models
from torchsummary import summary

net = models.resnet50()

summary(net, (3, 416, 416), device="cpu")




