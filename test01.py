import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torch

# img_path = r"C:\Users\liewei\Desktop\img_celeba\outputs\attachments/1.png"
# img_path = r"../unet\save_img\134_epoch.jpg"
# img_p = Image.open(img_path).convert("L")#.convert("P")#.convert("RGB").convert("P")
# print(img_p.mode)
# print(np.unique(img_p))
# plt.subplot(1,2,1),plt.imshow(img_p)
# plt.show()

# print("======================================")
# # a = np.random.uniform(0,255,(112,112))
# # a = np.ones((112,112))*112
# a = np.uint8(img_p)
# # a[a==1] = 128
# img = Image.fromarray(a).convert("P")
# print(img.mode)
# print(np.unique(img))
# plt.subplot(1,2,2),plt.imshow(img)
# plt.show()

# bce_loss = nn.BCELoss()
# # bce_loss = nn.CrossEntropyLoss()
# a = torch.rand(416,416)
# b = torch.randint(0,21,(416,416)).float()
# b = torch.FloatTensor(b)
# print(a==a)

# img_path = r"../Dataset/SegmentationClass/2007_000033.png"
# img = Image.open(img_path).convert("RGB")
# print(np.array(img).shape)
# plt.subplot(1,2,1),plt.imshow(img)
# print(np.unique(img))
# # print(bce_loss(a,b))
# a = np.zeros((416,416,3))
# a[:,:,0] = 128
# plt.subplot(1,2,2),plt.imshow(a)
# plt.show()
# a = np.random.randn(2,3)
# a[:,0]


# img_path = r"../Dataset/SegmentationClass/2007_000033.png"
# mask_img_ = Image.open(img_path)
# mask_img_.thumbnail((416, 416))
# mask_img = mask_img_.crop((0, 0, 416 + 0, 416 + 0))  # 裁剪
# mask_img = np.array(mask_img)
# mask_img[mask_img==255] = 0
#
# a = torch.rand(1,416,416)
# b = torch.zeros((20,416,416))
# c = torch.cat((a,b),dim=0)
# loss = nn.CrossEntropyLoss()
# print(f"loss:{loss(c.unsqueeze(0),torch.LongTensor(mask_img).unsqueeze(0))}")
#
# print(c.shape)
# mask_out = torch.argmax(c,dim=0)
# print(mask_out.shape,mask_img.shape)
# accuracy = torch.sum(torch.eq(mask_out,torch.tensor(mask_img)))/mask_img.size
# print(accuracy)
# a = torch.rand(3,4)
# print(a.shape[0]*a.shape[1])
# from matplotlib import cm
# from matplotlib.colors import LinearSegmentedColormap
#
# data = np.random.randint(0,255,(416,416,3))
#
# # 定义colorbar的颜色帧
# color_list = ['#0000FF', '#3333FF', '#00FF33', '#FFFF33', '#FF9900', '#FF0000', '#FF00FF', '#6600CC']
#
# # 线性补帧，并定义自定义colormap的名字，此处为rain
# my_cmap = LinearSegmentedColormap.from_list('rain', color_list)
#
# # 注册自定义的cmap，此后可以像使用内置的colormap一样使用自定义的rain
# cm.register_cmap(cmap=my_cmap)
#
# plt.imshow(data, vmin=0, vmax=10, cmap="rain")  # data是你要可视化的矩阵，使用时直接使用rain这个名字
# plt.show()

# from PIL import Image
# import os
#
# images_path = r"../data/SegmentationClass"
# for img_name in os.listdir(images_path):
#     print(img_name)
#     # img_path = os.path.join(images_path,img_name)
#     img_path = os.path.join(images_path,"111.9.162.61_20.50_40.66.png")
#     img = Image.open(img_path)
#     img_1 = img.copy()
#     img_1.thumbnail((224,224),resample=Image.NEAREST)
#     # img_1.thumbnail((224,224),resample=Image.AFFINE)
#     # img_1.thumbnail((224,224),resample=Image.USE_CFFI_ACCESS)
#     print(np.unique(np.array(img)))
#     plt.subplot(1,2,1),plt.imshow(img)
#     plt.subplot(1,2,2),plt.imshow(img_1)
#     plt.show()

# a = np.array([[1,2,3],[5],[8,9,6,5]])
# print([len(i)for i in a])
# print(a[np.argmax([len(i)for i in a])])
# print(np.argmax([3, 1, 4]))
import pandas as pd

# a = np.array([[140.7793, 49.648273], [120.50344, 47.958614], [121.06207, 41.25517], [141.33792, 42.944828]])
# a = pd.DataFrame(a)
# print(a)
# b = a.sort_values(by=0)
# # print(b)
# # print(b.iloc[0:2, :])
# # print(b.iloc[0:2, :].sort_values(by=1))
# c = b.iloc[:2, :].sort_values(by=1)
# d = b.iloc[2:, :].sort_values(by=1)
# # b = np.argsort(a,axis=0)
# # print(b)
# # print(c)
# # print(d)
#
# e = pd.concat([c.iloc[0:1,:],d,c.iloc[1:,:]],axis=0)
# print(np.array(e))
# b[0:2,:] = np.sort(b[0:2,:],axis=0)
# print(b)

# import torch
# import time
#
# a = torch.randint(0, 1, (416, 416), dtype=torch.int64)
# while True:
#     t1 = time.time()
#     a = a.cuda()
#     print("time:", time.time() - t1)
#     a = a.cpu()
#     print("time:", time.time() - t1)
#     print("===========================")
# torch.manual_seed(0)
# ce_loss = nn.CrossEntropyLoss()
#
# mask_out_reshape = torch.rand((1,3))
# mask_img_reshape = torch.tensor([2])
#
# print(mask_out_reshape)
# print(mask_img_reshape.dtype)  # torch.int64
#
# loss1 = ce_loss(mask_out_reshape, mask_img_reshape)
#
# print(loss1)

# a = np.random.randint(0, 5, (1, 2, 3))

# print(np.unique(a))

# a = np.array([1,2,3,4])
# b = a[::-1]
# print(b)
# torch.manual_seed(0)
# a = torch.randn((1, 6, 10, 10))
# print(torch.argmax(a.reshape(1, 6, -1), dim=2) // 10, torch.argmax(a.reshape(1, 6, -1), dim=2) % 10)
# b = torch.cat([torch.argmax(a.reshape(1, 6, -1), dim=2).T // 10, torch.argmax(a.reshape(1, 6, -1), dim=2).T % 10],
#               dim=1)
# print(b)
# print(b.shape)
# import os
# load_params = r"D:\my_program\study\heatmap_point\weights\u2net_p_weight\u2net_p_196_epoch.pth"
# onnx_name = os.path.basename(load_params).split(".")[0]+".onnx"
# print(onnx_name)

# print(np.array([3]).dtype)

import torch
import torch.nn.functional as F

a = torch.randn((4,6,416,416))
b =  F.interpolate(a, size=(224, 224), mode='nearest')
print(b.shape)

