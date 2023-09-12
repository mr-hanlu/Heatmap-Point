from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from Dataset import config
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MyDataset(Dataset):
    def __init__(self, data_path, transf):
        self.transforms = transf
        self.dataset = []
        masks_path = os.path.join(data_path, "SegmentationClass")  # 分割掩码总路径
        images_path = os.path.join(data_path, "JPEGImages")  # 图片总路径
        self.zero_img = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8)
        self.zero_mask = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, len(config.classes_list)))
        for mask_name in os.listdir(masks_path):
            # print(mask_name)
            # name = mask_name.split(".")[0]
            mask_path = os.path.join(masks_path, mask_name)
            image_path = os.path.join(images_path, mask_name.split(".")[0] + ".jpg")
            self.dataset.append([mask_path, image_path])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        mask_path, image_path = self.dataset[index]
        with open(mask_path, 'rb') as f:
            mask_img = pickle.load(f)
        real_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)  # h,w,c

        scale_min = min(config.IMG_HEIGHT / real_img.shape[0], config.IMG_WIDTH / real_img.shape[1])

        real_img_ = cv2.resize(real_img, (0, 0), fx=scale_min, fy=scale_min)
        real_img = self.zero_img.copy()
        real_img[:real_img_.shape[0], :real_img_.shape[1], :] = real_img_
        # print(real_img.shape)  # (224, 224, 3)

        mask_img_ = cv2.resize(mask_img, (0, 0), fx=scale_min, fy=scale_min)  # (171, 224, 6)
        mask_img = self.zero_mask.copy()
        mask_img[:mask_img_.shape[0], :mask_img_.shape[1], :] = mask_img_
        mask_img = mask_img.transpose([2, 0, 1])  # h,w,c-->c,h,w
        # print(mask_img.shape)  # (224, 224, 6)

        mask_img = torch.Tensor(np.array(mask_img))  # 转 tensor  [224, 224]
        # mask_img = self.transforms(mask_img)  # 转 tensor  [1, 224, 224]
        # print(np.uint8(np.unique(np.array(mask_img))*255))
        real_img = self.transforms(real_img)

        return mask_img, real_img


if __name__ == '__main__':
    data_path = r"../data"
    transforms = transforms.Compose([
        transforms.ToTensor(),  # 0-1
        # transforms.Normalize(mean=[0.5,], std=[0.5,])
    ])
    data = MyDataset(data_path, transforms)
    dataload = DataLoader(data, 1, shuffle=True)
    for mask_img, real_img in dataload:
        print(mask_img.shape)  # [1, 6, 224, 224]
        print(real_img.shape)  # [1, 3, 224, 224]
        # print(real_img)
        # print(torch.max(mask_img))
        # exit()
        plt.subplot(1, 2, 1), plt.imshow(np.max(mask_img[0].numpy(), axis=0), "gray")
        plt.subplot(1, 2, 2), plt.imshow(np.uint8(real_img[0].numpy().transpose([1, 2, 0]) * 255))
        plt.show()
        exit()

        out_img = torch.cat([mask_img, mask_img], dim=3)
        print(out_img.shape)

        save_image(out_img, r"out.jpg", nrow=1, padding=2, pad_value=255)

        exit()
        real_img_ = make_grid(real_img, nrow=2, padding=2, pad_value=255).numpy()
        mask_img_ = make_grid(mask_img, nrow=2, padding=2, pad_value=255).numpy()
        # print(mask_img_.shape)
        # print(np.max(mask_img_))
        # print(real_img_.shape)
        # print(np.max(real_img_))
        img1 = np.transpose(real_img_, (1, 2, 0))
        img2 = np.transpose(mask_img_, (1, 2, 0))
        # print(img1.shape)
        # print(img2.shape)
        # print(np.unique(np.array(img2)))  # 统计数值
        plt.subplot(2, 1, 1), plt.imshow(img1)
        plt.subplot(2, 1, 2), plt.imshow(img2)
        plt.show()
