import os
import cv2
import numpy as np
from model.U2net_p import U2NETP
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Detect():
    def __init__(self, load_params, classes_list, IMG_HEIGHT, IMG_WIDHT):
        self.transf = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.net = U2NETP(in_ch=3, out_ch=len(classes_list)).to(self.device)

        self.net.load_state_dict(torch.load(load_params))

        self.zero_img = np.zeros((IMG_HEIGHT, IMG_WIDHT, 3), dtype=np.uint8)
        self.net.eval()

        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDHT = IMG_WIDHT
        self.classes_list = classes_list

    def detect(self, image):
        scale_min = min(self.IMG_HEIGHT / image.shape[0], self.IMG_WIDHT / image.shape[1])

        real_img_ = cv2.resize(image, (0, 0), fx=scale_min, fy=scale_min)
        real_img = self.zero_img.copy()
        real_img[:real_img_.shape[0], :real_img_.shape[1], :] = real_img_
        # plt.imshow(np.array(real_img))
        # plt.show()
        real_img = self.transf(real_img)[None].to(self.device)  # [1, 3, 224, 224]

        mask_out = torch.sigmoid(self.net(real_img))[0].cpu().detach().numpy()  # [6, 224, 224]

        points = np.zeros((len(self.classes_list), 2), dtype=int)
        for i in range(mask_out.shape[0]):  # 每个通道
            points[i] = (np.argwhere(mask_out[i] == np.max(mask_out[i])) // scale_min)  # 最大值的位置

        points = points[:, ::-1]  # 关键点x,y

        mask_out_ = np.max(mask_out, axis=0)  # [224, 224]  输出关键点掩码图

        return points, mask_out_


if __name__ == '__main__':
    images_path = r"D:\my_program\study\heatmap_point\data\val\image"
    # images_path = r"D:\my_program\study\heatmap_point\data\JPEGImages"
    # images_path = r"D:\my_program\project\5.luosi\data\螺丝机OK图片\六角"
    load_params = r"D:\my_program\study\heatmap_point\weights\u2net_p_weight\u2net_p_196_epoch.pth"
    classes_list = ["0", "1", "2", "3", "4", "5"]
    IMG_HEIGHT = 224  # 416
    IMG_WIDHT = 224  # 416

    detect = Detect(load_params=load_params, classes_list=classes_list, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDHT=IMG_WIDHT)

    for img_name in os.listdir(images_path):
        image_path = os.path.join(images_path, img_name)

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)  # h,w,c

        t0 = time.time()
        points, mask_out = detect.detect(image)
        print("time:", time.time() - t0)

        # 显示关键点图和掩码图
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX  # cv2自带的英文字体
        for i, point in enumerate(points):  # 原图画关键点
            cv2.circle(image, point, 8, (255, 0, 0), thickness=-1)  # 画圆，圆心半径
            cv2.putText(image, f"{classes_list[i]}", point, font, 2, (0, 0, 255), 4, lineType=cv2.LINE_AA)
        plt.subplot(1, 2, 1), plt.imshow(image), plt.title("image")  # [224, 224, 3]
        plt.subplot(1, 2, 2), plt.imshow(mask_out, "gray"), plt.title("mask_out")  # [224, 224, 3]
        plt.show()
