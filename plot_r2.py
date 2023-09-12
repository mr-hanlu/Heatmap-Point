import json
import os
import cv2
from model.U2net_p import U2NETP
import torch
from torchvision import transforms
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示字符
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def labelme_point(json_path, labels):  # 一张图对应一组关键点
    d = json.load(open(json_path))
    imgh = d["imageHeight"]
    imgw = d["imageWidth"]
    points = [[] for i in range(len(labels))]  # 每个label点对应一个列表
    # points = []  # 每个label点对应一个列表
    for target in d["shapes"]:
        label = target["label"]
        point = target["points"]
        points[labels.index(label)] = point[0]

    return points


class Detect():
    def __init__(self, load_params, classes_list, IMG_HEIGHT, IMG_WIDHT):
        self.transf = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        real_img = self.transf(real_img)[None].to(self.device)  # [1, 3, 224, 224]

        mask_out = torch.sigmoid(self.net(real_img))[0].cpu().detach().numpy()  # [6, 224, 224]

        points = np.zeros((len(self.classes_list), 2), dtype=int)
        for i in range(mask_out.shape[0]):  # 每个通道
            points[i] = (np.argwhere(mask_out[i] == np.max(mask_out[i])) // scale_min)  # 最大值的位置

        points = points[:, ::-1]  # 关键点x,y

        # mask_out_ = np.max(mask_out, axis=0)  # [224, 224]  输出关键点掩码图

        return points


if __name__ == '__main__':
    images_path = r"D:\my_program\study\heatmap_point\data\val\image"
    jsons_path = r"D:\my_program\study\heatmap_point\data\val\json"
    # images_path = r"D:\my_program\study\heatmap_point\data\JPEGImages"
    # jsons_path = r"D:\my_program\study\heatmap_point\data\jsons"
    load_params = r"D:\my_program\study\heatmap_point\weights\u2net_p_weight\u2net_p_196_epoch.pth"
    classes_list = ["0", "1", "2", "3", "4", "5"]
    IMG_HEIGHT = 224
    IMG_WIDHT = 224

    detect = Detect(load_params=load_params, classes_list=classes_list, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDHT=IMG_WIDHT)

    r2_list = []
    for img_name in os.listdir(images_path):
        image_path = os.path.join(images_path, img_name)
        json_path = os.path.join(jsons_path, img_name.split(".")[0] + ".json")

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)  # h,w,c

        point_out = detect.detect(image)  # (6, 2)
        # print(point_out)

        point_target = labelme_point(json_path, classes_list)  # 图像w,h,关键点

        r2 = r2_score(np.array(point_target), point_out)
        r2_list.append(r2)

        # if r2 < 0.7:
        #     plt.imshow(image)
        #     plt.show()

    mean_r2 = np.mean(r2_list)
    y_lim = np.max(r2_list) - np.min(r2_list)  # y轴范围

    # plot
    plt.plot(range(len(r2_list)), r2_list, "r", marker="*", ms=10, label="r2")
    plt.xlabel("数据索引")
    plt.ylabel("r2分数")
    plt.title(f"平均r2分数为：{np.round(mean_r2, 4)}")
    # plt.legend(loc="upper left")
    for x1, y1 in zip(range(len(r2_list)), r2_list):
        plt.text(x1, y1 + y_lim * 0.015, str(np.round(y1, 4)), ha='center', va='bottom', fontsize=8, rotation=0)
    plt.show()
    # plt.savefig("r2分数.jpg")
