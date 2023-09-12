import os
import json
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import cv2


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)  # 产生等差数列 [start, stop, num]
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)  # 从坐标向量中返回坐标矩阵 返回列表存储x，y的值

    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)

    return heatmap


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

    return imgh, imgw, points


if __name__ == '__main__':
    jsons_path = r"../data/jsons"
    save_images_path = r"../data/SegmentationClass"
    labels = ["0", "1", "2", "3", "4", "5"]  # 关键点名字
    label_num = len(labels)
    sigma = 19

    for name in os.listdir(jsons_path):
        img_name = name.split(".")[0] + ".pkl"
        img_path = os.path.join(save_images_path, img_name)
        json_path = os.path.join(jsons_path, name)

        imgh, imgw, points = labelme_point(json_path, labels)  # 图像w,h,关键点

        # 生成一张(1236, 1616)需要0.62s
        heatmap = np.zeros((imgh, imgw, len(labels)))  # 掩码图 每个通道一个关键点
        for i in range(label_num):
            cx, cy = points[i]
            heatmap[:, :, i] = CenterLabelHeatMap(imgw, imgh, cx, cy, sigma)  # 归一化的值 0-1

            # print(np.max(heatmap[..., i]), np.min(heatmap[..., i]))  # 最大最小值
            # print(np.argwhere(heatmap[..., i] == np.max(heatmap[..., i])))  # 最大值的位置
            # plt.imshow(heatmap[..., i], "gray")
            # plt.show()

        with open(img_path, 'wb') as f:
            pickle.dump(heatmap, f)
        # exit()

        # heatmap = np.max(heatmap,axis=2)
        # print(heatmap.shape)
        # print(np.unique(heatmap))
        # exit()

        # print(np.max(heatmap), np.min(heatmap))  # 最大最小值
        # print(np.argsort(heatmap))
        # print(np.argwhere(heatmap == np.max(heatmap)))  # 最大值的位置
        # max_idx = np.array([(heatmap.argsort(axis=None, )[-label_num:]) // heatmap.shape[1],
        #               (heatmap.argsort(axis=None)[-label_num:]) % heatmap.shape[1]]).T[::-1]
        # print(max_idx)
        # plt.imshow(heatmap, "gray")
        # plt.show()
