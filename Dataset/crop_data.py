import os
import xml.etree.ElementTree as ET
import math
from config import classes_list, color_list
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# 转换成四点坐标
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return int(xc + pResx), int(yc + pResy)


xmls_path = r"../../data/xml"
images_path = r"../../data/JPEGImages"
save_crop_data = r"../../data/crop_image"  # 掩码图

for name in ["ip", "down_up"]:
    save_path = os.path.join(save_crop_data, f"{name}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

# os.remove(r"../../data/xml/desktop.ini")
# print("desktop.ini" in os.listdir(xmls_path))
# exit()

for j, xml_name in enumerate(os.listdir(xmls_path)):
    # print(xml_name)
    xml_path = os.path.join(xmls_path, xml_name)
    tree = ET.parse(xml_path)
    img_name = tree.find("path").text.replace("/", "\\").split("\\")[-1]  # 10.161.209.166_92.83_88.58.xml
    label = img_name[0:-4].split("_")  # ['10.161.209.166', '92.83', '88.58'] ip down up
    print(img_name)
    print("=================")
    frame = cv2.imread(os.path.join(images_path, f"{img_name}"))
    h, w, c = frame.shape
    img_mask = np.zeros((h, w))  # 生成掩码图
    objs = tree.findall("object")
    for i, obj in enumerate(objs):
        clas = obj.find("name").text  # 类别名
        # clas_index = classes_list.index(clas)  # 类别索引
        cx = float(obj.find("robndbox").find("cx").text)
        cy = float(obj.find("robndbox").find("cy").text)
        w = float(obj.find("robndbox").find("w").text)
        h = float(obj.find("robndbox").find("h").text)
        angle = float(obj.find("robndbox").find("angle").text)

        x0, y0 = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
        x1, y1 = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
        x2, y2 = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
        x3, y3 = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)
        box_point = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])  # 变化前四个点的坐标

        a = pd.DataFrame(box_point)  # 转pandas
        b = a.sort_values(by=0)  # 根据坐标的x排序
        c = b.iloc[:2, :].sort_values(by=1)  # x最小的两个根据y排序
        d = b.iloc[2:, :].sort_values(by=1)  # x最大的两个根据y排序
        e = pd.concat([c.iloc[0:1, :], d, c.iloc[1:, :]], axis=0)  # 找到顺序左上右上右下左下
        pts = np.float32(e)

        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)

        if xmax - xmin > ymax - ymin:  # 上下
            pts2 = np.float32([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])  # 变化后的四个点的坐标
        else:  # 左右
            pts2 = np.float32([[ymin, xmax], [ymin, xmin], [ymax, xmin], [ymax, xmax]])
        M = cv2.getPerspectiveTransform(pts, pts2)  # 获取透视变换矩阵

        dst4 = cv2.warpPerspective(frame, M, dsize=(frame.shape[1], frame.shape[0]))  # 透视变化(图片，变换矩阵，输出大小)
        img = Image.fromarray(cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB))

        if xmax - xmin > ymax - ymin:  # 上下
            img_num = img.crop(((int(xmin), int(ymin), int(xmax), int(ymax))))  # 裁剪
        else:  # 左右
            img_num = img.crop(((int(ymin), int(xmin), int(ymax), int(xmax))))

        if clas == "ip":
            save_path = os.path.join(save_crop_data, f"ip/{label[0]}_{j}.png")
            img_num.save(save_path)
        elif clas in ["down", "up"]:
            if clas == "down":
                save_path = os.path.join(save_crop_data, f"down_up/{label[1]}_{j}.png")
            elif clas == "up":
                save_path = os.path.join(save_crop_data, f"down_up/{label[2]}_{j}.png")
            img_num.save(save_path)
    #     plt.subplot(1, 2, 1), plt.imshow(img)
    #     plt.subplot(1, 2, 2), plt.imshow(img_num)
    #     plt.show()
    # exit()
    # cv2.imwrite(os.path.join(seg_path, f"{img_name}"), img_mask)

    # print(np.unique(img_mask))
    # cv2.namedWindow("", cv2.WINDOW_NORMAL)
    # cv2.imshow("", img_mask)
    # cv2.waitKey(0)
    # # print(x0, y0)
    # exit()
