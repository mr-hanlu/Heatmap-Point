import os
import xml.etree.ElementTree as ET
import math
from config import classes_list, color_list
import cv2
import numpy as np


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
seg_path = r"../../data/SegmentationClass"  # 掩码图

# os.remove(r"../../data/xml/desktop.ini")
# print("desktop.ini" in os.listdir(xmls_path))
# exit()

for xml_name in os.listdir(xmls_path):
    # print(xml_name)
    xml_path = os.path.join(xmls_path, xml_name)
    print(xml_path)
    tree = ET.parse(xml_path)
    img_name = tree.find("path").text.replace("/","\\").split("\\")[-1]  # 10.161.209.166_92.83_88.58.xml
    print(img_name)
    print("=================")
    img = cv2.imread(os.path.join(images_path, f"{img_name}"))
    h, w, c = img.shape
    img_mask = np.zeros((h, w))  # 生成掩码图
    objs = tree.findall("object")
    for i, obj in enumerate(objs):
        clas = obj.find("name").text  # 类别名
        clas_index = classes_list.index(clas)  # 类别索引
        cx = float(obj.find("robndbox").find("cx").text)
        cy = float(obj.find("robndbox").find("cy").text)
        w = float(obj.find("robndbox").find("w").text)
        h = float(obj.find("robndbox").find("h").text)
        angle = float(obj.find("robndbox").find("angle").text)

        x0, y0 = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
        x1, y1 = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
        x2, y2 = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
        x3, y3 = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)
        pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
        cv2.fillPoly(img_mask, [pts], color=clas_index)  # 填充多边形

    cv2.imwrite(os.path.join(seg_path,f"{img_name}"),img_mask)

    # print(np.unique(img_mask))
    # cv2.namedWindow("", cv2.WINDOW_NORMAL)
    # cv2.imshow("", img_mask)
    # cv2.waitKey(0)
    # # print(x0, y0)
    # exit()
