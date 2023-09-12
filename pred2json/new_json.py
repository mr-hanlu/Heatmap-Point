import os
import cv2
import numpy as np
from onnx_dolabel import heatmap_onnx
from txt2json import Txt2Json

onnx_path = r"../onnx/u2net_p_196_epoch.onnx"
# images_path = r"../data/JPEGImages"
images_path = r"../data/val/image"
save_json_path = images_path
clas_names = ["0", "1", "2", "3", "4", "5"]  # 类别

txt2json = Txt2Json(clas_names=clas_names)
heatmap_detect = heatmap_onnx(onnx_path, clas_names)

for img_name in os.listdir(images_path):
    json_name = img_name.split(".")[0] + ".json"
    json_path = os.path.join(save_json_path, json_name)
    image_path = os.path.join(images_path, img_name)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)  # h,w,c

    imageHeight, imageWidth = image.shape[:2]

    images = [image]
    output = heatmap_detect.detect(images)

    txt2json.save_json(img_name=img_name, json_path=json_path, imageWidth=imageWidth, imageHeight=imageHeight, points=output[0][0])
    # exit()
