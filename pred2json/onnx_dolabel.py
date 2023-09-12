import onnxruntime
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt


class heatmap_onnx():
    def __init__(self, onnx_path, classes_list):
        # 创建一个会话  类似于pytorch创建模型
        self.ort_session = onnxruntime.InferenceSession(onnx_path)
        # print(self.ort_session.get_outputs()[0])  # 输出结果  [1, 36288, 9]
        # print(self.ort_session.get_inputs()[0])  # 输入形状  [1, 3, 576, 1024]
        # print(self.ort_session.get_providers())  # device  ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])  # 使用cuda
        # self.ort_session.set_providers(['CPUExecutionProvider'])  # 使用cuda
        self.new_shape = self.ort_session.get_inputs()[0].shape  # 输入形状 [n, c, height, width]
        # self.zero_img = np.zeros((self.new_shape[1], self.new_shape[2], self.new_shape[3])).astype(np.float32)
        self.zero_img = np.zeros(self.new_shape).astype(np.float32)
        self.classes_list = classes_list

    def detect(self, images):
        real_img = self.zero_img.copy()
        scale = []
        for i, image in enumerate(images):
            scale_min = min(self.new_shape[2] / image.shape[0], self.new_shape[2] / image.shape[1])
            scale.append(scale_min)

            real_img_ = cv2.resize(image, (0, 0), fx=scale_min, fy=scale_min)
            real_img_ = real_img_[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)  # (3, 171, 224)
            real_img[i, :real_img_.shape[0], :real_img_.shape[1], :] = real_img_

        real_img = real_img / 255.  # [n, 3, 224, 224]

        ort_inputs = {self.ort_session.get_inputs()[0].name: real_img}
        mask = self.ort_session.run(None, ort_inputs)[0]
        mask_out_batch = (1 / (1 + np.exp(-mask)))  # (n, 6, 224, 224)  输出套sigmoid

        scale = np.array(scale)  # (1,)

        n, c, h, w = mask_out_batch.shape

        points_out = np.concatenate(((np.argmax(mask_out_batch.reshape(n, c, -1), axis=2) % w)[..., None],
                                     (np.argmax(mask_out_batch.reshape(n, c, -1), axis=2) // w)[..., None]),
                                    axis=2)  # (n, 6, 2)  2是坐标(x, y)

        points = (points_out // scale).astype(np.int32)  # (n, 6, 2)
        mask_out = np.max(mask_out_batch, axis=1)  # (n, 224, 224)

        return [points, mask_out]  # (n, 6, 2)  (n, 224, 224)


if __name__ == '__main__':
    classes_list = ["0", "1", "2", "3", "4", "5"]
    onnx_path = r"../onnx/u2net_p_196_epoch.onnx"
    # images_path = r"../data/JPEGImages"
    images_path = r"../data/val/image"

    detect = heatmap_onnx(onnx_path, classes_list)
    for img_name in os.listdir(images_path):
        image_path = os.path.join(images_path, img_name)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)  # h,w,c

        images = [image]
        t0 = time.time()
        output = detect.detect(images)
        print("time:", time.time() - t0)

        for i in range(len(images)):  # 一个batch的每张图
            points, mask_out = output[0][i], output[1][i]
            # 显示关键点图和掩码图
            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX  # cv2自带的英文字体
            for i in range(points.shape[0]):  # 原图画关键点
                cv2.circle(image, points[i], 8, (255, 0, 0), thickness=-1)  # 画圆，圆心半径
                cv2.putText(image, f"{classes_list[i]}", points[i], font, 2, (0, 0, 255), 4, lineType=cv2.LINE_AA)

            plt.subplot(1, 2, 1), plt.imshow(image), plt.title("image")  # [224, 224, 3]
            plt.subplot(1, 2, 2), plt.imshow(mask_out, "gray"), plt.title("mask_out")  # [224, 224, 3]
            plt.show()
