import pickle
import os
import numpy as np

if __name__ == '__main__':
    save_images_path = r"SegmentationClass"

    for name in os.listdir(save_images_path):
        img_name = os.path.join(save_images_path, name)
        with open(img_name, 'rb') as f:
            data = pickle.load(f)
        print(data.shape)  # (1236, 1616, 6)  [h,w,c]
