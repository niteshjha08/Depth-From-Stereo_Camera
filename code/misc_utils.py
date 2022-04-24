import cv2
import numpy as np
import os
import random

def get_images(images_dir):
    file_names = os.listdir(images_dir)
    img_names = []
    for name in file_names:
        if(name.endswith(tuple(['.png','.jpg','.jpeg']))):
            img_names.append(name)
    images = []
    for name in img_names:
        img = cv2.imread(os.path.join(images_dir, name))
        img = cv2.resize(img,(img.shape[1]//4, img.shape[0]//4))
        images.append(img)

    return images

def load_camera_matrix(data_dir):
    print(data_dir)
    data_name = data_dir.split('/')[-1]
    print(data_name)
    if(data_name == "pendulum"):
        K1 = np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0, 0, 1]])
        K2 = np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0, 0, 1]])
        baseline = 537.75


    elif (data_name == "octagon"):
        K1 = np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0, 0, 1]]) 
        K2 = np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0, 0, 1]])
        baseline = 221.76

    elif data_name == "curule":
        K1 = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0, 0, 1]])
        K2 = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0, 0, 1]])
        baseline = 88.39


    return K1, K2, baseline
