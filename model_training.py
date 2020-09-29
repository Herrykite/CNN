import os
import numpy as np
import torch
from Convolutional.readobj import loadObj
from Convolutional.my_dataload import DataSet


def tell_vertics(count):
    path = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/output_v2/total/'
    file_list = os.listdir(path)
    vertics_x, vertics_y, vertics_z = [], [], []
    if file_list[count].endswith('.obj'):
        vertics, faces = loadObj(path + file_list[count])
        for i in range(len(vertics)):
            vertics_x.append(vertics[i][0])
            vertics_y.append(vertics[i][1])
            vertics_z.append(vertics[i][2])
    return vertics_x, vertics_y, vertics_z


def tell_vertics_combine(vertics_x, vertics_y, vertics_z):
    v_data = []
    for j in range(len(vertics_x)):
        v_data.append(vertics_x[j])
        v_data.append(vertics_y[j])
        v_data.append(vertics_z[j])
    return v_data


def normalize(initial_x):
    x_mean = np.mean(initial_x)
    x_std = np.std(initial_x, ddof=1)  # 加入ddof=1则为无偏样本标准差
    normalized_x = (initial_x - x_mean) / x_std
    return normalized_x


def tell_labels(l_x, l_y, l_z):
    label = tell_vertics_combine(normalize(np.array(l_x)), normalize(np.array(l_y)), normalize(np.array(l_z)))
    return label


if __name__ == '__main__':
    x, y, z = tell_vertics(0)
    labels = tell_labels(x, y, z)
    train_labels = labels[:21000]
    test_labels = labels[21000:]
    train_labels = torch.Tensor(train_labels)
    test_labels = torch.Tensor(test_labels)
    image_path = 'D:/DIGISKY/CNN_input_image/'
    dataSet = DataSet(image_path)
