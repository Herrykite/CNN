# -*- coding: UTF-8 -*-
import math
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
from ConvNet.config.defaults import get_cfg_defaults
from ConvNet.tools.preprocess_data import get_vertics, mkdir
from ConvNet.tools.deal_with_obj import writeObj

cfg = get_cfg_defaults()
transform = transforms.Compose([
    transforms.ColorJitter(brightness=(0.75, 1.5), contrast=(0.83333, 1.2),
                           saturation=(0.83333, 1.2), hue=(-0.2, 0.2)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.95, 1.05), shear=(-5, 5, -5, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.3156022930958101, std=0.28214540372352737)
])

restore = transforms.Compose([
    transforms.Normalize(mean=-1.1185803097649145, std=3.5442718073830264),
    transforms.ToPILImage()
])


class DataSet(data.Dataset):
    def __init__(self, root):
        # 所有图片的绝对路径
        images = os.listdir(root)
        images.sort(key=lambda obj: len(obj))
        self.images = [os.path.join(root, k) for k in images]
        self.transforms = transform

    def __getitem__(self, item):
        img_path = self.images[item]
        figure = Image.open(img_path).convert('L')
        label, faces = get_vertics(item)
        if self.transforms:
            img_data = self.transforms(figure)
        else:
            figure = np.asarray(figure)
            img_data = torch.from_numpy(figure)
        lab_data = torch.from_numpy(label)
        # 下两行用于训练前测试标签与输入是否相互对应，正式训练时注释
        # check_before_train(img_data, lab_data, faces, item)
        # restore(img_data).show()
        return img_data, lab_data

    def __len__(self):
        return len(self.images)


def check_before_train(images, labels, faces, item):
    restore(images).convert('L')
    vertics = []
    for i in range(len(labels) // 3):
        vertics.append([labels[i], labels[i + len(labels) // 3], labels[i + 2 * len(labels) // 3]])
    mkdir(cfg.INPUT.CHECK)
    writeObj(cfg.INPUT.CHECK + str(item) + '_check.obj', vertics, faces)


def get_mean_std():
    image_list = os.listdir(cfg.DATASETS.SAVE_RESIZE_IMAGES)
    means, variances = 0, 0
    for i in range(len(image_list)):
        img_data = transform(Image.open(cfg.DATASETS.SAVE_RESIZE_IMAGES + image_list[i]).convert('L'))
        means += img_data.mean()
        variances += img_data.var()
    mean = np.asarray(means) / len(image_list)
    variance = np.asarray(variances) / len(image_list)
    std = math.sqrt(variance)
    print('mean= ', mean, 'std= ', std)
    return mean, std


class SingleTest:
    def __init__(self, img_path):
        self.transforms = transform
        self.img_path = img_path

    def output_data(self, img_path):
        figure = Image.open(img_path).convert('L')
        if self.transforms:
            img_data = self.transforms(figure)
        else:
            figure = np.asarray(figure, dtype=np.float32)
            img_data = torch.from_numpy(figure)
        return img_data
