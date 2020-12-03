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
    transforms.ColorJitter(brightness=(0.4, 1.2), contrast=(0.5, 2),
                           saturation=(0.83333, 1.2), hue=(-0.2, 0.2)),
    transforms.RandomRotation(3),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.99, 1.01), shear=(-2, 2, -2, 2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.3403131123687272, std=0.2668040246671969)
])

restore = transforms.Compose([
    transforms.Normalize(mean=-1.596812585100412, std=3.7238888795090084),
    transforms.ToPILImage()
])

transform_test = transforms.Compose([
    transforms.ColorJitter(brightness=(1.6, 1.6), contrast=(0.8, 0.8)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.4288024258422422, std=0.2685364768810849)
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
        # label, faces = get_vertics(item)
        if self.transforms:
            img_data = self.transforms(figure)
        else:
            figure = np.asarray(figure)
            img_data = torch.from_numpy(figure)
        # lab_data = torch.from_numpy(label)
        # 下两行用于训练前测试标签与输入是否相互对应，正式训练时注释
        # check_before_train(img_data, lab_data, faces, item)
        # restore(img_data).show()
        return img_data, item

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
    image_list = os.listdir(cfg.INPUT.SAVE_RESIZE_IMAGES)
    means, variances = 0, 0
    for i in range(len(image_list)):
        # img_data = transforms.ToTensor()(Image.open(cfg.INPUT.SAVE_RESIZE_IMAGES + image_list[i]).convert('L'))
        img_data = transform_test(Image.open(cfg.INPUT.SAVE_RESIZE_IMAGES + image_list[i]).convert('L'))
        restore(img_data).show()
        means += img_data.mean()
        variances += img_data.var()
    mean = np.asarray(means) / len(image_list)
    variance = np.asarray(variances) / len(image_list)
    std = math.sqrt(variance)
    print('mean= ', mean, 'std= ', std)
    return mean, std


class SingleTest:
    def __init__(self, img_path):
        self.transforms = transform_test
        self.img_path = img_path

    def output_data(self, img_path):
        figure = Image.open(img_path).convert('L')
        if self.transforms:
            img_data = self.transforms(figure)
        else:
            figure = np.asarray(figure, dtype=np.float32)
            img_data = torch.from_numpy(figure)
        return img_data
