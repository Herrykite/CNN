# -*- coding: UTF-8 -*-
import sys

sys.path.insert(0, '../../')
import math
import os
import torch
import yaml
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
from ConvNet.config.defaults import get_cfg_defaults
from ConvNet.tools.preprocess_data import mkdir
from ConvNet.tools.deal_with_obj import writeObj

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1])

transform = transforms.Compose([
    transforms.ColorJitter(brightness=(0.4, 1.2), contrast=(0.5, 2),
                           saturation=(0.83333, 1.2), hue=(-0.2, 0.2)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-2, 2, -2, 2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.DATASETS.TRANSFORM_MEAN, std=cfg.DATASETS.TRANSFORM_STD)
])

restore = transforms.Compose([
    transforms.Normalize(mean=-cfg.DATASETS.TRANSFORM_MEAN/cfg.DATASETS.TRANSFORM_STD,
                         std=1/cfg.DATASETS.TRANSFORM_STD),
    transforms.ToPILImage()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.DATASETS.TRANSFORM_MEAN, std=cfg.DATASETS.TRANSFORM_STD)
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
        # 下两行为编译时用于测试训练前标签与输入是否相互对应，正式训练时注释
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
    print('Calculating the mean and variance of the given dataset...')
    image_list = os.listdir(cfg.INPUT.SAVE_RESIZE_IMAGES)
    means, variances = 0, 0
    for i in range(len(image_list)):
        img_data = transforms.ToTensor()(Image.open(cfg.INPUT.SAVE_RESIZE_IMAGES + image_list[i]).convert('L'))
        # img_data = transform_test(Image.open(cfg.INPUT.SAVE_RESIZE_IMAGES + image_list[i]).convert('L'))
        # restore(img_data).show()
        means += img_data.mean()
        variances += img_data.var()
    mean = np.asarray(means) / len(image_list)
    variance = np.asarray(variances) / len(image_list)
    std = math.sqrt(variance)
    with open(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1], 'r') as f:
        menu = yaml.load(f, Loader=yaml.FullLoader)
        menu['DATASETS']['TRANSFORM_MEAN'] = float(mean)
        menu['DATASETS']['TRANSFORM_STD'] = std
    with open(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1], 'w') as f:
        yaml.dump(menu, f)
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
