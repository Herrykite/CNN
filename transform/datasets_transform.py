import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
from ConvNet.config.defaults import get_cfg_defaults
from ConvNet.tools.preprocess_data import get_vertics

cfg = get_cfg_defaults()
transform = transforms.Compose([
    transforms.Resize(cfg.DATASETS.TRANSFORM_RESIZE),
    # 缩放图片，保持长宽比不变，最短边的长为240像素,
    transforms.ToTensor(),
    # 将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=0.5, std=0.5)
    # 标准化至[-1,1]
])


class DataSet(data.Dataset):
    def __init__(self, root):
        # 所有图片的绝对路径
        images = os.listdir(root)
        self.images = [os.path.join(root, k) for k in images]
        self.transforms = transform

    def __getitem__(self, item):
        img_path = self.images[item]
        figure = Image.open(img_path).convert('L')
        label = get_vertics(item)
        if self.transforms:
            img_data = self.transforms(figure)
        else:
            figure = np.asarray(figure)
            img_data = torch.from_numpy(figure)
        lab_data = torch.from_numpy(label)
        return img_data, lab_data

    def __len__(self):
        return len(self.images)


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
