import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(32),  # 缩放图片，保持长宽比不变，最短边的长为32像素,
    transforms.CenterCrop(32),  # 从中间切出 32*32的图片
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化至[-1,1]
])


class DataSet(data.Dataset):
    def __init__(self, root):
        # 所有图片的绝对路径
        images = os.listdir(root)
        self.images = [os.path.join(root, k) for k in images]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.images[index]
        figure = Image.open(img_path).convert('L')
        if self.transforms:
            img_data = self.transforms(figure)
        else:
            figure = np.asarray(figure)
            img_data = torch.from_numpy(figure)
        return img_data

    def __len__(self):
        return len(self.images)
