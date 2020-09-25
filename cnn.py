from abc import ABC
from collections import OrderedDict
from torch import nn


class CNN(nn.Module, ABC):
    def __init__(self):
        super(CNN, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))
        #  定义后半部分的全连接层
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))
    # 定义网络的前向运算

    def forward(self, x):
        x = self.convnet(x)
        # 在第一个全连接层与卷积层连接的位置
        # 需要将特征图拉成一个一维向量
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
