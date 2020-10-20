import numpy as np
from collections import OrderedDict
from torch import nn


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('conv_net') != -1:
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(3))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(64, 96, kernel_size=(3, 3), stride=2)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(96, 144, kernel_size=(3, 3), stride=2)),
            ('relu6', nn.ReLU()),
            ('conv7', nn.Conv2d(144, 216, kernel_size=(3, 3), stride=2)),
            ('relu8', nn.ReLU()),
            ('conv9', nn.Conv2d(216, 324, kernel_size=(3, 3), stride=2)),
            ('relu10', nn.ReLU()),
            ('conv11', nn.Conv2d(324, 486, kernel_size=(3, 3), stride=2)),
            ('relu12', nn.ReLU())
            ]))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(3888, 22971))
        ]))

    def forward(self, x):
        # print('输入图片卷积前大小：', x.size())
        x = self.conv_net(x)
        # print('输入图片卷积后大小：', x.size())
        # 在第一个全连接层与卷积层连接的位置需要将特征图拉成一个一维向量
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
