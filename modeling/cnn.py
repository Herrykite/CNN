# -*- coding: UTF-8 -*-
import sys

sys.path.insert(0, '../../')
import os
from collections import OrderedDict
from torch import nn
from ConvNet.config.defaults import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1])


class PCAnet(nn.Module):
    def __init__(self):
        super(PCAnet, self).__init__()
        self.conv_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('conv3', nn.Conv2d(64, 96, kernel_size=(3, 3), stride=2, padding=1)),
            ('bn4', nn.BatchNorm2d(96)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(96, 144, kernel_size=(3, 3), stride=2, padding=1)),
            ('bn6', nn.BatchNorm2d(144)),
            ('conv7', nn.Conv2d(144, 216, kernel_size=(3, 3), stride=2, padding=1)),
            ('bn8', nn.BatchNorm2d(216)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(216, 324, kernel_size=(3, 3), stride=2, padding=1)),
            ('bn10', nn.BatchNorm2d(324)),
            ('conv11', nn.Conv2d(324, 486, kernel_size=(3, 3), stride=2, padding=1)),
            ('bn12', nn.BatchNorm2d(486)),
            ('relu3', nn.ReLU(inplace=True))
        ]))
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9720, cfg.INPUT.PCA_DIMENSION))
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # constant_用第二个参数值填充向量
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('输入图片卷积前大小：', x.size())
        x = self.conv_net(x)
        # print('输入图片卷积后大小：', x.size())
        # 在第一个全连接层与卷积层连接的位置需要将特征图拉成一个一维向量
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
