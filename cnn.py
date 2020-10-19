import math
from collections import OrderedDict
from torch import nn
from torch.nn import init


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_net = nn.Sequential(OrderedDict([
            ('conv1a', nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2)),
            ('relu1', nn.ReLU()),
            # ('conv1b', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)),
            # ('relu2', nn.ReLU()),
            ('conv2a', nn.Conv2d(64, 96, kernel_size=(3, 3), stride=2)),
            ('relu3', nn.ReLU()),
            # ('conv2b', nn.Conv2d(96, 96, kernel_size=(3, 3), stride=1, padding=1)),
            # ('relu4', nn.ReLU()),
            ('conv3a', nn.Conv2d(96, 144, kernel_size=(3, 3), stride=2)),
            ('relu5', nn.ReLU()),
            # ('conv3b', nn.Conv2d(144, 144, kernel_size=(3, 3), stride=1, padding=1)),
            # ('relu6', nn.ReLU()),
            ('conv4a', nn.Conv2d(144, 216, kernel_size=(3, 3), stride=2)),
            ('relu7', nn.ReLU()),
            # ('conv4b', nn.Conv2d(216, 216, kernel_size=(3, 3), stride=1, padding=1)),
            # ('relu8', nn.ReLU()),
            ('conv5a', nn.Conv2d(216, 324, kernel_size=(3, 3), stride=2)),
            ('relu9', nn.ReLU()),
            # ('conv5b', nn.Conv2d(324, 324, kernel_size=(3, 3), stride=1, padding=1)),
            # ('relu10', nn.ReLU()),
            ('conv6a', nn.Conv2d(324, 486, kernel_size=(3, 3), stride=2)),
            ('relu11', nn.ReLU())
            # ('conv6b', nn.Conv2d(486, 486, kernel_size=(3, 3), stride=1, padding=1)),
            # ('relu12', nn.ReLU())
            ]))
        # self.conv_net.apply(self.reset_parameters)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(3888, 22971))
        ]))

    def forward(self, x):
        # print('输入图片卷积前大小：', x.size())
        x = self.conv_net(x)
        # print('输入图片卷积后大小：', x.size())
        # 在第一个全连接层与卷积层连接的位置需要将特征图拉成一个一维向量
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

    # def reset_parameters(self, _):
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(3))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         init.uniform_(self.bias, -bound, bound)
