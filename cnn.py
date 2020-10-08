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
#             self.conv_net = nn.Sequential(OrderedDict([
#             ('conv1a', nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2)),
#             ('relu1', nn.ReLU()),
#             ('conv1b', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)),
#             ('relu2', nn.ReLU()),
#             ('conv2a', nn.Conv2d(64, 96, kernel_size=(3, 3), stride=2)),
#             ('relu3', nn.ReLU()),
#             ('conv2b', nn.Conv2d(96, 96, kernel_size=(3, 3), stride=1)),
#             ('relu4', nn.ReLU()),
#             ('conv3a', nn.Conv2d(96, 144, kernel_size=(3, 3), stride=2)),
#             ('relu5', nn.ReLU()),
#             ('conv3b', nn.Conv2d(144, 144, kernel_size=(3, 3), stride=1)),
#             ('relu6', nn.ReLU()),
#             ('conv4a', nn.Conv2d(144, 216, kernel_size=(3, 3), stride=2)),
#             ('relu7', nn.ReLU()),
#             ('conv4b', nn.Conv2d(216, 216, kernel_size=(3, 3), stride=1)),
#             ('relu8', nn.ReLU()),
#             ('conv5a', nn.Conv2d(216, 324, kernel_size=(3, 3), stride=2)),
#             ('relu9', nn.ReLU()),
#             ('conv5b', nn.Conv2d(324, 324, kernel_size=(3, 3), stride=1)),
#             ('relu10', nn.ReLU()),
#             ('conv6a', nn.Conv2d(324, 486, kernel_size=(3, 3), stride=2)),
#             ('relu11', nn.ReLU()),
#             ('conv6b', nn.Conv2d(486, 486, kernel_size=(3, 3), stride=1)),
#             ('relu12', nn.ReLU())
#         ]))
#         # self.drop = np.random.rand(486) < 0.2
#         # self.conv_net *= self.drop
#         #  定义后半部分的全连接层
#         self.fc = nn.Sequential(OrderedDict([
#             ('fc', nn.Linear(9720, 22971))
#         ]))
    # 定义网络的前向运算
    def forward(self, x):
        x = self.convnet(x)
        # 在第一个全连接层与卷积层连接的位置
        # 需要将特征图拉成一个一维向量
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
