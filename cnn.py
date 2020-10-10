from collections import OrderedDict
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_net = nn.Sequential(OrderedDict([
            ('conv1a', nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv1b', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)),
            ('relu2', nn.ReLU()),
            ('conv2a', nn.Conv2d(64, 96, kernel_size=(3, 3), stride=2, padding=1)),
            ('relu3', nn.ReLU()),
            ('conv2b', nn.Conv2d(96, 96, kernel_size=(3, 3), stride=1, padding=1)),
            ('relu4', nn.ReLU()),
            ('conv3a', nn.Conv2d(96, 144, kernel_size=(3, 3), stride=2, padding=1)),
            ('relu5', nn.ReLU()),
            ('conv3b', nn.Conv2d(144, 144, kernel_size=(3, 3), stride=1, padding=1)),
            ('relu6', nn.ReLU()),
            ('conv4a', nn.Conv2d(144, 216, kernel_size=(3, 3), stride=2, padding=1)),
            ('relu7', nn.ReLU()),
            ('conv4b', nn.Conv2d(216, 216, kernel_size=(3, 3), stride=1, padding=1)),
            ('relu8', nn.ReLU()),
            ('conv5a', nn.Conv2d(216, 324, kernel_size=(3, 3), stride=2, padding=1)),
            ('relu9', nn.ReLU()),
            ('conv5b', nn.Conv2d(324, 324, kernel_size=(3, 3), stride=1, padding=1)),
            ('relu10', nn.ReLU()),
            ('conv6a', nn.Conv2d(324, 486, kernel_size=(3, 3), stride=2, padding=1)),
            ('relu11', nn.ReLU()),
            ('conv6b', nn.Conv2d(486, 486, kernel_size=(3, 3), stride=1, padding=1)),
            ('relu12', nn.ReLU())
            ]))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(9720, 22971)),
            ('relu', nn.ReLU()),
        ]))

    def forward(self, x):
        print(x.size())
        x = self.conv_net(x)
        # 在第一个全连接层与卷积层连接的位置需要将特征图拉成一个一维向量
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
