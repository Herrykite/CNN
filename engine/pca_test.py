# -*- coding: UTF-8 -*-
import sys

sys.path.insert(0, '../../')
import os
import torch
import pickle
import numpy as np
from ConvNet.tools.deal_with_obj import writeObj
from ConvNet.transform.datasets_transform import SingleTest
from ConvNet.modeling.cnn import PCAnet
from torchvision.models.resnet import resnet50
from ConvNet.tools.preprocess_data import mkdir, get_vertics
from ConvNet.config.defaults import get_cfg_defaults

# 调用配置文件
print('Start testing...')
cfg = get_cfg_defaults()
# 神经网络初始化
net = PCAnet()
# net = resnet50(pretrained=True)
# net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2))
# net.fc = torch.nn.Linear(2048, cfg.INPUT.PCA_DIMENSION )
net.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_NET_FILENAME))
print('loaded net successfully!')
device = cfg.MODEL.DEVICE1
net = net.to(device)
# 定义损失函数与优化器参数
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.BASE_LR)
optimizer.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_OPTIMIZER_FILENAME))
print('loaded optimizer successfully!')
# 定义路径
label_path = cfg.INPUT.VERTICS_PATH
image_path = cfg.INPUT.SAVE_RESIZE_IMAGES
label_list = os.listdir(label_path)
label_list.sort(key=lambda x: int(x[:-4]))
image_list = os.listdir(image_path)
image_list.sort(key=lambda x: int(x[:-4]))
# 加载预处理数据
data = np.load('../tools/data.npy')
pca = pickle.load(open('../tools/pca.pkl', 'rb'))
pca_coefficient = np.load('../tools/coefficient.npy')
print('PCA data has been loaded!')


def whole_proofread():
    net.eval()
    for rand in range(len(image_list)):
        print('测试图片为:', image_list[rand])
        test = SingleTest(img_path=image_path + image_list[rand])
        image = test.output_data(img_path=image_path + image_list[rand])
        image = image.expand(cfg.TEST.IMAGE_BATCH, cfg.TEST.IMAGE_CHANNEL, cfg.TEST.IMAGE_LENGTH, cfg.TEST.IMAGE_HEIGHT)
        image = image.to(device)
        mkdir(cfg.TEST.SAVE_OBJ)
        with torch.no_grad():
            predict = net(image)[0]
            vertics, faces = get_vertics(0)
            predict = predict.cpu().numpy()
            predict = torch.tensor(pca.inverse_transform(predict))
        # 输出为.obj文件
        pre_vertics = []
        for order in range(len(predict) // 3):
            pre_vertics.append([predict[3 * order], predict[3 * order + 1], predict[3 * order + 2]])
        writeObj(cfg.TEST.SAVE_OBJ + '/' + str(rand) + '.obj', pre_vertics, faces)


if __name__ == '__main__':
    whole_proofread()  # Use the saved model to test all data
