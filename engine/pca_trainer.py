# -*- coding: UTF-8 -*-
# import sys
# sys.path.insert(0, '/home/digisky/wanghairui')
import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from ConvNet.tools.deal_with_obj import writeObj
from ConvNet.transform.datasets_transform import DataSet, SingleTest
from ConvNet.modeling.cnn import PCAnet
from ConvNet.modeling.newcnn import CNN
from ConvNet.tools.preprocess_data import mkdir, get_vertics
from ConvNet.config.defaults import get_cfg_defaults
# from sklearn.decomposition import PCA

# 调用配置文件
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cfg = get_cfg_defaults()
# 神经网络初始化
net = CNN()
cnn = PCAnet()
cnn.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_NET_FILENAME))
print('loaded net successfully!')
model_dict = cnn.state_dict()
pretrained_dict = {k: v for k, v in net.state_dict().items() if k in model_dict}
model_dict.update(pretrained_dict)
cnn.load_state_dict(model_dict)
net = cnn
# 检查可训练的参数
for name, param in net.named_parameters():
    if param.requires_grad:
        print(name)
device = cfg.MODEL.DEVICE1
net = net.to(device)
# 定义损失函数与优化器参数
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.ADJUST_LR)
optimizer.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_OPTIMIZER_FILENAME))
print('loaded optimizer successfully!')
# 定义路径
label_path = cfg.INPUT.VERTICS_PATH
image_path = cfg.INPUT.SAVE_RESIZE_IMAGES
label_list = os.listdir(label_path)
label_list.sort(key=lambda x: int(x[:-4]))
image_list = os.listdir(image_path)
image_list.sort(key=lambda x: int(x[:-4]))

train_loss = []
loader = DataLoader(DataSet(image_path), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=True)
data = np.load('data.npy')
pca = pickle.load(open('pca.pkl', 'rb'))
pca_coefficient = np.load('coefficient.npy')
print('PCA data has been loaded!')


def train(number):
    for i, (images, itemSet) in enumerate(loader, start=1):  # index为所采用图片的位序，对应.obj的文件名
        labels = []
        for j in itemSet:
            labels.append(pca_coefficient[j])   # 调取降维系数
        labels = torch.Tensor(labels)
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)  # 网络前向运行
        print('epoch:', number, '  batch:', i)
        loss = criterion(output*1000, labels*1000)  # 计算网络的损失函数
        if i % 60 == 0:
            with open(cfg.OUTPUT.LOGGING, 'a') as f:
                print('第', number, '轮  第', i, '次 Loss =', loss.item(), file=f)
        print('Loss =', loss.item())
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 优化更新权重


def proofread():
    # 开始测试。此句是防止改变权值。
    net.eval()
    rand = np.random.randint(0, len(image_list))
    with open(cfg.OUTPUT.LOGGING, 'a') as f:
        print('测试图片为:', image_list[rand], '\n对应Obj为：', label_list[rand], file=f)
    test = SingleTest(img_path=image_path + image_list[rand])
    image = test.output_data(img_path=image_path + image_list[rand])
    image = image.expand(cfg.TEST.IMAGE_BATCH, cfg.TEST.IMAGE_CHANNEL, cfg.TEST.IMAGE_LENGTH, cfg.TEST.IMAGE_HEIGHT)
    image = image.to(device)
    mkdir(cfg.TEST.SAVE_OBJ)
    with torch.no_grad():
        predict = net(image)[0]
        vertics, faces = get_vertics(rand)
        vertics = torch.tensor(vertics.reshape(3, 7657).T.reshape(22971)).to(device)
        predict = predict.cpu().numpy()
        predict = torch.tensor(pca.inverse_transform(predict)).to(device)
        loss = criterion(vertics, predict)
        with open(cfg.OUTPUT.LOGGING, 'a') as f:
            print('Loss =', loss.item(), file=f)
    # 输出为.obj文件
    pre_vertics = []
    for order in range(len(predict) // 3):
        pre_vertics.append([predict[3*order], predict[3*order + 1], predict[3*order + 2]])
    writeObj(cfg.TEST.SAVE_OBJ + '/' + label_list[rand], pre_vertics, faces)


def save(number):
    mkdir(cfg.OUTPUT.PARAMETER)
    torch.save(net.state_dict(), cfg.OUTPUT.PARAMETER + '/' + str(number + 1) + '_CNN.pkl',
               _use_new_zipfile_serialization=False)
    torch.save(optimizer.state_dict(),  cfg.OUTPUT.PARAMETER + '/' + str(number + 1) + '_opt.pkl',
               _use_new_zipfile_serialization=False)
    print('\n网络参数已保存!\n')


def adjust_learning_rate(number):
    if number < cfg.SOLVER.FIRST_ADJUST_LIMIT:
        learning = cfg.SOLVER.BASE_LR
    elif number > cfg.SOLVER.SECOND_ADJUST_LIMIT:
        learning = cfg.SOLVER.ADJUST_LR
    else:
        learning = 1 / (100 * number)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning
    with open(cfg.OUTPUT.LOGGING, 'a') as f:
        print('学习率已更新为：', learning, file=f)


def run():
    epoch = cfg.INPUT.BASE_EPOCH
    while True:
        adjust_learning_rate(epoch)
        train(epoch)
        proofread()
        if epoch % cfg.DATASETS.SAVE_INTERVAL == cfg.DATASETS.SAVE_INTERVAL-1:
            save(epoch)
        epoch += 1


if __name__ == '__main__':
    run()

