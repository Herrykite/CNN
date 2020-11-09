# -*- coding: UTF-8 -*-
# import sys
# sys.path.insert(0, '/home/digisky/wanghairui')
import os
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ConvNet.tools.deal_with_obj import writeObj
from ConvNet.transform.datasets_transform import DataSet, SingleTest
# from ConvNet.modeling.cnn import CNN
# from torchvision.models.resnet import resnet50, Bottleneck
from ConvNet.modeling.newcnn import CNN
from ConvNet.tools.preprocess_data import mkdir, get_vertics
# from ConvNet.tools.draw import draw_train_process
from ConvNet.config.defaults import get_cfg_defaults
import time

# 调用配置文件
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cfg = get_cfg_defaults()
# 神经网络初始化
# net = resnet50(pretrained=True)
# cnn = CNN(Bottleneck, [3, 4, 6, 3])
# model_dict = cnn.state_dict()
# pretrained_dict = {k: v for k, v in net.state_dict().items() if k in model_dict}
# model_dict.update(pretrained_dict)
# cnn.load_state_dict(model_dict)
# net = cnn
net = CNN()
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
# tb = SummaryWriter('./')


def train(number):
    for i, (images, labels) in enumerate(loader, start=1):  # index为所采用图片的位序，对应.obj的文件名
        start = time.time()
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)  # 网络前向运行
        print('epoch:', number, '  batch:', i, '\npredictive labels：\n', output)
        loss = criterion(output, labels)  # 计算网络的损失函数
        with open(cfg.OUTPUT.LOGGING, 'a') as f:
            print('Loss =', loss.item(), file=f)
        print('Loss =', loss.item())
        train_loss.append(loss.item())
        # TensorBoard(images, loss, number * cfg.INPUT.BATCH_SIZE + i)
        optimizer.zero_grad()
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 优化更新权重
        # if len(train_loss) % 100 == 0 and len(train_loss) < 30000:
        #     iters = range(len(train_loss))
        #     draw_train_process(cfg.VISUAL.TITLE, iters, train_loss, cfg.VISUAL.LINE_LABEL)
        end = time.time()
        with open(cfg.OUTPUT.LOGGING, 'a') as f:
            print('第', number + 1, '轮  第', i, '次用时为: ', end - start, file=f)


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
    with torch.no_grad():
        predict = net(image)[0]
        vertics, faces = get_vertics(rand)
        vertics = torch.Tensor(vertics).to(device)
        loss = criterion(vertics, predict)
        with open(cfg.OUTPUT.LOGGING, 'a') as f:
            print('测试图片输出数据Loss =', loss.item(), file=f)
    # 输出为.obj文件
    mkdir(cfg.TEST.SAVE_OBJ)
    predict = predict.cpu().numpy()
    pre_vertics = []
    for order in range(len(predict) // cfg.TEST.VERTICS_DIMENSION):
        pre_vertics.append([predict[order], predict[order + len(predict) // cfg.TEST.VERTICS_DIMENSION],
                            predict[order + len(predict) // cfg.TEST.VERTICS_DIMENSION * 2]])
    writeObj(cfg.TEST.SAVE_OBJ + '/' + label_list[rand], pre_vertics, faces)


# def TensorBoard(tb, images, loss, i):
#     grid = torchvision.utils.make_grid(images)
#     tb.add_image('Image', grid, 0)
#     tb.add_graph(net, images)
#     tb.add_scalar('Loss', loss, i)
#     tb.add_histogram('conv1.weight', net.conv_net.conv1.weight)
#     tb.add_histogram('bn2.weight', net.conv_net.bn2.weight)
#     tb.add_histogram('bn2.bias', net.conv_net.bn2.bias)
#     tb.add_histogram('conv3.weight', net.conv_net.conv3.weight)
#     tb.add_histogram('conv3.bias', net.conv_net.conv3.bias)
#     tb.add_histogram('bn4.weight', net.conv_net.bn4.weight)
#     tb.add_histogram('bn4.bias', net.conv_net.bn4.bias)
#     tb.add_histogram('conv5.weight', net.conv_net.conv5.weight)
#     tb.add_histogram('conv5.bias', net.conv_net.conv5.bias)
#     tb.add_histogram('bn6.weight', net.conv_net.bn6.weight)
#     tb.add_histogram('bn6.bias', net.conv_net.bn6.bias)
#     tb.add_histogram('conv7.weight', net.conv_net.conv7.weight)
#     tb.add_histogram('conv7.bias', net.conv_net.conv7.bias)
#     tb.add_histogram('bn8.weight', net.conv_net.bn8.weight)
#     tb.add_histogram('bn8.bias', net.conv_net.bn8.bias)
#     tb.add_histogram('conv9.weight', net.conv_net.conv9.weight)
#     tb.add_histogram('conv9.bias', net.conv_net.conv9.bias)
#     tb.add_histogram('bn10.weight', net.conv_net.bn10.weight)
#     tb.add_histogram('bn10.bias', net.conv_net.bn10.bias)
#     tb.add_histogram('conv11.weight', net.conv_net.conv11.weight)
#     tb.add_histogram('conv11.bias', net.conv_net.conv11.bias)
#     tb.add_histogram('bn12.weight', net.conv_net.bn12.weight)
#     tb.add_histogram('bn12.bias', net.conv_net.bn12.bias)
#     tb.add_histogram('fc.weight', net.fc.fc.weight)
#     tb.add_histogram('fc.bias', net.fc.fc.bias)
#     # tensorboard --logdir=./
# def TensorBoard(images, loss, i):
#     grid = torchvision.utils.make_grid(images)
#     tb.add_image('Image', grid, 0)
#     tb.add_graph(net, images)
#     tb.add_scalar('Loss', loss, i)
#     tb.add_histogram('conv.weight', net.conv.weight)
#     tb.add_histogram('bn1.weight', net.bn1.weight)
#     tb.add_histogram('bn1.bias', net.bn1.bias)
#     tb.add_histogram('fc1.weight', net.fc1.weight)
#     tb.add_histogram('fc1.bias', net.fc1.bias)
#     # tensorboard --logdir=./


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
    net.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_NET_FILENAME))
    print('loaded net successfully!')
    while True:
        adjust_learning_rate(epoch)
        train(epoch)
        proofread()
        if epoch % cfg.DATASETS.SAVE_INTERVAL == cfg.DATASETS.SAVE_INTERVAL-1:
            save(epoch)
        epoch += 1


if __name__ == '__main__':
    run()

