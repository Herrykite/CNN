import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from ConvNet.tools.deal_with_obj import loadObj, writeObj
from ConvNet.transform.datasets_transform import DataSet, SingleTest
from ConvNet.modeling.cnn import CNN, conv_init
from ConvNet.tools.preprocess_data import mkdir
from ConvNet.tools.draw import draw_train_process
from ConvNet.config.defaults import get_cfg_defaults
from sklearn.metrics import mean_squared_error
import time

# 调用配置文件
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cfg = get_cfg_defaults()
# 神经网络初始化
net = CNN()
net.apply(conv_init)
device = cfg.MODEL.DEVICE1
net = net.to(device)
# 定义损失函数与优化器参数
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.BASE_LR)
# 定义路径
label_path = cfg.INPUT.VERTICS_PATH
image_path = cfg.DATASETS.SAVE_RESIZE_IMAGES
label_list = os.listdir(label_path)
label_list.sort(key=lambda x: len(x))
image_list = os.listdir(image_path)
image_list.sort(key=lambda x: len(x))

train_loss = []
loader = DataLoader(DataSet(image_path), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=True)


def train(number):
    for i, (images, labels) in enumerate(loader, start=1):  # index为所采用图片的位序，对应.obj的文件名
        start = time.time()
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)  # 网络前向运行
        print('epoch:', number+1, '  batch:', i, '\npredictive labels：\n', output)
        loss = criterion(output, labels)  # 计算网络的损失函数
        with open(cfg.OUTPUT.LOGGING, 'a') as f:
            print('Loss =', loss.item(), file=f)
        train_loss.append(loss.item())
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 优化更新权重
        if len(train_loss) % 100 == 0 and len(train_loss) < 30000:
            iters = range(len(train_loss))
            draw_train_process(cfg.VISUAL.TITLE, iters, train_loss, cfg.VISUAL.LINE_LABEL)
        end = time.time()
        with open(cfg.OUTPUT.LOGGING, 'a') as f:
            print('第', number + 1, '轮  第', i, '次用时为: ', end - start, file=f)


def proofread():
    net.eval()  # 必须有此句，否则有输入数据，即使不训练也会改变权值。
    pre_vertics = []
    rand = np.random.randint(0, len(image_list))
    with open(cfg.OUTPUT.LOGGING, 'a') as f:
        print('测试图片为:', image_list[rand], '\n对应Obj为：', label_list[rand], file=f)
    test = SingleTest(img_path=image_path + image_list[rand])
    image = test.output_data(img_path=image_path + image_list[rand])
    image = image.expand(cfg.TEST.IMAGE_BATCH, cfg.TEST.IMAGE_CHANNEL, cfg.TEST.IMAGE_LENGTH, cfg.TEST.IMAGE_HEIGHT)
    image = image.to(device)
    with torch.no_grad():
        predict = net(image)
        predict = predict[0].cpu().numpy()
        vertics, faces = loadObj(label_path + label_list[rand])
        for order in range(len(predict) // cfg.TEST.VERTICS_DIMENSION):
            pre_vertics.append(
                [predict[order], predict[order + len(predict) // cfg.TEST.VERTICS_DIMENSION],
                 predict[order + len(predict) // cfg.TEST.VERTICS_DIMENSION * 2]])
        loss = mean_squared_error(vertics, pre_vertics)
        with open(cfg.OUTPUT.LOGGING, 'a') as f:
            print('测试图片输出数据Loss =', loss, file=f)
        mkdir(cfg.TEST.SAVE_OBJ)
        writeObj(cfg.TEST.SAVE_OBJ + '/' + str(rand) + '_test.obj', pre_vertics, faces)


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
        learning = 1 / (10 * number)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning
    with open(cfg.OUTPUT.LOGGING, 'a') as f:
        print('学习率已更新为：', learning, file=f)


def run():
    epoch = cfg.INPUT.BASE_EPOCH
    net.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_NET_FILENAME))
    print('loaded net successfully!')
    optimizer.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_OPTIMIZER_FILENAME))
    print('loaded optimizer successfully!')
    while True:
        train(epoch)
        proofread()
        if epoch % cfg.DATASETS.SAVE_INTERVAL == 0:
            save(epoch)
        adjust_learning_rate(epoch)
        epoch += 1


if __name__ == '__main__':
    run()
