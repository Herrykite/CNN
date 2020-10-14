from __future__ import division
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from ConvNet.deal_with_obj import loadObj
from ConvNet.input_transform import DataSet
from ConvNet.cnn import CNN

# 初始化网络
net = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
# 定义损失函数：交叉熵或MSE
criterion = torch.nn.MSELoss()
# 定义网络优化方法：Adam
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# 定义路径
path = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/output_v2/total/'
image_path = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/c2_rot/'
file_list = os.listdir(path)


def normalize(initial_x):
    x_mean = np.mean(initial_x)
    # x_min = np.min(initial_x)
    x_std = np.std(initial_x, ddof=1)  # 加入ddof=1则为无偏样本标准差
    normalized_x = (initial_x - x_mean) / x_std
    # normalized_x = initial_x - x_min
    return normalized_x


def tell_vertics(count):
    vertics_x, vertics_y, vertics_z = [], [], []
    if file_list[count].endswith('.obj'):
        vertics, faces = loadObj(path + file_list[count])
        for i in range(len(vertics)):
            vertics_x.append(vertics[i][0])
            vertics_y.append(vertics[i][1])
            vertics_z.append(vertics[i][2])
    return vertics_x, vertics_y, vertics_z


def tell_vertics_combine(vertics_x, vertics_y, vertics_z):
    v_data = []
    for j in range(len(vertics_x)):
        v_data.append(vertics_x[j])
    for j in range(len(vertics_y)):
        v_data.append(vertics_y[j])
    for j in range(len(vertics_z)):
        v_data.append(vertics_z[j])
    v_data = np.array(v_data, dtype=np.float32)
    return v_data


def train(number):
    for i, (images, index) in enumerate(loader, start=1):
        images = images.to(device)
        vertics = []  # 每一批Batch以后重置Obj获取的顶点
        for batch in range(len(images)):  # data[0]为图片数据，data[1]为图片索引数组，data[0]大小为64*1*240*320
            x, y, z = tell_vertics(index[0][batch])
            vertics.append(tell_vertics_combine(x, y, z))
        labels = torch.tensor(vertics)
        labels = labels.to(device)
        print('第', number+1, '轮   第', i, '组原始数据：\n', labels)
        print('初始数据维度：', len(labels))
        optimizer.zero_grad()
        output = net(images)              # 网络前向运行
        print('预测值：\n', output, '\n预测值维度：', len(output))
        loss = criterion(output, labels)  # 计算网络的损失函数
        loss.backward()                   # 反向传播梯度
        optimizer.step()                  # 优化更新权重
        print('Loss =', loss)
        if i % 10 == 9:
            torch.save(net.state_dict(), 'D:/DIGISKY/CNNTEST/' + str(number + 1) + '_CNN.pkl')
            print('\n网络参数已保存!\n')


if __name__ == '__main__':
    # net.load_state_dict(torch.load('D:/DIGISKY/CNNTEST/' + pkl_list[len(pkl_list) - 1]))
    pkl_list = os.listdir('D:/DIGISKY/CNNTEST')
    image_address = DataSet(image_path)
    loader = DataLoader(image_address, batch_size=128, shuffle=True)
    for epoch in range(10):
        train(epoch)

