import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from Convolutional.readobj import loadObj
from Convolutional.my_dataload import DataSet
from Convolutional.cnn import CNN

# 初始化网络
net = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
# 定义损失函数：交叉熵
criterion = torch.nn.MSELoss()
# 定义网络优化方法：Adam
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# 定义路径
path = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/output_v2/total/'
image_path = 'D:/DIGISKY/CNN_input/'
file_list = os.listdir(path)


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
        v_data.append(vertics_y[j])
        v_data.append(vertics_z[j])
    return np.array(v_data)


def normalize(initial_x):
    x_mean = np.mean(initial_x)
    x_std = np.std(initial_x, ddof=1)  # 加入ddof=1则为无偏样本标准差
    normalized_x = (initial_x - x_mean) / x_std
    return normalized_x


def tell_labels(l_x, l_y, l_z):
    label = tell_vertics_combine(normalize(l_x), normalize(l_y), normalize(l_z))
    return label


def train(epoch):
    net.train()
    for i, images in enumerate(data_train_loader, start=1):
        images, labels = images.to(device), train_labels.to(device)
        print(images, '\n', len(images))
        # 初始0梯度
        optimizer.zero_grad()
        # 网络前向运行
        output = net(images)
        # 计算网络的损失函数
        loss = criterion(output, labels)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        predicted = output.detach().max(1)[1]
        if epoch % 500 == 0:
            print('训练轮次： %d, Loss值: %f' % (epoch, loss.detach().cuda().item()))
            print('预测值如下：\n', predicted)
        # 反向传播梯度
        loss.backward()
        # 优化更新权重
        optimizer.step()


if __name__ == '__main__':
    train_labels, test_labels = [], []
    for num in range(len(file_list) - 5738):
        x, y, z = tell_vertics(num)
        vertics = tell_vertics_combine(x, y, z)
        train_labels = vertics[:7000 * 3]
        test_labels = vertics[7000 * 3:]
        data_train = DataSet(image_path + 'train')
        data_test = DataSet(image_path + 'test')
        train_labels = torch.tensor(train_labels)
        test_labels = torch.tensor(test_labels)
        data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
        data_test_loader = DataLoader(data_test, batch_size=64, num_workers=8)
        for e in range(10):
            train(train_labels)
