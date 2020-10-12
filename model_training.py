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
    return np.array(v_data, dtype=np.float32)


def train(order):
    pkl_list = os.listdir('D:/DIGISKY/CNNTEST')
    # if os.path.exists('D:/DIGISKY/CNNTEST/' + str(order) + '_CNN.pkl'):
    #     net.load_state_dict(torch.load('D:/DIGISKY/CNNTEST/' + pkl_list[len(pkl_list) - 1]))
    net.train()
    for i, images in enumerate(data_train_loader, start=1):
        images, labels = images.to(device), train_labels.to(device)
        print('第', order + 1, '组原始数据：\n', labels)
        optimizer.zero_grad()
        # 网络前向运行
        output = net(images)
        print('预测值：\n', sum(output) / 8)
        # 计算网络的损失函数
        loss = criterion(sum(output) / 8, labels)
        # 反向传播梯度
        loss.backward()
        # 优化更新权重
        optimizer.step()
        train_loss.append(loss.detach().cuda().item())
        print('Loss =', train_loss[len(train_loss)-1])
        if len(train_loss) > 1:
            if train_loss[len(train_loss) - 1] - train_loss[len(train_loss) - 2] < 0:
                torch.save(net.state_dict(), 'D:/DIGISKY/CNNTEST/' + str(order + 1) + '_CNN.pkl')
            else:
                pass
        else:
            pass
        break


if __name__ == '__main__':
    train_labels, train_loss = [], []
    for num in range(len(file_list) - 1):
        x, y, z = tell_vertics(num)
        vertics = tell_vertics_combine(x, y, z)
        data_train = DataSet(image_path)
        train_labels = torch.tensor(vertics)
        data_train_loader = DataLoader(data_train, batch_size=8, shuffle=True)
        train(num)
