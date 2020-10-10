import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from Convolutional.deal_with_obj import loadObj
from Convolutional.input_transform import DataSet
from Convolutional.cnn import CNN

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
    return np.array(v_data, dtype=np.float32)


def normalize(initial_x):
    x_mean = np.mean(initial_x)
    x_std = np.std(initial_x, ddof=1)  # 加入ddof=1则为无偏样本标准差
    normalized_x = (initial_x - x_mean) / x_std
    return normalized_x


def tell_labels(l_x, l_y, l_z):
    label = tell_vertics_combine(normalize(l_x), normalize(l_y), normalize(l_z))
    return label


def train(epoch):
    pkl_list = os.listdir('D:/DIGISKY/CNNTEST')
    net.load_state_dict(torch.load('D:/DIGISKY/CNNTEST/' + file_list[len(pkl_list) - 1]))
    net.train()
    train_loss, batch_list = [], []
    for i, images in enumerate(data_train_loader, start=1):
        images, labels = images.to(device), train_labels.to(device)
        print('图片：\n', images[epoch][0])
        print(labels, '\n标签个数：', len(labels))
        # 初始0梯度
        optimizer.zero_grad()
        # 网络前向运行
        output = net(images)
        print('输出值：\n', output[epoch], '\n输出值维度:', len(output[epoch]))
        # 计算网络的损失函数
        loss = criterion(output[epoch], labels)
        train_loss.append(loss.detach().cuda().item())
        print('\nloss:', loss, '\ntrain_loss:', train_loss)
        # 反向传播梯度
        loss.backward()
        # 优化更新权重
        optimizer.step()
    torch.save(net.state_dict(), 'D:/DIGISKY/CNNTEST/' + str(epoch) + '_CNN.pkl')
    return train_loss


if __name__ == '__main__':
    train_labels, test_labels = [], []
    for num in range(len(file_list) - 5738):
        x, y, z = tell_vertics(num)
        vertics = tell_vertics_combine(x, y, z)
        data_train = DataSet(image_path + 'train')
        train_labels = torch.tensor(vertics)
        data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True)
        for e in range(5):
            train_loss = train(e)

