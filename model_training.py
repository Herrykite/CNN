import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from ConvNet.deal_with_obj import loadObj, writeObj
from ConvNet.input_transform import DataSet, SingleTest
from ConvNet.cnn import CNN
from ConvNet.preprocess_data import get_vertics, mkdir
from sklearn.metrics import mean_squared_error

# 初始化网络
net = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 定义损失函数：交叉熵或MSE
criterion = torch.nn.MSELoss()
# 定义网络优化方法：Adam
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
# 定义路径
path = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/output_v2/total/'
image_path = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/c2_rot/'
label_list = os.listdir(path)
image_list = os.listdir(image_path)

# def normalize(initial_x):
#     x_mean = np.mean(initial_x)
#     # x_min = np.min(initial_x)
#     x_std = np.std(initial_x, ddof=1)  # 加入ddof=1则为无偏样本标准差
#     normalized_x = (initial_x - x_mean) / x_std
#     # normalized_x = initial_x - x_min
#     return normalized_x


# def tell_vertics(count):
#     vertics_x, vertics_y, vertics_z = [], [], []
#     if file_list[count].endswith('.obj'):
#         vertics, faces = loadObj(path + file_list[count])
#         for i in range(len(vertics)):
#             vertics_x.append(vertics[i][0])
#             vertics_y.append(vertics[i][1])
#             vertics_z.append(vertics[i][2])
#     return vertics_x, vertics_y, vertics_z


# def tell_vertics_combine(vertics_x, vertics_y, vertics_z):
#     v_data = []
#     for j in range(len(vertics_x)):
#         v_data.append(vertics_x[j])
#     for j in range(len(vertics_y)):
#         v_data.append(vertics_y[j])
#     for j in range(len(vertics_z)):
#         v_data.append(vertics_z[j])
#     v_data = np.array(v_data, dtype=np.float32)
#     return v_data


def save(number):
    mkdir('./CNN_output_parameter')
    torch.save(net.state_dict(), './CNN_output_parameter/' + str(number + 1) + '_CNN.pkl')
    torch.save(optimizer.state_dict(), './CNN_output_parameter/opt.pkl')
    print('\n网络参数已保存!\n')


def train(number):
    net.train()
    for i, (images, index) in enumerate(loader, start=1):
        vertics = []  # 每一批Batch以后重置Obj获取的顶点
        for batch in range(len(images)):  # images为图片数据，index为图片索引数组，images大小为128*1*240*320
            vertics.append(get_vertics(index[0][batch]))
        images = images.to(device)
        labels = torch.tensor(vertics)
        labels = labels.to(device)
        print('第', number + 1, '轮   第', i, '组：')
        print('初始数据：\n', labels)
        optimizer.zero_grad()
        output = net(images)  # 网络前向运行
        predict = sum(output) / 128
        print('预测值：\n', output, '\n预测值维度：', len(predict))
        loss = criterion(output, labels)  # 计算网络的损失函数
        print('Loss =', loss.item())
        train_loss.append(loss.item())
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 优化更新权重
        iters = range(len(train_loss))
        if len(train_loss) % 90 == 0:
            draw_train_process('The Training Process', iters, train_loss, 'Loss')


def proofread(number):
    net.eval()  # 必须有此句，否则有输入数据，即使不训练也会改变权值。
    pre_vertics = []
    loss = 0
    rand = np.random.randint(0, len(image_list))
    test = SingleTest(img_path=image_path+image_list[rand])
    image = test.output_data(img_path=image_path+image_list[rand])
    image = image.expand(64, 1, 320, 240)
    image = image.to(device)
    with torch.no_grad():
        predict = net(image)
    predict = predict[0].cpu().numpy()
    vertics, faces = loadObj(path+label_list[rand])
    for order in range(len(predict) // 3):
        pre_vertics.append([predict[order], predict[order + len(predict) // 3], predict[order + len(predict) // 3 * 2]])
        loss += mean_squared_error(vertics[order], pre_vertics[order])
    print('测试图片输出数据Loss =', loss)
    mkdir('./CNN_test_output')
    writeObj('./CNN_test_output/' + str(number) + '_test.obj', pre_vertics, faces)


def draw_train_process(title, i, loss, label):
    plt.title(title, fontsize=24)
    plt.xlabel("Batch", fontsize=20)
    plt.plot(i, loss, color='c', label=label)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    train_loss = []
    net.load_state_dict(torch.load('./CNN_saved_parameter/_CNN.pkl'))
    optimizer.load_state_dict(torch.load('./CNN_saved_parameter/opt.pkl'))
    image_address = DataSet(image_path)
    loader = DataLoader(image_address, batch_size=128, shuffle=True)
    epoch = 0
    while True:
        train(epoch)
        proofread(epoch)
        if epoch % 10 == 0:
            save(epoch)
        epoch += 1
