# 本代码为训练与测试数据集相关的第一个版本
import torch
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from Convolutional.cnn import CNN
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 读取数据
data_train = MNIST('../data/MNIST',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('../data/MNIST',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

# num_workers=8 使用多进程加载数据
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

# 初始化网络
net = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

# 定义损失函数：交叉熵
criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, reduce=None, reduction='mean')
# 定义网络优化方法：Adam
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)


# 定义训练阶段
def train(epoch):
    net.train()
    train_accs, train_loss, batch_list = [], [], []
    for i, (images, labels) in enumerate(data_train_loader, start=1):
        images, labels = images.to(device), labels.to(device)
        # 初始0梯度
        optimizer.zero_grad()
        # 网络前向运行
        output = net(images)
        # 计算网络的损失函数
        loss = criterion(output, labels)
        # 存储每一次的梯度与迭代次数
        train_loss.append(loss.detach().cuda().item())
        batch_list.append(i)
        # 训练曲线的绘制单个batch中的准确率
        _, predicted = torch.max(output.data, 1)
        # print('第', i, '次predicted', predicted, len(predicted))
        # torch.max(x, 1)中的1表示行，0表示列
        # 单个的predicted对数字判断的结果矩阵是有train对应的Batch=256这么多，此处为234组256,最后一次取尽剩余，为第235组
        total = labels.size(0)  # labels 的长度
        correct = (predicted == labels).sum().item()  # 对预测正确的数字项进行累加计数
        train_accs.append(100 * correct / total)

        if i % 25 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cuda().item()))
        # 反向传播梯度
        loss.backward()
        # 优化更新权重
        optimizer.step()
    # 保存网络模型结构
    torch.save(net.state_dict(), 'D:/DIGISKY/CNNTEST/' + str(epoch) + '_model.pkl')
    return train_loss, train_accs


def loss_test():
    # 验证阶段
    net.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            total_loss += criterion(output, labels).sum()
            # 计算准确率,pred取行最大值及其索引
            pred = output.detach().max(1)[1]
            # 单个的pred对数字判断的结果矩阵是有test对应的Batch=1024这么多，此处为9组1024，最后一次取尽剩余，为第10组
            total_correct += pred.eq(labels.view_as(pred)).sum()
        train_accs = float(total_correct) / len(data_test)
        avg_loss = total_loss / len(data_test)
        print('测试平均Loss值: %f, 准确率: %f' % (avg_loss, train_accs))


def show():
    images, lables = next(iter(data_train_loader))
    # 通过iter()函数获取这些可迭代对象的迭代器
    img = utils.make_grid(images)
    # transpose 转置函数(x=0,y=1,z=2),新的x是原来的y轴大小，新的y是原来的z轴大小，新的z是原来的x大小
    img = img.numpy().transpose(1, 2, 0)
    # img是用来控制像素色彩的参数，以显示出数据对应的图形
    # img = img * [0.5] + [0.5]
    for i in range(len(lables)):
        print(lables[i], end=" ")
        # 依据图像，以8个数字为一行
        if i % 8 == 7:
            print(end='\n')
    plt.imshow(img)
    plt.show()


def draw_train_process(title, iters, train_loss, train_accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("Batch", fontsize=20)
    plt.ylabel('Accuracy(%)', fontsize=20)
    plt.plot(iters, train_loss, color='r', label=label_cost)
    plt.plot(iters, train_accs, color='c', label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()


def main():
    loss, accu = [], []
    for e in range(1, 2):
        rec_data = train(e)
        for i in range(len(rec_data[0])):
            loss.append(rec_data[0][i])
            accu.append(rec_data[1][i])
            # print('show前print次数为：', i)
        show()
        loss_test()
    test_iters = range(len(loss))
    draw_train_process('The Training Process', test_iters, loss, accu, 'Loss', 'Accuracy')


if __name__ == '__main__':
    main()
