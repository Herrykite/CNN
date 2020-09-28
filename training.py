import torch
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from Convolutional.cnn import CNN
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 读取数据
data_train = MNIST('../data/MNIST', download=True,
                   transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
data_test = MNIST('../data/MNIST', train=False, download=True,
                  transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

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
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


# 定义训练阶段
def loss_train(epoch):
    file_list = os.listdir('D:/DIGISKY/CNNTEST')
    net.load_state_dict(torch.load('D:/DIGISKY/CNNTEST/' + file_list[len(file_list) - 1]))
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
        _, train_predicted = torch.max(output.data, 1)
        # torch.max(x, 1)中的1表示行，0表示列
        # 单个的predicted对数字判断的结果矩阵是有train对应的Batch=256这么多，此处为234组256,最后一次取尽剩余，为第235组
        total = labels.size(0)  # labels 的长度
        correct = (train_predicted == labels).sum().item()  # 对预测正确的数字项进行累加计数
        train_accs.append(100 * correct / total)
        if i % 47 == 0:
            data_show(images)
            print('训练轮次： %d, 样本序号: %d, Loss值: %f' % (epoch, i, loss.detach().cuda().item()))
            print('此段用于检测对应关系：\n原始数据如下：\n', labels, '\n预测值如下：\n', train_predicted)
            print('训练数据准确度:', train_accs[i - 1], '%\n\n')
        # 反向传播梯度
        loss.backward()
        # 优化更新权重
        optimizer.step()
    # 保存网络模型结构
    torch.save(net.state_dict(), 'D:/DIGISKY/CNNTEST/' + str(epoch) + '_model.pkl')
    return train_loss, train_accs


def loss_test():
    net.eval()
    # 必须有此句，否则有输入数据，即使不训练也会改变权值。这是net中含有Batch Normalization层所带来的的性质
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            total_loss += criterion(output, labels).sum()
            # 计算准确率,pred取行最大值及其索引
            test_predicted = output.detach().max(1)[1]
            # print('此段用于检验长度为', len(test_predicted), '的测试集的输出值：\n', test_predicted)
            # 单个的pred对数字判断的结果矩阵是有test对应的Batch=1024这么多，此处为9组1024，最后一次取尽剩余，为第10组
            total_correct += test_predicted.eq(labels.view_as(test_predicted)).sum()
        train_accs = float(total_correct) / len(data_test)
        avg_loss = total_loss / len(data_test)
        print('\n测试数据集的平均Loss值: %f, 准确率: %f' % (avg_loss, train_accs))


def data_show(images):
    images = images.cpu()
    # labels = labels.cpu()
    img = utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    # img是用来控制像素色彩的参数，以显示出数据对应的图形, img = img * [0.5] + [0.5]
    # for i in range(len(labels)):
    #     print(labels[i], end=" ")
    #     if i % 8 == 7:
    #         print(end='\n')
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
    for e in range(1, 6):
        rec_data = loss_train(e)
        for i in range(len(rec_data[0])):
            loss.append(rec_data[0][i])
            accu.append(rec_data[1][i])
        # print('show前print次数为：', len(rec_data[0])-1)
        # data_show()
        loss_test()
    test_iters = range(len(loss))
    draw_train_process('The Training Process', test_iters, loss, accu, 'Loss', 'Accuracy')


if __name__ == '__main__':
    main()
