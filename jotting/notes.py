import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ConvNet.config.defaults import get_cfg_defaults
from sklearn.decomposition import PCA
from ConvNet.tools.deal_with_obj import loadObj

# if np.any(np.isnan(input.cpu().numpy())):   # 判断输入数据是否存在nan
#     print('Input data has NaN!')
#
# if np.isnan(loss.item()):                   # 判断损失是否为nan
#     print('Loss value is NaN!')


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    image_path = cfg.INPUT.SAVE_RESIZE_IMAGES
    # net = CNN()
    # optimizer = torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.BASE_LR)
    # net.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_NET_FILENAME))
    # print('loaded net successfully!')
    # optimizer.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_OPTIMIZER_FILENAME))
    # print('loaded optimizer successfully!')
    # loader = DataLoader(DataSet(image_path), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=True)
    # for i, (images, labels) in enumerate(loader, start=1):
    input_path = cfg.INPUT.VERTICS_PATH
    file_list = os.listdir(input_path)
    file_list.sort(key=lambda x: len(x))
    data = []
    for i in range(len(file_list)):
        print(i, 'finished')
        vertics, faces = loadObj(input_path + file_list[i])
        data.append(np.array(vertics).reshape((7657, 3)))
    data = np.array(data)
    data = data.reshape(len(file_list)-4000, 3*7657).T
    fig = plt.figure()
    ax = Axes3D(fig)
    pca = PCA(n_components=826)
    pca_feature = pca.fit_transform(data)
    ax.scatter(pca_feature[0, :], pca_feature[1, :], pca_feature[2, :], alpha=0.9, edgecolors='white')
    feature_info = np.zeros(826)
    for j in range(826):
        feature_info[j] = sum(pca.explained_variance_ratio_[0:j])
    A = pca.inverse_transform(pca_feature)


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
