import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from ConvNet.config.defaults import get_cfg_defaults
from ConvNet.modeling.newcnn import CNN
from ConvNet.transform.datasets_transform import DataSet
from ConvNet.engine.trainer import proofread

# if np.any(np.isnan(input.cpu().numpy())):   # 判断输入数据是否存在nan
#     print('Input data has NaN!')
#
# if np.isnan(loss.item()):                   # 判断损失是否为nan
#     print('Loss value is NaN!')


def normalize(initial_x):
    x_mean = np.mean(initial_x)
    # x_min = np.min(initial_x)
    x_std = np.std(initial_x, ddof=1)  # 加入ddof=1则为无偏样本标准差
    normalized_x = (initial_x - x_mean) / x_std
    # normalized_x = initial_x - x_min
    return normalized_x


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


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    image_path = cfg.INPUT.SAVE_RESIZE_IMAGES
    net = CNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.BASE_LR)
    net.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_NET_FILENAME))
    print('loaded net successfully!')
    optimizer.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_OPTIMIZER_FILENAME))
    print('loaded optimizer successfully!')
    loader = DataLoader(DataSet(image_path), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=True)
    for i, (images, labels) in enumerate(loader, start=1):
        tb = SummaryWriter('./')
        grid = torchvision.utils.make_grid(images)
        tb.add_image('Image', grid, 0)
        tb.add_graph(net, images)
        # tb.add_scalar('Loss', loss.item(), i)
        tb.add_histogram('conv1.weight', net.conv_net.conv1.weight)
        tb.add_histogram('bn2.weight', net.conv_net.bn2.weight)
        tb.add_histogram('bn2.bias', net.conv_net.bn2.bias)
        tb.add_histogram('conv3.weight', net.conv_net.conv3.weight)
        tb.add_histogram('conv3.bias', net.conv_net.conv3.bias)
        tb.add_histogram('bn4.weight', net.conv_net.bn4.weight)
        tb.add_histogram('bn4.bias', net.conv_net.bn4.bias)
        tb.add_histogram('conv5.weight', net.conv_net.conv5.weight)
        tb.add_histogram('conv5.bias', net.conv_net.conv5.bias)
        tb.add_histogram('bn6.weight', net.conv_net.bn6.weight)
        tb.add_histogram('bn6.bias', net.conv_net.bn6.bias)
        tb.add_histogram('conv7.weight', net.conv_net.conv7.weight)
        tb.add_histogram('conv7.bias', net.conv_net.conv7.bias)
        tb.add_histogram('bn8.weight', net.conv_net.bn8.weight)
        tb.add_histogram('bn8.bias', net.conv_net.bn8.bias)
        tb.add_histogram('conv9.weight', net.conv_net.conv9.weight)
        tb.add_histogram('conv9.bias', net.conv_net.conv9.bias)
        tb.add_histogram('bn10.weight', net.conv_net.bn10.weight)
        tb.add_histogram('bn10.bias', net.conv_net.bn10.bias)
        tb.add_histogram('conv11.weight', net.conv_net.conv11.weight)
        tb.add_histogram('conv11.bias', net.conv_net.conv11.bias)
        tb.add_histogram('bn12.weight', net.conv_net.bn12.weight)
        tb.add_histogram('bn12.bias', net.conv_net.bn12.bias)
        tb.add_histogram('fc.weight', net.fc.fc.weight)
        tb.add_histogram('fc.bias', net.fc.fc.bias)
        tb.close()
    # tensorboard --logdir=./
