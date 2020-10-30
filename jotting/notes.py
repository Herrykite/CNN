import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from ConvNet.config.defaults import get_cfg_defaults
from ConvNet.modeling.cnn import CNN
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
    # image_path = cfg.DATASETS.SAVE_RESIZE_IMAGES
    # net = CNN()
    # optimizer = torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.BASE_LR)
    # net.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_NET_FILENAME))
    # print('loaded net successfully!')
    # optimizer.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_OPTIMIZER_FILENAME))
    # print('loaded optimizer successfully!')
    # loader = DataLoader(DataSet(image_path), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=True)
    # for i, (images, labels) in enumerate(loader, start=1):
    #     writer = SummaryWriter('./')
    #     grid = torchvision.utils.make_grid(images)
    #     writer.add_image('Image', grid, 0)
    #     writer.add_graph(net, images)
    #     writer.close()
