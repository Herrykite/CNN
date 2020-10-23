import numpy as np
import os
from ConvNet.config.defaults import get_cfg_defaults
from PIL import Image
from torchvision import transforms

# if np.any(np.isnan(input.cpu().numpy())):   # 判断输入数据是否存在nan
#     print('Input data has NaN!')
#
# if np.isnan(loss.item()):                   # 判断损失是否为nan
#     print('Loss value is NaN!')


# def tell_vertics(count):
#     vertics_x, vertics_y, vertics_z = [], [], []
#     if file_list[count].endswith('.obj'):
#         vertics, faces = loadObj(path + file_list[count])
#         for i in range(len(vertics)):
#             vertics_x.append(vertics[i][0])
#             vertics_y.append(vertics[i][1])
#             vertics_z.append(vertics[i][2])
#         normalized_x = normalize(np.array(vertics_x))
#         normalized_y = normalize(np.array(vertics_y))
#         normalized_z = normalize(np.array(vertics_z))
#         return normalized_x, normalized_y, normalized_z

# torch.nn.init.kaiming_normal_(images, a=0, mode='fan_in', nonlinearity='leaky_relu')

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
#
#     device = cfg.MODEL.DEVICE1
#     batch_size = cfg.INPUT.SIZE_TRAIN
#     print(batch_size)

transform = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(
        0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.25, 0.25)),
    # 亮度、对比度、饱和度、色相
    transforms.RandomRotation(5),
    # 随机旋转
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-5, 5, -5, 5), fillcolor=0),
    # 旋转角度、平移的宽高区间比例系数、缩放比例、错切角度、填充颜色
    transforms.ToTensor(),
    transforms.RandomErasing(p=1, scale=(0.001, 0.01), ratio=(0.5, 2.0), value=0),
    # 按均匀分布概率抽样，遮挡区域的面积 = image * scale, ratio为遮挡区域的宽高比范围, value为遮挡区域的像素值
    transforms.ToPILImage()
])

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    image_path = cfg.INPUT.IMAGES_PATH_MINI
    images = os.listdir(image_path)
    figure = Image.open(image_path + images[0]).convert('L')
    new_figure = transform(figure)
    new_figure.show()
