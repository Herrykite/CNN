# -*- coding: UTF-8 -*-
import sys

sys.path.insert(0, '../../')
from ConvNet.config.defaults import get_cfg_defaults

cfg = get_cfg_defaults()

print('输入用于训练的人脸照片地址(注意将地址中的斜杠改为/或双斜杠)：')
images_path = input().split(' ')[0]
while True:
    if images_path == '':
        images_path = input().split(' ')[0]
    else:
        break
cfg.DATASETS.IMAGES_PATH = images_path + '/'

print('输入用于训练的.obj文件地址(注意将地址中的斜杠改为/或双斜杠)：')
vertics_path = input().split(' ')[0]
while True:
    if vertics_path == '':
        vertics_path = input().split(' ')[0]
    else:
        break
cfg.INPUT.VERTICS_PATH = vertics_path + '/'

print('输入用于保存规整后相片文件的地址(勿与前两次输入的原始照片数据地址一致)：')
saved_resized_path = input().split(' ')[0]
while True:
    if saved_resized_path == '':
        saved_resized_path = input().split(' ')[0]
    else:
        break
cfg.INPUT.SAVE_RESIZE_IMAGES = saved_resized_path + '/'

print('输入用于保存训练拟合所得.obj文件的地址(勿与前三次输入的地址一致)：')
save_obj = input().split(' ')[0]
while True:
    if save_obj == '':
        save_obj = input().split(' ')[0]
    else:
        break
cfg.TEST.SAVE_OBJ = save_obj


def save_cfg_defaults():
    return cfg.clone()
