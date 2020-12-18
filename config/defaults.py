# -*- coding: UTF-8 -*-
from yacs.config import CfgNode


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()
_C.MODEL = CfgNode()
_C.MODEL.DEVICE1 = 'cuda'
_C.MODEL.DEVICE2 = 'cpu'
_C.MODEL.CONFIG = '../config/config_cache/'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASETS = CfgNode()
_C.DATASETS.IMAGES_PATH = '//192.168.20.63/ai/double_camera_data/2020-08-21/160810/c1_rot/'
_C.DATASETS.SAVE_INTERVAL = 100
# 保存参数轮次间隔
_C.DATASETS.TRANSFORM_RESIZE = 240
# 统一缩放
_C.DATASETS.BRIGHTNESS = (0.75, 1.5)
# 亮度
_C.DATASETS.CONTRAST = (0.83333, 1.2)
# 对比度
_C.DATASETS.SATURATION = (0.83333, 1.2)
# 饱和度
_C.DATASETS.HUE = (-0.2, 0.2)
# 色相
_C.DATASETS.RANDOMROTATION = 5
# 随机旋转角度（-5°, 5°）
_C.DATASETS.DEGREES = 0
# 仿射变换不进行再次旋转
_C.DATASETS.TRANSLATE = (0.2, 0.2)
# 仿射变换进行平移时长宽区间的比例系数
_C.DATASETS.SCALE = (0.95, 1.05)
# 仿射变换缩放比例
_C.DATASETS.SHEAR = (-5, 5, -5, 5)
# 仿射变换错切角度范围
_C.DATASETS.RANDOMERASING_P = 0.5
# 随机擦除概率
_C.DATASETS.RANDOMERASING_SCALE = (0.001, 0.01)
# 随机擦除按均匀分布概率抽样，遮挡区域的面积 = image * scale
_C.DATASETS.RANDOMERASING_RATIO = (0.5, 2.0)
# 随机擦除遮挡区域的宽高比范围
_C.DATASETS.RANDOMERASING_VALUE = 0
# 随机擦除遮挡区域的像素值
_C.DATASETS.TRANSFORM_MEAN = 0.3376989895544583
_C.DATASETS.TRANSFORM_STD = 0.2616380854654215

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

_C.SOLVER = CfgNode()
_C.SOLVER.NUM_WORKERS = 8
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.ADJUST_LR = 1e-6
_C.SOLVER.FIRST_ADJUST_LIMIT = 10
_C.SOLVER.SECOND_ADJUST_LIMIT = 10000

# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #

_C.VISUAL = CfgNode()
_C.VISUAL.TITLE_FRONT_SIZE = 24
_C.VISUAL.LABEL_FRONT_SIZE = 20
_C.VISUAL.X_LABEL = 'The Number of Training Iterations'
_C.VISUAL.LINE_COLOR = 'c'
_C.VISUAL.TITLE = 'The Training Process'
_C.VISUAL.LINE_LABEL = 'Loss'

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #

_C.INPUT = CfgNode()
_C.INPUT.VERTICS_PATH = '//192.168.20.63/ai/double_camera_data/2020-08-21/160810/output_2/total/'
# _C.INPUT.VERTICS_PATH = 'D:/DIGISKY/CNN_1193Dataset/labels/'
_C.INPUT.SAVE_RESIZE_IMAGES = 'D:/DIGISKY/CNN_12987_20200821_160810_C1/'
# _C.INPUT.SAVE_RESIZE_IMAGES = 'D:/DIGISKY/CNN_1193Dataset/images/'
_C.INPUT.CHECK = '../output/check/'
_C.INPUT.BATCH_SIZE = 16
_C.INPUT.BASE_EPOCH = 1
_C.INPUT.VERTICS_NUM = 22971
_C.INPUT.PCA_DIMENSION = 638


# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #

_C.OUTPUT = CfgNode()
# 保存、加载模型参数的地址
_C.OUTPUT.PARAMETER = '../output/CNN_output_parameter'
# 数据预处理后的读取地址
_C.OUTPUT.CNN_INITIAL_DATA_PATH = '../output/20200821_160810_C1_data'
# _C.OUTPUT.CNN_INITIAL_DATA_PATH = '../output/CNN_output_data_MAX'
# 数据预处理所需的.json模板
_C.OUTPUT.CNN_FORMAT_JSON_FILENAME = 'CNN_Format_Camera2.data'
# 加载模型网络参数的文件名
_C.OUTPUT.SAVE_NET_FILENAME = '/20200908_162524_C2_CNN.pkl'
# 加载模型优化器参数的文件名
_C.OUTPUT.SAVE_OPTIMIZER_FILENAME = '/20200908_162524_C2_opt.pkl'
# 记录打印内容
_C.OUTPUT.LOGGING = '../output/log.txt'
# _C.OUTPUT.LOGGING = '../output/log_MINI.txt'

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #

_C.TEST = CfgNode()
_C.TEST.BASE_LOSS = 0
_C.TEST.IMAGE_BATCH = 1
_C.TEST.IMAGE_CHANNEL = 1
_C.TEST.IMAGE_LENGTH = 320
_C.TEST.IMAGE_HEIGHT = 240
# 测试输出路径
# _C.TEST.SAVE_OBJ = 'D:/DIGISKY/OBJ/64_MINI'
_C.TEST.SAVE_OBJ = 'D:/DIGISKY/OBJ/20200821_160810_C1'


def get_cfg_defaults():
    return _C.clone()

