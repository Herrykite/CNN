from yacs.config import CfgNode


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()
_C.MODEL = CfgNode()
_C.MODEL.DEVICE1 = 'cuda'
_C.MODEL.DEVICE2 = 'cpu'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASETS = CfgNode()
_C.DATASETS.SAVE_INTERVAL = 500
# 保存参数轮次间隔
_C.DATASETS.TRANSFORM_RESIZE = 240
# 统一缩放
_C.DATASETS.BRIGHTNESS = (0.5, 1.5)
# 亮度
_C.DATASETS.CONTRAST = (0.5, 1.5)
# 对比度
_C.DATASETS.SATURATION = (0.5, 1.5)
# 饱和度
_C.DATASETS.HUE = (-0.25, 0.25)
# 色相
_C.DATASETS.RANDOMROTATION = 5
# 随机旋转角度（-5°, 5°）
_C.DATASETS.DEGREES = 0
# 仿射变换不进行再次旋转
_C.DATASETS.TRANSLATE = (0.05, 0.05)
# 仿射变换进行平移时长宽区间的比例系数
_C.DATASETS.SCALE = (0.95, 1.05)
# 仿射变换缩放比例
_C.DATASETS.SHEAR = (-5, 5, -5, 5)
# 仿射变换错切角度范围
_C.DATASETS.FILLCOLOR = 0
# 填充颜色为黑色
_C.DATASETS.RANDOMERASING_P = 1.0
# 随机擦除概率
_C.DATASETS.RANDOMERASING_SCALE = (0.001, 0.01)
# 随机擦除按均匀分布概率抽样，遮挡区域的面积 = image * scale
_C.DATASETS.RANDOMERASING_RATIO = (0.5, 2.0)
# 随机擦除遮挡区域的宽高比范围
_C.DATASETS.RANDOMERASING_VALUE = 0
# 随机擦除遮挡区域的像素值

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

_C.SOLVER = CfgNode()
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.ADJUST_LR = 1e-5
_C.SOLVER.FIRST_ADJUST_LIMIT = 100
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
_C.INPUT.VERTICS_PATH = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/output_v2/total/'
_C.INPUT.VERTICS_PATH_MINI = 'D:/DIGISKY/CNNTEST/labels/'
_C.INPUT.IMAGES_PATH = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/c2_rot/'
_C.INPUT.IMAGES_PATH_MINI = 'D:/DIGISKY/CNNTEST/images/'
_C.INPUT.INITIALIZE_NET = '../output/CNN_saved_parameter/_CNN.pkl'
_C.INPUT.INITIALIZE_OPTIMIZER = '../output/CNN_saved_parameter/opt.pkl'
_C.INPUT.CHECK = '../output/check/'
_C.INPUT.BATCH_SIZE = 2
_C.INPUT.BASE_EPOCH = 0

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #

_C.OUTPUT = CfgNode()
# 保存、加载模型参数的地址
_C.OUTPUT.PARAMETER = '../output/CNN_output_parameter'
# 数据预处理后的读取地址
_C.OUTPUT.CNN_INITIAL_DATA_PATH = '../output/CNN_output_data'
_C.OUTPUT.CNN_INITIAL_DATA_PATH_MINI = '../output/CNN_output_data_MINI'
# 数据预处理所需的.json模板
_C.OUTPUT.CNN_FORMAT_JSON_FILENAME = 'CNN_Format_Camera2.data'
# 加载模型网络参数的文件名
_C.OUTPUT.SAVE_NET_FILENAME = '/1001_CNN.pkl'
# 加载模型优化器参数的文件名
_C.OUTPUT.SAVE_OPTIMIZER_FILENAME = '/1001_opt.pkl'
# 记录打印内容
_C.OUTPUT.LOGGING = '../output/log.txt'

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #

_C.TEST = CfgNode()
_C.TEST.BASE_LOSS = 0
_C.TEST.IMAGE_BATCH = 1
_C.TEST.IMAGE_CHANNEL = 1
_C.TEST.IMAGE_LENGTH = 320
_C.TEST.IMAGE_HEIGHT = 240
_C.TEST.VERTICS_DIMENSION = 3
# 测试输出路径
_C.TEST.SAVE_OBJ = '../output/CNN_test_output'


def get_cfg_defaults():
    return _C.clone()
