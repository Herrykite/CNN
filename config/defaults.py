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
_C.DATASETS.SAVE_INTERVAL = 10
_C.DATASETS.TRANSFORM_RESIZE = 240

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
_C.INPUT.IMAGES_PATH = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/c2_rot/'
_C.INPUT.INITIALIZE_NET = '../CNN_saved_parameter/_CNN.pkl'
_C.INPUT.INITIALIZE_OPTIMIZER = '../CNN_saved_parameter/opt.pkl'
_C.INPUT.BATCH_SIZE = 128
_C.INPUT.BASE_EPOCH = 0

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #

_C.OUTPUT = CfgNode()
# 保存、加载模型参数的地址
_C.OUTPUT.PARAMETER = '../CNN_output_parameter'
# 数据预处理后的读取地址
_C.OUTPUT.CNN_INITIAL_DATA_PATH = '../CNN_output_data'
# 数据预处理所需的.json模板
_C.OUTPUT.CNN_FORMAT_JSON_FILENAME = 'CNN_Format_Camera2.data'
# 加载模型网络参数的文件名
_C.OUTPUT.SAVE_NET_FILENAME = '/31_CNN.pkl'
# 加载模型优化器参数的文件名
_C.OUTPUT.SAVE_OPTIMIZER_FILENAME = '/31_opt.pkl'

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #

_C.TEST = CfgNode()
_C.TEST.BASE_LOSS = 0
_C.TEST.IMAGE_BATCH = 64
_C.TEST.IMAGE_CHANNEL = 1
_C.TEST.IMAGE_LENGTH = 320
_C.TEST.IMAGE_HEIGHT = 240
_C.TEST.VERTICS_DIMENSION = 3
# 测试输出路径
_C.TEST.SAVE_OBJ = '../CNN_test_output'


def get_cfg_defaults():
    return _C.clone()

