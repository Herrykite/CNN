# -*- coding: UTF-8 -*-
import sys

sys.path.insert(0, '../../')
import os
from ConvNet.config.defaults import get_cfg_defaults
from ConvNet.transform.datasets_transform import get_mean_std
cfg = get_cfg_defaults()
cfg.merge_from_file(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1])
get_mean_std()
