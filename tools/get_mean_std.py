# -*- coding: UTF-8 -*-
import sys

sys.path.insert(0, '../../')
from ConvNet.config.defaults import get_cfg_defaults
from ConvNet.transform.datasets_transform import get_mean_std
cfg = get_cfg_defaults()
get_mean_std()
