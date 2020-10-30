import json
import os
import re
import numpy as np
from ConvNet.tools.deal_with_obj import loadObj
from ConvNet.config.defaults import get_cfg_defaults

cfg = get_cfg_defaults()
path = cfg.OUTPUT.CNN_INITIAL_DATA_PATH


def mkdir(creat_path):
    folder = os.path.exists(creat_path)
    if not folder:
        os.makedirs(creat_path)
        print('对应路径文件创建完毕。')
    else:
        pass


def Woodblock(order, data):  # 活字印刷术
    listdir = os.listdir('../')
    listdir.sort(key=lambda obj: len(obj))
    if cfg.OUTPUT.CNN_FORMAT_JSON_FILENAME in listdir:
        f = open('../' + cfg.OUTPUT.CNN_FORMAT_JSON_FILENAME)
        model = json.load(f)
        f.close()
        for j in range(3):
            for i in range(len(data[0][0])):
                model['camera2']['mesh'][0]['points'].append(data[0][0][i][j])
        for i in range(len(data[0][1])):
            model['camera2']['mesh'][1]['points'].append(data[0][1][i])
        content = json.dumps(model, indent=4, separators=(',', ': '))
        new_file = open(path + '/' + str(order) + '.data', 'w')
        new_file.write(content)
        new_file.close()


def initial_load():
    input_path = cfg.INPUT.VERTICS_PATH
    file_list = os.listdir(input_path)
    file_list.sort(key=lambda x: len(x))
    mkdir(cfg.OUTPUT.CNN_INITIAL_DATA_PATH)
    for count in range(len(file_list)):
        datas = []
        if file_list[count].endswith('.obj'):
            vertics, faces = loadObj(input_path + file_list[count])
            datas.append([vertics, faces])
            print(count, '.obj finished')

        Woodblock(count, datas)  # 此句执行完以后均存储为.data文件


def get_vertics(count):
    f = open(path + '/' + re.sub("\D", "", str(count)) + '.data')
    model = json.load(f)
    f.close()
    vertics = np.array(model['camera2']['mesh'][0]['points'], dtype=np.float32)
    faces = model['camera2']['mesh'][1]['points']
    return vertics, faces


if __name__ == '__main__':
    initial_load()
