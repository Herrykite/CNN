import json
import os
import numpy as np
from ConvNet.deal_with_obj import loadObj


def Woodblock(order, data):            # 活字印刷术
    path1 = 'D:/DIGISKY/'
    listdir1 = os.listdir(path1)
    if 'CNN_Format_Camera2.data' in listdir1:
        f = open(path1 + 'CNN_Format_Camera2.data')
        model = json.load(f)
        f.close()
        for i in range(len(data[0][0])):
            model['camera2']['mesh'][0]['points'].append(data[0][0][i])
        for i in range(len(data[0][1])):
            model['camera2']['mesh'][1]['points'].append(data[0][1][i])
        content = json.dumps(model, indent=4, separators=(',', ': '))
        new_file = open(path1 + 'CNN_output_data/' + str(order) + '.data', 'w')
        new_file.write(content)
        new_file.close()


def initial_load():
    path = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/output_v2/total/'
    file_list = os.listdir(path)
    for count in range(len(file_list)):
        datas = []
        if file_list[count].endswith('.obj'):
            vertics, faces = loadObj(path + file_list[count])
            datas.append([vertics, faces])
        Woodblock(count, datas)  # 此句执行完以后均存储为.data文件


def get_vertics(count):
    f = open('D:/DIGISKY/CNN_output_data/'+count+'.data')
    model = json.load(f)
    vertics = []
    for j in range(3):
        for i in range(len(model['camera2']['mesh'][0]['points'])):
            vertics.append(model['camera2']['mesh'][0]['points'][i][j])
    return np.array(vertics, dtype=np.float32)
