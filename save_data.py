import json
import os
from Convolutional.readobj import loadObj


def Woodblock(order, data):
    path1 = 'D:/DIGISKY/'
    listdir1 = os.listdir(path1)
    if 'CNN_Format_Camera1.data' in listdir1:
        f = open(path1 + 'CNN_Format_Camera1.data')
        model = json.load(f)
        f.close()
        for i in range(len(data[0][0])):
            model['camera1']['mesh'][0]['points'].append(data[0][0][i])
        for i in range(len(data[0][1])):
            model['camera1']['mesh'][1]['points'].append(data[0][1][i])
        content = json.dumps(model, indent=4, separators=(',', ': '))
        new_file = open(path1 + 'CNN_output_data/' + str(order) + '.data', 'w')
        new_file.write(content)
        new_file.close()


if __name__ == '__main__':
    path = '//192.168.20.63/ai/double_camera_data/2020-08-21/161240/output_v2/total/'
    file_list = os.listdir(path)
    for count in range(len(file_list)):
        datas = []
        if file_list[count].endswith('.obj'):
            vertics, faces = loadObj(path + file_list[count])
            datas.append([vertics, faces])
        Woodblock(count, datas)
