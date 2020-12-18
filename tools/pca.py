import sys

sys.path.insert(0, '../../')
import os
import pickle
import yaml
import numpy as np
from ConvNet.config.defaults import get_cfg_defaults
from sklearn.decomposition import PCA
from ConvNet.tools.deal_with_obj import loadObj


if __name__ == '__main__':
    print('The vertex data is being dimensionally reduced...')
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1])
    input_path = cfg.INPUT.VERTICS_PATH
    file_list = os.listdir(input_path)
    file_list.sort(key=lambda x: int(x[:-4]))
    data = []
    for i in range(len(file_list)):
        print(i, 'finished')
        vertics, faces = loadObj(input_path + file_list[i])
        data.append(np.array(vertics).reshape(cfg.INPUT.VERTICS_NUM//3, 3))
    data = np.array(data)
    data = data.reshape(len(file_list), cfg.INPUT.VERTICS_NUM)
    print('The retrieval has completed. Data is being processed...')
    pca = PCA(n_components=0.9999999)
    pca_coefficient = pca.fit_transform(data)
    feature_info = np.zeros(len(file_list))
    for j in range(len(file_list)):
        feature_info[j] = sum(pca.explained_variance_ratio_[0:j])
    base_matrix = pca.components_
    data_reduction = pca.inverse_transform(pca_coefficient)
    np.save('data.npy', data)
    np.save('coefficient.npy', pca_coefficient)
    file = open('pca.pkl', 'wb')
    pickle.dump(pca, file)
    print('pca data has been saved!The dimension is', pca_coefficient.shape[1])
    with open(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1], 'r') as f:
        menu = yaml.load(f, Loader=yaml.FullLoader)
        menu['INPUT']['PCA_DIMENSION'] = int(pca.n_components_.T)
    with open(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1], 'w') as f:
        yaml.dump(menu, f)
