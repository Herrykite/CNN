import os
import pickle
import numpy as np
from ConvNet.config.defaults import get_cfg_defaults
from sklearn.decomposition import PCA
from ConvNet.tools.deal_with_obj import loadObj


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    input_path = cfg.INPUT.VERTICS_PATH
    file_list = os.listdir(input_path)
    file_list.sort(key=lambda x: int(x[:-4]))
    data = []
    for i in range(len(file_list)):
        print(i, 'finished')
        vertics, faces = loadObj(input_path + file_list[i])
        data.append(np.array(vertics).reshape(7657, 3))
    data = np.array(data)
    data = data.reshape(len(file_list), 3 * 7657)
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
