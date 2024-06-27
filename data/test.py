import scipy.io as scio

import pandas as pd

data_path="Amazon.mat"

#Method 1

data = scio.loadmat(data_path)
print(data)

data_train_label=data.get('label')#取出字典里的label
# print(data_train_label)
print(data_train_label.shape)

# data_train_data=data.get('data')#取出字典里的data
# print(data_train_data)
