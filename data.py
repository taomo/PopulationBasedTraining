import torch
from torch import nn
# import torch.utils.tensorboard as tensorboard


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset, TensorDataset

import time
import argparse
import torch.optim as optim
import torch.nn.functional as F


usecols=[0, 1, 2, 3, 4, 5]
k = 1 # 向前的时刻
n = len(usecols) # 多少个变量


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def load_data(file_name, sequence_length=10, split=0.8):
       
    # load dataset
    # df = pd.read_csv(file_name, sep=',', usecols=[1])
    # names =  [r'$x1(t)$', r'$x2(t)$', r'$x1c(t)$', r'$x2c(t)$', r'$x1cc(t)$', r'$x2cc(t)$']
    # usecols=[0, 1, 2, 3, 4, 5]
    # usecols=[5]
    df = pd.read_csv(file_name,header=None,sep=',',  skiprows=1, usecols=usecols)
  
    print(df)

   # load dataset
    values = df.values
    # print('values',values)
    # ensure all data is float
    values = values.astype('float')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # print('scaled',scaled)
    # frame as supervised learning
    # k = 1 # 向前的时刻
    # n = len(usecols) # 多少个变量
    # target = 'var6'
    # df, target,preds = ts_dataframe_to_supervised(df, target, 1, 0, False)

    reframed = series_to_supervised(scaled, k, 1)

    # print(type(reframed))
    # print(reframed.size)

    split_boundary = int(reframed.shape[0] * split)

    #fram
    train_x = reframed.ix[: split_boundary, :k*n]
    # train_x = train_x.values.reshape(6,-1)
    # print('train_x',train_x)
    test_x = reframed.ix[split_boundary:, :k*n]
    # print(test_x)
    train_y = reframed.ix[: split_boundary, k*n:]
    # print('train_y',train_y)
    test_y = reframed.ix[split_boundary:, k*n:]

    train = reframed.ix[: split_boundary, :]
    test = reframed.ix[split_boundary:, :]    

    print('reframed',reframed.head())
 

    return train, test, train_x, train_y, test_x, test_y, scaler    




class CustomDataset(Dataset):

    def __init__(self, tensor_x, tensor_y):
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y


    def __len__(self):
        return len(self.tensor_x)

    def __getitem__(self, index):
        # data = self.tensor_data[index]
        # label = 1
        train_x = self.tensor_x[index,:,:]
        train_y = self.tensor_y[:,index]
        return train_x, train_y


def get_data():   

    datapath = "taomo10k.csv"


    train_data, test_data, train_x, train_y, test_x, test_y, scaler  = load_data(datapath)
    
    train_x = torch.from_numpy(train_x.values).float()
    train_x = train_x.reshape(-1,n,k)  # 10 # 向前的时刻  6个变量

    # A =  np.arange(60)
    # A = A.reshape(-1,10)
    # print(A)
    # A = A.reshape(-1,5)
    # print(A)

    train_y = torch.from_numpy(train_y.values).float()
    train_y = train_y[:, -1]

    test_x = torch.from_numpy(test_x.values).float()
    test_x = test_x.reshape(-1,n,k)
    print(test_x.size())


    test_y = torch.from_numpy(test_y.values).float()
    test_y = test_y[:, -1]
    # 数据变形 准备
    # train_x = train_x.unsqueeze(1)
    train_y = train_y.unsqueeze(1) 

    # test_x = test_x.unsqueeze(1)
    test_y = test_y.unsqueeze(1) 

    # train = torch.from_numpy(train.values).float()
    # train = train.unsqueeze(1)    
    # test = torch.from_numpy(test.values).float()

    print(train_x.size())
    print(train_y.size())        

    BATCH_SIZE = 100

    train_dataset = CustomDataset(tensor_x = train_x, tensor_y = train_y)    
    # 将数据转换为torch的dataset格式    
    # torch_dataset = TensorDataset(train_x.values.T, train_y.values.T)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = CustomDataset(tensor_x = test_x, tensor_y = test_y) 


    print(1+2)

    return train_dataset, test_dataset







if __name__ == "__main__":
    get_data()    