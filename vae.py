import ipca
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import torch

import time
import torch.nn.functional as F
from model import *
from matrix_process import *
#超参数
max_depth = 10
n_components = 30#降维以后的数据维度
lr = 1e-5
n_epochs = 60
check = 1
kl_weight = 0.00025
device = torch.device('cpu')
use_pretrained = True


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    try:
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar)))
    except:
        print(logvar)
    loss = recons_loss + kl_loss * kl_weight
    return loss

def eval(model, dataset, label):
    loss = 0
    for i in range(dataset.shape[0]):
        loss += loss_fn(model(torch.tensor(dataset[i], dtype=torch.float32)), label[i])
    print(loss)
    return

def getdata():
    df = pd.read_csv('./HCP_s1200.csv')
    print(np.array(df['Loneliness_Unadj']))
    #读取训练数据
    data = []
    with open('./rsfc_atlas400_753_4.txt') as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            line = line.split(' ')
            line = list(map(float, line))
            data.append(line)
        f.close()
    #读取标签
    label = []
    with open('./HCP_list_Yeo.txt') as f:
        lines = f.readlines()
        for line in lines:
            for i in range(len(df['Subject'])):
                if df['Subject'][i] == int(line):
                    label.append(int(df['Loneliness_Unadj'][i]))
                break
    #print(label)
        f.close()
    data = np.array(data)
    #print(data.shape)
    label = np.array(label).astype(np.int64)

    return data, label

if __name__ == '__main__':
    
    structure_path = './scfp_atlas400_753.txt'  # 结构连接数据
    function_path = './rsfc_atlas400_753_4.txt'  # 功能连接数据
    behaviour_path = './HCP_753final.csv'  # 行为数据

    predictor = 'MMSE_Score'  # 我想预测指标的名称，若输入不合法则默认按'MMSE_Score'指标处理

    index = get_column_index(behaviour_path, predictor)
    X_train, Y_train, X_test, Y_test = make_cnn_dataset(structure_path, behaviour_path, index)
    print("...")
    

    model = VAE()

    optimizer = torch.optim.Adam(model.parameters(), lr)
    for i in range(n_epochs):
        print('start train')
        begin_time = time.time()
        loss_sum = 0
        for x, y in zip(X_train, Y_train):
            x = torch.tensor(x.reshape(-1, 1, 399, 399), dtype=torch.float32)
            y_hat, label_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar) + (label_hat.reshape(1)-torch.Tensor(y))**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        #if i % check ==0:
            #eval(model, data[700:, :], label[700:])

        train_time = time.time()-begin_time
        print(f'epoch {i}: loss {loss_sum} {train_time}')
        torch.save(model.state_dict(), './model5_'+str(i)+'.pth')

