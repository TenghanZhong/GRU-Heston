import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import os
def preprocess_data(dataset_train,timesteps,objective):
    dataset_train['date'] = pd.to_datetime(dataset_train['date'].astype(str)).dt.strftime('%Y%m%d')
    train_set=dataset_train.sort_values(by='datetime',ascending=True)
    train_set=train_set.reset_index(drop=True)

    train_set = train_set[list(objective)].values

    indicators = train_set.shape[1]  # 特征个数,即为列数
    sc = MinMaxScaler(feature_range=(0, 1))
    train_set_scaled = sc.fit_transform(train_set)

    # 每60分钟基于前60分钟数据来预测下一分钟的数据，通过试错法得到（60 timesteps刚好是三个月的交易日数量和）
    X_train = []  # 每60分钟数据
    y_train = []  # T+1分钟预测数据
    timesteps = timesteps
    for i in range(timesteps, len(train_set_scaled)):
        X_train.append(train_set_scaled[(i - timesteps):i, :])  # 加入前60天的股价，期权价格等数据,左闭右开
        y_train.append(train_set_scaled[i, 0])  # 第61天期权价格
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping--增加更多的指标来预测股价,并且和lstm input_shape相吻合  indicators=train_set.shape[1]#特征个数,即为列数
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], indicators))#batch_size,timesteps,indicators
    return X_train,y_train,indicators,sc

#Grid_search:
def build_regressor(optimizer='adam',units=128,dropout_rate=0.2):
    regressor = Sequential()

    regressor.add(GRU(units=units, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], indicators)))
    regressor.add(Dropout(rate=dropout_rate))
    regressor.add(GRU(units=units, activation='tanh', return_sequences=True))
    regressor.add(Dropout(rate=dropout_rate))
    regressor.add(GRU(units=units, activation='tanh', return_sequences=True))
    regressor.add(Dropout(rate=dropout_rate))
    regressor.add(GRU(units=units, activation='tanh', return_sequences=False))
    regressor.add(Dropout(rate=dropout_rate))
    # 输出层
    regressor.add(Dense(units=1))  # 预测一个值
    # 编译模型
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    return regressor

if __name__ == "__main__":
    dataset_train1 = pd.read_csv(
        r'D:\Option_data\上证50ETF\实证选择数据\LSTM_(20231227,2.608,20160)\(20231227,2.608,20160)_8月训练集.csv')
    dataset_train2 = pd.read_csv(
        r'D:\Option_data\上证50ETF\实证选择数据\LSTM_(20231227,2.608,20160)\(20231227,2.608,20160)_9月训练集.csv')
    dataset_train3 = pd.read_csv(
        r'D:\Option_data\上证50ETF\实证选择数据\LSTM_(20231227,2.608,20160)\(20231227,2.608,20160)_10月训练集.csv')
    dataset = pd.concat([dataset_train1, dataset_train2, dataset_train3], axis=0)

    # 分阶段优化
    objective = ('error', 's', 'ttm_ratio', 'heston_price')  # 要预测的值
    timesteps = 60  # 预测步长
    datasets = [dataset_train1, dataset_train2, dataset_train3]
    best_params = None
    combo_count = 0

    X_train, y_train, indicators, sc = preprocess_data(dataset, timesteps, objective)
    regressor = KerasRegressor(build_fn=build_regressor, units=128, dropout_rate=0.2)
    parameters_dictionary = {'batch_size': [16, 32],
                             'epochs': [75, 50],
                             'optimizer': ['adam'],
                             'units': [256],
                             'dropout_rate': [0.2, 0.4,0.5]}  # 要检验的值，经验主义 ,rmsprop一般是常用的随机梯度下降方法
    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters_dictionary,
                               scoring='neg_mean_squared_error',
                               cv=5)
    grid_search = grid_search.fit(X_train, y_train)
    #当前数据集最优参数:{'batch_size': 32, 'dropout_rate': 0.4, 'epochs': 75,
    # 'optimizer': 'adam', 'units': 256}

    # 更新计数器
    combo_count += len(grid_search.cv_results_['params'])
    print(f'当前尝试了{combo_count}种参数组合')

    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_  # 最小误差
    print(f"当前数据集最优参数: {best_parameters}")
    print(f"当前数据集最小MSE: {-best_accuracy}")

'''
数据集之间有两天的间隔，但仍然可以考虑合并它们进行优化，尤其是当您寻找一组能够适用于所有数据集的统一参数时。以下是一些考虑因素：

市场连续性：金融市场的特性是连续的，即使存在短暂的间隔。如果这两天没有发生重大的市场事件或其他显著的市场变化，那么合并数据进行优化的影响可能是微小的。

统计特性：如果您的各个数据集在统计特性上相似（比如分布、波动性等），合并进行优化是合理的。这意味着模型在所有数据集上学习的模式可能是一致的。

模型鲁棒性：合并数据集进行优化可以增强模型对不同市场条件的适应能力，有助于提高模型的鲁棒性。

过渡处理：考虑在合并的数据集中对两天间隔的部分进行适当处理，例如通过数据插值来估计这两天的数据。

实验对比：您可以尝试两种方法——分别对数据集进行优化和合并优化，然后比较两种方法的模型性能，以确定哪种方法更适合您的需求。
'''