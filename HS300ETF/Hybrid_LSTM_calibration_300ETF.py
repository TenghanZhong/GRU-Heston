import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
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
        y_train.append(train_set_scaled[i, 0])  # 第61天期权价格,第一列数据为y
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping--增加更多的指标来预测股价,并且和lstm input_shape相吻合  indicators=train_set.shape[1]#特征个数,即为列数
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], indicators))#batch_size,timesteps,indicators
    return X_train,y_train,indicators,sc

#Grid_search:
def build_regressor(units=128,dropout_rate=0.2):
    regressor = Sequential()
    regressor.add(LSTM(units=units, activation='tanh', return_sequences=True, input_shape=(60, indicators)))
    # return_sequences=True，允许添加另一层LSTM，input_shape=X_train的后两个维度（timesteps,indicators）
    regressor.add(Dropout(rate=dropout_rate))  # 一般从0.2开始试起
    regressor.add(LSTM(units=units, activation='tanh', return_sequences=True))  # 加入第二层时就不用input_shape了
    regressor.add(Dropout(rate=dropout_rate))
    regressor.add(LSTM(units=units, activation='tanh', return_sequences=True))
    regressor.add(Dropout(rate=dropout_rate))
    regressor.add(LSTM(units=units, activation='tanh', return_sequences=False))  # 后面不叠加LSTM了就制定为False
    regressor.add(Dropout(rate=dropout_rate))

    regressor.add(Dense(units=1))#Dense:Final_objective(error)
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor

from keras.callbacks import EarlyStopping
if __name__ == "__main__":
    dataset_train1 = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_8月训练集.csv')
    dataset_train2 = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_9月训练集.csv')
    dataset_train3 = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_10月训练集.csv')
    dataset_all = pd.concat([dataset_train1, dataset_train2, dataset_train3], axis=0)

    # 分阶段优化
    objective = ('error', 's', 'ttm_ratio', 'heston_price')  # 要预测的值
    timesteps = 60  # 预测步长
    datasets = [dataset_train1, dataset_train2, dataset_train3]
    best_params = None
    combo_count=0

    X_train, y_train, indicators, sc = preprocess_data(dataset_train3, timesteps, objective)
    regressor = KerasRegressor(build_fn=build_regressor, units=256, dropout_rate=0.3)
    parameters_dictionary = {'batch_size': [32,64],
                             'epochs': [75],
                             'optimizer': ['adam'],
                             'units': [256],
                             'dropout_rate': [0.3,0.5]}  # 要检验的值，经验主义 ,rmsprop一般是常用的随机梯度下降方法
    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters_dictionary,
                               scoring='neg_mean_squared_error',
                               cv=3)#cv=cross_validation k折交叉验证
    #{'batch_size': 64, 'dropout_rate': 0.3, 'epochs': 75, 'optimizer': 'adam', 'units': 256}

    grid_search = grid_search.fit(X_train, y_train)

    # 更新计数器
    combo_count += len(grid_search.cv_results_['params'])
    print(f'当前尝试了{combo_count}种参数组合')

    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_  # 最小误差
    print(f"当前数据集最优参数: {best_parameters}")
    print(f"当前数据集最小MSE: {-best_accuracy}")

    #第一个训练集：{'batch_size': 32, 'dropout_rate': 0.4(0.3), 'epochs': 75(25,50), 'units': 256}
    #第二个{'batch_size': 16, 'dropout_rate': 0.4(0.5), 'epochs': 75(50), 'units': 256}
    #第三个 当前数据集最优参数: {'batch_size': 32, 'dropout_rate': 0.4(0.2), 'epochs': 75(100), 'units': 256}


    '''
    for dataset in datasets:
        X_train,y_train,indicators,sc=preprocess_data(dataset,timesteps,objective)
        regressor = KerasRegressor(build_fn=build_regressor, units=128, dropout_rate=0.2)
        parameters_dictionary = {'batch_size': [32, 64],
                                 'epochs': [25,50],
                                 'optimizer': ['adam'],
                                 'units': [128, 256],
                                 'dropout_rate': [0.2, 0.4]}  # 要检验的值，经验主义 ,rmsprop一般是常用的随机梯度下降方法
        grid_search = GridSearchCV(estimator=regressor,
                                   param_grid=parameters_dictionary,
                                   scoring='neg_mean_squared_error',
                                   cv=10)
        grid_search = grid_search.fit(X_train, y_train)

        # 更新计数器
        combo_count += len(grid_search.cv_results_['params'])
        print(f'当前尝试了{combo_count}种参数组合')

        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_  # 最小误差
        print(f"当前数据集最优参数: {best_parameters}")
        print(f"当前数据集最小MSE: {-best_accuracy}")

        if not best_params or best_accuracy > best_params['score']:
            best_params = {'params': best_parameters, 'score': best_accuracy}
    print(f"所有数据集最优参数: {best_params}")
    '''