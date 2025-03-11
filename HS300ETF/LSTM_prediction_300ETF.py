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
from keras.callbacks import EarlyStopping
import os

def preprocess_first_data(dataset_train, timesteps,objective):
    dataset_train['date'] = pd.to_datetime(dataset_train['date'].astype(str)).dt.strftime('%Y%m%d')
    train_set=dataset_train.sort_values(by='datetime',ascending=True)
    train_set=train_set.reset_index(drop=True)

    train_set = train_set[list(objective)].values

    indicators = len(objective) # 特征个数,即为列数
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
    return X_train,y_train,sc

def preprocess_later_data(dataset_train, timesteps,objective,sc):
    dataset_train['date'] = pd.to_datetime(dataset_train['date'].astype(str)).dt.strftime('%Y%m%d')
    train_set=dataset_train.sort_values(by='datetime',ascending=True)
    train_set=train_set.reset_index(drop=True)

    train_set = train_set[list(objective)].values

    indicators = len(objective) # 特征个数,即为列数
    train_set_scaled = sc.transform(train_set)

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
    return X_train,y_train,sc

def build_lstm_model(indicators,timesteps,X_train,y_train, units, dropout_rate,epochs,batch_size):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 若设置为‘-1’则是使用CPU，忽略GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices(
        'GPU')))  # 如果输出是 Num GPUs Available: 0，这意味着 TensorFlow 将只使用 CPU 运行。

    regressor = Sequential()

    input_shape = (timesteps, indicators)
    #X_train.shape[1]=timesteps
    regressor.add(LSTM(units=units, activation='tanh', return_sequences=True, input_shape=input_shape))
    # return_sequences=True，允许添加另一层LSTM，input_shape=X_train的后两个维度（timesteps,indicators）
    regressor.add(Dropout(rate=dropout_rate))  # 一般从0.2开始试起
    regressor.add(LSTM(units=units, activation='tanh', return_sequences=True))  # 加入第二层时就不用input_shape了
    regressor.add(Dropout(rate=dropout_rate))
    regressor.add(LSTM(units=units, activation='tanh', return_sequences=True))
    regressor.add(Dropout(rate=dropout_rate))
    regressor.add(LSTM(units=units, activation='tanh', return_sequences=False))  # 后面不叠加LSTM了就制定为False
    regressor.add(Dropout(rate=dropout_rate))

    regressor.add(Dense(units=1))  # DENSE的units=输出特征个数 这里只有第一个值
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error', )
    # loss='mean_squared_error'因为预测的是股票价格，是一个数值，类似于回归问题，就使用mse。（分类问题用cross——entropy）
    # 这里不用,metrics=['accuracy']，也是因为预测的是股票价格（而不是分类是否正确的问题）

    #设置早停策略
    #作用：防止过拟合，通过设置monitor参数，你可以选择监控哪个性能指标（如验证集上的损失或准确率）。当监控指标不再改善时，EarlyStopping将停止训练。
    #patience参数定义了在性能不再改善时，要等待多少个epoch才停止训练。例如，如果patience=3，则在连续3个epoch中，监控指标都没有改善时，训练将停止。
    #过设置restore_best_weights=True，在停止训练后，EarlyStopping将恢复到性能最好的模型的权重。
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    #regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[early_stopping])

    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[early_stopping])

    return regressor

def Incremental_Learning_lstm_model(regressor,X_new_train, y_new_train,new_epochs,new_batch_size):
    regressor.fit(X_new_train, y_new_train, epochs=new_epochs, batch_size=new_batch_size)
    return regressor

def make_predictions(dataset_test,dataset_train,indicators,regressor, objective,timesteps,sc):
    dataset_test['date'] = pd.to_datetime(dataset_test['date'].astype(str)).dt.strftime('%Y%m%d')

    real_option_price = dataset_test[[objective[0]]].values  # objective第一列为要预测的目标

    # Make predictions
    dataset_total = pd.concat((dataset_train,dataset_test),axis=0)#axis=0竖着链接  # 设置整体日期
    dataset_total = dataset_total[list(objective)]
    inputs = dataset_total[(dataset_total.shape[0] - dataset_test.shape[0] - timesteps):].values  # timesteps=60

    # 得到test前60天的特征数据
    inputs = sc.transform(inputs)  # 使用训练集所用的缩放方式应用于input（这时只用sc.transform，而不用fit)
    X_test = []
    for i in range(timesteps, timesteps + dataset_test.shape[0]):
        X_test.append(inputs[(i - timesteps):i, :])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], indicators))
    predicted_option_price = regressor.predict(X_test)
    # 创建一个形状与预测结果相同的数组
    dummy_array = np.zeros((predicted_option_price.shape[0], inputs.shape[1]))
    dummy_array[:, 0] = predicted_option_price[:, 0]  # 假设目标列是第一列

    # 应用逆变换
    predicted_option_price = sc.inverse_transform(dummy_array)[:, 0]
    dataset_test['predicted_price'] = np.array(predicted_option_price).flatten()

    return real_option_price,predicted_option_price,dataset_test

def plot_predictions(real_price, predicted_price):
    plt.figure(figsize=(18,8))
    plt.plot(real_price, color='red', label='Real')
    plt.plot(predicted_price, color='blue', label='Predicted')
    plt.title('Error Prediction')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
def calibration_report_MSE(real_price, predicted_price):

    real_price = np.array(real_price).flatten()#铺平所有值
    predicted_price = np.array(predicted_price).flatten()

    if len(real_price) != len(predicted_price):
        raise ValueError("The lengths of real_price and predicted_price must be the same")

    # Creating a DataFrame to store real and predicted prices and their error
    data = pd.DataFrame({
        'real_price': real_price,
        'predicted_price': predicted_price
    })
    data['error'] = data['real_price'] - data['predicted_price']

    # Calculating Mean Squared Error
    mse = (data['error'] ** 2).mean()
    # Formatting the error summary
    error_summary = "Mean Square Error of real and predicted (%%) : %5.9f" % (mse * 100)  # multiplied by 100 if you want it in percentage
    return error_summary

if __name__ == "__main__":
    dataset_train1 = pd.read_csv(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_8月训练集.csv')
    dataset_train2 = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_9月训练集.csv')
    dataset_train3 = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_10月训练集.csv')
    dataset_test = pd.read_csv(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_测试集.csv')

    # 设置训练目标和步长
    objective = ('c')  # #要预测的值放在第一个,其他特征放后面 (构建模型只预测第一个值，Dense(units=1))
    timesteps = 60  # 预测步长
    indicators = len(objective)  # 训练特征个数


    # 预处理训练集
    X_train1, y_train1, sc = preprocess_first_data(dataset_train1, timesteps, objective)
    X_train2, y_train2, sc = preprocess_later_data(dataset_train2, timesteps, objective, sc)
    X_train3, y_train3, sc = preprocess_later_data(dataset_train3, timesteps, objective, sc)

    # 训练模型,分别进行增量学习
    regressor = build_lstm_model(indicators, timesteps, X_train1, y_train1,
                                 units=256, dropout_rate=0.3, epochs=50, batch_size=64)
    regressor = Incremental_Learning_lstm_model(regressor, X_train2, y_train2, new_epochs=50, new_batch_size=64)
    regressor = Incremental_Learning_lstm_model(regressor, X_train3, y_train3, new_epochs=50, new_batch_size=64)
    '''
    当前尝试了36种参数组合
    当前数据集最优参数: {'batch_size': 64, 'dropout_rate': 0.3, 'epochs': 50, 'optimizer': 'adam', 'units': 256}
    当前数据集最小MSE: 0.0003477227473154705
    '''



    # 得到predicted_error 绘制对比图
    real_price, predicted_price, dataset_test = make_predictions(dataset_test, dataset_train3, indicators, regressor,
                                                                 objective, timesteps, sc)
    plot_predictions(real_price, predicted_price)
    print(calibration_report_MSE(real_price, predicted_price))

    #保存训练后的测试集文件
    dataset_test['lstm_price']=dataset_test['predicted_price']
    file_path1 = os.path.join(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)', f"(20231227, 3.6, 20160)_LSTM直接预测测试集.csv")
    dataset_test.to_csv(file_path1, index=False)
    # 保存模型
    model_path = r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\lstm_price_model.h5'
    regressor.save(model_path)