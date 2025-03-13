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

def preprocess_data(dataset, query_date_start, query_date_end, timesteps,objective):
    dataset['date'] = pd.to_datetime(dataset['date'].astype(str)).dt.strftime('%Y%m%d')
    train_set = dataset.query(f"'{query_date_start}' <= date <= '{query_date_end}'")
    train_set=train_set.sort_values(by='datetime',ascending=True)
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

    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    history = regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return regressor, history


def make_predictions(indicators,regressor, dataset,test_date_start, test_date_end,objective,timesteps,sc):
    dataset['date'] = pd.to_datetime(dataset['date'].astype(str)).dt.strftime('%Y%m%d')
    dataset_test = dataset.query(f"'{test_date_start}' <= date <= '{test_date_end}'") # 与之前时间错开

    real_option_price = dataset_test[[objective[0]]].values  # objective第一列为要预测的目标
    #dataset_train1['error'].shape
    #Out[17]: (4320,)
    #dataset_train1[['error']].shape
    #Out[18]: (4320, 1)

    # Make predictions
    dataset_total = dataset.query(f"'{train_date_start}' <= date <= '{test_date_end}'")  # 设置整体日期
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

    return real_option_price,predicted_option_price

def plot_predictions(real_price, predicted_price):
    plt.figure(figsize=(18,8))
    plt.plot(real_price, color='red', label='Real Price')
    plt.plot(predicted_price, color='blue', label='Predicted Price')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
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

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')  # 注意横坐标标签是 'Epoch'
    plt.legend()
    plt.show()




if __name__ == "__main__":
    dataset = pd.read_csv(r'E:\Option_data\上证50ETF\实证选择数据\50ETF看涨(20231227,2.608,20160).csv')
    train_date_start, train_date_end= '20230807','20230825'      #'20230807','20230930'
    test_date_start, test_date_end=  '20230826','20230831'           #'20231001','20231016'
    objective=('c','s') # #要预测的值放在第一个,其他特征放后面 (构建模型只预测第一个值，Dense(units=1))
    timesteps=60#预测步长

    X_train,y_train,indicators,sc=preprocess_data(dataset,train_date_start, train_date_end,
                                                  timesteps,objective)
    regressor, history=build_lstm_model(indicators,timesteps,X_train,y_train,
                               units=256, dropout_rate=0.2,epochs=80,batch_size=32)

    real_price, predicted_price=make_predictions(indicators,regressor, dataset,test_date_start,
                                                 test_date_end,objective,timesteps,sc)
    plot_predictions(real_price, predicted_price)
    print(calibration_report_MSE(real_price, predicted_price))
    # 绘制历史记录中的损失
    plot_history(history)