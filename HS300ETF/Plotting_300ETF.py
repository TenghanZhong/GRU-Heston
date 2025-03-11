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
def plot_gru_heston_price_comparison_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['error'], label='Real Error', color='blue')
    plt.plot(data['predicted_error'], label='GRU-Heston Model Predicted_error', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Error vs GRU-Heston Model Predicted Error(date: {day1}-to-{day2} )')
    plt.ylabel('Error')
    plt.legend()
    #month_day1,month_day2=day1[-4:],day2[-4:]
    #plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()


file_path1 = os.path.join(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)',
                              f"(20231227, 3.6, 20160)_测试集_GRU训练后.csv")
data=pd.read_csv(file_path1)
plot_gru_heston_price_comparison_chart(data)