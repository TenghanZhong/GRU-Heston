import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#提供MSE校准报告
def calibration_heston_lstm_report_MSE(data):
    avg=0
    data['lstm_heston_error']=data['c']-data['lstm_heston_price']#计算误差
    for i in range(len(data)):
        err=data.loc[i,'lstm_heston_error']
        avg+=err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error of LSTM-Heston Model (%%) : %5.9f" % (avg)
    return data,error_summary #返回计算error后的data

def calibration_heston_gru_report_MSE(data):
    avg=0
    data['gru_heston_error']=data['c']-data['gru_heston_price']#计算误差
    for i in range(len(data)):
        err=data.loc[i,'gru_heston_error']
        avg+=err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error of GRU-Heston Model (%%) : %5.9f" % (avg)
    return data,error_summary #返回计算error后的data

def calibration_heston_report_MSE(data):
    avg=0
    for i in range(len(data)):
        err=data.loc[i,'error']
        avg+=err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error of Heston Model (%%) : %5.9f" % (avg)
    return data,error_summary #返回计算error后的data

def calibration_BS_report_MSE(data):
    avg=0
    for i in range(len(data)):
        err=data.loc[i,'BS_error']
        avg+=err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error of BS Model (%%) : %5.9f" % (avg)
    return data,error_summary #返回计算error后的data

def calibration_lstm_report_MSE(data):
    avg=0
    for i in range(len(data)):
        err=data.loc[i,'c']-data.loc[i,'lstm_price']
        avg+=err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error of LSTM Model (%%) : %5.9f" % (avg)
    return data,error_summary #返回计算error后的data

def calibration_gru_report_MSE(data):
    avg=0
    for i in range(len(data)):
        err=data.loc[i,'c']-data.loc[i,'gru_price']
        avg+=err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error of GRU Model (%%) : %5.9f" % (avg)
    return data,error_summary #返回计算error后的data


#提供MAPE校准报告

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calibration_heston_lstm_report_MAPE(data):
    mape = calculate_mape(data['c'], data['lstm_heston_price'])
    error_summary = "Mean Absolute Percentage Error of LSTM-Heston Model (%%) : %5.9f" % (mape)
    return error_summary

def calibration_heston_gru_report_MAPE(data):
    mape = calculate_mape(data['c'], data['gru_heston_price'])
    error_summary = "Mean Absolute Percentage Error of GRU-Heston Model (%%) : %5.9f" % (mape)
    return error_summary

def calibration_heston_report_MAPE(data):
    mape = calculate_mape(data['c'], data['c'] - data['error'])
    error_summary = "Mean Absolute Percentage Error of Heston Model (%%) : %5.9f" % (mape)
    return error_summary

def calibration_BS_report_MAPE(data):
    mape = calculate_mape(data['c'], data['c'] - data['BS_error'])
    error_summary = "Mean Absolute Percentage Error of BS Model (%%) : %5.9f" % (mape)
    return error_summary

def calibration_lstm_report_MAPE(data):
    mape = calculate_mape(data['c'], data['lstm_price'])
    error_summary = "Mean Absolute Percentage Error of LSTM Model (%%) : %5.9f" % (mape)
    return error_summary

def calibration_gru_report_MAPE(data):
    mape = calculate_mape(data['c'], data['gru_price'])
    error_summary = "Mean Absolute Percentage Error of GRU Model (%%) : %5.9f" % (mape)
    return error_summary


#画对比图
def plot_lstm_heston_price_comparison_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['c'], label='Market Price', color='blue')
    plt.plot(data['lstm_heston_price'], label='LSTM-Heston Model Price', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Price vs LSTM-Heston Model Price(date: {day1}-to-{day2} )')
    plt.ylabel('Price')
    plt.legend()
    #month_day1,month_day2=day1[-4:],day2[-4:]
    #plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()
def plot_heston_price_comparison_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['c'], label='Market Price', color='blue')
    plt.plot(data['heston_price'], label='Heston Model Price', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Price vs Heston Model Price(date: {day1}-to-{day2} )')
    plt.ylabel('Price')
    plt.legend()
    #month_day1,month_day2=day1[-4:],day2[-4:]
    #plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()

def plot_BS_price_comparison_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['c'], label='Market Price', color='blue')
    plt.plot(data['BS_price'], label='BS Model Price', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Price vs BS Model Price(date: {day1}-to-{day2} )')
    plt.ylabel('Price')
    plt.legend()
    #month_day1,month_day2=day1[-4:],day2[-4:]
    #plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()

def plot_gru_heston_price_comparison_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['c'], label='Market Price', color='blue')
    plt.plot(data['gru_heston_price'], label='GRU-Heston Model Price', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Price vs GRU-Heston Model Price(date: {day1}-to-{day2} )')
    plt.ylabel('Price')
    plt.legend()
    #month_day1,month_day2=day1[-4:],day2[-4:]
    #plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()

def plot_lstm_price_comparison_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['c'], label='Market Price', color='blue')
    plt.plot(data['lstm_price'], label='LSTM Model Price', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Price vs LSTM Model Price(date: {day1}-to-{day2} )')
    plt.ylabel('Price')
    plt.legend()
    #month_day1,month_day2=day1[-4:],day2[-4:]
    #plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()

def plot_gru_price_comparison_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['c'], label='Market Price', color='blue')
    plt.plot(data['gru_price'], label='GRU Model Price', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Price vs GRU Model Price(date: {day1}-to-{day2} )')
    plt.ylabel('Price')
    plt.legend()
    #month_day1,month_day2=day1[-4:],day2[-4:]
    #plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()

def plot_all_price_comparison_chart(data_lstm,data_gru,data_bs,lstm_price,gru_price):
    plt.figure(figsize=(10, 6), dpi=100)

    # 绘制BS模型价格，虚线
    plt.plot(data_bs['BS_price'], label='BS模型', color='blue', linestyle='--')

    # 绘制Heston模型价格，虚线
    plt.plot(data_lstm['heston_price'], label='Heston模型', color='magenta', linestyle='--')

    # 绘制LSTM模型价格，虚线
    plt.plot(lstm_price['lstm_price'], label='LSTM模型', color='red', linestyle='--')

    # 绘制GRU模型价格，虚线
    plt.plot(gru_price['gru_price'], label='GRU模型', color='cyan', linestyle='--')

    # 绘制LSTM-Heston模型价格，虚线
    plt.plot(data_lstm['lstm_heston_price'], label='Heston-LSTM模型', color='green', linestyle='--')

    # 绘制GRU-Heston模型价格，实线
    plt.plot(data_gru['gru_heston_price'], label='Heston-GRU模型', color='orange')

    # 绘制市场价格，实线
    plt.plot(data_lstm['c'], label='实际期权价格', color='black')

    date=data_lstm['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.ylabel('价格(元)')
    plt.legend(loc='best')
    #month_day1,month_day2=day1[-4:],day2[-4:]
    #plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()


#误差图
def plot_error_comparison_chart(data_lstm,data_gru):
    plt.figure(figsize=(10, 6),dpi=150)

    # Plot Heston model actual error
    plt.plot(data_lstm['error'], label='Heston模型实际误差', color='blue')

    # Plot LSTM model predicted error
    plt.plot(data_lstm['predicted_error'], label='Heston-LSTM模型预测误差', color='red')

    # Plot GRU model predicted error
    plt.plot(data_gru['predicted_error'], label='Heston-GRU模型预测误差', color='green')

    plt.ylabel('误差值')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    data_lstm = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_测试集_LSTM训练后.csv')
    data_gru = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_测试集_GRU训练后.csv')
    data_bs = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_测试集_bsm.csv')
    lstm_price = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_LSTM直接预测测试集.csv')
    gru_price = pd.read_csv(
        r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)\(20231227, 3.6, 20160)_GRU直接预测测试集.csv')

    # 计算MSE误差
    data_lstm, lstm_heston_error_summary = calibration_heston_lstm_report_MSE(data_lstm)
    data_gru, gru_heston_error_summary = calibration_heston_gru_report_MSE(data_gru)
    data_lstm, heston_error_summary = calibration_heston_report_MSE(data_lstm)
    data_bs, bs_error_summary = calibration_BS_report_MSE(data_bs)
    lstm_price, lstm_error_summary = calibration_lstm_report_MSE(lstm_price)
    gru_price, gru_error_summary = calibration_gru_report_MSE(gru_price)

    # 打印误差
    print(bs_error_summary)
    print(heston_error_summary)
    print(lstm_error_summary)
    print(gru_error_summary)
    print(lstm_heston_error_summary)
    print(gru_heston_error_summary)



    # 计算MAPE误差
    lstm_heston_MAPE_summary = calibration_heston_lstm_report_MAPE(data_lstm)
    gru_heston_MAPE_summary = calibration_heston_gru_report_MAPE(data_gru)
    heston_MAPE_summary = calibration_heston_report_MAPE(data_lstm)
    bs_MAPE_summary = calibration_BS_report_MAPE(data_bs)
    lstm_MAPE_summary = calibration_lstm_report_MAPE(lstm_price)
    gru_MAPE_summary = calibration_gru_report_MAPE(gru_price)

    # 打印误差
    print(bs_MAPE_summary)
    print(heston_MAPE_summary)
    print(lstm_MAPE_summary)
    print(gru_MAPE_summary)
    print(lstm_heston_MAPE_summary)
    print(gru_heston_MAPE_summary)




    # 画对比图
    # 设置中文支持字体，例如使用宋体
    matplotlib.rcParams['font.family'] = 'SimSun'
    '''
    plot_lstm_heston_price_comparison_chart(data_lstm)
    plot_gru_heston_price_comparison_chart(data_gru)

    plot_heston_price_comparison_chart(data_lstm)
    plot_BS_price_comparison_chart(data_bs)

    plot_lstm_price_comparison_chart(lstm_price)
    plot_gru_price_comparison_chart(gru_price)
    '''
    plot_error_comparison_chart(data_lstm, data_gru)

    plot_all_price_comparison_chart(data_lstm, data_gru, data_bs, lstm_price, gru_price)




