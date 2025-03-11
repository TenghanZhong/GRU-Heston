import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as npr
from datetime import datetime
from scipy.interpolate import griddata
import random
import math
import scipy.stats as scs
from math import log,sqrt,exp
import os
import multiprocessing
import QuantLib as ql
from scipy import optimize
import tqdm

def calculate_single_Heston_value(df,params):
    option_type = ql.Option.Call
    strike_price = df['k']
    call_payoff = ql.PlainVanillaPayoff(option_type, strike_price)  # 创造一个看涨期权收益结构

    day_count = ql.Actual365Fixed()  # 一年按365天
    calendar = ql.China(ql.China.SSE)  # 设定交易所的交易日历

    maturity = ql.Date(str(df['maturity_date']), '%Y%m%d')  # 15th June 2023
    today = ql.Date(str(df['date']), '%Y%m%d')  # 15th December 2022
    ql.Settings.instance().evaluationDate = today

    call_exercise = ql.EuropeanExercise(maturity)  # 创造一个欧式行权方式对象，指定期权到期日maturity
    option = ql.VanillaOption(call_payoff, call_exercise)  # 创造一个看涨期权对象（结合其收益结构的行权方式）

    # Option input value
    spot_price = df['s']  # 标的市场价格,这里是hs300
    rf = df['rf']  # 无风险利率
    dividend = df['dividend']  # 股息率,在最后应用模型的时候考虑股息拟合会更好
    theta, kappa, sigma, rho, v0=params

    initial_value_spot_price = ql.QuoteHandle(ql.SimpleQuote(spot_price))  # 设定标的市场价格

    # 设置无风险利率曲线
    rf_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, rf, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, day_count))

    heston_process = ql.HestonProcess(rf_curve, dividend_yield, initial_value_spot_price, v0,
                                       kappa, theta, sigma, rho)
    hestonModel = ql.HestonModel(heston_process)  # 创建赫斯顿模型
    engine = ql.AnalyticHestonEngine(hestonModel)  # 为赫斯顿模型设置一个分析型引擎
    option.setPricingEngine(engine)  # 设置定价引擎，将赫斯顿引擎应用于之前定义的期权对象，以便进行定价。
    h_price = option.NPV()  # 期权的理论市场价值。
    return h_price

def calculate_all_Heston_value(df,params):
    pool=multiprocessing.Pool(8)
    results = pool.starmap(calculate_single_Heston_value, [(df.iloc[i],params) for i in range(len(df))])
    #pool.starmap 是 pool.map 的一个变体，允许被映射的函数接受多个参数。
    pool.close()  # 关闭进程池，不再接受新的任务
    pool.join()  # 等待所有子进程结束
    return results

def calibration_report_MSE(data):
    avg=0
    data['error']=data['c']-data['heston_price']#计算误差
    for i in range(len(data)):
        err=data.loc[i,'error']
        avg+=0.5*err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error (%%) : %5.9f" % (avg)
    return data,error_summary #返回计算error后的data

def calibration_report_MAPE(data):
    data['MAPE_error']=np.abs(data['c']-data['heston_price'])/data['c']
    total_error=data['MAPE_error'].sum()
    avg=total_error*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Absolute Percentage Error (%%) : %5.9f" % (avg)
    return data,error_summary

def plot_price_comparison_chart(data,params):
    plt.figure(figsize=(12, 6))
    plt.plot(data['c'], label='Market Price', color='blue')
    plt.plot(data['heston_price'], label='Heston Model Price', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Price vs Heston Model Price(date: {day1}-to-{day2} )')
    plt.suptitle(f'Parameters={params}')
    plt.ylabel('Price')
    plt.legend()
    month_day1,month_day2=day1[-4:],day2[-4:]
    plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_完整期({month_day1}-{month_day2}).png')
    plt.show()


if __name__ == "__main__":
    data= pd.read_csv(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\300ETF看涨期权(20231227, 3.6, 20160).csv')

    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data['maturity_date'], data['date'] = (pd.to_datetime(data['maturity_date'].astype(str)).
                                                     dt.strftime('%Y%m%d'),
                                                     pd.to_datetime(data['date'].astype(str)).dt.strftime(
                                                         '%Y%m%d'))

    # theta, kappa, sigma, rho, v0 = params
    calibrated_params=(0.001733 , 24.766057 , 0.292921 , 0.99933 , 0.137305)
    # 8月  0807-0808：  最优参数：(0.001184 , 23.377164 , 0.166967 , 0.414889 , 0.228492)#Square Error (%) :0.000775622
    #9月 0904-0905 ：  最优参数：(0.054042 , 1.214735 , 0.353305, -0.934994 , 0.010846)  Mean Square Error (%) :0.002703705
    #10月 1009-1010 ：  最优参数：(0.001733 , 24.766057 , 0.292921 , 0.99933 , 0.137305) Mean Square Error (%) :0.000630459

    data_train = data.query("'20231011' <= date <= '20231022'")
    month=10
    # 调参两天（0807-0808） 训练集：0809-0903      调参两天（0904-0905）  训练集：0906-1008
    # 调参两天（1009-1010） 训练集：1011-1022      为测试集：1023-1031
    # 华夏上证50ETF：整个0807-1031 时间段共13440个样本（行数），到期时间：(57, 142) s:(2.422, 2.725)
    #调参日为每个月第一个星期一和其相连的星期二

    file_path1 = os.path.join(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)', f"(20231227, 3.6, 20160)_{month}月训练集.csv")

    data_train = data_train.sort_values(by=['datetime'])
    data_train = data_train.reset_index(drop=True)
    print(f'训练集data有{data_train.shape[0]}行数据')

    #计算heston值并绘图
    data_train['heston_price']=calculate_all_Heston_value(data_train,calibrated_params)
    print(data_train)  # 参数校准结果
    plot_price_comparison_chart(data_train,calibrated_params)  # 绘制对比图
    #计算误差，同时输出报告
    data_train,error_summary=calibration_report_MSE(data_train)
    print(error_summary)

    #添加训练集
    data_train.to_csv(file_path1, index=False)

    #添加测试集  十月为测试集
    #调参两天（1009-1010） 1011-1022------1023-1031为测试集
    data_test = data.query("'20231023' <= date <= '20231031'")
    data_test = data_test.sort_values(by=['datetime'])
    data_test = data_test.reset_index(drop=True)
    print(f'测试集data有{data_test.shape[0]}行数据')

    data_test['heston_price']=calculate_all_Heston_value(data_test,calibrated_params)
    data_test,error_summary=calibration_report_MSE(data_test)
    plot_price_comparison_chart(data_test, calibrated_params)

    file_path2 = os.path.join(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)',
                              f"(20231227, 3.6, 20160)_测试集.csv")
    data_test.to_csv(file_path2, index=False)



