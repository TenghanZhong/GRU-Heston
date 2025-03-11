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

def calculate_single_BS_value(df):
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

    initial_value_spot_price = ql.QuoteHandle(ql.SimpleQuote(spot_price))  # 设定标的市场价格

    # 设置无风险利率曲线
    rf_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, rf, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, day_count))

    ConstantVol_volatility = ql.BlackConstantVol(today, ql.NullCalendar(), 0.1748,
                                                 ql.Actual365Fixed())  # 这里设定了一个年化波动率为10%的恒定波动率结构。
    volatility = ql.BlackVolTermStructureHandle(ConstantVol_volatility)  # 将波动率结构转换为QuantLib可以识别的形式：
    BSM_process = ql.BlackScholesMertonProcess(initial_value_spot_price, dividend_yield, rf_curve, volatility)
    engine = ql.AnalyticEuropeanEngine(BSM_process)  # 为BSM模型设置一个分析型引擎
    option.setPricingEngine(engine)  # 设置定价引擎，将BSM引擎应用于之前定义的期权对象，以便进行定价。
    price_BSM = option.NPV()  # 期权的理论市场价值。
    return price_BSM

def calculate_all_BS_value(df):
    pool=multiprocessing.Pool(8)
    results = pool.map(calculate_single_BS_value, [df.iloc[i] for i in range(len(df))])
    #pool.starmap 是 pool.map 的一个变体，允许被映射的函数接受多个参数。
    pool.close()  # 关闭进程池，不再接受新的任务
    pool.join()  # 等待所有子进程结束
    return results

def calibration_report_MSE(data):
    avg=0
    data['BS_error']=data['c']-data['BS_price']#计算误差
    for i in range(len(data)):
        err=data.loc[i,'BS_error']
        avg+=0.5*err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error (%%) : %5.9f" % (avg)
    return data,error_summary #返回计算error后的data

def calibration_report_MAPE(data):
    data['MAPE_BS_error']=np.abs(data['c']-data['BS_price'])/data['c']
    total_error=data['MAPE_BS_error'].sum()
    avg=total_error*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Absolute Percentage Error (%%) : %5.9f" % (avg)
    return data,error_summary

def plot_price_comparison_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['c'], label='Market Price', color='blue')
    plt.plot(data['BS_price'], label='BS price', color='red')
    date=data['date'].unique().tolist()
    day1,day2=date[0],date[-1]
    plt.title(f'Market Price vs BS Model Price(date: {day1}-to-{day2} )')
    plt.ylabel('Price')
    plt.legend()
    month_day1,month_day2=day1[-4:],day2[-4:]
    plt.show()

if __name__ == "__main__":
    data= pd.read_csv(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\300ETF看涨期权(20231227, 3.6, 20160).csv')

    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data['maturity_date'], data['date'] = (pd.to_datetime(data['maturity_date'].astype(str)).
                                                     dt.strftime('%Y%m%d'),
                                                     pd.to_datetime(data['date'].astype(str)).dt.strftime(
                                                         '%Y%m%d'))
    #implied_vol=data['implied_volatility'].mean()  #=0.16016665374402594

    #添加测试集  十月为测试集
    #调参两天（1009-1010） 1011-1022------1023-1031为测试集
    data_test = data.query("'20231023' <= date <= '20231031'")
    data_test = data_test.sort_values(by=['datetime'])
    data_test = data_test.reset_index(drop=True)
    print(f'测试集data有{data_test.shape[0]}行数据')

    data_test['BS_price']=calculate_all_BS_value(data_test)
    data_test,error_summary=calibration_report_MSE(data_test)
    plot_price_comparison_chart(data_test)

    file_path2 = os.path.join(r'E:\Option_data\沪深300ETF\300ETF实证选择数据\LSTM_(20231227, 3.6, 20160)',
                              f"(20231227, 3.6, 20160)_测试集_bsm.csv")
    data_test.to_csv(file_path2, index=False)

