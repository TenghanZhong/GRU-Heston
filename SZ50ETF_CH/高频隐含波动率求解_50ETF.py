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

def cal_single_dividend(df):  # 计算隐含分红率
    # q = -log((df.settle+df.exercise_price*exp(-df.interest*df.delta)-df.settle_p)/(df.s0))/df.delta
    q = -log((df.c + df.k * exp(-df.rf * (df.ttm_ratio)) - df.c_p) / (df.s)) / (df.ttm_ratio)
    return q

def cal_ALL_dividend(df_c,df_p):  # 计算隐含分红率
    # 计算隐含分红率
    df_p = df_p.rename(columns={'c': 'c_p', 'ts_code': 'ts_code_p',
                                'call_put': 'call_put_p'})
    df = pd.merge(df_c, df_p, how='left', on=['datetime','date','k', 'ttm_ratio','ttm_days','rf', 's'])
    # 把相同日期，执行价，时间段，无风险利率和市标的物价格的期权merge到一起，以便求隐含分红率

    pool=multiprocessing.Pool(8)
    print('可用CPU核心为{},开始计算分红率'.format(multiprocessing.cpu_count()))
    results=pool.map(cal_single_dividend,[df.iloc[i] for i in range(len(df))])
    pool.close()  # 关闭进程池，不再接受新的任务
    pool.join()  # 等待所有子进程结束

    df['dividend'] = results

    df_c = df[
        ['ts_code','datetime', 'call_put', 'k', 'date', 'c', 's', 'rf', 'ttm_ratio', 'ttm_days', 'dividend', 'maturity_date_x']]
    df_c = df_c.rename(columns={'maturity_date_x': 'maturity_date'})
    df_p = df[['ts_code_p','datetime','date', 'c_p', 'call_put_p', 'k', 'ttm_ratio', 'ttm_days', 'rf', 's', 'dividend',
               'maturity_date_y']]
    df_p = df_p.rename(columns={'c_p': 'c', 'ts_code_p': 'ts_code',
                                'call_put_p': 'call_put', 'maturity_date_y': 'maturity_date'})

    return df_c,df_p

def cal_single_implied_volatility(df):
    if df['call_put'] == 'P':
        option_type = ql.Option.Put
    elif df['call_put'] == 'C':
        option_type = ql.Option.Call
    strike_price = df['k']
    call_payoff = ql.PlainVanillaPayoff(option_type, strike_price)  # 创造一个看涨期权收益结构

    day_count = ql.Actual365Fixed()  # 一年按365天
    calendar = ql.China(ql.China.SSE)  # 设定交易所的交易日历

    maturity = ql.Date(df['maturity_date'], '%Y%m%d')  # 15th June 2023
    today = ql.Date(df['date'], '%Y%m%d')  # 15th December 2022
    ql.Settings.instance().evaluationDate = today

    call_exercise = ql.EuropeanExercise(maturity)  # 创造一个欧式行权方式对象，指定期权到期日maturity
    option = ql.VanillaOption(call_payoff, call_exercise)  # 创造一个看涨期权对象（结合其收益结构的行权方式）

    # Option input value
    spot_price = df['s']  # 标的市场价格,这里是hs300
    rf = df['rf']  # 无风险利率
    dividend = df['dividend']  # 股息率
    '''
    yearly_historical_volatility = 0.1
    variance = 0.01  # 初始方差
    kappa = 2  # 均值回归速度
    theta = 0.01  ##波动率的长期均值水平
    sigma = 0.1  # 波动率的波动率（volatility of volatility）
    rho = 0.5  # 资产价格和波动率的相关性
    '''

    initial_value = ql.QuoteHandle(ql.SimpleQuote(spot_price))  # 设定标的市场价格

    # 设置无风险利率曲线
    rf_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, rf, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, day_count))

    # 用BSM求隐含波动率
    estimated_volatility = ql.BlackConstantVol(today, ql.NullCalendar(), 0.5,
                                               day_count)  # 这里设定了一个年化波动率为50%的恒定波动率结构,初始值
    volatility = ql.BlackVolTermStructureHandle(estimated_volatility)  # 将波动率结构转换为QuantLib可以识别的形式：
    BSM_process = ql.BlackScholesMertonProcess(initial_value, dividend_yield, rf_curve, volatility)
    engine = ql.AnalyticEuropeanEngine(BSM_process)  # 为BSM模型设置一个分析型引擎
    option.setPricingEngine(engine)  # 设置定价引擎，将BSM引擎应用于之前定义的期权对象，以便进行定价。

    target_value = df['c']  # 目标值
    try:
        implied_volatility = option.impliedVolatility(target_value,
                                                      BSM_process)
    except:
        implied_volatility = 0
    return implied_volatility

def cal_ALL_implied_volatility(data):
    pool = multiprocessing.Pool(8)
    print(f'可用CPU核心为{multiprocessing.cpu_count()},开始计算隐含波动率')
    results = pool.map(cal_single_implied_volatility, [data.iloc[i] for i in range(len(data))])
    pool.close()  # 关闭进程池，不再接受新的任务
    pool.join()  # 等待所有子进程结束
    return results #返回一个list

def vol_plot(data):  # 画微笑曲线
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Extracting the columns for the axes
    x = data['k']
    y = data['ttm_ratio']
    z = data['implied_volatility']

    # Creating a grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolating z values on the grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    # Creating a 3D scatter plot
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')

    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel('exercise_price')
    ax.set_ylabel('time_to_maturity')
    ax.set_zlabel('Implied_Volatility')

    plt.title('3D Surface Plot of Implied Volatility')
    plt.colorbar(surf)
    plt.show()  ##画微笑曲线

if __name__ == '__main__':
    ts.set_token('dfb6e9f4f9a3db86c59a3a0f680a9bdc46ed1b5adbf1e354c7faa761')  # 请替换成你自己的token
    pro = ts.pro_api()
    pd.set_option('display.max_columns', None)

    data1 = pd.read_csv('E:\Option_data\上证50ETF\上证50ETF看涨期权(1).csv')
    data2 = pd.read_csv('E:\Option_data\上证50ETF\上证50ETF看涨期权(2).csv')
    data3 = pd.read_csv('E:\Option_data\上证50ETF\上证50ETF看跌期权(1).csv')
    data4 = pd.read_csv('E:\Option_data\上证50ETF\上证50ETF看跌期权(2).csv')

    #链接文件
    data_call=pd.concat([data1,data2],axis=0,ignore_index=True)
    data_put=pd.concat([data3,data4],axis=0,ignore_index=True)
    data_call['rf'] = data_call['rf'] / 100
    data_put['rf'] = data_put['rf'] / 100
    #日期格式处理
    data_call['maturity_date'], data_call['date']= (pd.to_datetime(data_call['maturity_date'].astype(str)).
                                            dt.strftime('%Y%m%d') , pd.to_datetime(data_call['date'].astype(str)).dt.strftime('%Y%m%d'))
    data_put['maturity_date'], data_put['date']= (pd.to_datetime(data_put['maturity_date'].astype(str)).
                                            dt.strftime('%Y%m%d') , pd.to_datetime(data_put['date'].astype(str)).dt.strftime('%Y%m%d'))
    #计算分红率
    data_call,data_put=cal_ALL_dividend(data_call,data_put)
    missing_values1 = data_call.isnull().any()
    print(missing_values1)
    missing_values2 = data_put.isnull().any()
    print(missing_values2)
    '''
    file_path1 = os.path.join('E:\Option_data\上证50ETF', f"上证50ETF_看涨期权_dividend.csv")
    file_path2 = os.path.join('E:\Option_data\上证50ETF', f"上证50ETF_看跌期权_dividend.csv")
    data_call.to_csv(file_path1, index=False)
    data_put.to_csv(file_path2, index=False)
    
    #计算隐含波动率
    data_call = pd.read_csv('E:\Option_data\上证50ETF\上证50ETF_看涨期权_dividend.csv')
    data_put = pd.read_csv('E:\Option_data\上证50ETF\上证50ETF_看跌期权_dividend.csv')
    '''
    data_call['maturity_date'], data_call['date'] = (pd.to_datetime(data_call['maturity_date'].astype(str)).
                                                     dt.strftime('%Y%m%d'),
                                                     pd.to_datetime(data_call['date'].astype(str)).dt.strftime('%Y%m%d'))
    data_put['maturity_date'], data_put['date'] = (pd.to_datetime(data_put['maturity_date'].astype(str)).
                                                   dt.strftime('%Y%m%d'),
                                                   pd.to_datetime(data_put['date'].astype(str)).dt.strftime('%Y%m%d'))
    data_call['implied_volatility'] = cal_ALL_implied_volatility(data_call)
    data_put['implied_volatility'] = cal_ALL_implied_volatility(data_put)
    #画图
    vol_plot(data_call)
    vol_plot(data_put)
    #保存文件
    file_path1 = os.path.join('E:\Option_data\上证50ETF', f"上证50ETF_看涨期权_iv.csv")
    file_path2 = os.path.join('E:\Option_data\上证50ETF', f"上证50ETF_看跌期权_iv.csv")
    data_call.to_csv(file_path1, index=False)
    data_put.to_csv(file_path2, index=False)
