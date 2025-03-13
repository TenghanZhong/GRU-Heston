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

def constrain_f(params):
    theta, kappa, sigma, rho, v0 = params
    return 2 * kappa * theta - sigma * sigma

def setup_model(i,option_data,params):
    #设置Heston模型
    day_count = ql.Actual365Fixed()
    calendar = ql.China(ql.China.SSE)
    theta, kappa, sigma, rho, v0 = params
    option_type = ql.Option.Call
    strike_price = option_data.loc[i, 'k']
    call_payoff = ql.PlainVanillaPayoff(option_type, strike_price)  # 创造一个看涨期权收益结构
    today = ql.DateParser.parseFormatted(str(option_data.loc[i, "date"]), '%Y%m%d')
    maturity = ql.DateParser.parseFormatted(str(option_data.loc[i, "maturity_date"]), '%Y%m%d')
    ql.Settings.instance().evaluationDate = today
    call_exercise = ql.EuropeanExercise(maturity)  # 创造一个欧式行权方式对象，指定期权到期日maturity
    option = ql.VanillaOption(call_payoff, call_exercise)  # 创造一个看涨期权对象（结合其收益结构的行权方式）

    rf = option_data.loc[i, 'rf']
    dividend = option_data.loc[i, 'dividend']
    spot_price = option_data.loc[i, 's']
    initial_value_spot_price = ql.QuoteHandle(ql.SimpleQuote(spot_price))  # 设定标的市场价格

    # 设置无风险利率曲线
    rf_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, rf, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, day_count))

    heston_process = ql.HestonProcess(rf_curve, dividend_yield, initial_value_spot_price, v0,
                                      kappa, theta, sigma, rho)
    hestonModel = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(hestonModel)

    return option,engine

def calculate_single_option_error_for_multiprocessing(i, option_data, params):
    option, Heston_engine = setup_model(i, option_data, params)
    option.setPricingEngine(Heston_engine)
    model_value = option.NPV()
    market_value = option_data.loc[i, 'c']
    err = ((model_value - market_value) / market_value) ** 2
    return err

def cost_function_generator_multiprocessing(option_data, norm=True):
    def cost_function(params):
        theta, kappa, sigma, rho, v0 = params

        # 打印当前参数值
        print(f"当前参数: theta={theta}, kappa={kappa}, sigma={sigma}, rho={rho}, v0={v0}")

        # 检查 Feller 条件
        if 2 * kappa * theta <= sigma ** 2:
            print("Feller 条件不满足")

        pool=multiprocessing.Pool(8)
        error=pool.starmap(calculate_single_option_error_for_multiprocessing,
                           [(i, option_data, params) for i in range(len(option_data))])
        pool.close()  # 关闭进程池，不再接受新的任务
        pool.join()  # 等待所有子进程结束

        if norm:
            total_error = np.sqrt(np.sum(error))  ##误差可改
            print(f"总误差: {total_error}")
            return total_error
        else:
            return error

    return cost_function

def cost_function_generator(option_data, norm=True):
    def cost_function(params):
        theta, kappa, sigma, rho, v0 = params

        # 打印当前参数值
        print(f"当前参数: theta={theta}, kappa={kappa}, sigma={sigma}, rho={rho}, v0={v0}")

        # 检查 Feller 条件
        if 2 * kappa * theta <= sigma ** 2:
            print("Feller 条件不满足")

        error = []
        for i in range(len(option_data)):
            option, Heston_engine=setup_model(i,option_data,params)# 为这一时刻期权数据设置一个Heston分析型引擎

            option.setPricingEngine(Heston_engine)  # 设置定价引擎，将赫斯顿引擎应用于之前定义的期权对象，以便进行定价。
            model_value = option.NPV()  # 期权这一时刻的理论市场价值。
            market_value=option_data.loc[i,'c']
            err = ((model_value-market_value)/market_value)**2 #加权均方误差
            error.append(err)

        if norm:
            total_error = np.sqrt(np.sum(error))  ##误差可改
            print(f"总误差: {total_error}")
            return total_error
        else:
            return error

    return cost_function


def calibration_report_MSE(data):
    avg=0
    data['error']=data['c']-data['heston_price']
    for i in range(len(data)):
        err=data.loc[i,'error']
        avg+=0.5*err**2#sum of mean squared error
    avg=avg*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Square Error (%%) : %5.9f" % (avg)
    return data,error_summary

def calibration_report_MAPE(data):
    data['MAPE_error']=np.abs(data['c']-data['heston_price'])/data['c']
    total_error=data['MAPE_error'].sum()
    avg=total_error*100/len(data)#mean squared error(sum÷len)
    error_summary = "Mean Absolute Percentage Error (%%) : %5.9f" % (avg)
    return data,error_summary


def get_heston_param_TRR(data,initial_params,bounds):
    #param = (theta#长期均值波动率, kappa, sigma, rho, v0  # 初始方差)

    cost_function = cost_function_generator(data,norm=True)
    #添加限制条件
    nlc=optimize.NonlinearConstraint(constrain_f,0 , np.inf)

    def callback(xk, convergence):
        print(f"收敛度: {convergence}")

    #信赖域反射
    sol = optimize.minimize(cost_function, initial_params, method='trust-constr', bounds=bounds,
                            constraints=[nlc],callback=callback) ###
    #模拟退火法：
    #sol = optimize.basinhopping(cost_function, initial_params, niter=100, callback=callback)

    theta, kappa, sigma, rho, v0=list(sol.x)
    params = tuple(round(param, 6) for param in [theta, kappa, sigma, rho, v0])

    return params

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
    plt.savefig(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\{params[0]}_校准期({month_day1}-{month_day2}).png')
    plt.show()


if __name__ == "__main__":
    data= pd.read_csv(r'E:\Option_data\上证50ETF\实证选择数据\50ETF看涨(20231227,2.608,20160).csv')
    data['maturity_date'], data['date'] = (pd.to_datetime(data['maturity_date'].astype(str)).
                                                     dt.strftime('%Y%m%d'),
                                                     pd.to_datetime(data['date'].astype(str)).dt.strftime(
                                                         '%Y%m%d'))
    data = data.query( "'20231009' <= date <= '20231010'")#选择参数估计的日子
    data = data.sort_values(by=['datetime'])
    data = data.reset_index(drop=True)
    pd.set_option('display.max_columns', None)
    initial_params=(0.02, 0.3, 0.1, 0, 0.01)#初始设置的参数值
    bounds = [(0.001, 1.0), (0.1, 25.0), (0.01, 4), (-1.0, 1.0), (0.001, 1.0)]#设置参数边界
    #theta, kappa, sigma, rho, v0 = params

    calibrated_params = get_heston_param_TRR(data,initial_params,bounds)
    result_df = pd.DataFrame([calibrated_params], columns=['Theta', 'Kappa', 'Sigma', 'Rho', 'V0'])
    print(result_df)

    data['heston_price']=calculate_all_Heston_value(data,calibrated_params)
    plot_price_comparison_chart(data,calibrated_params)  # 绘制对比图
    data, error_summary = calibration_report_MSE(data)
    print(error_summary)
    date=data['date'].unique().tolist()
    month_day1, month_day2 = date[0][-4:], date[-1][-4:]
    with open(f'E:\Option_data\上证50ETF\实证选择数据\拟合图像\error_summary({month_day1}-{month_day2}).txt', 'w') as file:
        file.write(str(error_summary))