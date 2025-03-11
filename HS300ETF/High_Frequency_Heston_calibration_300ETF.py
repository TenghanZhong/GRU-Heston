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

def constrain_f(param):#添加限制条件
    theta, kappa, sigma, rho, v0 = param
    return 2 * kappa * theta - sigma * sigma

def setup_model(rf_curve, dividend_yield,spot_price,initial_params=(0.02, 0.2, 0.5, 0, 0.01)):
    initial_value = ql.QuoteHandle(ql.SimpleQuote(spot_price))  # 设定标的市场价格
    theta, kappa, sigma, rho, v0 = initial_params
    HestonProcess = ql.HestonProcess(rf_curve, dividend_yield,
                                     initial_value, v0, kappa, theta, sigma, rho)
    HestonModel = ql.HestonModel(HestonProcess)  # 创建赫斯顿模型
    engine = ql.AnalyticHestonEngine(HestonModel)  # 为赫斯顿模型设置一个分析型引擎

    return HestonModel,engine

def setup_helpers(option_data):
    day_count = ql.Actual365Fixed()  # 一年按365天
    calendar = ql.China(ql.China.SSE)  # 设定交易所的交易日历
    heston_helpers = []
    grid_data = []

    for i in range(option_data.shape[0]):
        today = ql.DateParser.parseFormatted(str(option_data.loc[i,"date"]), '%Y%m%d')
        ql.Settings.instance().evaluationDate = today
        t= int(option_data.loc[i,'ttm_days'])
        spot_price=option_data.loc[i,'s']#标的市场价格
        period = ql.Period(t, ql.Days)# 创建一个Period对象，表示这些天数
        volatility=option_data.loc[i,'implied_volatility']#做heston校准波动率用的是BS反推得到的IV
        rf_curve, dividend_yield = option_data.loc[i, 'rf'],option_data.loc[i, 'dividend']
        rf_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, rf_curve, day_count))
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_yield, day_count))

        helper = ql.HestonModelHelper(#这是创建的HestonModelHelper对象，用于期权定价模型的校准。
            period, calendar, spot_price, option_data.loc[i,"k"] \
            , ql.QuoteHandle(ql.SimpleQuote(volatility)),
            rf_curve, dividend_yield)
        heston_helpers.append(helper)#helper存储period,spot,stike,volatility,rf,dividend信息
        grid_data.append((ql.DateParser.parseFormatted(str(option_data.loc[i,"maturity_date"]), '%Y%m%d') \
                              , option_data.loc[i,"k"]))#保存到期日和行权价信息，两个两个为1元祖
    return heston_helpers, grid_data

def cost_function_generator_fine(heston_helpers, option_data, norm=True):
    def cost_function(params):
        theta, kappa, sigma, rho, v0 = params

        # 打印当前参数值
        print(f"当前参数: theta={theta}, kappa={kappa}, sigma={sigma}, rho={rho}, v0={v0}")

        # 检查 Feller 条件
        if 2 * kappa * theta <= sigma ** 2:
            print("Feller 条件不满足")

        day_count = ql.Actual365Fixed()
        calendar = ql.China(ql.China.SSE)
        error = []
        for i, helper in enumerate(heston_helpers):
            today = ql.DateParser.parseFormatted(str(option_data.loc[i,"date"]), '%Y%m%d')
            ql.Settings.instance().evaluationDate = today
            rf = option_data.loc[i, 'rf']
            dividend = option_data.loc[i, 'dividend']
            spot_price = option_data.loc[i, 's']
            rf_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, rf, day_count))
            dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, day_count))

            # 使用当前的参数创建局部Heston过程
            local_model, local_engine = setup_model(rf_curve, dividend_yield, spot_price,
                                                    initial_params=(theta, kappa, sigma, rho, v0))
            helper.setPricingEngine(local_engine)
            err = helper.calibrationError()
            error.append(err)

        if norm:
            total_error = np.sum(np.abs(error))##误差可改
            print(f"总误差: {total_error}")
            return total_error
        else:
            return error

    return cost_function


def calibration_report_MSE(heston_helpers,grid_data):
    avg=0
    for i,helper in enumerate(heston_helpers):
        err=(helper.modelValue()-helper.marketValue())
        data,strike=grid_data[i]
        avg+=0.5*err**2#sum of mean squared error
    avg=avg*100/len(heston_helpers)#mean squared error(sum÷len)
    summary = "Square Error (%%) : %5.9f" % (avg)
    return avg

def calibration_report_MAPE(heston_helpers, grid_data):
    total_percentage_error = 0
    for helper in heston_helpers:
        actual_value = helper.marketValue()
        predicted_value = helper.modelValue()

        # 避免除以零
        if actual_value != 0:
            percentage_error = abs((actual_value - predicted_value) / actual_value)
            total_percentage_error += percentage_error

    mape = (total_percentage_error / len(heston_helpers)) * 100  # MAPE
    return mape


def get_heston_param_TRR(data,initial_params,bounds):
    #get calibration result summary
    summary=[]
    heston_helpers, grid_data = setup_helpers(data)

    #Set parameter bounds(feller condition)
    #param = (theta#长期均值波动率, kappa, sigma, rho, v0  # 初始方差)
    #cost_function=cost_function_generator_1(HestonModel, heston_helpers, norm=True)#在这里Hestionmodel设置了param，形成了error
    cost_function = cost_function_generator_fine( heston_helpers, data, norm=True)
    nlc=optimize.NonlinearConstraint(constrain_f,0,np.inf)

    def callback(xk, convergence):
        print(f"收敛度: {convergence}")

    #use differential_evolution(差分进化算法优化)：

    sol=optimize.minimize(cost_function, initial_params,method='trust-constr', bounds=bounds, constraints=[nlc],callback=callback)
    #sol = optimize.basinhopping(cost_function, initial_params, niter=100)

    theta, kappa, sigma, rho, v0=list(sol.x)
    error = calibration_report_MSE(heston_helpers, grid_data)#报告最小误差
    # 准备数据以创建 DataFrame
    summary = [error] + list(sol.x)
    result_df = pd.DataFrame([summary], columns=['MSE_Error', 'Theta', 'Kappa', 'Sigma', 'Rho', 'V0'])
    # 更新 params
    params = (theta, kappa, sigma, rho, v0)
    return result_df,params

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
    plt.show()


if __name__ == "__main__":
    data= pd.read_csv(r'E:\Option_data\上证50ETF\实证选择数据\50ETF看涨(20231227,2.608,20160).csv')
    data['maturity_date'], data['date'] = (pd.to_datetime(data['maturity_date'].astype(str)).
                                                     dt.strftime('%Y%m%d'),
                                                     pd.to_datetime(data['date'].astype(str)).dt.strftime(
                                                         '%Y%m%d'))
    data = data.query("'20230829' <= date <= '20230830'")
    data = data.sort_values(by=['datetime'])
    data = data.reset_index(drop=True)
    option_data=data
    pd.set_option('display.max_columns', None)
    initial_params=(0.02, 0.3, 0.1, 0, 0.01)#初始设置的参数值
    bounds = [(0.001, 1.0), (0.1, 25.0), (0.01, 4), (-1.0, 1.0), (0.001, 1.0)]#设置每个值边界
    #theta, kappa, sigma, rho, v0 = params
    result_df,calibrated_params = get_heston_param_TRR(option_data,initial_params,bounds)

    data['heston_price']=calculate_all_Heston_value(option_data,calibrated_params)
    print(result_df)  # 参数校准结果
    plot_price_comparison_chart(data,calibrated_params)  # 绘制对比图

'''
优化校准：①minimize算法选择,全局算法（差分）,或者TRR   还可以考虑使用 SLSQP 或 L-BFGS-B 等算法
②损失函数的选择   np.sum(np.abs(error))    print(f"总误差: {total_error}")        return total_error
或者 total_error = sum(helper.calibrationError() for helper in heston_helpers)
return np.sqrt(total_error / len(heston_helpers)) if norm else total_error
③寻找损失函数好minimize算法的不同组合关系
'''


'''

在现实中，对于每个期权数据点，理想情况下应该使用对应时间点的无风险利率和股息率来构建Heston模型和定价引擎。
但是，QuantLib中的AnalyticHestonEngine需要一个固定的Heston过程，而Heston过程又是基于单一的无风险利率和股息率创建的。
这就带来了一个问题：我们不能为每个数据点创建一个单独的Heston模型和引擎，因为这样做在计算上是不切实际的。
一种可能的解决方案是使用整个数据集的代表性值（如平均值）来创建初始的Heston模型和引擎，
然后在计算每个期权数据点的误差时考虑其特定的无风险利率，股息率，和标的价格。具体来说，可以在计算误差时，
用每个数据点的无风险利率和股息率和标的价格来调整期权的理论价格，产生误差。
'''
