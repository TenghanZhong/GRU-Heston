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
data=pd.read_csv(r'E:\Option_data\沪深300ETF\300ETF_看涨期权_iv.csv')

#检查有多少不同到期日，样本数分别为多少
maturity_date=list(data['maturity_date'].unique())
number_of_maturity_date=len(maturity_date)
print(f'到期日一共有{number_of_maturity_date}种不同天数')
for ma_date in maturity_date:
    cc=len(data.loc[data['maturity_date']==ma_date,'maturity_date'])
    print(f'到期日为{ma_date}有{cc}个样本')

#检查有多少不同行权价，样本数分别为多少
exercise_price=list(data['k'].unique())
number_of_exercise_price=len(exercise_price)
print(f'行权价一共有{number_of_exercise_price}种不同的值')
for k in exercise_price:
    cc=len(data.loc[data['k']==k,'k'])
    print(f'行权价为{k}有{cc}个样本')

#检查 到期日和行权价 不同组合的样本数
combo=[]
for k in exercise_price:
    for day in  maturity_date:
        tuple=(day,k)
        combo.append(tuple)

#
count_cumbo=[]
for combo in combo:
    matu_date,k=combo
    mask=(data['maturity_date']==matu_date)&(data['k']==k)
    count_cumbo.append((matu_date,k,len(data.loc[mask])))
count_cumbo.sort(key=lambda x: x[2],reverse=True)#倒序，按照样本数排列
print(count_cumbo)#(maturity_date,k,样本数)


#选择参数：(20231227, 3.6, 20160) (20231227, 3.8, 20160) (20231227, 3.7, 20160),
mask=(data['maturity_date']==20231227)&(data['k']==3.7)
data1=data.loc[mask]
data1.sort_values(by='datetime',inplace=True)#得到指定样本并排序
data1['moneyness']=data1['s']/data1['k']

info=(20231227, 3.7, 20160)
filepath=os.path.join(r'E:\Option_data\沪深300ETF\300ETF实证选择数据',f"300ETF看涨期权{info}.csv")
data1.to_csv(filepath, index=False)
''