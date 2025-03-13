import tushare as ts
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import time
import tushare as ts
import multiprocessing

pd.set_option('display.max_columns', None)
ts.set_token('d689cb3c1d8c8a618e49ca0bb64f4d6de2f70e28ab5f76a867b31ac7')  # 请替换成你自己的token
pro = ts.pro_api()

file_name1='D:\python_data\上证50ETF\华夏上证50ETF看跌期权(pure1).csv'
file_name2='D:\python_data\上证50ETF\华夏上证50ETF看跌期权(pure2).csv'
data1=pd.read_csv(file_name1)
data2=pd.read_csv(file_name2)
df_etf=pd.read_csv(r'D:\python_data\上证50ETF\510050.XSHG.csv')

data1=data1.rename(columns={'order_book_id':'ts_code'})#data.rename(columns={'a':'b'})
data1['ts_code']=data1['ts_code'].astype(str)+'.SH' #不能用str（data1['ts_code']），str会把整列算到一行中
data1=data1.drop(columns=['high','low'])

data2=data2.rename(columns={'order_book_id':'ts_code'})
data2['ts_code']=data2['ts_code'].astype(str)+'.SH'
data2.drop(columns=['high','low'],inplace=True)#data2.drop(columns=['high','low'],replace=True)


#补充合约信息到分钟数据中
ts_code_list=[]
ts_1=list(data1['ts_code'].unique())#筛选出独立的期权代码.   data[].unique() is array
ts_2=list(data2['ts_code'].unique())
ts_code_list=ts_1+ts_2

columns_to_update=['exercise_type','exercise_price','maturity_date','call_put']
for i in columns_to_update:#将要补充的期权信息列名字补充进去
    data1[i]=np.nan
    data2[i]=np.nan

for code in ts_1:#同时遍历两个列表
    mask=(data1['ts_code']==code)#
    # 提取合约基础信息
    option_data = pro.opt_basic(ts_code=code,exchange='SSE',
                                field='exercise_type,exercise_price,maturity_date,call_put')
    option_data=option_data[['exercise_type','exercise_price','maturity_date','call_put']]
    for column in columns_to_update:
        data1.loc[mask, column] = option_data[column].iloc[0]#让code的一列column全为一个值（同一个code的合约信息一定相同）
for code in ts_2:
    mask = (data2['ts_code'] == code)
    # 提取合约基础信息
    option_data = pro.opt_basic(ts_code=code, exchange='SSE',
                                field='exercise_type,exercise_price,maturity_date,call_put')
    option_data = option_data[['exercise_type', 'exercise_price', 'maturity_date','call_put']]
    for column in columns_to_update:
        data2.loc[mask, column] = option_data[column].iloc[0]

#Add spot_price to option data
#上面已经 df_etf=pd.read_csv(r'E:\Option_data\上证50ETF\510050.XSHG.csv')
df_etf=df_etf[['datetime','open']]
df_etf=df_etf.rename(columns={'open':'s'})#df.rename(columns={'old_columns':'new_columns'})
data1=pd.merge(data1,df_etf,on='datetime')#merge前后关系
data2=pd.merge(data2,df_etf,on='datetime')

# 直接计算两个日期列的差异
data1['ttm_days'] = (pd.to_datetime(data1['maturity_date']) - pd.to_datetime(data1['trading_date'])).dt.days#计算日期间隔 .dt就是一列一起算
data2['ttm_days'] = (pd.to_datetime(data2['maturity_date']) - pd.to_datetime(data2['trading_date'])).dt.days
data1['ttm_ratio']=data1['ttm_days']/365
data2['ttm_ratio']=data2['ttm_days']/365
data1 = data1[data1['ttm_days'] != 0]#或者data1 = data1.query('ttm_days != 0')，，data1 = data1.loc[data1['ttm_days'] != 0]
data2 = data2[data2['ttm_days'] != 0] #三种方法：直接mask法，query法,data.loc+掩码法

##计算Rf，线性插值法（估算两个已知数据点之间的数值的方法）
df_rf = pro.shibor(start_date='20230807', end_date='20231208')#无风险利率dataframe
def calculate_linear_interpolation(days, short_rate, long_rate, short_days, long_days):
    return short_rate + ((long_rate - short_rate) * (days - short_days)) / (long_days - short_days)


def get_rate_interval(ttm_days):
    if 1 < ttm_days < 7:
        return ('on', '1w', 1, 7)
    elif 7 < ttm_days < 14:
        return ('1w', '2w', 7, 14)
    elif 14 < ttm_days < 30:
        return ('2w', '1m', 17, 30)
    elif 30 < ttm_days < 90:
        return ('1m', '3m', 30, 90)
    elif 90 < ttm_days < 180:
        return ('3m', '6m', 90, 180)
    elif 180 < ttm_days < 270:
        return ('6m', '9m', 180, 270)
    elif 270 < ttm_days < 365:
        return ('9m', '1y', 270, 365)

def calculate_single_rf(data):
    '''
    :param data: 是一个单独的值，不是series，后面用multiprocessing作用在每一行
    :return: rf for the single row
    '''
    #估算任意到期时间对应的无风险利率
    today=pd.to_datetime(data['trading_date']).strftime('%Y%m%d')
    # 划分标准分区间，用字典形式映射
    ttm_to_shibor = {1: 'on',7: '1w',14: '2w',30: '1m',90: '3m',180: '6m',270: '9m',365: '1y'}
    # 为每一行的ttm_days找到相应的SHIBOR列名
    shibor_col = ttm_to_shibor.get(data['ttm_days'])
    # 如果找到了相应的SHIBOR列名，就从df_rf中提取利率
    if shibor_col:
        rf = df_rf.loc[df_rf['date'] == today, shibor_col].iloc[0]
        #注意df_rf列名为利率日，行名为实际日期，每一天都有不同期限的利率
    else:
        interval=get_rate_interval(data['ttm_days'])
        days=data['ttm_days']
        short_col, long_col, short_days, long_days = interval
        short_rate=df_rf.loc[df_rf['date'] == today, short_col].iloc[0]
        long_rate = df_rf.loc[df_rf['date'] == today, long_col].iloc[0]
        rf=calculate_linear_interpolation(days, short_rate, long_rate, short_days, long_days)

    return rf

def calculate_all_rf(data):
    pool=multiprocessing.Pool(8)#并行的行数 pool=multiprocessing.Pool()
    results=pool.map(calculate_single_rf,[data.iloc[i] for i in range(len(data))])# pool.map(func,[data.iloc[i] for i in range(data.shape[0])])
    #result is a list of rf from first row to the last row
    pool.close()  # 关闭进程池，不再接受新的任务
    pool.join()  # 等待所有子进程结束
    return results #返回一个list

data1['rf']=calculate_all_rf(data1)# 另dataframe一列为一个列表
data2['rf']=calculate_all_rf(data2)

#修改数据列名
data1=data1[['ts_code','datetime','open','trading_date','exercise_price','maturity_date','rf','s','ttm_days','ttm_ratio','call_put']]
data2=data2[['ts_code','datetime','open','trading_date','exercise_price','maturity_date','rf','s','ttm_days','ttm_ratio','call_put']]
data1['trading_date']=pd.to_datetime(data1['trading_date']).dt.strftime('%Y%m%d')#对于一列的时间数据，要想整体改变一列就要用pd.to_datetime(data['column']).dt.days/.strftime('%Y%m%d')
data2['trading_date']=pd.to_datetime(data2['trading_date']).dt.strftime('%Y%m%d')
data1=data1.rename(columns={'open':'c','trading_date':'date','exercise_price':'k'})
data2=data2.rename(columns={'open':'c','trading_date':'date','exercise_price':'k'})
'''
#保存数据
file_path = os.path.join('E:\Option_data\上证50ETF', f"上证50ETF看跌期权(1).csv")
file_path2 = os.path.join('E:\Option_data\上证50ETF', f"上证50ETF看跌期权(2).csv")
data1.to_csv(file_path, index=False)
data2.to_csv(file_path2, index=False)
'''

