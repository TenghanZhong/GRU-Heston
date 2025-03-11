import pandas as pd
import os

data=pd.read_csv('E:\Option_data\沪深300ETF\华泰柏瑞沪深300ETF看涨期权.csv')
data_die=pd.read_csv('E:\Option_data\沪深300ETF\华泰柏瑞沪深300ETF看跌期权.csv')


data1=data.iloc[0:2880]
data2=data.iloc[2880:]
data3=data_die.iloc[0:960]
data4=data_die.iloc[960:]

file_path = os.path.join('E:\Option_data\沪深300ETF', f"300ETF看涨期权(pure1).csv")
file_path2 = os.path.join('E:\Option_data\沪深300ETF', f"300ETF看涨期权(pure2).csv")
data1.to_csv(file_path, index=False)
data2.to_csv(file_path2, index=False)

file_path3 = os.path.join('E:\Option_data\沪深300ETF', f"300ETF看跌期权(pure1).csv")
file_path4 = os.path.join('E:\Option_data\沪深300ETF', f"300ETF看跌期权(pure2).csv")
data3.to_csv(file_path3, index=False)
data4.to_csv(file_path4, index=False)




'''

data=pd.read_csv('E:\Option_data\上证50ETF\上证50ETF看跌期权(1).csv')
missing_values = data.isnull().any()
print(missing_values)
'''