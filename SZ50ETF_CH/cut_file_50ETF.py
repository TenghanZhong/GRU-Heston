import pandas as pd
import os

data=pd.read_csv('D:\python_data\上证50ETF\华夏上证50ETF看跌期权.csv')

data1=data.iloc[0:1035600]
data2=data.iloc[1035600:]

file_path = os.path.join('E:\Option_data\上证50ETF', f"华夏上证50ETF看跌期权(pure1).csv")
file_path2 = os.path.join('E:\Option_data\上证50ETF', f"华夏上证50ETF看跌期权(pure2).csv")
data1.to_csv(file_path, index=False)
data2.to_csv(file_path2, index=False)

data=pd.read_csv('E:\Option_data\上证50ETF\上证50ETF看跌期权(1).csv')
missing_values = data.isnull().any()
print(missing_values)