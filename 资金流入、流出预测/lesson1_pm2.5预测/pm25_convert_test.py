from pandas import read_csv
from datetime import datetime

#读取数据，并对时间列进行格式转化
dataset = read_csv('./raw.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=lambda x : datetime.strptime(x, "%Y %m %d %H"))
#删除第一列
dataset.drop('No', axis=1, inplace=True)
#重命名列
dataset.columns = ['pllution', 'dew', 'tem', 'pre', 'cbw', 'ws', 'is', 'ir']
#索引改名为date
dataset.index.name = 'date'
#缺失数据用0填充
dataset['pllution'].fillna(0, inplace=True)
#移除前24条数据（1天数据）
dataset = dataset[24:]
#浏览前五条数据
print(dataset.head(5))

dataset.to_csv('./pllution_test.csv')

