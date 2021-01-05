import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# 序贯模型
from tensorflow.keras.models import Sequential
#全连接层和LSTM
from tensorflow.keras.layers import Dense, LSTM


dataset = pd.read_csv('./pllution_test.csv', index_col = 0)
# print(dataset)
values = dataset.values
# print(values)

#绘制8个原始特征图
# i = 1
# for item in range(8):
# 	plt.subplot(8, 1, i)
# 	plt.plot(values[:, item])
# 	plt.title(dataset.columns[item])
# 	i += 1
# plt.show()

# print(dataset['cbw'].value_counts())
# 2.将分类特征wnd_dir进行标签编码，将数据集中的文本转化成0或1的数值，LabelEncoder 和 OneHotEncoder    
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype('float32')
# print(values)

# 3.统一数据维度（数据规划），可采用正态分布或01规范化
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)
# print(scaled.shape)

"""
 将时间序列数据转换为适用于监督学习的数据  ？？？？？？？？？？？
 
 data: 观察序列
 n_in: 观察数据input(x)的步长（时间长度）， 范围 [1, len(data)], 默认为1
 n_out: output(y)的步长,默认为1
 dropnan: 是否删除NaN行
 
 return: 适用于监督学习的DataFrame
"""
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #预测序列
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    #去掉NaN行
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 4.数据转换，前一时刻的8各特征预测下一时刻8各特征
reframed = series_to_supervised(scaled, 1, 1)
reframed.to_csv('reframed-1.csv')
# print(reframed)

# 5.去掉不需要预测的列，即var2(t)-var8(t).基于var1(t-1)-var8(t-1)前一时刻8个特征预测现在时刻的一个结果var1(t)
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
reframed.to_csv('reframed-2.cvs')
# print(reframed)


# 6.数据集切分，80%训练集，20%测试集
values = reframed.values
#LSTM 不能采用train_test_split()随机切分数据，因为时间序列会变的不连续
#XGBosst 可以使用train_test_split()切分，因为样本是相互独立的
n_train_hours = int(len(values) * 0.8)
#训练集
#:n_train_hours代表前:n_train_hours行， 后面的:代表所有列
train = values[:n_train_hours, :]
#测试集
test = values[n_train_hours:, :]

# :-1 表示从0到数组最后一位， -1代表数组最后一位
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

#有35039个样本，特征数有8个
# train_X.shape

# 7.转换为3D格式 [样本数， 时间步， 特征数]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

'''
Sequential 堆叠模型 通过堆叠多model图层，构建复杂神经网络
更多见：https://blog.csdn.net/mogoweb/article/details/82152174
'''
# 8.设置网络模型 为堆叠模型  
model = Sequential()
#50个神经元，input_shape参数用于指定输入数据的形状。train_X.shape[1]代表时间步， train_X.shape[2]代表特征数
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#全连接到一个结果中
model.add(Dense(1))
#设置网络优化器 optimizer：指定优化器 loss：指定损失函数
model.compile(optimizer='adam', loss='mse')
# 9.模型训练 epochs：批次大小(超参)  epochs:迭代次数(超参) 这两个参数需根据数据规模确定
result = model.fit(train_X, train_y, epochs=10, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 10.模型预测
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)
print(test_predict)

#模型评估
score = model.evaluate(test_X，test_y，batch_size = 64)
print(score)

'''呈现原始数据，训练结果，预测结果'''
def plot_img(source_data_set, train_predict, test_predict):
    #原始数据 蓝色
    plt.plot(source_data_set[:, -1], label='real', c='b')
    #训练数据 绿色
    plt.plot(train_predict[:, -1], label='train_predict', c='g')
    #预测结果 红色
    plt.plot([None for _ in train_predict] + [x for x in test_predict], label='test_predict', c='r')
    plt.legend(loc='best')
    plt.show()
# 11.绘制预测结果与实际结果对比
plot_img(values, train_predict, test_predict)