# -*- coding: utf-8 -*-

#mnist数据集
from sklearn.datasets import load_digits
#数据预处理标准化
from sklearn import preprocessing
#数据集切分
from sklearn.model_selection import train_test_split
#CART模型
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#评估模型准确率
from sklearn.metrics import accuracy_score
#绘图
from matplotlib import pyplot as plt

#1.加载数据
digits = load_digits()
data = digits.data
target = digits.target

print('shape:', data.shape)
print('图形数据:', digits.images[1])
print('图形含义:', digits.target[1])

#绘图
plt.gray()
plt.title('数字')
plt.imshow(digits.images[1])
plt.show()

#2.数据标准化

transfer = preprocessing.StandardScaler()
data = transfer.fit_transform(data)

#3.拆分数据集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=30)

#4.创建CART决策分类器
#基尼系数最小为准
cart = DecisionTreeClassifier(criterion='gini')

#5.模型训练
cart.fit(x_train, y_train)

#6.使用训练好的模型预测
y_pred = cart.predict(x_test)

#7.准确率评估
acc = accuracy_score(y_test, y_pred)
print('CAST准确率: %0.4lf' % acc)