"""
李宏毅机器学习Spring 2020--Homework1. regression
作业内容：根据前9天的天气情况，预测第10天的PM2.5的值。
Jerry Sun 20201022
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random


months = 12
days = 20
hours = 24
features = 18
pm25 = 9


def print_file():
    f = open('./train.csv', 'r', encoding='big5')
    for i in range(10):
        print(f.readline())
    f.close()


"""
## **load train.csv**
train.csv包含了训练数据，12个月，每个月20天，每天24小时，18项指标的值。
故可以数据维度为：12*20*24行，18列。每行为1小时的天气特征feature
但是文件并不是以此格式存储，而是列是24小时，每18行是一天，所有天再按行排列。
所以文件格式是18*20*12行，24+3列，多出来的3列是标题（日期、测站、测项）
"""
def load_train_csv(csv):
    data = pd.read_csv(csv, encoding='big5')
    data = data.iloc[:,3:]
    # use pandas to set all rainfall value to 0
    data[data == 'NR'] = 0
    # use for iteration to set rainfall values to 0
    # rain_feature = 10
    # for i in range(months * days):
    #     for j in range(hours):
    #         data.iloc[rain_feature,j]=0
    #     rain_feature += features
    # 低版本用data.values，高版本用data.to_numpy()
    data = data.to_numpy()
    return data


"""
## **extract features**
将数据排列成为12\*20\*24行，18列。这样每列是1小时的天气feature，而每24行是1天，每20\*24行是1月，全部的是1年。<br/>
现在的数据是18个feature分布在行上，24个hour分布在列上，把每18个行作为一个单位，放到当前列的最后，这样再转置就得到上述形状的数据。
"""
def extract_features(data):
    raw_data = data
    train_data = np.empty([features, months*days*hours])
    for i in range(months*days):
        train_data[:,i*hours:(i+1)*hours] = raw_data[i*features:(i+1)*features,:]
    # i = 0
    # print(train_data.shape)
    # print(raw_data.shape)
    # print(train_data[:, i * hours:(i + 1) * hours].shape)
    # print(raw_data[i * features:(i + 1) * features, :].shape)
    return train_data


def convert2samples(train_data):
    # 数据9笔+1笔循环打包为x和y，共有数据量减9个样本
    sample_num = months * days * hours - 9
    x = np.empty([sample_num, 9],dtype=np.float)
    y = np.empty(sample_num,dtype=np.float)
    for i in range(months * days * hours - 9):
        sample = train_data[pm25, i:i + 9]
        x[i] = sample
        y[i] = train_data[pm25, i + 9]
    x = x.astype(dtype=np.float)
    y = y.astype(dtype=np.float)
    return x,y


def load_test_csv(csv):
    """
    读取测试数据文件
    :param csv: 文件名
    :return:
    """
    data = pd.read_csv(csv, encoding='utf8')
    data = data.iloc[:,2:]
    # use pandas to set all rainfall value to 0
    data[data == 'NR'] = 0
    # 低版本用data.values，高版本用data.to_numpy()
    data = data.to_numpy().astype(dtype=np.float32)
    # data = data.to_numpy().astype(dtype=np.float32)
    data = data[8:-1:18,:]
    return data


def train_with_sklearn(x,y):
    # print_file()
    # 加载训练数据生成训练样本
    # 创建线性回归模型进行训练
    model = LinearRegression()
    model.fit(x, y)
    score = model.score(x,y)
    print('coef:',model.coef_)
    print('score:', score)
    return model.coef_


def predict(x,w):
    return np.matmul(x,w)


def show_result(x,y,w):
    # 对训练数据中的前100项进行预测，看看效果如何
    test_data = x[0:9]
    # result = model.predict(test_data)
    result = predict(test_data, w)
    for i in range(len(test_data)):
        print('predict:{0:.2f}, actual:{1:.2f}, error:{2:.2f}'.format(result[i],y[i], y[i]-result[i]))
        temp_data = test_data[i]
        temp_data = temp_data.tolist()
        temp_data.append(y[i])
        plt.subplot(3,3,i+1)
        plt.plot(temp_data)
        plt.scatter([9],[result[i]], marker='o', c='r')
        plt.ylim(0,60)
    plt.show()

    # 加载测试数据
    test_data = load_test_csv('test.csv')
    # 从测试数据中随机选取9项进行预测
    for i in range(9):
        # 随机抽取一个测试样本
        test_one = random.choice(test_data)
        # test_one = test_data[i]
        test_one = test_one.reshape(1,-1)
        # 预测
        # result = model.predict(test_one)
        result = predict(test_one,w)
        data = test_one[0].tolist()
        data.append(result[0])
        # 显示预测表格
        plt.subplot(3,3,i+1)
        plt.plot(data)
        # plt.ylim(0, 60)
    plt.show()


def my_train(epochs=200):
    data = load_train_csv('train.csv')
    train_data = extract_features(data)
    x,y = convert2samples(train_data)
    # w = np.random.random(10)
    w = np.random.random(9)
    learning_rate = 1e-4
    total = len(y)
    for i in range(epochs):
        loss = 0
        gradient = 0
        for j in range(total):
            sample = x[j].copy()
            # sample = np.insert(sample,0,1)
            error = y[j] - np.matmul(w,sample)
            loss += error*error
            gradient += -2*error*sample
        loss /= total
        print('epoch {0}, loss {1}'.format(i, loss))
        gradient /= total
        w = w - learning_rate*gradient
    return w


if __name__ == '__main__':
    data = load_train_csv('train.csv')
    train_data = extract_features(data)
    x,y = convert2samples(train_data)
    print(x.shape)
    print(y.shape)
    # w = train_with_sklearn(x,y)
    w = my_train()
    print('train weights:', w)
    show_result(x, y, w)
