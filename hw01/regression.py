"""
李宏毅机器学习Spring 2020--Homework1. regression
作业内容：根据前9天的天气情况，预测第10天的PM2.5的值。
Jerry Sun 20201022
"""
import pandas as pd
import numpy as np

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
    x = np.empty([sample_num, 9])
    y = np.empty([sample_num, 0])
    for i in range(months * days * hours - 9):
        sample = train_data[pm25, i:i + 9]
        x[i] = sample
        y[i] = train_data[pm25, i + 9]
    return x,y


if __name__ == '__main__':
    print_file()