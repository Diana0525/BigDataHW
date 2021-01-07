import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
import sys
# import matplotlib.pyplot as plt

class Model():
    def __init__(self,**kwargs):
        #模型初始化
        self.model = DTC(criterion='gini', splitter="best", max_depth=15)
    def train(self,X,y):
        ##模型构建
        self.model.fit(X,y.astype('str'))
    def predict(self,X):
        pred_y=self.model.predict(X)
        return pred_y

class preparation():
    def __init__(self):
        self.data_source_file = "E:\pythonWorkSpace\BigDataHomework/f_train.csv"
        self.write_data_file = "E:\pythonWorkSpace\BigDataHomework/f_train_afterclear.csv"
    def clear(self):
        print(sys.argv)
        df = pd.read_csv(sys.argv[1], encoding='gbk')
        cols = df.columns.values
        # 获取所有的数据并转为二维数组的形式
        datas = []
        for i in range(len(df)):
            s = df.iloc[i].values
            datas.append(s)
        datas = np.array(datas)
        # 将数据与表头转化为DataFrame格式
        df_frame = {}
        for i in range(len(cols)):
            df_frame[cols[i]] = datas[:, i]
        df = pd.DataFrame(df_frame)
        # 筛选重复值并删除
        df.duplicated()
        df.drop_duplicates()
        # 删除ID列
        del (df['id'])
        # plt.plot(df['年龄'].value_counts().sort_index())
        # plt.show()
        # 处理缺失值,均用均值填充
        cols = df.columns.values
        for col in cols:
            df[col] = df[col].fillna(df[col].mean())

        # 写入新文件
        df.to_csv(self.write_data_file, encoding='gbk', index=False)

if __name__=='__main__':
    ##读取和处理X，y
    #需要在命令行输入训练文件所在位置
    pre = preparation()
    pre.clear()
    data_source_file = "E:\pythonWorkSpace\BigDataHomework/f_train_afterclear.csv"
    df = pd.read_csv(data_source_file, encoding='gbk')
    cols = list(df.columns.values)
    cols.remove('label')
    X = df[cols]
    y = df[['label']]
    # 划分训练集与测试集
    X_train = X[:700]
    y_train = y[:700]
    X_test = X[700:len(df)]
    y_test = y[700:len(df)]

    model=Model()
    model.train(X_train,y_train)
    pred_y=model.predict(X_test)
    for i in range(len(pred_y)):
        pred_y[i] = int(float(pred_y[i]))

    ##计算损失函数
	##最后需要输出预测的准确率或者均方误差等指标值
    sum = 0
    y_test = np.array(y_test).flatten().astype('int')

    for i in range(len(y_test)):
        sum = sum + (pred_y[i] - y_test[i]) ** 2
    sum = sum / (2*len(pred_y))
    print(sum)
    # 计算准确率
    a = 0 # 预测正确的正样本
    for i in range(len(pred_y)):
        if(pred_y[i] == y_test[i] and pred_y[i] ==  1):
            a = a+1
    b = 0 # 预测的正样本数
    for i in range(len(pred_y)):
        if(pred_y[i] == 1):
            b = b+1
    c = 0 # 总正样本数
    for i in range(len(pred_y)):
        if(y_test[i] == 1):
            c = c+1
    P = a/b
    R = a/c
    F1 = (2*P*R)/(P+R)
    print(F1)

