import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


class Model():
    def __init__(self,**kwargs):
        #模型初始化
        # self.model = GradientBoostingRegressor()
        self.model = linear_model.LinearRegression()
        # self.model = DTC(criterion='entropy',splitter="best",max_depth=5)  # 基于信息熵
    def train(self,X,y):
        ##模型构建
        self.model.fit(X,y.astype('str'))
    def predict(self,X):
        pred_y=self.model.predict(X).astype('float')
        return pred_y

class preparation():
    def __init__(self):
        self.data_source_file = "E:\pythonWorkSpace\BigDataHomework/d_train.csv"
        self.write_data_file = "E:\pythonWorkSpace\BigDataHomework/d_train_afterclear.csv"
    def clear(self):
        # 读取csv文件并获取所有的表头
        df = pd.read_csv(self.data_source_file,encoding='gbk')
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
        # 将男/女分类映射为0/1分类
        class_mapping = {'男': 0, '女': 1}
        df['性别'] = df['性别'].map(class_mapping)

        # 将年龄数据离散化
        bins = [18, 25, 35, 45, 55, 65, 75, 100]  # 指定的年龄分界点
        df['年龄'] = pd.cut(df['年龄'], bins, labels=False)
        # 删除ID和体检日期
        del(df['id'])
        del(df['体检日期'])
        # 处理缺失值,除乙肝相关指标用0填充，其余用平均值填充
        cols = df.columns.values
        for col in cols:
            if((col != '乙肝表面抗原')
            and (col != '乙肝表面抗体')
            and (col != '乙肝e抗原')
            and (col != '乙肝e抗体')
            and (col != '乙肝核心抗体')):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna('0')
        # 写入新文件
        df.to_csv(self.write_data_file,encoding='gbk',index= False)

if __name__=='__main__':
    ##读取和处理X，y
    pre = preparation()
    pre.clear()
    data_source_file = "E:\pythonWorkSpace\BigDataHomework/d_train_afterclear.csv"
    df = pd.read_csv(data_source_file, encoding='gbk')
    cols = list(df.columns.values)
    cols.remove('血糖')
    X = df[cols]
    y = df[['血糖']]
    # 划分训练集与测试集
    X_train = X[:5000]
    y_train = y[:5000]
    X_test = X[5000:len(df)]
    y_test = y[5000:len(df)]

    model=Model()
    model.train(X_train,y_train)
    pred_y=model.predict(X_test)
    ##计算损失函数
    sum = 0
    y_test = np.array(y_test).flatten()
    for i in range(len(y_test)):
        sum = sum + (pred_y[i]-y_test[i])**2
    sum = sum/(2*len(pred_y))
    print(sum)

    #绘制算法性能曲线图
    y_test = np.array(list(y_test)) #实际值
    pred_y = np.array(list(pred_y))
    plt.figure(figsize=(10,3))
    plt.plot(y_test, color='green', label='y_test')
    plt.plot(pred_y, color = 'red', label='pred_y')
    plt.legend()
    plt.show()