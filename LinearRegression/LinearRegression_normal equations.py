import numpy as np


class LinearRegression(object):
    """
    根据解析解求线性回归
    解析解不需要任何超参数
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        X = self.add_x0(X)
        self.w_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.cost_ = (y - X @ self.w_).sum() / (2 * len(X))
        return self

    def predict(self, X):
        """ 计算数据预测label """
        return X @ self.w_

    def add_x0(self, X):
        return np.insert(X, 0, values=1, axis=1)

def loadDataSet():
    """ 读取数据 """
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    return data[:, :-1], data[:, -1]

if __name__ == '__main__':
    # 吴恩达机器学习数据
    X, y = loadDataSet()
    X = np.array(X)
    y = np.reshape(np.array(y), (len(X), 1))
    print(X.shape, y.shape)

    # 训练模型
    model = LinearRegression()
    model.fit(X, y)
    print("模型系数:", model.w_)
    print("模型误差", model.cost_)
