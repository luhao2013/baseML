import numpy as np


class LinearRegression(object):
    """
    根据批量梯度下降法求线性回归
    """
    def __init__(self, alpha, max_iter):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        X = self.normalize_data(X)  # 标准化数据，消除量纲影响
        X = self.add_x0(X)  # 增加系数w0的数据,就不需要对w区别对待了
        self.w_ = np.zeros((X.shape[1], 1))
        self.cost_ = []

        for i in range(self.max_iter):
            output = self.predict(X)  # 预测值
            errors = output - y  # 误差
            gradient = X.T @ errors  # 梯度值
            self.w_ -= self.alpha * gradient / len(X)
            # self.w_ = self.w_ - X.T @ (X @ self.w_ - y) * self.alpha / len(X)
            cost = (errors**2).sum() / (2.0 * len(X))
            self.cost_.append(cost)
        return self

    def predict(self, X):
        """ 计算数据预测label """
        return X @ self.w_

    def normalize_data(self, X):
        return (X - X.mean()) / X.std()

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
    model = LinearRegression(alpha=0.02, max_iter=2000)
    model.fit(X, y)
    print("模型系数:", model.w_)
    print("模型误差", model.cost_[-1])
    # 两个数据集得到最后的损失值与解析解相等

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    standard = StandardScaler()
    X = standard.fit_transform(X)
    # X = (X - X.mean()) / X.std()
    model = LinearRegression()
    model.fit(X, y)
    print("模型系数:", model.coef_)
    print("模型误差", model.intercept_)
