import numpy as np


class LogisticRegression(object):
    def __init__(self, alpha=0.01, max_iter=10):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        self.w_ = np.zeros((X.shape[1]+1, 1))
        self.cost_ = []

        for _ in range(self.max_iter):
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.alpha * np.dot(X.T, errors)
            self.w_[0] += self.alpha * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def predict(self, X):
        """ 计算数据预测label """
        return np.where(self.activation(X) >= 0.0, 1, 0)

    def net_input(self, X):
        """ 计算下一层神经元输入"""
        # 计算向量点乘
        # print(X.shape, self.w_[1:].shape, self.w_[0].shape)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ 逻辑回归的激活函数为sigmoid函数 """
        return 1.0 / (1+np.exp(-1*self.net_input(X)))


def loadDataSet():
    """ 读取机器学习实战数据 """
    dataMat = []
    labelMat = []
    fr = open('TestSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        for _ in range(len(lineArr)):
            if '' in lineArr:
                lineArr.remove('')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat


if __name__ == '__main__':
    # 机器学习实战数据
    X, y = loadDataSet()
    X = np.array(X)
    y = np.reshape(np.array(y), (100, 1))


    # 训练模型
    model = LogisticRegression(alpha=0.01, max_iter=300)
    model.fit(X, y)
    print("模型系数:", model.w_)

    from matplotlib import pyplot as plt
    # 画出正例和反例的散点图
    for (x1, x2), (label,) in zip(X, y):
        if label == 0:
            plt.scatter(x1, x2, c='red', marker='x')
        else:
            plt.scatter(x1, x2, marker='o', facecolors='none', edgecolors='blue')
    plt.xlabel('x$^{1}$')
    plt.ylabel('x$^{2}$')
    # plt.xlim(-0.1,6)
    # plt.ylim(-0.1,6)
    # 画出超平面（在本例中即是一条直线）
    line_x = np.arange(-4,4)
    line_y = line_x * (-model.w_[1] / model.w_[2]) - (model.w_[0]/ model.w_[2])
    plt.plot(line_x, line_y)
    plt.show()
