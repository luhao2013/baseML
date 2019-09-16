"""
因子分解机回归简单实现
未来优化方向：向量化
"""
import numpy as np


class FM(object):
    def __init__(self, alpha=0.01, max_iter=10, factor_dim=7):
        self.alpha = alpha
        self.max_iter = max_iter
        self.factor_dim = factor_dim

    def fit(self, X, y):
        self.w_ = np.zeros((X.shape[1]+1, 1))
        # 交叉项参数
        self.v = np.random.normal(0, 0.1, size=(X.shape[1], self.factor_dim))
        self.cost_ = []

        for _ in range(self.max_iter):
            error = 0
            # 每次迭代遍历一遍数据集
            for xi, target in zip(X, y):  # 获取每条样本和label
                output = self.activation(xi)
                errors = (target - output).sum()
                # 更新常规项
                self.w_[1:] += self.alpha * xi.reshape((xi.shape[0], 1)) * errors
                self.w_[0] += self.alpha * errors
                # 更新交叉项
                for fea_num in range(X.shape[1]):
                    for k in range(self.factor_dim):
                        interaction_df = xi[fea_num] * np.dot(xi, self.v)[k] - self.v[fea_num,k] * xi[fea_num]**2
                        self.v[fea_num, k] = self.v[fea_num, k] + self.alpha * errors * interaction_df

                cost = (errors**2).sum() / 2.0
                self.cost_.append(cost)
        return self

    def predict(self, X):
        """ 计算数据预测label """
        return np.where(self.activation(X) >= 0.0, 1, 0)

    def net_input(self, x):
        """ 计算下一层神经元输入"""
        # 计算向量点乘
        # print(X.shape, self.w_[1:].shape, self.w_[0].shape)
        # 交叉项中和的平方
        square_of_sum = np.dot(x, self.v)**2

        # 交叉项中平方的和
        sum_of_square = np.dot(np.multiply(x, x), np.multiply(self.v, self.v))

        # 交叉项汇总
        interaction = np.sum(square_of_sum - sum_of_square)/2
        return np.dot(x, self.w_[1:]) + self.w_[0] + interaction

    def activation(self, X):
        """ 回归没有激活函数 """
        return self.net_input(X)


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
    # X, y = loadDataSet()
    # X = np.array(X)
    # y = np.reshape(np.array(y), (100, 1))
    X = np.array([[1, 0, 0], [0, 0, 0], [4, 0, 0]])
    y = np.array([[1], [0], [4]])


    # 训练模型
    model = FM(alpha=0.01, max_iter=300, factor_dim=7)
    model.fit(X, y)
    print("模型系数:", model.w_, model.v)

    for xi, target in zip(X, y):
        output = model.activation(xi)
        print(output, target)

"""
输出结果
[1.00291049] [1]
[0.004213] [0]
[3.99900295] [4]
"""