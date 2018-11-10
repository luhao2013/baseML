import numpy as np


class Perceptron(object):
    """
    实现感知机算法的原始形式

    Parameters参数
    ------------
    alpha学习率:float,Learning rate (between 0.0 and 1.0)
    max_iter迭代次数:int,Passes over the training dataset.

    Attributes属性
    -------------
    w_权重: 1d-array,Weights after fitting.
    errors_分类错误样本数量: list,Numebr of misclassifications in every epoch.
    """
    def __init__(self, alpha=0.01, max_iter=10):
        self.alpha = alpha
        self.max_iter = max_iter

    def net_input(self, X):
        """ 计算下一层神经元输入"""
        # 计算向量点乘
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ 预测类别标记 """
        # 相当于符号函数或者称为阶跃函数
        return np.where(self.net_input(X)>=0.0, 1, -1)

    def fit(self, X, y):
        """Fit training data.先对权重参数初始化，然后对训练集中每一个样本循环，根据感知机算法学习规则对权重进行更新
        Parameters
        ------------
        X: {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_featuers is the number of features.
        y: array-like, shape=[n_smaples]
            Target values.
        Returns
        ----------
        self: object
        """
        # 初始化权重。数据集特征维数+1
        self.w_ = np.zeros(X.shape[1]+1)
        # 用于记录每一次迭代中误分类的样本数
        self.errors_ = []

        for _ in range(self.max_iter):
            error = 0
            # 每次迭代遍历一遍数据集
            for xi, target in zip(X, y):  # 获取每条样本和label
                # 调用了predict()函数,预测正确update值为0
                update = self.alpha * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self


if __name__ == '__main__':
    # 统计学习方法例题数据
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([[1], [1], [-1]])

    # 训练模型
    model = Perceptron(alpha=0.1, max_iter=10)
    model.fit(X, y)
    print("模型系数:", model.w_)

    from matplotlib import pyplot as plt
    # 画出正例和反例的散点图
    for (x1, x2), (label,) in zip(X, y):
        if label == -1:
            plt.scatter(x1, x2, c='red', marker='x')
        else:
            plt.scatter(x1, x2, marker='o', facecolors='none', edgecolors='blue')
    plt.xlabel('x$^{1}$')
    plt.ylabel('x$^{2}$')
    plt.xlim(-0.1,6)
    plt.ylim(-0.1,6)
    # 画出超平面（在本例中即是一条直线）
    line_x = np.arange(0,4)
    line_y = line_x * (-model.w_[1] / model.w_[2]) - (model.w_[0]/ model.w_[2])
    plt.plot(line_x, line_y)
    plt.show()