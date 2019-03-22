import numpy as np
import math

def GaussianNB():
    def __init__(self):
        self.label_frequency = {}
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return X_mean = np.mean(X, axis=0)

    # 标准差（方差）
    def stdev(self, X):
        return np.std(X, axis=0)

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        summaries = zip(list(self.mean(train_data)), list(self.stdev(train_data)))
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        # 1.求先验概率P(y)
        labels = list(set(y))
        for label in labels:
            self.label_frequency[label] = labels.count(label) / float(len(labels))
        # 2.求条件概率P(X|y)
        for label in labels:
            label_index = list(np.where(y == label))
            self.model = {label: self.summarize(X[label_index])}
        print('gaussianNB train done!')
        return self

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        from copy import deepcopy
        probabilities = deepcopy(self.label_frequency)
        for label, value in self.model.items():
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))