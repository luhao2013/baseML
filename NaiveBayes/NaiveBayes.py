import numpy as np


class NaiveBayes(object):
    """
    朴素贝叶斯实现，生成模型, 实现了拉普拉斯平滑
    只能应用于分类特征
    鉴于朴素贝叶斯没有什么优化的参数
    所以训练后只需要保存先验概率和条件概率就可以
    所以将第三步计算联合概率放到预测未知样本时做
    """
    def __init__(self, lambd=1):
        # 先验概率
        self.label_frequency = {}
        # 条件概率
        self.value_label_frequency = {}
        self.lambd = lambd

    def fit(self, X, y):
        # 1.求先验概率P(y)
        labels = list(set(y))
        for label in labels:
            self.label_frequency[label] = labels.count(label) / float(len(labels))
        # 2.求条件概率P(X|y)
        for label in labels:
            label_index = set(np.where(y==label))
            for i in range(X.shape[1]):
                feature = X[:, i]  # 第i个特征
                feature_values = list(set(feature))  # 第i个特征的特征值集合
                for value in feature_values:
                    value_index = set(np.where(feature == value))
                    # 每类特征值在指定label值下的频数
                    value_label_count = len(value_index & label_index) + self.lambd
                    # 条件概率,每类特征值在指定label值下的频率
                    self.value_label_frequency[str(i)+"'"+str(value)+"|"+str(label)] = value_label_count / \
                                                                                       float(len(label_index)) + self.lambd *X.shape[1]

    def predict(self, X):
        # 3. 求联合概率P(X, y)
        from copy import deepcopy
        predict_value = deepcopy(self.label_frequency)
        for label in predict_value:  # 计算每个类别的联合概率
            for i, value in enumerate(X):
                predict_value[label] *= self.value_label_frequency[str(i)+"'"+str(value)+"|"+str(label)]]
                predict_label = max(predict_value, key=predict_value.get)  # 概率最大值对应的类别
                return predict_label

if __name__ == '__main__':
    model = NaiveBayes()


