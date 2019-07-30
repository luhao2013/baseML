def discret(lisan_list, test_val):
    """
    连续数据等间距离散化
    """
    max_val, min_val = max(lisan_list), min(lisan_list)
    val33 = (max_val - min_val) / 3 + min_val
    val66 = (max_val - min_val) / 3 * 2 + min_val
    for i in range(len(lisan_list)):
        if lisan_list[i] < val33:
            lisan_list[i] = 0
        elif lisan_list[i] < val66:
            lisan_list[i] = 1
        else:
            lisan_list[i] = 2
    if test_val < val33:
        test_val = 0
    elif test_val < val66:
        test_val = 1
    else:
        test_val = 2
    return lisan_list, test_val


def load_data():
    train_num = 9
    train_y = list(map(int, '0 0 0 0 1 1 1 1 1'.strip().split()))
    train_X = [[0, 0, 30, 450, 7],
               [1, 1, 5, 500, 3],
               [1, 0, 10, 150, 1],
               [0, 1, 40, 300, 6],
               [1, 0, 20, 100, 10],
               [0, 1, 25, 180, 12],
               [0, 0, 32, 50, 11],
               [1, 0, 23, 120, 9],
               [0, 0, 27, 200, 8]]
    test_X = [list(map(int, '0 0 40 180 8'.strip().split()))]
    for col in range(2, 5):
        lisan_list = [row[col] for row in train_X]
        lisan_list, test_X[0][col] = discret(lisan_list, test_X[0][col])
        for row in range(len(lisan_list)):
            train_X[row][col] = lisan_list[row]
    return train_X, train_y, test_X


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
        self.label_value_frequency = {}
        # 条件概率
        self.feature_value_label_value_frequency = {}
        self.lambd = lambd  # 拉普拉斯平滑

    def fit(self, X, y):
        # 1.求先验概率P(y)
        label_values_set = list(set(y))
        for label_value in label_values_set:
            self.label_value_frequency[label_value] = y.count(label_value) / float(len(y))
        # 2.求条件概率P(X|y)
        for label_value in label_values_set:
            #             label_value_index = set(np.where(y==label_value))
            # 求给定label的行索引值集合
            label_value_index = set([row_index for row_index in range(len(y)) if y[row_index] == label_value])
            for i in range(len(X[0])):
                #                 feature = X[:, i]  # 第i个特征
                feature = [row[i] for row in X]
                feature_values_set = list(set(feature))  # 第i个特征的特征值集合
                for feature_value in feature_values_set:
                    #                     feature_value_index = set(np.where(feature == feature_value))
                    # 第i个特征，给定特征值的行索引值集合
                    feature_value_index = set([index for index in range(len(y)) if feature[index] == feature_value])
                    # 每类特征值在指定label_value值下的频数
                    feature_value_count_given_label_value = self.lambd
                    for every in feature_value_index:
                        if every in label_value_index:
                            feature_value_count_given_label_value += 1
                    #                     feature_value_count_given_label_value = len(feature_value_index & label_value_index) + self.lambd
                    # 条件概率,每类特征值在指定label_value值下的频率
                    self.feature_value_label_value_frequency[str(i) + "'" + str(feature_value) + "|" + str(
                        label_value)] = feature_value_count_given_label_value / \
                                        float(len(label_value_index)) + self.lambd * len(X[0])

    def predict(self, X):
        # 3. 求联合概率P(X, y)
        from copy import deepcopy
        predict_value = deepcopy(self.label_value_frequency)  # 预测值初始值为先验
        X = X[0]
        for label_value in predict_value:  # 计算每个类别的联合概率
            for i, feature_value in enumerate(X):
                predict_value[label_value] *= self.feature_value_label_value_frequency[
                    str(i) + "'" + str(feature_value) + "|" + str(label_value)]
        predict_label = max(predict_value, key=predict_value.get)  # 概率最大值对应的类别
        # print(predict_value[1] / predict_value[0])
        return predict_label


if __name__ == '__main__':
    model = NaiveBayes(0)
    train_X, train_y, test_X = load_data()
    model.fit(train_X, train_y)
    print(model.predict(test_X))


