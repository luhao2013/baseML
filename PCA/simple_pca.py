import numpy as np

def simple_pca(X, k):
    """
    :param X: 样本数据
    :param k: 需要的主成分个数
    :return:  降维后的样本数据
    协方差阵的理解：对于一个特征，最能表现这个特征本质的就是这条特征下取的值
    两个特征是不是相似，就可以用两个特征的内积来表现出来，不线性相关内积为0，
    也可以称为维度与维度之间的关系，每个特征视为空间中的一个基。协方差矩阵
    就是计算特征之间两两相关性的；
    协方差阵对角化理解：协方差阵对角线上元素表现了该特征的重要性，方差越大，
    特征包含信息越多，而协方差阵上的非对角线元素应该越小越好。达到这个目的就要是用对角化，
    对角化后维度之间相互正交，特征值表现了维度的重要性。
    PCA要对数据先中心化的几何理解：因为线性代数的基坐标以及线性变换都要求以原点为基准
    PCA最后的变换理解：将视角1下的数据转化为视角2下的数据
    可参考：https://zhuanlan.zhihu.com/p/21580949
    """
    n_samples, n_features = X.shape
    # 1.样本数据中心化
    X_mean = np.mean(X, axis=0)
    norm_X = X - X_mean
    # 2.计算散度矩阵
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # 3.计算特征值与特征向量
    eigen_values, eigen_vectors = np.linalg.eig(scatter_matrix)
    # 4.根据特征值的绝对大小，将特征值和特征向量降序排列
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(n_features)]
    eigen_pairs.sort(reverse=True)
    # 5.选出前k个主成分
    components = np.transpose(np.array([pair[1] for pair in eigen_pairs[: k]]))
    # 6.得到降维后的数据
    data = np.dot(X, components)

    return data

if __name__ == '__main__':
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    print(simple_pca(X, 1))

    # 用sklearn的PCA, 和simple_pca一样
    from sklearn.decomposition import PCA
    import numpy as np

    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=1)
    pca.fit(X)
    print(pca.transform(X))

"""输出结果
[[ 0.50917706]
 [ 2.40151069]
 [ 3.7751606 ]
 [-1.20075534]
 [-2.05572155]
 [-3.42937146]]
"""
