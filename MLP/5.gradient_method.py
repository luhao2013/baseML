import matplotlib.pylab as plt
from utils import numerical_gradient_2d
import numpy as np


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    梯度下降法的简单实现
    :param f: 函数
    :param init_x:初始自变量值
    :param lr: 学习率，即步长
    :param step_num: 迭代次数
    :return:(学习到的极值点，每次迭代值列表)
    """
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient_2d(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2


if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0])
    lr = 0.1
    step_num = 20

    x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(x_history[:, 0], x_history[:, 1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
