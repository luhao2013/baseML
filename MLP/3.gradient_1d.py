import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    """
    计算数值中心微分，即导数
    :param f: 函数
    :param x: 求导数的点
    :return: 求得的导数值
    """
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)  # 导数的定义


def function_l(x):
    """
    计算函数值
    :param x:自变量
    :return: 函数值
    """
    return 0.01*x**2 + 0.1*x  # 定义的函数


def tangent_line(f, x):
    """
    画出点x处的切线
    :param f:函数
    :param x:自变量
    :return: 切线函数
    """
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x  # y是截距
    return lambda t: d*t + y


if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)
    y = function_l(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    tf = tangent_line(function_l, 5) 
    y2 = tf(x)

    plt.plot(x, y)   # 画函数曲线
    plt.plot(x, y2)  # 画出切线
    plt.show()