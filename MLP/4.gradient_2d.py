import numpy as np
import matplotlib.pyplot as plt


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.001
    grad = np.zeros_like(x)  # 生成和x形状相同的驻足

    for idx in range(x.size):  # size返回数组元素数量
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h  # 只微小的增加某一维的数值
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:  # 返回数组维度
        return _numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    """
    画出点x处的切线
    :param f:函数
    :param x:自变量
    :return: 切线函数
    """
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x  # y是截距
    return lambda t: d*t + y


if __name__ == "__main__":
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()
