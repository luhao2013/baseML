import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int)


if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    y = step_function(x)
    print(y)

    # 画阶跃函数图像
    import matplotlib.pyplot as plt
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.ylim(-0.1, 1.1)
    plt.plot(x, y)
    plt.show()
