import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))  # 这里用到了numpy的广播


if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    print(sigmoid(x))

    # 画sigmoid图像
    x = np.arange(-5.0, 5.0, 0.1)
    import matplotlib.pyplot as plt
    plt.plot(x, sigmoid(x))
    plt.ylim(-0.1, 1.1)
    plt.show()
