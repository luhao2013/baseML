import numpy as np


def relu(x):
    return np.maximum(0, x)


if __name__ == '__main__':
    x = np.arange(-5, 5, 0.1)
    y = relu(x)
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()
