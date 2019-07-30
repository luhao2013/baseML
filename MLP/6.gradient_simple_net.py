# import sys, os
# sys.path.append(os.pardir)
import numpy as np
from utils import softmax, cross_entropy_error
from utils import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == "__main__":
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    net = simpleNet()

    # # lambda等价于下面这个函数，w应该无意义
    # def f(x, t):
    #     return net.loss(x, t)
    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)

    print(dW)