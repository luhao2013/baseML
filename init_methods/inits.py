"""
实现了一些常见的初始化方法。
如果是用numpy，直接返回numpy数组；
如果使用Tensorflow，用tf.Variable()将numpy数组变为Tensorflow的变量。
"""

import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):  # 应该是keras API中的glorot_uniform初始化方法
    """Glorot & Bengio (AISTATS 2010) init."""
    # 正向传播时，激活值的方差保持不变；反向传播时，关于状态值的梯度的方差保持不变。
    # 上面这句话意思就是无论正向传播还是反向传播，输入x的方差都假设不变，应该是为1
    # 我们只需要对参数方差w进行初始化调整就可以保持方差不变了
    # 而均值是一直不变的
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def he_uniform(shape):
    """He kaiming uniform"""
    init_range = np.sqrt(6.0 / shape[0])
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
