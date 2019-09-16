# baseML
基础机器学习算法实现
***
### 一、感知机
1. [感知机-原始形式](https://github.com/luhao2013/baseML/blob/master/Perceptron/Perceptron.py)
### 二、逻辑回归
1. [逻辑斯特回归-梯度下降法](https://github.com/luhao2013/baseML/blob/master/LogisticRegression/LogisticRegression.py)
### 三、PCA降维
1. [simple_pca](https://github.com/luhao2013/baseML/blob/master/PCA/simple_pca.py)
### 四、线性回归
1. [根据批量梯度下降法求线性回归](https://github.com/luhao2013/baseML/blob/master/LinearRegression/LinearRegression_bgd.py)
2. [根据解析法求解线性回归](https://github.com/luhao2013/baseML/blob/master/LinearRegression/LinearRegression_normal%20equations.py)
### 五、决策树
1. [ID3](https://github.com/luhao2013/baseML/blob/master/DecisionTree/ID3.py)
### 六、朴素贝叶斯
1. [GaussianNB](https://github.com/luhao2013/baseML/blob/master/NaiveBayes/GaussianNB.py)
2. [朴素贝叶斯](https://github.com/luhao2013/baseML/blob/master/NaiveBayes/NaiveBayes.py)
3. [朴素贝叶斯不使用numpy](https://github.com/luhao2013/baseML/blob/master/NaiveBayes/NBwithoutNumpy.py)
### 七、激活函数
本目录下实现几种神经元激活函数：
##### 1.概览
1. **恒等函数**(identity function)  

    $$a(x) = x​$$

    恒等函数的输入和输出相同，没有什么特定含义，只是为了保证和之前的流程一致。

2. **阶跃函数**(step function)  
    $$a(x) =  \begin{cases}  1, & x >0 \\  0, &otherwise \end{cases}​$$

3. **逻辑斯特函数**(sigmoid function)

    $$a(x) = \dfrac{1}{1+exp(-x)}$$

4. **ReLU函数**

    $$a(x) = \begin{cases} x, &x>0 \\ 1, &x\leq0 \end{cases}​$$

##### 2.线性函数和非线性函数

神经网络激活函数使用线性函数，加深神经网络的层数就没有意义了。

**线性函数的问题**：

+ 不管如何加深层数，总是存在与之等效的 “无隐藏层的神经网络”。
  如 $y(x) = c \times c \times c \times  x $ 与$y=ax(a = c ^3)$等效



##### 3.sigmoid函数和阶跃函数的比较

**不同点**：

1. 平滑性不同；sigmoid是一条平滑的曲线，阶跃函数以0为界，输出发生急剧性变化，sigmoid函数的平滑性对神经网络的学习具有重要意义。
2. 返回值不同；阶跃函数只能返回0或1，sigmoid函数可以返回0-1之间的实数值。

**相同点**：

1. 虽然平滑性有差异，但都是输入小时，输出接近0(为0)，随着输出增大，输出向1靠近(变成1)；也就是说，当输入为不重要的信息时，都输出较小的值，输入信号为重要信息时，都输出较大的值。

2. 输出值都在0到1之间。
### 八、MLP
1. [两层全连接训练mnist](https://github.com/luhao2013/baseML/blob/master/MLP/8.train_neural_net.py) 准确率约97%
### 九、CNN
1. [卷积神经网络](https://github.com/luhao2013/baseML/blob/master/CNN/SimpleConvNet.py)

### 十.梯度下降
#### 1.[初始化方法代码](https://github.com/luhao2013/baseML/blob/master/init_methods/inits.py) 
**1.1 全0初始化/全1初始化**
会使模型相当于是一个线性模型，因为如果将权重初始化为零，那么损失函数对每个 w 的梯度都会是一样的，这样在接下来的迭代中，同一层内所有神经元的梯度相同，梯度更新也相同，所有的权重也都会具有相同的值，这样的神经网络和一个线性模型的效果差不多。将 biases 设为零不会引起多大的麻烦，即使 bias 为 0，每个神经元的值也是不同的。

**1.2 随机初始化**
将权重进行随机初始化，使其服从标准正态分布 np.random.randn(size_l, size_l-1)。在训练深度神经网络时可能会造成两个问题，**梯度消失和梯度爆炸**。

**1.3 均匀分布初始化**
会产生**梯度消失或梯度爆炸**。

**1.4 截断初始化**
tf.initializers.truncated_normal(0, 1)会根据均值为0，方差为1的正态分布产生一个随机数。如果生成的随机数超出了均值+/-2倍标准差的临界值，那么该值会被丢弃，重新产生一个随机数。TF官方文档推荐使用这个方法来初始化神经网络的权重，而不是普通的正态分布生成器tf.initializaers.random_normal。

**1.5 LeCun初始化***
Yann LeCun在1998年的一篇文章上率先提出了一种初始化方法。  

- 首先，要求节点的输出所属的分布必须标准差接近1，因此首先需要输出的方差为1，这可以通过一个归一化操作来解决。  
- 然后，假设某个神经元的输入yi不相关且方差为1，那么该单元的所有权重之和的标准差

**1.6 Glorot均匀分布初始化，又称Xavier均匀初始化**

- 对使用tanh或sigmoid激活的网络，建议使用Xavier初始化
- 若对于一层网络的入出和输出方差不变或相近，这样就可以避免输出趋向于0，从而避免梯度弥散情况。

**1.7 He初始化**

- 对使用ReLU激活的网络，建议使用He初始化
- 初始化基本思想是，当使用ReLU做为激活函数时，Xavier的效果不好，原因在于，当RelU的输入小于0时，其输出为0，相当于该神经元被关闭了，影响了输出的分布模式。
- 因此He初始化，在Xavier的基础上，假设每层网络有一半的神经元被关闭，于是其分布的方差也会变小。经过验证发现当对初始化值缩小一半时效果最好，故He初始化可以认为是Xavier初始/2的结果。
- 实验表明，在较深（例如30层）的卷积神经网络上，使用传统的初始化和Xavier初始化都会出现梯度消失的情况，而He初始化不会
- Kaiming初始化的推导过程和Xavier初始化的推导过程最重要的差别在于：即激活函数的期望和激活函数的导数的期望不再为0。

**初始化值不能太大或太小**
    普通的随机初始化方法有缺陷，不适合复杂网络。主要原因是对于非线性激活函数，其导数通常都有一大部分非常平坦的区域。
    如果初始化得到的权重落在了这个区域里，神经元会饱和，权重很难得到更新。例如，对于常见的激活函数tanh，如果输入x的绝对值大于2，就会落入到饱和区域。

- 对于sigmoid、tanh激活函数，初始化值太大，使得流入激活函数的值过大，造成饱和现象，当反相传播时，会使得梯度极小，导致梯度弥散。（ReLU函数不受此影响）
- 初始化太小时，经过多层网络，输出值变得极小，在反向传播时也会使得梯度极小，导致梯度弥散。（ReLU函数也受此影响！）

### 十一、因子分解机
1. [FM](https://github.com/luhao2013/baseML/blob/master/FM/FM.py) 

### 十二、指标
1. [AUC计算](https://github.com/luhao2013/baseML/blob/master/metric/auc.py)