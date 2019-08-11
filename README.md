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
