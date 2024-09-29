## EM算法

### EM算法的引入

概率模型有时既含观测变量，又含隐变量。如果概率模型的变量都是观测变量，那么给定数据，可以直接用极大似然估计法，或贝叶斯估计法估计模型参数。但是，当模型含有隐变量时，就不能简单地使用这些估计方法。**EM算法**就是**含有隐变量的概率模型参数的极大似然估计法**，或**极大后验概率估计法**。我们仅讨论极大似然估计，极大后验概率估计与其类似。

#### 三硬币模型

假设有3枚硬币，分别记作$A, B, C$。这些硬币正面出现的概率分别是$\pi, p, q$。进行如下投掷试验：先掷硬币$A$，根据其结果选出硬币$B$或硬币$C$，正面选硬币$B$，反而选硬币$C$；然后掷选出的硬币，掷硬币的结果，出现正面记作1，出现反面记作0；独立地重复$n$次试验(这里$n=10$)，预测结果如下：1, 1, 0, 1, 0, 0, 1, 0, 1, 1。

假设只能观测到掷硬币的结果，不能观测掷硬币的过程。问如何估计三枚硬币分别出现正面的概率，即三硬币模型的参数。

三硬币模型可以写作：
$$
\begin{aligned}
P(y|\theta) &=\sum_{z} P(y, z|\theta)=\sum_{z} P(z|\theta) P(y|z, \theta) \\
&=\pi p^{y}(1-p)^{1-y}+(1-\pi) q^{y}(1-q)^{1-y}
\end{aligned}
$$
其中，随机变量$y$是观测变量，表示一次试验观测的结果是0或1；随机变量$z$为隐变量，表示**未观测到**的硬币$A$的结果；$\theta=(\pi,p,q)$是模型参数。这一模型是以上数据的生成模型。注意，随机变量$y$的数据可以观测，随机变量$z$的数据**不可观测**。

### EM算法过程

将观测数据表示为$Y=(Y_1,Y_2,\cdots,Y_n)^\text T$，未观测数据表示为$Z=(Z_1,Z_2,\cdots,Z_n)^\text T$，则**观测数据的似然函数**为(下式可以理解为**全概率公式**)：
$$
P(Y|\theta)=\sum_Z P(Z|\theta)P(Y|Z,\theta)
$$
考虑到每次试验是独立的，则整个观测数据$Y$的似然函数为$Y$中**所有独立重复试验的似然的乘积**：
$$
P(Y|\theta)=\prod_{j=1}^{n}[\pi p^{y_j}(1-p)^{1-y_j}+(1-\pi)q^{y_j}(1-q)^{1-y_j}]
$$
考虑求模型参数$\theta=(\pi,p,q)$的极大似然估计，即
$$
\hat \theta=\arg\max_\theta \log P(Y|\theta)
$$
这个问题没有解析解，只有通过迭代的方法求解。EM算法就是可以用于求解这个问题的一种迭代算法，下面给出针对三硬币问题的EM算法：

EM算法首先选取参数的初值，记作$\theta^{(0)}=(\pi^{(0)},p^{(0)},q^{(0)})$，然后通过下面的步骤迭代计算参数的估计值，直至收敛为止。第$i$次迭代参数的估计值为$\theta^{(i)}=(\pi^{(i)},p^{(i)},q^{(i)})$。EM算法第$i+1$次迭代如下：

**E步**：计算在模型参数$\pi^{(i)},p^{(i)},q^{(i)}$下观测数据$y_j$来自掷硬币$B$的概率
$$
\mu^{(i+1)}=\frac{\pi^{(i)}(p^{(i)})^{y_{j}}(1-p^{(i)})^{1-y_{j}}}{\pi^{(i)}(p^{(i)})^{y_{j}}(1-p^{(i)})^{1-y_{j}}+(1-\pi^{(i)})(q^{(i)})^{y_{j}}(1-q^{(i)})^{1-y_{j}}}
$$
**M步**：计算模型参数的新估计值
$$
\begin{aligned}
\pi^{(i+1)}=\frac{1}{n} \sum_{j=1}^{n} \mu_{j}^{(i+1)} \\
p^{(i+1)}=\frac{\sum_{j=1}^{n} \mu_{j}^{(i+1)} y_{j}}{\sum_{j=1}^{n} \mu_{j}^{(i+1)}} \\
q^{(i+1)}=\frac{\sum_{j=1}^{n}(1-\mu_{j}^{(i+1)}) y_{j}}{\sum_{j=1}^{n}(1-\mu_{j}^{(i+1)})}
\end{aligned}
$$
可以算得，如果取初值$\pi^{(0)}=0.4,p^{(0)}=0.6,q^{(0)}=0.7$，那么得到的模型参数的极大似然估计中，三者的值分别为0.4064, 0.5368和0.6432。**EM算法与初值的选择有关**，选择不同的初值可能得到不同的参数估计值。

一般用$Y$表示观测随机变量的数据，$Z$表示隐随机变量的数据。$Y$和$Z$连在一起称为**完全数据(complete data)**，观测数据$Y$又称为**不完全数据(incomplete data)**。假设给定观测数据$Y$，其概率分布是$P(Y|\theta)$，其中$\theta$是需要估计的模型参数，那么不完全数据$Y$的似然函数是$P(Y|\theta)$，对数似然函数$L(\theta)=\log P(Y|\theta)$；假设$Y$和$Z$的联合概率分布是$P(Y,Z|\theta)$，那么完全数据的对数似然函数是$\log P(Y,Z|\theta)$。

EM算法通过迭代求$L(\theta)=\log P(Y|\theta)$的极大似然估计。每次迭代包含两步：E步，求期望；M步，求极大化。以下为**EM算法的描述**：

输入：观测变量数据$Y$，隐变量数据$Z$，联合分布$P(Y,Z|\theta)$，条件分布$P(Z|Y,\theta)$；

输出：模型参数$\theta$。

(1) 选择参数的初值$\theta^{(0)}$，开始迭代；

(2) E步：记$\theta^{(i)}$为第$i$词迭代参数$\theta$的估计值，在第$i+1$次迭代的E步，计算
$$
\begin{aligned}
Q(\theta,\theta^{(i)})&=E_Z[\log P(Y,Z|\theta)|Y,\theta^{(i)}]\\
&=\sum_Z \log P(Y,Z|\theta)P(Z|Y,\theta^{(i)})
\end{aligned}
$$
这里，$P(Z|Y,\theta^{(i)})$是在给定观测数据$Y$和当前的参数估计$\theta^{(i)}$下隐变量数据$Z$的条件概率分布；

(3) M步：求使$Q(\theta,\theta^{(i)})$极大化的$\theta$，确定第$i+1$次迭代的参数的估计值$\theta^{(i+1)}$
$$
\theta^{(i+1)}=\arg\max_\theta Q(\theta,\theta^{(i)})
$$
(4) 重复(2)和(3)，直到收敛。

其中，$Q(\theta,\theta^{(i)})$为EM算法的核心，称为$Q$函数。完全数据的对数似然函数$\log P(Y,Z|\theta)$关于在给定观测数据$Y$和当前参数$\theta^{(i)}$下对为观测数据$Z$的条件概率分布$P(Z|Y,\theta^{(i)})$的期望称为Q函数，即
$$
Q(\theta,\theta^{(i)})=E_Z[\log P(Y,Z|\theta)|Y,\theta^{(i)}]
$$

下面关于EM算法作几点说明：

步骤(1) 参数的初值可以任意选择，但需注意EM算法对初值是敏感的。

步骤(2) E步求$Q$函数时，$Q$函数式中$Z$是未观测数据，$Y$是观测数据。注意，$Q(\theta,\theta^{(i)})$的第一个变元表示**要极大化的参数**，第二个变元表示**参数的当前估计值**。每次迭代实际在求$Q$函数及其极大。

步骤(3) M步求$Q$函数的极大化，得到$\theta^{(i+1)}$，完成一次迭代。**每次迭代使似然函数增大或达到局部极值**。

步骤(4) 给出停止迭代的条件，一般是对较小的正数$\varepsilon_1,\varepsilon_2$，若满足
$$
\|\theta^{(i+1)}-\theta^{(i)}\|<\varepsilon_1
$$
或
$$
\|Q(\theta^{(i+1)},\theta^{(i)})-Q(\theta^{(i)},\theta^{(i)})\|<\varepsilon_2
$$
则停止迭代。

### EM算法的Python实现

```python
import numpy as np
import random
import math
import time


def loadData(mu0, sigma0, mu1, sigma1, alpha0, alpha1):
    """
    初始化数据集
    这里通过服从高斯分布的随机函数来伪造数据集
    :param mu0: 高斯0的均值
    :param sigma0: 高斯0的方差
    :param mu1: 高斯1的均值
    :param sigma1: 高斯1的方差
    :param alpha0: 高斯0的系数
    :param alpha1: 高斯1的系数
    :return: 混合了两个高斯分布的数据
    """
    length = 1000  # 定义数据集长度为1000

    # 初始化第一个高斯分布，生成数据，数据长度为length * alpha系数，以此来满足alpha的作用
    data0 = np.random.normal(mu0, sigma0, int(length * alpha0))
    # 第二个高斯分布的数据
    data1 = np.random.normal(mu1, sigma1, int(length * alpha1))

    # 初始化总数据集，两个高斯分布的数据混合后会放在该数据集中返回
    dataSet = []
    dataSet.extend(data0)  # 将第一个数据集的内容添加进去
    dataSet.extend(data1)  # 添加第二个数据集的数据
    random.shuffle(dataSet)  # 对总的数据集进行打乱
    return dataSet  # 返回数据集


def calcGauss(dataSetArr, mu, sigmod):
    """
    根据高斯密度函数计算值
    注：在公式中y是一个实数，但是在EM算法中，需要对每个j都求一次yjk，在本实例
    中有1000个可观测数据，因此需要计算1000次。考虑到在E步时进行1000次高斯计
    算，程序上比较不简洁，因此这里的y是向量，在numpy的exp中如果exp内部值为向
    量，则对向量中每个值进行exp，输出仍是向量的形式。所以使用向量的形式1次计算
    即可将所有计算结果得出，程序上较为简洁。
    :param dataSetArr: 可观测数据集
    :param mu: 均值
    :param sigmod: 方差
    :return: 整个可观测数据集的高斯分布密度（向量形式）
    """
    result = (1 / (math.sqrt(2 * math.pi) * sigmod ** 2)) * np.exp(
        -1 * (dataSetArr - mu) * (dataSetArr - mu) / (2 * sigmod ** 2))
    return result


def E_step(dataSetArr, alpha0, mu0, sigmod0, alpha1, mu1, sigmod1):
    """
    EM算法中的E步
    依据当前模型参数，计算分模型k对观数据y的响应度
    :param dataSetArr: 可观测数据y
    :param alpha0: 高斯模型0的系数
    :param mu0: 高斯模型0的均值
    :param sigmod0: 高斯模型0的方差
    :param alpha1: 高斯模型1的系数
    :param mu1: 高斯模型1的均值
    :param sigmod1: 高斯模型1的方差
    :return: 两个模型各自的响应度
    """
    # 计算y0的响应度
    gamma0 = alpha0 * calcGauss(dataSetArr, mu0, sigmod0)  # 模型0的响应度的分子
    gamma1 = alpha1 * calcGauss(dataSetArr, mu1, sigmod1)  # 模型1响应度的分子
    sum = gamma0 + gamma1  # 两者相加为E步中的分布
    gamma0 = gamma0 / sum  # 各自相除，得到两个模型的响应度
    gamma1 = gamma1 / sum

    return gamma0, gamma1  # 返回两个模型响应度


def M_step(muo, mu1, gamma0, gamma1, dataSetArr):
    mu0_new = np.dot(gamma0, dataSetArr) / np.sum(gamma0)
    mu1_new = np.dot(gamma1, dataSetArr) / np.sum(gamma1)

    sigmod0_new = math.sqrt(np.dot(gamma0, (dataSetArr - muo) ** 2) / np.sum(gamma0))
    sigmod1_new = math.sqrt(np.dot(gamma1, (dataSetArr - mu1) ** 2) / np.sum(gamma1))

    alpha0_new = np.sum(gamma0) / len(gamma0)
    alpha1_new = np.sum(gamma1) / len(gamma1)

    # 将更新的值返回
    return mu0_new, mu1_new, sigmod0_new, sigmod1_new, alpha0_new, alpha1_new


def EM_Train(dataSetList, iter=500):
    """
    根据EM算法进行参数估计
    :param dataSetList:数据集（可观测数据）
    :param iter: 迭代次数
    :return: 估计的参数
    """
    # 将可观测数据y转换为数组形式，主要是为了方便后续运算
    dataSetArr = np.array(dataSetList)

    # 对参数取初值，开始迭代
    alpha0 = 0.5
    mu0 = 0
    sigmod0 = 1
    alpha1 = 0.5
    mu1 = 1
    sigmod1 = 1

    step = 0  # 开始迭代
    while step < iter:
        step += 1  # 每次进入一次迭代后迭代次数加1
        # E步：依据当前模型参数，计算分模型k对观测数据y的响应度
        gamma0, gamma1 = E_step(dataSetArr, alpha0, mu0, sigmod0, alpha1, mu1, sigmod1)
        # M步
        mu0, mu1, sigmod0, sigmod1, alpha0, alpha1 = M_step(mu0, mu1, gamma0, gamma1, dataSetArr)
    return alpha0, mu0, sigmod0, alpha1, mu1, sigmod1  # 迭代结束后将更新后的各参数返回


if __name__ == '__main__':
    start = time.time()

    alpha0 = 0.3  # 系数α
    mu0 = -2  # 均值μ
    sigmod0 = 0.5  # 方差σ

    alpha1 = 0.7  # 系数α
    mu1 = 0.5  # 均值μ
    sigmod1 = 1  # 方差σ

    # 初始化数据集
    dataSetList = loadData(mu0, sigmod0, mu1, sigmod1, alpha0, alpha1)

    # 打印设置的参数
    print('---------------------------')
    print('the Parameters set is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f' % (
        alpha0, alpha1, mu0, mu1, sigmod0, sigmod1
    ))

    # 开始EM算法，进行参数估计
    alpha0, mu0, sigmod0, alpha1, mu1, sigmod1 = EM_Train(dataSetList)

    # 打印参数预测结果
    print('----------------------------')
    print('the Parameters predict is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f' % (
        alpha0, alpha1, mu0, mu1, sigmod0, sigmod1
    ))

    print('----------------------------')
    print('time span:', time.time() - start)
```

### 参考资料

- 李航. 统计学习方法. 北京: 清华大学出版社, 2019.

- EM算法的实现：https://www.cnblogs.com/chenxiangzhen/p/10435969.html

