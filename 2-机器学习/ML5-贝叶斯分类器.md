## 机器学习5 贝叶斯分类器

### 朴素贝叶斯的学习与分类

**朴素贝叶斯(naive Bayse)**算法是基于**贝叶斯定理**与**特征条件独立假设**的分类方法。设输入空间$\mathcal X \subseteq \mathbb R^n$为$n$维向量的集合，输出空间为类标记集合$\mathcal Y={c_1,c_2,\cdots,c_K}$。输入为特征向量$x \in \mathcal X$，输出为类标记$y \in \mathcal Y$。$P(X,Y)$是输入空间和输出空间上的随机变量$X$和$Y$的联合概率分布，训练数据集(含$N$个数据)由$P(X,Y)$独立同分布产生。朴素贝叶斯在数据集上学习**联合概率分布**$P(X,Y)$。具体地，先学习以下先验概率分布及条件概率分布：

(1) **先验概率分布**：$P(Y=c_k),k=1,2,\cdots,K$。

(2) **条件概率分布**：$P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},\cdots,X^{(n)}=x^{(n)}|Y=c_k),k=1,2,\cdots,K$。

朴素贝叶斯对条件概率分布作了**条件独立性假设**：
$$
P(X=x | Y=c_{k})=P(X^{(1)}=x^{(1)}, \ldots, X^{(n)}=x^{(n)} | Y=c_{k})=\prod_{j=1}^{n} P(X^{(j)}=x^{(j)} | Y=c_{k})
$$
朴素贝叶斯算法实际上学习到生成数据的机制，属于**生成模型**。条件独立性假设等于是说用于分类的特征**在类确定的条件下**都是条件独立的。这一假设使朴素贝叶斯算法变得简单，但会牺牲一定的分类准确率。

朴素贝叶斯分类时，对给定的输入$x$，通过学习到的模型计算后验概率，最后将**后验概率最大的类别**作为$x$的类输出(**后验概率最大化**)：
$$
y=f(x)=\arg \max _{c_{k}} P(Y=c_{k}) \prod_{j=1}^{n} P(X^{(j)}=x^{(j)} | Y=c_{k})
$$
朴素贝叶斯将实例分到后验概率最大的类，这**等价于期望风险最小化**。假设使用0-1损失函数：

$$
L(Y, f(X))=\left\{\begin{array}{ll}
1, & Y \neq f(X) \\
0, & Y=f(X)
\end{array}\right.
$$
式中的$f(X)$是分类决策函数。这时，**期望风险函数**是为$R_{\exp}(f)=E[L(Y,f(X))]$ 。

此期望是对联合分布$P(X,Y)$取的。由此取**条件期望**：

$$
R_{\exp }(f)=E_{X} \sum_{k=1}^{K}[L(c_{k}, f(X))] P(c_{k} | X)
$$
上式**条件期望的推导过程**如下：
$$
\begin{aligned}
R_{\exp}(f)&=E[L(Y,f(X))]\\
&=\int_{\mathcal X \times \mathcal Y}L(y,f(x))P(x,y)\text dx \text dy\\
&=\int_{\mathcal X \times \mathcal Y}L(y,f(x))P(y|x)P(x)\text dx \text dy\\
&=\int_{\mathcal X} \left( \int_{\mathcal Y} L(y,f(x))P(y|x)\text dy \right)P(x)\text dx\\
&=E_{X} \sum_{k=1}^{K}[L(c_{k}, f(X))] P(c_{k} | X)
\end{aligned}
$$
为了使期望风险最小化，只需对$X=x$逐个极小化，由此得到：
$$
\begin{aligned}
f(x) &=\arg \min _{y \in Y} \sum_{k=1}^{K} L(c_{k}, y) P(c_{k} | X=x)\\
&=\arg \min _{y \in Y} \sum_{k=1}^{K} P(c_{k} \neq Y | X=x) \\
&=\arg \min _{y \in Y} \sum_{k=1}^{K}(1-P(c_{k}=Y | X=x))\\
&=\arg \max _{y \in Y} \sum_{k=1}^{K} P(c_{k}=Y | X=x)
\end{aligned}
$$
通过以上推导，根据期望风险最小化得到了后验概率最大化准则：
$$
f(x)=\arg \max _{c_{k}} P\left(c_{k} | X=x\right)
$$

### 朴素贝叶斯的参数估计

#### 极大似然估计

在朴素贝叶斯算法中，学习意味着估计$P(Y=c_k)$和$P(X=x|Y=c_k)$。可以应用**极大似然估计**法估计相应的概率。先验概率$P(Y=c_k)$的极大似然估计是第$k$类的样本数除以总样本数：
$$
P(Y=c_{k})=\frac{\sum_{i=1}^{N}I(y_{i}=c_{k})}{N}, k=1,2, \cdots, K
$$
设第$j$个特征$x^{(j)}$**可能取值的集合**为$\{a_{j1},a_{j2},\cdots,a_{jS_j}\}$，条件概率$P(X^{(j)}=a_{jl}|Y=c_k)$的极大似然估计第$k$类中$x^{(j)}$特征的值为$a_{jl}$的样本个数除以第$k$类总样本数：
$$
P(X^{(j)}=a_{j l} | Y=c_{k})=\frac{\sum_{i=1}^{N} I(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k})}{\sum_{i=1}^{N} I(y_{i}=c_{k})}\\
j=1,2,\cdots,n;\quad l=1,2,\cdots,S_j;\quad k=1,2,\cdots,K
$$
式中，$x_i^{(j)}$是第$i$个样本的第$j$个特征，$a_{jl}$是第$j$个特征可能取的第$l$个值，$I$为指示函数。

注意，以上是输入特征为离散值的情况。若输入数据特征的数值是连续值，应进行**离散化等处理**。

#### 贝叶斯估计

用极大似然估计可能会出现所要估计的概率值为0的情况，这会影响到后验概率的计算结果，使分类产生偏差。解决方法是采用贝叶斯估计。具体地，条件概率的贝叶斯估计是：
$$
P_{\lambda}(X^{(j)}=a_{j l} | Y=c_{k})=\frac{\sum_{i=1}^{N} I(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k})+\lambda}{\sum_{i=1}^{N} I(y_{i}=c_{k})+S_{j} \lambda}
$$
式中$\lambda \geqslant 0$。当$\lambda=0$时为极大似然估计。常取$\lambda=1$，这时称为拉普拉斯平滑。

同样，先验概率的贝叶斯估计是：
$$
P_\lambda(Y=c_k)=\frac{\sum_{i=1}^N I(y_i=c_k)+\lambda}{N+K\lambda}
$$

### 高斯朴素贝叶斯

如果要处理的是**连续数据**，一种通常的假设是这些连续数值为**高斯分布**。 例如，假设训练集中有一个连续属性$X_i$。我们首先对数据根据类别分类，然后计算每个类别中$X_i$的均值和方差，即计算$X_i$在某一个类别$y$类内的均值$μ_y$，和$X_i$在$y$类内的方差$σ_y^2$。计算$y$类中$X_i$取值为$x_i$的概率的公式如下：
$$
P(x_{i} | y)=\frac{1}{\sqrt{2 \pi \sigma_{y}^{2}}} \exp \left(-\frac{(x_{i}-\mu_{y})^{2}}{2 \sigma_{y}^{2}}\right)
$$
处理连续数值问题的另一种常用的技术是**离散化连续数值**。通常，当训练样本数量较少或者是精确的分布已知时，通过概率分布的方法是一种更好的选择。**在大量样本的情形下离散化的方法表现更优**，因为大量的样本可以学习到数据的分布。由于朴素贝叶斯是一种典型的用到大量样本的方法(越大计算量的模型可以产生越高的分类精确度)，所以**朴素贝叶斯方法一般都使用离散化方法**，而不是概率分布估计的方法。

### 半朴素贝叶斯分类器

为了方便计算估计条件概率，朴素贝叶斯分类器采用了属性条件独立性假设，但在现实任务中这个假设往往很难成立。半朴素贝叶斯分类器的基本想法是适当考虑一部分属性间的相互依赖信息，从而既不需要进行完全联合概率计算，又不至于彻底忽略了比较强的属性依赖关系。

**独依赖估计(one-dependent estimator, ODE)**是半朴素贝叶斯分类器最常用的一种策略，即假设每个属性在类别之外最多仅依赖于其他一个属性，该属性称为父属性。因此，问题的关键转化为如何确定每个属性的父属性，不同的做法产生不同的独依赖分类器：

(1) **SPODE(super-parent ODE)**：最直接的做法是假设所有属性都依赖于同一个属性“超父”(super-parent)，然后通过交叉验证等方法选择超父。

(2) **TAN(tree augmented naive Bayes)**：在最大带权生成树算法的基础上，通过计算属性之间的条件互信息来计算属性间的相关性。TAN实际上仅保留了强相关属性间的依赖性。

(3) **AODE(averated one-dependent estimator)**：一种基于集成学习机制、更为强大的独依赖分类器。

既然将条件独立性假设放松为独依赖假设可能获得泛化性能的提升，那么，能否通过考虑属性间的**高阶依赖**来进一步提升泛化性能？需注意的是，随着依赖属性数目的增加，在样本有限的条件下，又会陷入估计高阶联合概率的泥沼。根据对属性间依赖的涉及程度，贝叶斯分类器形成了一个“谱”：**朴素贝叶斯分类器**不考虑属性间的依赖性，**贝叶斯网**能表示任意属性间的依赖性，介于两者之间的则是一系列**半朴素贝叶斯分类器**，它们基于各种假设和约束来对属性间的部分依赖性进行建模。

### 基于numpy的朴素贝叶斯实现

```python
import numpy as np
import pandas as pd

# 构造数据集 (来自于李航统计学习方法表4.1)
x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
x2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
df.head()

X = df[['x1', 'x2']]
y = df[['y']]


def nb_fit(X, y):
    classes = y[y.columns[0]].unique()
    class_count = y[y.columns[0]].value_counts()
    class_prior = class_count / len(y)

    prior = dict()
    for col in X.columns:
        for j in classes:
            p_x_y = X[(y == j).values][col].value_counts()
            for i in p_x_y.index:
                prior[(col, i, j)] = p_x_y[i] / class_count[j]
    return classes, class_prior, prior


classes, class_prior, prior = nb_fit(X, y)
print(classes, class_prior, prior)

X_test = {'x1': 2, 'x2': 'S'}
classes, class_prior, prior = nb_fit(X, y)


def predict(X_test):
    res = []
    for c in classes:
        p_y = class_prior[c]
        p_x_y = 1
        for i in X_test.items():
            p_x_y *= prior[tuple(list(i) + [c])]
        res.append(p_y * p_x_y)
    return classes[np.argmax(res)]


print(predict(X_test))
```

### 使用scikit-learn中的多项式朴素贝叶斯算法对新闻文本进行分类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 加载数据，vectorized数据表示已将文本变为向量表示(这里使用tf-idf作为特征)
newsgroups_train = fetch_20newsgroups_vectorized('train')
X_train = newsgroups_train['data']
y_train = newsgroups_train['target']

newsgroups_test = fetch_20newsgroups_vectorized('test')
X_test = newsgroups_test['data']
y_test = newsgroups_test['target']

y_train = np.array(y_train)  # 变为numpy数组
y_test = np.array(y_test)  # 变为numpy数组

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# (11314, 130107) (7532, 130107) (11314,) (7532,)

nb = MultinomialNB()  # 定义多项式朴素贝叶斯分类器
nb.fit(X_train, y_train)

print(accuracy_score(y_test, nb.predict(X_test)))  # 打印分类准确率
print(classification_report(y_test, nb.predict(X_test)))  # 分类报告中包含precision/recall/f1-score
```

### 参考资料

- 李航. 统计学习方法. 北京: 清华大学出版社, 2019.
- 周志华. 机器学习. 北京: 清华大学出版社, 2016.
- 鲁伟. 机器学习：公式推导与代码实现. 北京: 人民邮电出版社, 2022.

