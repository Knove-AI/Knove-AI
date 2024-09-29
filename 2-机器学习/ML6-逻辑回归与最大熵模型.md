## 逻辑回归与最大熵模型

### 逻辑回归模型

**逻辑回归(logistic regression, LR)**模型是一种处理**二分类**问题的线性模型。逻辑回归模型由logistic分布(logistic distribution)导出。设$X$是连续随机变量，$X$服从logistic分布是指$X$具有下列**分布函数**和**密度函数**：
$$
\begin{aligned}
F(x)&=P(X\leqslant x)=\frac{1}{1+e^{-(x-\mu)/\gamma}}\\
f(x)&=F'(x)=\frac{e^{-(x-\mu)/\gamma}}{\gamma(1+e^{-(x-\mu)/\gamma})^2}
\end{aligned}
$$
式中，$\mu$为未知参数，$\gamma>0$为形状参数。其分布函数$F(x)$以点$(\mu,\frac 1 2)$为中心对称。

二分类的逻辑回归模型是如下的条件概率分布：
$$
\begin{aligned}
P(Y=1|x)&=\frac{\exp(w\cdot x+b)}{1+\exp(w\cdot x+b)}=\frac{1}{1+e^{-(w \cdot x+b)}}\\
P(Y=0|x)&=\frac{1}{1+\exp(w\cdot x+b)}=1-P(Y=1|x)
\end{aligned}
$$
二分类逻辑回归模型比较两个条件概率值$P(Y=1|x)$和$P(Y=0|x)$的大小，将实例$x$分到概率值较大的那一类。将$w$和$x$变为增广形式，可得：
$$
\begin{aligned}
P(Y=1|x)&=\frac{\exp(w\cdot x)}{1+\exp(w\cdot x)}=\frac{1}{1+e^{-(w \cdot x)}}\\
P(Y=0|x)&=\frac{1}{1+\exp(w\cdot x)}=1-P(Y=1|x)
\end{aligned}
$$
将上式变形后得到：
$$
\begin{aligned}
w\cdot x&=\log \frac{p(Y=1|x)}{1-p(Y=1|x)}\\
&=\log \frac{p(Y=1|x)}{p(Y=0|x)}
\end{aligned}
$$
其中，$\log$中的形式称为**几率(odds)**，指一个事件发生与不发生的比值。

逻辑回归采用**交叉熵(cross entropy)**作为损失函数，而**交叉熵损失函数可以直接利用极大似然估计推到得到**。

对于给定的训练集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$，其中$x_i \in \mathbb R^n,y_i \in \{0,1\}$，令$P(Y=1|x)=\pi(x)$，$P(Y=0|x)=1-\pi(x)$，由训练集可以得到似然函数：
$$
L(w)=\prod_{i=1}^{N}[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}
$$

取对数，得到对数似然函数：
$$
LL(w)=\sum_{i=1}^{N}[y_i\log\pi(x_i)+(1-y_i)\log(1-\pi(x_i))]
$$
将上式取负号并除以$N$，便在训练集上的二分类交叉熵风险函数$\mathcal R(w)$。最小化风险函数和最大化似然函数是等价的，在逻辑回归中通常通过梯度下降法实现：
$$
\begin{aligned}
\frac{\partial \mathcal{R}({w})}{\partial {w}} &=-\frac{1}{N} \sum_{i=1}^{N}\left(y_i \frac{\pi(x_i)(1-\pi(x_i))}{\pi(x_i)} {x}_i-(1-y_i) \frac{\pi(x_i)(1-\pi(x_i))}{1-\pi(x_i)} {x}_i\right) \\
&=-\frac{1}{N} \sum_{i=1}^{N}(y_i(1-\pi(x_i)) {x}_i-(1-y_i) \pi(x_i) {x}_i) \\
&=-\frac{1}{N} \sum_{i=1}^{N} {x}_i(y_i-\pi(x_i))
\end{aligned}
$$
从上式可知，逻辑回归的风险函数是关于参数$w$的连续可到的凸函数。因此除了梯度下降法之外，逻辑回归还可以用**高阶的优化方法(比如牛顿法)**来进行优化。

### softmax回归模型

**softmax回归(softmax regression)**，也称多项逻辑回归，是**逻辑回归在多分类问题上的推广**。对于多类问题，类别标签$y \{\in 1,2,\cdots,C\}$可以有$C$个取值。给定一个样本$x$，softmax回归预测$x$术语类别$c$的条件概率为：
$$
\begin{aligned}
p(y=c | {x}) &=\operatorname{softmax}({w}_{c}^{\mathrm{T}} {x}) \\
&=\frac{\exp ({w}_{c}^{\mathrm{T}} {x})}{\sum_{c^{\prime}=1}^{C} \exp ({w}_{c^{\prime}}^{\mathrm{T}} {x})}
\end{aligned}
$$
其中，$w_c$是第$c$类的权重向量。softmax回归的决策函数可以表示为：
$$
\hat{y} =\underset{c=1}{\arg\max}\ p(y=c | {x}) =\underset{c=1}{\arg\max}\ {w}_{c}^{\mathrm{T}} {x}
$$
softmax可以用**向量形式**写为：
$$
\hat{\boldsymbol{y}}=\operatorname{softmax}(W^{\mathrm{T}} {x})=\frac{\exp (W^{\mathrm{T}} {x})}{1^{\mathrm{T}} \exp (W^{\mathrm{T}} {x})}
$$
其中，$W=[w_1,w_2,\cdots,w_C]$是由$C$个类的权重向量组成的矩阵，$\boldsymbol 1$为全1向量，$\hat{ \boldsymbol y} \in \mathbb R^C$为**所有类别的预测条件概率组成的向量**，即第$c$维的值是第$c$类的预测条件概率。

softmax回归也使用交叉熵损失函数来学习最优的参数矩阵$W$。其风险函数为：
$$
\begin{aligned}
\mathcal{R}(W) &=-\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} {y}_{c}^{(n)} \log \hat{{y}}_{c}^{(n)} \\
&=-\frac{1}{N} \sum_{n=1}^{N}({y}^{(n)})^{\mathrm{T}} \log \hat{{y}}^{(n)}
\end{aligned}
$$
可求得风险函数$\mathcal R(W)$关于$W$的梯度为：
$$
\frac{\partial \mathcal{R}(W)}{\partial W}=-\frac{1}{N} \sum_{n=1}^{N} {x}^{(n)}({y}^{(n)}-\hat{{y}}^{(n)})^{\mathrm{T}}
$$
求得梯度后，便可采用梯度下降法对$W$进行迭代更新：
$$
W_{t+1} \leftarrow W_{t}+\alpha\left(\frac{1}{N} \sum_{n=1}^{N} {x}^{(n)}({y}^{(n)}-\hat{{y}}_{W_{t}}^{(n)})^{\mathrm{T}}\right)
$$

### 最大熵模型

#### 最大熵原理

**最大熵模型(maximum entropy model)**由**最大熵原理**推导实现。最大熵原理是概率模型学习的一个准则。最大熵原理认为，学习概率模型时，在所有可能的概率模型(分布)中，熵最大的模型是最好的模型。假设离散随机变量$X$的概率分布式$P(X)$，则其熵为：
$$
H(P)=-\sum_x P(x)\log P(x)
$$

熵满足下列不等式：
$$
0 \leqslant H(P) \leqslant \log |X|
$$
式中，$|X|$是$X$的取值个数，当且仅当$X$为均匀分布时右边的等号成立。也就是说，当$X$服从均匀分布时熵最大。

直观地，最大熵原理认为要选择的概率模型首先必须满足已有的事实，即约束条件。在没有更多信息的情况下，那些不确定的部分都是“等可能的”。最大熵原理通过熵的最大化来表示等可能性。“等可能”不容易操作，而**熵则是一个可优化的数值指标**。

首先通过一个例子来引入最大熵原理。假设随机变量$X$有5个取值$\{A,B,C,D,E\}$，要估计取各个值的概率$P(A),P(B),P(C),P(D),P(E)$。这些概率满足约束条件：$P(A)+P(B)+P(C)+P(D)+P(E)=1$。满足这个约束条件的概率分布有无穷多个，如果没有任何其他信息，仍要对概率分布进行估计，一个办法就是认为这个分布中取各个值的概率是相等的，即：
$$
P(A)=P(B)=P(C)=P(D)=P(E)=\frac{1}{5}
$$
等概率表示了对事实的无知，因为没有更多信息，这种判断是合理的。

有时，能从一些先验知识中得到一些对概率值的约束条件，例如：
$$
P(A)+P(B)=\frac{3}{10}\\
P(A)+P(B)+P(C)+P(D)+P(E)=1
$$
满足这两个约束条件的概率分布仍然有无数多个。在缺少其他信息的情况下，可以认为$A$与$B$是等概率的，$C$，$D$与$E$是等概率的，于是$P(A)=P(B)=\frac{3}{10}$，$P(C)=P(D)=P(E)=\frac{7}{30}$。其他约束条件可以继续按照满足约束条件下求等概率的方法估计概率分布。上述概率模型的学习方法正是遵循了最大熵原理。

#### 最大熵模型的定义

**最大熵原理是统计学习的一般原理**，将最大熵原理应用到分类问题，即得到最大熵模型。假设**分类模型是一个条件概率分布**$P(Y|X)$，$X \in \mathcal X \subseteq \mathbb R^n$表示输入，$Y \in \mathcal Y$表示输出，$\mathcal X$和$\mathcal Y$分别是输入和输出的集合。这个模型表示的是对于给定的输入$X$，以条件概率$P(Y|X)$输出$Y$。

给定一个训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$，学习的目标是用最大熵原理选择最好的分类模型。首先考虑模型应该满足的条件。给定训练数据集，可以确定联合分布$P(X,Y)$的经验分布和边缘分布$P(X)$的经验分布，分别以$\tilde P(X,Y)$和$\tilde P(X)$表示，这里：
$$
\begin{aligned}
&\tilde P(X=x,Y=y)=\frac{\nu(X=x,Y=y)}{N}\\
&\tilde P(X=x)=\frac{\nu(X=x)}{N}
\end{aligned}
$$
其中，$\nu(X=x,Y=y)$表示训练数据中样本$(x,y)$出现的频数，$\nu(X=x)$表示训练数据中输入$x$出现的频数，$N$表示训练样本的容量。

用**特征函数(feature function)**$f(x,y)$描述输入$x$和输出$y$之间的**某一个事实**。其定义为若$x$和$y$满足某一事实，则$f(x,y)$值为1，否则为0。它是一个二值函数。

**特征函数**$f(x,y)$关于**经验分布**$\tilde P(X,Y)$的**期望值**为：
$$
E_{\tilde P}(f)=\sum_{x,y}\tilde P(x,y)f(x,y)
$$
**特征函数**$f(x,y)$关于**模型**$P(Y|X)$与**经验分布**$\tilde P(X)$的**期望值**为：
$$
E_P(f)=\sum_{x,y}\tilde P(x)P(y|x)f(x,y)
$$
如果模型能够获取训练数据中的信息，那么就可以假设这两个期望值相等，即$E_P(f)=E_{\tilde P}(f)$，或：
$$
\sum_{x,y}\tilde P(x)P(y|x)f(x,y)=\sum_{x,y}\tilde P(x,y)f(x,y)
$$
上式即为模型学习的约束条件。假如有$n$个特征函数$f_i(x,y),i=1,2,\cdots,n$，那么就有$n$个约束条件。假设满足所有约束条件的模型集合为：
$$
\mathcal C = \{P\in \mathcal P|E_P(f_i)=E_{\tilde P}(f_i),\ \ i=1,2,\cdots,n\}
$$
定义在条件概率分布$P(Y|X)$上的条件熵为：
$$
H(P)=-\sum_{x,y}\tilde P(x)P(y|x)\log P(y|x)
$$
则**模型集合**$\mathcal C$中条件熵$H(P)$最大的模型称为**最大熵模型**。式中的对数为**自然对数**。


#### 最大熵模型的学习

最大熵模型的学习过程就是求解最大熵模型的过程。最大熵模型的学习可以形式化为约束最优化问题。

对于给定的训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$以及特征函数$f_i(x,y),\ \ i=1,2,\cdots,n$，最大熵模型的学习等价于约束最优化问题：
$$
\begin{aligned}
\max _{P \in \mathcal C}\quad & -\sum_{x, y} \widetilde{P}(x) P(y|x) \log P(y|x)\\
\text {s.t.}\quad & \ E_{p}(f_{i})=E_{\tilde{P}}(f_{i}),\ \ i=1,2,\cdots,n\\
&\sum_{y} P(y|x)=1
\end{aligned}
$$
按照最优化问题的习惯，**将求最大值问题改写为等价的求最小值问题**：
$$
\begin{aligned}
\min _{P \in \mathcal C}\quad & \sum_{x, y} \widetilde{P}(x) P(y|x) \log P(y|x)\\
\text {s.t.}\quad & \ E_{p}(f_{i})-E_{\tilde{P}}(f_{i})=0,\ \ i=1,2,\cdots,n\\
&\sum_{y} P(y|x)=1
\end{aligned}
$$
求解上式约束最优化问题所得出的解就是最大熵模型学习的解。具体地，将约束最优化问题转换为无约束最优化的对偶问题。通过求解对偶问题来求解原始问题。

首先引入拉格朗日乘子$w_0, w_1, w_2, \cdots, w_n$，定义拉格朗日函数$L(P, w)$：
$$
\begin{aligned}
L(P, w) &=-H(P)+w_{0}(1-\sum_{y} P(y | x))+\sum_{i=1}^{n} w_{i}(E_{\tilde{p}}(f_{i})-E_{p}(f_{i})) \\
&=\sum_{x, y} \tilde{P}(x) P(y | x) \log P(y | x)+w_{0}(1-\sum_{y} P(y | x))\\
&+\sum_{i=1}^{n} w_{i}(\sum_{x, y} \tilde{P}(x, y) f(x, y)-\sum_{x, y} \tilde{P}(x) p(y | x) f(x, y))
\end{aligned}
$$
最优化的原始问题是：
$$
\min _{P \in C} \max _{w} L(P, w)
$$
对偶问题是：
$$
\max _{w} \min _{P \in C} L(P, w)
$$
由于拉格朗日函数$L(P, w)$是$P$的凸函数，因此原始问题的解与对偶问题的解是等价的。因此可以通过求解对偶问题来求解原始问题。首先求解对偶问题内部的极小化问题$\min _{P \in C} L(P, w)$。$\min _{P \in C} L(P, w)$是$w$的函数，记作
$$
\Psi(w)=\min _{P \in C} L(P, w)=L\left(P_{w}, w\right)
$$
$\Psi(w)$称为对偶函数。同时，将其解$P_w$记作
$$
P_{w}=\arg \min _{P \in C} L(P, w)=P_{w}(y | x)
$$
具体地，求$L(P, w)$对$P(y|x)$的偏导数：
$$
\begin{aligned}
\frac{\partial L(P, w)}{\partial P(y | x)} &=\sum_{x, y} \widetilde{P}(x)(\log P(y | x)+1)-\sum_{y} w_{0}-\sum_{x, y}(\widetilde{P}(x) \sum_{i=1}^{n} w_{i} f_{i}(x, y)) \\
&=\sum_{x, y} \widetilde{P}(x)(\log P(y | x)+1-w_{0}-\sum_{i=1}^{n} w_{i} f_{i}(x, y))
\end{aligned}
$$
令偏导数等于0，解得：
$$
P(y | x) = \frac{1}{\exp \left(1-w_{0}\right)} \sum_{y} \exp (\sum_{i=1}^{n} w_{i} f_{i}(x, y))
$$
由于$\sum_y P(y | x) = 1$，得
$$
\begin{aligned}
P_{w}(y | x) &=\frac{1}{Z_{w}(x)} \exp (\sum_{i=1}^{n} w_{i} f_{i}(x, y)) \\
Z_{w}(x) &=\sum_{y} \exp (\sum_{i=1}^{n} w_{i} f_{i}(x, y))
\end{aligned}
$$
上式表示的模型$P_W = P_w(y|x)$就是最大熵模型。之后，求解对偶问题外部的极大化问题
$$
\max_w \Psi(w)
$$
将其解记为$w^*$，即
$$
w^* = \arg \max_w \Psi(w)
$$
便可以应用最优化算法(梯度下降法、拟牛顿法等)求对偶函数的极大化，得到$w^*$，用来表示$P^* \in \mathcal C$。这里，$P^* = P_{w^*} = P_{w^*}(y | x)$是学习到的最优模型，即最大熵模型。**最大熵的模型可以归结为对偶函数的极大化**。

### 基于numpy的逻辑回归实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report


# 定义sigmoid函数
def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


# 定义参数初始化函数
def initialize_params(dims):
    W = np.zeros((dims, 1))
    b = 0
    return W, b


# 定义对数几率回归模型主体
def logistic(X, y, W, b):
    num_train = X.shape[0]  # 训练样本量
    num_feature = X.shape[1]  # 训练特征数
    a = sigmoid(np.dot(X, W) + b)  # 逻辑回归模型输出
    cost = -1 / num_train * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))  # 交叉熵损失
    dW = np.dot(X.T, (a - y)) / num_train  # 权值梯度
    db = np.sum(a - y) / num_train  # 偏置梯度
    cost = np.squeeze(cost)  # 压缩损失数组维度
    return a, cost, dW, db


# 定义对数几率回归模型训练过程
def logistic_train(X, y, learning_rate, epochs):
    W, b = initialize_params(X.shape[1])  # 初始化模型参数
    cost_list = []  # 初始化损失列表
    for i in range(epochs):  # 迭代训练
        a, cost, dW, db = logistic(X, y, W, b)
        # 参数更新
        W = W - learning_rate * dW
        b = b - learning_rate * db
        # 记录损失
        if i % 100 == 0:
            cost_list.append(cost)
        if i % 100 == 0:
            print('epoch %d cost %f' % (i, cost))
    params = {'W': W, 'b': b}  # 保存参数
    grads = {'dW': dW, 'db': db}  # 保存梯度
    return cost_list, params, grads


# 定义预测函数
def predict(X, params):
    # 模型预测值
    y_prediction = sigmoid(np.dot(X, params['W']) + params['b'])
    # 基于分类阈值对概率预测值进行类别转换
    for i in range(len(y_prediction)):
        if y_prediction[i] > 0.5:
            y_prediction[i] = 1
        else:
            y_prediction[i] = 0

    return y_prediction


# 生成100*2的模拟二分类数据集
X, labels = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2)

# 设置随机数种子
rng = np.random.RandomState(2)
# 对生成的特征数据添加一组均匀分布噪声
X += 2 * rng.uniform(size=X.shape)
# 标签类别数
unique_lables = set(labels)
# 根据标签类别数设置颜色
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
# 绘制模拟数据的散点图
for k, col in zip(unique_lables, colors):
    x_k = X[labels == k]
    plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
             markersize=14)
plt.title('Simulated binary data set')
plt.show()

print(X.shape, labels.shape)
labels = labels.reshape((-1, 1))
data = np.concatenate((X, labels), axis=1)
print(data.shape)
# 训练集与测试集的简单划分
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], labels[:offset]
X_test, y_test = X[offset:], labels[offset:]
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

print('X_train=', X_train.shape)
print('X_test=', X_test.shape)
print('y_train=', y_train.shape)
print('y_test=', y_test.shape)

cost_list, params, grads = logistic_train(X_train, y_train, 0.01, 1000)
y_pred = predict(X_test, params)
print(classification_report(y_test, y_pred))
```

### 使用scikit-learn中的逻辑回归算法完成自定义数据集上的分类任务

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def plot_decision_boundary(model, axis):
    """
    在axis范围内绘制模型model的决策边界
    :param model: classification model which must have 'predict' function
    :param axis: [left, right, down, up]
    """
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


# 使用numpy库构建自定义数据集
np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array((X[:, 0] ** 2 + X[:, 1]) < 1.5, dtype='int')
for _ in range(20):
    y[np.random.randint(200)] = 1

plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

log_reg = LogisticRegression()  # 定义逻辑回归类的对象
log_reg.fit(X_train, y_train)  # 训练

print('classification accuracy of original logistic regression: ', log_reg.score(X_test, y_test))  # 评分函数

# 绘制原始逻辑回归模型的决策边界
plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()


# 使用Pipeline添加多项式特征、归一化后再应用逻辑回归算法
def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])


poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_train, y_train)

print('classification accuracy of polynomial logistic regression: ', poly_log_reg.score(X_test, y_test))

# 绘制添加了多项式特征后的逻辑回归算法的决策边界
plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
```

### 参考资料

- 李航. 统计学习方法. 北京: 清华大学出版社, 2019.
- 邱锡鹏. 神经网络与深度学习. 北京: 机械工业出版社, 2020.
- 鲁伟. 机器学习：公式推导与代码实现. 北京: 人民邮电出版社, 2022.

