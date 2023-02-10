# 贝叶斯分类器

在参考资料 [1] 中，有专门讲解贝叶斯定理的章节，在此就不对此定理的具体内容进行阐述，下面仅列出定理的表达式：

## 贝叶斯定理

**定理：** 如果事件 $A_1,A_2,\cdots,A_n$ 互不相容， $B\subset\cup_{j=1}^nA_j$ ，则 $P(B)\gt0$ 时，有：
$$
\displaystyle{P(A_j|B)=\frac{P(A_j)P(B|A_j)}{\sum_{i=1}^nP(A_i)P(B|A_i)}}\tag{1}
$$


其中 $1\le j\le n$ 。

对于分类问题，假设有 $K$ 种类别标签，即 ${\cal{Y}}=\{c_1,c_2,\cdots,c_K\}$ （对应于（1）式中的互不相容的事件 $A_j$ ）。对于样本 $\pmb{x}$ ，要计算 $P(c_j|\pmb{x})$ ，根据（1）式，有：
$$
P(c_j|\pmb{x})=\frac{P(c_j)P(\pmb{x}|c_j)}{P(\pmb{x})}\tag{2}
$$
其中：

- $P(c_j)$ 是先验概率。当训练集中包含充足的独立同分布样本时，可以用各类样本出现的频率估计此概率。

- $P(\pmb{x}|c_j)$ 是样本 $\pmb{x}$ 相对类别 $c_j$ 的条件概率，称为“似然”。

  **注意：**有的资料中认为 $P(\pmb{x}|c_j)$ 可以用频率来估计$^{[2]}$ ，实则不然，参考资料 [3] 中对这个问题的完整说明。假设样本有 $d$ 个特征，并且都是二值类型的数据，那么样本空间所有可能取值为 $2^d$ 个。在现实应用中，这个值往往远大于训练集的样本数。也就是，很多样本取值在训练集中根本没有出现。**“未被观测到”与“出现概率为零”通常是不同的**，所以，不能用频率来估计概率 $P(\pmb{x}|c_j)$ 。

  如果从概率的角度来看，得到的训练集样本都具有随机性，如果要能够用频率估计概率，必须满足样本与总体是同分布的。但是，在样本数不是很充足的时候，就不能满足。所以，对于似然，不能用频率来估计。

- $P(\pmb{x})$ 与类别无关，对于一个训练集而言，它是一个常量。从（1）式中，分母对一个试验而言，是一个常量。所以，（2）式可以转化为：
  $$
  P(c_j|\pmb{x})\propto\!P(c_j)P(\pmb{x}|c_j)\tag{3}
  $$
  由此可以，如果能够得到似然 $P(\pmb{x}|c_j)$ 的值，就可以根据（3）式得到后验概率 $P(c_j|\pmb{x})$ 的值，从而能够判断出样本所属的类别。

如何计算（3）式中的似然 $P(\pmb{x}|c_j)$ ，一种常用方法就是最大似然估计。

## 最大似然估计

在参考资料 [1] 中第6.2.1节，专门讲解了最大似然估计，这里使用其中的结论。

按照如下步骤计算 $P(\pmb{x}|c_j)$ ：

1. 假设样本数据独立同分布，且为某种概率分布，但是不知道此概率分布的参数。
2. 根据训练集样本数据，对概率分布的参数进行估计。假设 $P(\pmb{x}|c_j)$ 的概率分布的参数向量是 $\pmb{\theta}$ 

根据参考资料 [1] 中的结论，可以得到如下似然：
$$
L(\pmb{X}_{c_j}|\pmb{\theta})=\prod_{\pmb{x}\in\pmb{X}_{c_j}}P(\pmb{x}|\pmb{\theta})\tag{4}
$$
其中：$\pmb{X}_{c_j}$ 是数据集中类别为 $c_j$ 的样本集合。

在具体计算的时候，可以对（4）式取对数。例如参考资料 [1] 的358页中给出了对于数据符合正态分布的参数 $\mu$ 和 $\sigma^2$ （总体均值和方差）的估计。

设总体 $X \sim N(\mu,\sigma^2)$ （正态分布），$\mu、\sigma^2$ 是未知参数，$x_1,\cdots,x_n$ 是来自 $X$ 的样本值，求 $\mu、\sigma^2$ 的最大似然估计值。

1. 写出 $X$ 的概率密度函数：$f(x;\mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)$

2. 写出似然函数（4）式：

   $$
   L = \prod_{i=1}^n\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{1}{2\sigma^2}(x_i-\mu)^2\right)=(2\pi)^{-\frac{n}{2}}(\sigma^2)^{-\frac{n}{2}}\exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2\right)
   $$
   

3. 对上式取对数

   $$
   \log L = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2-\frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2
   $$
   

4. 将 $\log L$ 分别对 $\mu$ 和 $\sigma^2$ 求偏导数，并令其为 $0$ （注意，$\sigma^2$ 视作一个整体）

   $$
    \begin{cases}\frac{\partial}{\partial \mu}\log L &= \frac{1}{\sigma^2}(\sum_{i=1}^nx_i - n\mu)=0 \\ \frac{\partial}{\partial\sigma^2}\log L &= -\frac{n}{2\sigma^2}+ \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n(x_i-\mu)^2=0 \end{cases}
   $$
   

5. 解方程组，分别得到 $\mu$ 和 $\sigma^2$ 的极大似然估计

   $$
   \begin{split}\hat\mu &= \frac{1}{n}\sum_{i=1}^nx_i=\overline x \\ \hat\sigma^2 &= \frac{1}{n}\sum_{i=1}^n(x_i-\overline x)^2\end{split}\tag{5}
   $$

在参考资料 [1] 中还以预测足球队比赛胜负概率为例，详细介绍了最大似然估计的应用。请参阅。

## 朴素贝叶斯分类器

如果进一步假设“特征相互独立”，即每个特征独立地对分类结果产生影响。

假设一个样本 $\pmb{x}$ 有 $d$ 个特征，即： $\pmb{x}=[x_1,x_2,\cdots,x_d]$ ，则条件概率为：
$$
\begin{split}
P(\pmb{x}|c_j)&=P(x_1,x_2,\cdots,x_d|c_j)
\\&=\prod_{i=1}^dP(x_i|c_j),~(j=1,\cdots,n)
\end{split}\tag{6}
$$
将（6）式代入到（2）式，则：
$$
P(c_j|\pmb{x})=\frac{P(c_j)P(\pmb{x}|c_j)}{P(\pmb{x})}=\frac{P(c_j)}{P(\pmb{x})}\prod_{i=1}^dP(x_i|c_j)\tag{7}
$$

- 对于（7）式中的先验概率 $P(c_j)$ ，按照之前所讲，可以用该类别样本数量占全体数据集样本数量的比例来估计，即用频率估计概率，用下面的方式表示：
  $$
  P(c_j)=\frac{1}{K}\sum_{i=1}^KI(y_i=c_j),~(j=1,2\cdots,n)\tag{8}
  $$
  其中 $I(\cdot)$ 表示函数：$\displaystyle{I=\begin{cases}&1,(y=c)\\&0,(others)\end{cases}}$ 。

- 对于 $\prod_{i=1}^dP(x_i|c_j)$ ，则是利用（4）式的最大似然估计计算。针对不同的概率分布，分别有不同的计算结果。

### 高斯朴素贝叶斯分类器

即特征的条件概率分布满足高斯分布：
$$
p(x_i|c_j)=\frac{1}{\sqrt{2\pi\sigma^2_j}}\text{exp}\left(-\frac{(x_i-\mu_j)^2}{2\sigma^2_j}\right)\tag{9}
$$

### 伯努利朴素贝叶斯分类器

即特征的条件概率分布满足伯努利分布：
$$
P(x_i|c_j)=px_i+(1-p)(1-x_i),~(其中:p=P(x_i=1|c_j),x_i\in\{0,1\})\tag{10}
$$
对（8）式和（9）式，利用最大似然估计，均可以估计到其中的参数，从而得到条件概率 $P(x_i|c_j)$ ，最大似然估计的方法见参考资料 [1] 。

## 最大后验估计$^{[1]}$

前面用最大似然估计，能够计算出条件概率，在利用（2）式，得到后验概率。这种方法，背后隐藏着一个基本观点，即认为分布的总体参数虽然未知，但是它是客观存在的一个固定值，因此可以通过优化似然函数获得。这就是所谓的**频率主义学派**的观点。

此外，还有另外一种观点，把参数也看成随机变量，它们也有一定的分布。于是就可以假定参数服从某种分布，即所谓先验分布。然后基于观测到的数据，计算参数的后验分布。并且获得的数据越多，后验分布可以得到不断的修正。持这种观点的人，也形成了一个学派，就是贝叶斯统计学。

贝叶斯学派强调“观察者”所掌握的知识（即对被观察对象的认识）。如果“观察者”知识完备，则能准确而唯一的判断事件的结果，不需要概率。

对于先验分布，假设为参数 $\theta_1,\cdots,\theta_k$ ，在已有的认识中，这些参数具有某种规律，设概率密度函数为 $g(\theta_1,\cdots,\theta_k)$ （简写为 $g(\pmb{\theta})$ 。此处以连续型分布为例，如果是离散型，可记作 $p(\theta_1, \cdots, \theta_k)$ ）。

先验分布 $g(\theta_1,\cdots,\theta_k)$ 中的参数也是未知的（或部分未知）——这就是知识不完备。为了能准确判断，还需要结合观测数据得到的知识，也就是似然函数 $f(x_1,\cdots,x_n|\theta_1,\cdots,\theta_k)$  ，简写作 $f(\pmb{x}|\pmb{\theta})$（如果是离散型，则可写作 $p(x_1,\cdots,x_n | \theta_1,\cdots,\theta_k)$ ）。

然后将先验分布和似然函数，根据（1）式的贝叶斯定理，可得：
$$
\displaystyle\!f(\pmb{\theta}|\pmb{x}) = \frac{f(\pmb{x}|\pmb{\theta})g(\pmb{\theta})}{\int_{\pmb{\Theta}}f(\pmb{x}|\boldsymbol{\theta})g(\pmb{\theta})d\pmb{\theta}}  \tag{11}
$$

-  $f(\pmb{\theta}|\pmb{x})$ 就是**后验概率**或**后验分布**——“试验之后”。
-  $\pmb{\Theta}$ 是 $g(\pmb{\theta})$ 的值域，且 $\pmb\theta \in \pmb\Theta$ 。分母 $\int_{\pmb\Theta}f(\pmb{x}|\pmb\theta)g(\pmb\theta)d\pmb\theta = p(\pmb{x})$ ，是观测到的数据的边缘分布，与 $\pmb\theta$ 无关，在此相当于一个常数，故：

$$
f(\pmb\theta|\pmb{x}) \propto f(\pmb{x}|\pmb\theta)g(\pmb\theta)\tag{12}
$$

在（10）式中，似然函数 $f(\pmb{x}|\pmb\theta)$ 的函数形式可以根据观测数据确定（注意，参数 $\pmb\theta$ 未知），

那么先验分布 $g(\pmb\theta)$ 的形式应该如何确定？

在贝叶斯统计学中，如果先验分布 $g(\pmb\theta)$ 和后验分布 $f(\pmb\theta|\pmb{x})$ 为同种类型的分布，称它们为**共轭分布**（conjugate distributions），此时的先验分布称为似然函数 $f(\pmb{x}|\pmb\theta)$ 的**共轭先验**（conjugate prior）。

显然，要对后验分布 $f(\pmb\theta|\pmb{x})$ 求最大值。依据（12）式，进而计算 $f(\pmb{x}|\pmb\theta)g(\pmb\theta)$ 的最大值，最终得到估计量 $\hat{\pmb\theta}$ 。
$$
arg\max_{\theta_1,\cdots, \theta_k} f(\theta_1,\cdots,\theta_k|x_1,\cdots,x_n) \propto arg\max_{\theta_1,\cdots,\theta_k} f(x_1,\cdots,x_n|\theta_1,\cdots,\theta_k)g(\theta_1,\cdots,\theta_k)\tag{13}
$$
对上式右侧去对数：

$$
\begin{split}& arg\max_{\theta_1,\cdots,\theta_k} \log\prod_{i=1}^nf(x_i|\theta_1,\cdots,\theta_k)+\log(g(\theta_1,\cdots,\theta_k))\\ = & arg \max_{\theta_1,\cdots,\theta_k}\sum_{i=1}^n(\log f(x_i|\theta_1,\cdots,\theta_k)) + \log(g(\theta_1,\cdots,\theta_k))\end{split}
$$


这样，通过计算上式的最大值，就得到了参数的估计量 $\hat{\pmb\theta}_{MAP}$ ，这个估计方法称为**最大后验估计**（maximum a posteriori estimation，MAP）。

不难看出， $\displaystyle{arg\max_{\theta_1,\cdots,\theta_k}\sum_{i=1}^n(\log f(x_i|\theta_1,\cdots,\theta_k))}$ 就是最大似然的估计量 $\hat{\pmb\theta}_{MLE}$ 。所以，我们可以说，$\log g(\pmb\theta)$ 就是对 $\hat{\pmb\theta}_{MLE}$ 增加的正则项，此修正来自于我们的主观认识。注意一种特殊情况，如果先验分布式均匀分布，例如 $g(\theta) = 0.8$ ，那么最大后验估计就退化为最大似然估计了。 

下面使用参考资料 [1] 中已经证明的一个结论：

二项分布 $p(x|\theta)=\begin{pmatrix}n\\x\end{pmatrix}\theta^x(1-\theta)^{n-x}$ 的共轭服从 $\text{B}$ 分布（Beta 分布），即：
$
g(\theta)=p(\theta) = \text{B}(\alpha, \beta)= \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}\tag{14}
$
其中 $\Gamma(\cdot)$ 是 Gamma 函数（ $\Gamma(n) = (n-1)!$ ），$\alpha$ 和 $\beta$ 是与样本无关的超参数。

并得到：
$$
p(\theta|x) \propto \text{B}(x+\alpha, n-x+\beta)\tag{15}
$$
即后验分布也是 $\text{B}$ 分布，与先验分布构成了共轭分布。

并且可以求得：
$$
\hat{\theta} = \frac{x+\alpha-1}{n+\alpha+\beta-2}\tag{16}
$$
以上结论见参考资料 [1] 的6.2.3节。

如果，对于 $\theta$ 的先验估计是 $\theta_0$ ，可以令：
$$
\begin{split}
\alpha&=\lambda\theta_0+1
\\\beta&=\lambda(1-\theta_0)+1
\end{split}\tag{17}
$$
注意：（17）式是为了后面的目的而凑出来的一种假设，并引入了变量 $\lambda$ 。

将（17）式代入（16）式，得到：
$$
\hat{\theta}=\frac{x+\lambda\theta_0}{n+\lambda}\tag{18}
$$

这就是所谓的**拉普拉斯平滑**，或曰**拉普拉斯修正** 。

### 多项朴素贝叶斯分类器

即特征的条件概率分布满足多项分布，其参数 $\theta$ 的估计值就是经过拉普拉斯修正之后的值$^{[4]}$：
$$
\hat{\theta}_{y_i}=\frac{N_{y_i}+\alpha}{N_y+\alpha\!n}\tag{19}
$$
其中 $\displaystyle{N_{y_i}=\Sigma_{x\in~\!T}}x_i$ 是测试集类别标签为 $y$ 的样本中，特征 $i$ 出现的次数。$N_y=\Sigma_{i=1}^nN_{y_i}$ 是所有类别标签是 $y$ 的特征数量。

（21）式中的 $\alpha$ ，称为**平滑先验**：

- 若 $\alpha\ge0$ ，考虑了学习样本中不存在的特征，并防止在进一步计算中出现零概率。 
- 若 $\alpha=1$ ，称为拉普拉斯平滑。
- 若 $\alpha\lt0$ ，称为 **Lidstone 平滑**。 

## 朴素贝叶斯实现

使用 scikit-learn 提供的模块实现朴素贝叶斯分类器，网址见参考资料 [4] 。

常见的三种：高斯朴素贝叶斯，伯努利朴素贝叶斯和多项朴素贝叶斯。

### 高斯朴素贝叶斯

1. 加载数据

   ```python
   # 加载数据
   from sklearn import datasets
   wine = datasets.load_wine()
   ```

2. 了解数据

   ```python
   # 数据集特征（13个）
   wine.feature_names
   ```

   ```python
   # 样本的类别标签（3个）
   wine.target_names
   ```

   ```python
   # 数据集（特征）形状
   wine.data.shape
   ```

   ```python
   # 查看前2条样本
   wine.data[:2]
   ```

   ```python
   # 样本标签的值：
   wine.target
   ```

3. 划分数据集

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
                                                       test_size=0.3,
                                                       random_state=20
                                                      )
   ```

4. 训练模型

   ```python
   # 训练模型
   from sklearn.naive_bayes import GaussianNB
   
   gnb = GaussianNB()
   gnb.fit(X_train, y_train)
   ```

5. 简单评估

   ```python
   from sklearn import metrics
   
   # 预测
   y_pred = gnb.predict(X_test)
   metrics.accuracy_score(y_test, y_pred)
   ```

### 多项朴素贝叶斯

适合于离散特征，特别是文本分类。通常，要求特征下的数值是整数，但实际上，小数亦可以，例如 tf-idf 的数值。

案例：对新闻数据进行分类

1. 引入模块并加载数据、划分数据集

   ```python
   from sklearn.naive_bayes import MultinomialNB
   from sklearn.datasets import fetch_20newsgroups
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.model_selection import train_test_split
   
   # 获取数据
   news = fetch_20newsgroups(subset="all")
   
   # 划分数据集
   X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.2, random_state=20)
   ```

2. 特征工程：从文本中计算 tf-idf

   ```python
   transfer = TfidfVectorizer()
   X_train = transfer.fit_transform(X_train)
   X_test = transfer.transform(X_test)
   ```

3. 训练模型

   ```python
   mnb = MultinomialNB()  # 默认 alpha=1.0
   mnb.fit(X_train, y_train)
   ```

4. 评估模型:拟合优度

   ```python
   mnb.score(X_test, y_test)
   ```

5. 观察 $\alpha$ 对预测结果的影响

   ```python
   # alpha的值对模型的影响
   import numpy as np
   alphas = np.logspace(-2, 5, num=200)  # 10^-2 到 10^5
   scores = []
   for alpha in alphas:
       mnb = MultinomialNB(alpha=alpha)
       mnb.fit(X_train, y_train)
       scores.append(mnb.score(X_test, y_test))
   ```

   ```python
   # 绘图
   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(1,1,1)
   
   ax.plot(alphas, scores)
   
   ax.set_xlabel(r"$\alpha$")
   ax.set_ylabel(r"score")
   ax.set_ylim(0, 1.0)
   ax.set_xscale('log')
   ```

### 伯努利朴素贝叶斯

适用于二分类问题。

案例：鉴别垃圾邮件

1. 引入模块，加载数据

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.model_selection import train_test_split
   from sklearn.naive_bayes import BernoulliNB
   
   data = pd.read_csv("./data/spam.csv", encoding='latin-1')
   data = data[['class', 'message']]
   ```

2. 训练模型并评估

   ```python
   # 特征 X，标签 y
   X = np.array(data["message"])
   y = np.array(data["class"])
   
   # 邮件内容向量化
   cv = CountVectorizer()
   X = cv.fit_transform(X)
   
   # 划分数据集
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
   
   # 训练模型
   bnb = BernoulliNB(binarize=0.0)  # 参数说明 1
   bnb.fit(X_train, y_train)
   
   # 模型评估
   print(bnb.score(X_test, y_test))
   ```

   参数说明：

   - `binarize` ：
     - 如果为 `None` ，则假定原始数据已经二值化。
     - 如果是浮点数，则以该数值为临界，特征取值大于此浮点数的作为 1，小于的作为 0 。用这种方式将特征数据二值化。



## 参考资料

[1]. 齐伟. 机器学习数学基础[M]. 北京：电子工业出版社

[2]. 谈继勇. 深度学习500问[M]. 北京:电子工业出版社, 2021:73.

[3]. 周志华. 机器学习[M]. 北京:清华大学出版社, 2016:148-149

[4]. Naive Bayes[EB/OL]. https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes . 2022.09.20