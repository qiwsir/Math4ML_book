# 线性判别分析

线性判别分析（Linear Discriminant Analysis，LDA），一般的机器学习资料$$^{[1]}$$将其笼统地归类为“分类算法”，实则这是一个值得探究的问题。

在参考资料 [1] 中，已经比较详细地讲解了**费雪的线性判别分析**，此处则从贝叶斯定理出发，阐述 LDA 基本原理，推导过程所用数学知识可以参考 [3] 。

## 优势 Odds

- Probability，译为“**概率**”，这是公认的，也是概率论教材中普遍采用的翻译。另外，还有“几率”、“机率”等翻译，皆为此单词的中文译名。
- Odds，翻译为“**优势**”。对此单词，有诸多译名，比如“赔率”、“几率”（周志华教授的《机器学习》一书中用此翻译）、“比值比”、“发生比”等。此处，采信“优势”的翻译（感谢苏州大学唐煜教授指导）。

概率，在统计学中有严格的定义，请参阅《机器学习数学基础》。

以抛硬币为例，正面朝上的概率为：

$$p=\frac{\{H\}}{\{H,T\}}=\frac{1}{2}=0.5$$

优势 Odds 不是概率。

优势定义式：

$$
\text{odds}=\frac{p}{1-p}\tag{1}
$$
其中 $$p$$ 是概率。

对数优势，即计算
$$
\text{log}(\text{odds})=\text{log}\frac{p}{1-p}\tag{2}
$$
 通常是以 $$e$$ 为底的对数。

## 分类问题

### 数据集

假设数据集 $$\mathcal{X}=\{\pmb{x}_1,\cdots,\pmb{x}_n\}$$ ，样本数量是 $$n$$ ，特征数量（维数）是 $$d$$ ，即样本数据 $$\pmb{x}_1,\cdots,\pmb{x}_n$$ 在 $$\mathbb{R}^d$$ 空间。

以 $$C_1,\cdots, C_k$$ 作为样本所属的类别，或者说是样本标签。若 $$\pmb{x}_i$$ 属于第 $$j$$ 类或属于 $$C_j$$ 类，即其标签是 $$C_j$$ ，亦可表示为 $$i\in C_j$$ 。

**问题：**现有给定一个样本 $$\pmb{x}$$ ，判断它属于哪一个类别？

### 贝叶斯定理

根据贝叶斯定理$$^{[4]}$$，有：

$$
P(C_j|\pmb{x})=\frac{p(\pmb{x}|C_j)P(C_j)}{p(\pmb{x})}\tag{3}
$$
其中：

- $$P(C_j)$$ 是类别 $$C_j$$ 出现的概率，称为**先验概率**。

- $$p(x|C_j)$$ 是给定类别 $$C_j$$ ，样本数据 $$\pmb{x}$$ 的概率密度函数，称为**似然函数**（likelihood，“看起来像”）。

- $$p(\pmb{x})$$ 是样本数据 $$\pmb{x}$$ 的概率密度函数，称为**证据**，根据全概率公式，算式为：
  $$
  p(\pmb{x})=p(\pmb{x}|C_1)P(C_1)+\cdots+p(\pmb{x}|C_k)P(C_k)\tag{4}
  $$

- $$P(C_j|\pmb{x})$$ 是给定样本数据 $$\pmb{x}$$ 情况下，该样本属于 $$C_j$$ 的概率，称为**后验概率**。 

对贝叶斯定理的全面理解，参阅《机器学习数学基础》5.2.3节。

由贝叶斯定理，判定：样本数据 $$\pmb{x}$$ 应归属于具有最大后验概率的类别，即：若 $$P(C_l|\pmb{x})=\text{max}_{1\le j\le k}P(C_j|\pmb{x})$$ ，则 $$\pmb{x}$$ 属于类别 $$C_l$$ 。

又因为对任何后验概率 $$P(C_j|\pmb{x})$$ ，（3）式的 $$p(x)$$ 都相同，所以：$$P(C_j|\pmb{x})\propto p(\pmb{x}|C_j)P(C_j)$$ ，即：若 $$P(\pmb{x}|C_l)P(C_l)=\text{max}_{1\le j\le k}P(\pmb{x}|C_j)P(C_j)$$ ，则 $$\pmb{x}$$ 属于类别 $$C_l$$ 。

## S 形函数

考虑二分类问题，即 $$k=2$$ ，则（3）式中的后验概率有两个：$$P(C_1|\pmb{x}),P(C_2|\pmb{x})$$ 。

于是（4）式即为：$$p(\pmb{x})=p(\pmb{x}|C_1)P(C_1)+p(\pmb{x}|C_2)P(C_2)$$ 。

根据（3）式，计算 $$P(C_1|\pmb{x})$$ ：
$$
\begin{split}P(C_1|\pmb{x})&=\frac{p(\pmb{x}|C_1)P(C_1)}{p(\pmb{x})}\\&=\frac{p(\pmb{x}|C_1)P(C_1)}{p(\pmb{x}|C_1)P(C_1)+p(\pmb{x}|C_2)P(C_2)}\\&=\frac{1}{1+\frac{p(\pmb{x}|C_2)P(C_2)}{p(\pmb{x}|C_1)P(C_1)}}\end{split}\tag{5}
$$
（5）式中最后得到的分母：$$\frac{p(\pmb{x}|C_2)P(C_2)}{p(\pmb{x}|C_1)P(C_1)}$$ 。

根据（3）式，可得（注意，下面的（6）式跟上面的（5）式分母相比，互为倒数，分子分母位置倒过来了）：
$$
\frac{P(C_1|\pmb{x})}{P(C_2|\pmb{x})}=\frac{p(\pmb{x}|C_1)P(C_1)}{p(\pmb{x}|C_2)P(C_2)}\tag{6}
$$


即 $$P(C_1|\pmb{x})$$ 相对 $$P(C_2|\pmb{x})$$ 的优势，若取优势对数，得：
$$
a=\text{log}\frac{P(C_1|\pmb{x})}{P(C_2|\pmb{x})}=\text{log}\frac{p(\pmb{x}|C_1)P(C_1)}{p(\pmb{x}|C_2)P(C_2)}\tag{7}
$$
（7）式是对数优势（log odds）函数，也称为 **logit 函数**。

由于对数优势中的对数是以 $$e$$ 为底，由（7）式可得：
$$
\frac{p(\pmb{x}|C_1)P(C_1)}{p(\pmb{x}|C_2)P(C_2)}=e^a=\text{exp}(a)\tag{8}
$$
（8）式的倒数为：
$$
\frac{p(\pmb{x}|C_2)P(C_2)}{p(\pmb{x}|C_1)P(C_1)}=\frac{1}{e^{a}}=e^{-a}=\text{exp}(-a)\tag{9}
$$
将（9）式代入到（5）式，得：
$$
\begin{split}P(C_1|\pmb{x})&=\frac{1}{1+\text{exp}(-a)}=f(a)\\\\即：&f(a)=\frac{1}{1+\text{exp}(-a)}\end{split}\tag{10}
$$
函数 $$f(a)$$ 是 S 形函数（sigmoid function），也称为 **logistic 函数**。它具有如下形式：

- 对称性：$$f(-a)=1-f(a)$$
- 反函数：$$a=\text{log}\left(\frac{f}{1-f}\right)$$ ，即（7）式中的 logit 函数
- 导数：$$\frac{df}{da}=f(1-f)$$ 

在神经网络中，logistic 函数常作为激活函数（参阅《机器学习数学基础》4.4.1节、4.4.4节），且喜欢用符号 $$\sigma$$ 表示，则其导数：$$\sigma'=\sigma(1-\sigma)$$ 。 

## Softmax 函数

如果考虑多分类问题，即 $$k\gt2$$ ，将（4）式代入（3）式，并将分母（4）式用求和符号表示：
$$
P(C_j|\pmb{x})=\frac{p(\pmb{x}|C_j)P(C_j)}{\sum_{l=1}^kp(\pmb{x}|C_l)P(C_l)}\tag{11}
$$
令：
$$
a_j=\text{log}\left(p(\pmb{x}|C_j)P(C_j)\right),\quad(1\le j\le k)\tag{12}
$$
其中对数让然是以 $$e$$ 为底，则：
$$
p(\pmb{x}|C_j)P(C_j)=\text{exp}(a_j)\tag{13}
$$
将（13）式代入到（11）式，则：
$$
P(C_j|\pmb{x})=\frac{\text{exp}(a_j)}{\sum_{l=1}^k\text{exp}(a_l)},\quad(1\le j\le k)\tag{14}
$$
（14）式所得函数即称为 **Softmax 函数**。

## 二分类的线性判别分析

### 构建线性判别分析的分类器

根据贝叶斯定理构建分类器。

线性判别分析：linear discriminant analysis，LDA。

二分类的线性判别分析，即 $$k=2$$

**假设：**（此假设是 LDA 的重要适用条件）

- $$p(\pmb{x}|C_j)$$ 服从正态分布
- 所有类别有相同的协方差矩阵

即：
$$
p(\pmb{x}|C_j)=\frac{1}{(2\pi)^{n/2}|\pmb{\Sigma}|^{1/2}}\text{exp}\left(-\frac{1}{2}(\pmb{x}-\pmb{\mu}_j)^\text{T}\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu}_j)\right)\tag{15}
$$
其中：

- $$\pmb{\mu}_j\in\mathbb{R}^d$$ 是类别 $$C_j$$ 的均值（向量）
- $$\pmb{\Sigma}$$ 是 $$(d\times d)$$ 的协方差矩阵，$$|\pmb{\Sigma}|$$ 表示协方差矩阵对应的行列式
- $$j=1,2$$ ，二分类

由 logit 函数（对数优势函数）（7）式可知，$$a=a(\pmb{x})$$ ，即
$$
\begin{split}a(\pmb{x})&=\text{log}\frac{p(\pmb{x}|C_1)P(C_1)}{p(\pmb{x}|C_2)P(C_2)}\\&=\text{log}p(\pmb{x}|C_1)+\text{log}P(C_1)-\text{log}p(\pmb{x}|C_2)-\text{log}P(C_2) \end{split}\tag{16}
$$
对（15）式取对数，得到：
$$
\begin{split}\text{log}p(\pmb{x}|C_j)&=\text{log}\left[\frac{1}{(2\pi)^{n/2}|\pmb{\Sigma}|^{1/2}}\text{exp}\left(-\frac{1}{2}(\pmb{x}-\pmb{\mu}_j)^\text{T}\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu}_j)\right)\right]\\&=-\text{log}\left((2\pi)^{n/2}|\pmb{\Sigma}|^{1/2}\right)+ \left(-\frac{1}{2}(\pmb{x}-\pmb{\mu}_j)^\text{T}\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu}_j)\right),\quad(j=1,2)\end{split}\tag{17}
$$
将（17）式代入到（16）式中，得到：
$$
\begin{split}a(\pmb{x})
&=&&-\frac{1}{2}(\pmb{x}-\pmb{\mu}_1)^\text{T}\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu}_2)
+\frac{1}{2}(\pmb{x}-\pmb{\mu}_2)^\text{T}\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu}_2)+\text{log}P(C_1)-\text{log}P(C_2)
\\
&=&&-\frac{1}{2}\pmb{x}^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}
+\frac{1}{2}\pmb{x}^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1
+\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}
-\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1
\\
& &&+\frac{1}{2}\pmb{x}^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}
-\frac{1}{2}\pmb{x}^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_2
-\frac{1}{2}\pmb{\mu}_2^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}
+\frac{1}{2}\pmb{\mu}_2^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_2+\text{log}P(C_1)-\text{log}P(C_2)
\\
&=&&\frac{1}{2}\pmb{x}^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{\mu}_1-\pmb{\mu}_2)+\frac{1}{2}(\pmb{\mu}_1^{\text{T}}-\pmb{\mu}_2^{\text{T}})\pmb{\Sigma}^{-1}\pmb{x}
\\
& &&-\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1+\frac{1}{2}\pmb{\mu}_2^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_2+\text{log}P(C_1)-\text{log}P(C_2)
\\
&=&&\frac{1}{2}\pmb{x}^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{\mu}_1-\pmb{\mu}_2)+\frac{1}{2}(\pmb{\mu}_1-\pmb{\mu}_2)^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}
\\
& &&-\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1+\frac{1}{2}\pmb{\mu}_2^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_2+\text{log}P(C_1)-\text{log}P(C_2)
\end{split}
\tag{18}
$$
在（18）式中，二次项 $$\pmb{x}^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}$$ 之和等于 $$0$$ 。

由于 $$\pmb{\Sigma}$$ 是对称矩阵，$$\pmb{\Sigma}^{\text{T}}=\pmb{\Sigma}$$ ，$$(\pmb{\Sigma}^{-1})^{\text{T}}=(\pmb{\Sigma}^{\text{T}})^{-1}=\pmb{\Sigma}^{-1}$$ ，有：

$$\begin{split}\frac{1}{2}\pmb{x}^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{\mu}_1-\pmb{\mu}_2)&=\frac{1}{2}(\pmb{\Sigma}^{-1}\pmb{x})^{\text{T}}(\pmb{\mu}_1-\pmb{\mu}_2)\\&=\frac{1}{2}(\pmb{\Sigma}^{-1}\pmb{x})\cdot(\pmb{\mu}_1-\pmb{\mu}_2)\\&=\frac{1}{2}(\pmb{\mu}_1-\pmb{\mu}_2)\cdot(\pmb{\Sigma}^{-1}\pmb{x}) \\&=\frac{1}{2}(\pmb{\mu}_1-\pmb{\mu}_2)^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}\end{split}$$ 

故，（18）式等于：
$$
\begin{split}a(\pmb{x})&=&&(\pmb{\mu}_1-\pmb{\mu}_2)^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}
\\
& &&-\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1+\frac{1}{2}\pmb{\mu}_2^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_2+\text{log}P(C_1)-\text{log}P(C_2)
\end{split}\tag{19}
$$
令：

- $$\pmb{w}=\pmb{\Sigma}^{-1}(\pmb{\mu}_1-\pmb{\mu}_2)$$
- $$w_0=-\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1+\frac{1}{2}\pmb{\mu}_2^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_2+\text{log}P(C_1)-\text{log}P(C_2)$$

则（19）式可以写成：
$$
a(\pmb{x})=\pmb{w}^{\text{T}}\pmb{x}+w_0\tag{20}
$$
将（20）式代入到（10），得到后验概率 $$P(C_1|\pmb{x})$$ 的重新表述：
$$
P(C_1|\pmb{x})=\frac{1}{1+\text{exp}(-a)}=\frac{1}{1+\text{exp}(-(\pmb{w}^{\text{T}}\pmb{x}+w_0))}\tag{21}
$$
由于是二分类，则 $$P(C_1|\pmb{x})=P(C_2|\pmb{x})=\frac{1}{2}$$ ，根据（21）式，故两个类别的边界（超平面）为：
$$
\pmb{w}^{\text{T}}\pmb{x}+w_0=0\tag{22}
$$
也就是说，如果 $$\pmb{w}^{\text{T}}\pmb{x}\ge-w_0$$ ，则样本 $$\pmb{x}$$ 属于 $$C_1$$ ，否则属于 $$C_2$$ 。

这就是所谓的“线性判别”，因为（22）式是“线性”函数。

所以，问题就转化为如何确定 $$\pmb{w}$$ 和 $$w_0$$ 。由（19）式可知，必须要知道：

- 先验概率：$$P(C_j),j=1,2$$
- 均值向量：$$\pmb{\mu}_j,j=1,2$$
- 协方差矩阵：$$\pmb{\Sigma}$$

### 用最大似然估计求解

设样本数据集 $$\mathcal{X}=\{\pmb{x}_i,y_i\}_{i=1}^n$$ ，其中：

- $$\pmb{x}_i\in\mathbb{R}^n$$ 表示一个样本，共有 $$d$$ 个特征
- $$y_i=1$$ 表示 $$i\in C_1$$ ；$$y_i=0$$ 表示 $$i\in C_2$$

并设先验概率 $$P(C_1)=\pi$$ ，则 $$P(C_2)=1-\pi$$ 。

按照前述假设：$$p(\pmb{x}|C_j)$$ 服从正态分布，用 $$N(\pmb{x}_i|\pmb{\mu}_j,\pmb{\Sigma})$$ 表示正态分布的概率密度函数：

- 如果 $$\pmb{x}_i$$  来自 $$C_1$$ 类，则：
  $$
  p(\pmb{x}_i,C_1)=p(\pmb{x}_i|C_1)P(C_1)=N(\pmb{x}_i|\pmb{\mu}_1,\pmb{\Sigma})\cdot \pi=\pi N(\pmb{x}_i|\pmb{\mu}_1,\pmb{\Sigma})\tag{23}
  $$

- 如果 $$\pmb{x}_i$$ 来自 $$C_2$$ 类，则：
  $$
  p(\pmb{x}_i,C_2)=p(\pmb{x}_i|C_2)P(C_2)=(1-\pi)N(\pmb{x}_i|\pmb{\mu}_2,\pmb{\Sigma})\tag{24}
  $$

写出似然函数：
$$
\begin{split}L(\pi,\pmb{mu}_1,\pmb{\mu}_2,\pmb{\Sigma}|\mathcal{X})&=p(\mathcal{X}|\pi,\pmb{\mu}_1,\pmb{\mu}_2,\pmb{\Sigma})
\\
&=\prod_{i=1}^n\left[p(\pmb{x}_i,C_1)\right]^{y_i}\left[p(\pmb{x}_i,C_2)\right]^{1-y_i}
\\
&=\prod_{i=1}^n\left[\pi N(\pmb{x}_i|\pmb{\mu}_1,\pmb{\Sigma})\right]^{y_i}\left[(1-\pi) N(\pmb{x}_i|\pmb{\mu}_2,\pmb{\Sigma})\right]^{1-y_i}
\end{split}\tag{25}
$$
因为对数是单调递增函数，所以，最大化 $$L$$ 就等价于最大化 $$\text{log}L$$ （通常取以 $$e$$ 为底的对数），于是得：
$$
\text{log}L=\sum_{i=1}^n\left(y_i[\text{log}\pi+\text{log} N(\pmb{x}_i|\pmb{\mu}_1,\pmb{\Sigma})]+(1-y_i)[\text{log}(1-\pi)+\text{log} N(\pmb{x}_i|\pmb{\mu}_2,\pmb{\Sigma})]\right)\tag{26}
$$
接下来**分别将（26）式对 $$\pi,\pmb{\mu}_j,\pmb{\Sigma}$$ 求导，并令其导数为零，即可得到最优解**。

**1. 计算 $$\hat{\pi}$$**
$$
\frac{\partial}{\partial\pi}(\text{log}L)=\sum_{i=1}^n\left(\frac{y_i}{\pi}-\frac{1-y_i}{1-\pi}\right)\tag{27}
$$
在（27）式求导中，使用了自然对数的导数：$$\frac{d\text{ln}x}{dx}=\frac{1}{x}$$ 。

令 $$\frac{\partial\text{log}L}{\partial\pi}=0$$ ，则可得到 $$\pi$$ 的最大似然估计，即：
$$
\begin{split}\sum_{i=1}^n\left(\frac{y_i}{\pi}-\frac{1-y_i}{1-\pi}\right)=0
\\
\frac{1}{\pi}\sum_{i=1}^ny_i-\frac{1}{1-\pi}\sum_{i=1}^n(1-y_i)=0
\\
\frac{1}{\pi}\sum_{i=1}^ny_i-\frac{n}{1-\pi}+\frac{1}{1-\pi}\sum_{i=1}^ny_i=0
\\
\because y_i=1\quad or \quad y_i=0
\\
\therefore\quad 令：n_1=\sum_{i=1}^ny_i，表示类别 C_1 的样本数量
\\同理，n_2=n-n_1，表示类别 C_2 的样本数量
\\
得：\hat\pi=\frac{n_1}{n}=\frac{n_1}{n_1+n_2}
\end{split}\tag{28}
$$
**2. 计算 $$\hat{\pmb{\mu}}_1$$ 和 $$\hat{\pmb{\mu}}_2$$**

首先计算 $$\frac{\partial\text{log}L}{\partial\pmb{\mu}_1}$$ ，由（26）式可知，只有第一个方括号中的项含有 $$\pmb{\mu}_1$$ ，并参考（15）式，故：
$$
\begin{split}
\frac{\partial\text{log}L}{\partial\pmb{\mu}_1}&=\sum_{i=1}^ny_i\frac{\partial}{\partial\pmb{\mu}_1}\left(\text{log}N(\pmb{x}_i|\pmb{\mu}_1,\pmb{\Sigma})\right)
\\
&=\sum_{i\in C_1}\frac{\partial}{\partial\pmb{\mu}_1}\left(\text{log}\left[\frac{1}{(2\pi)^{n/2}|\pmb{\Sigma}|^{1/2}}\text{exp}\left(-\frac{1}{2}(\pmb{x}_i-\pmb{\mu}_1)^\text{T}\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_1)\right)\right]\right)
\\
&=\sum_{i\in C_1}\frac{\partial}{\partial\pmb{\mu}_1}\left(-\frac{1}{2}(\pmb{x}_i-\pmb{\mu}_1)^\text{T}\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_1)-\text{log}((2\pi)^{n/2}|\pmb{\Sigma}|^{1/2})\right)
\\
&=\sum_{i\in C_1}\frac{\partial}{\partial\pmb{\mu}_1}\left(-\frac{1}{2}\pmb{x}_i^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}_i+\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}_i+\frac{1}{2}\pmb{x}_i^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1-\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1-\text{log}((2\pi)^{n/2}|\pmb{\Sigma}|^{1/2})\right)
\\
\because\quad&\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}_i=\pmb{x}_i^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1
\\&上式进一步写成：
\\
\frac{\partial\text{log}L}{\partial\pmb{\mu}_1}&=\sum_{i\in C_1}\frac{\partial}{\partial\pmb{\mu}_1}\left(\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}_i-\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1-\frac{1}{2}\pmb{x}_i^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}_i-\text{log}((2\pi)^{n/2}|\pmb{\Sigma}|^{1/2})\right)
\\
&=\sum_{i\in C_1}\left(\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_1)\right)
\\
&=\pmb{\Sigma}^{-1}\sum_{i\in C_1}(\pmb{x}_i-\pmb{\mu}_1)
\end{split}
\tag{29}
$$
在上面计算导数的时候，使用了矩阵导数的公式（参考《机器学习数学基础》213页）：

- $$\because\frac{\partial\pmb{a}^{\text{T}}\pmb{x}}{\pmb{x}}=\frac{\pmb{x}^{\text{T}}\pmb{a}}{\pmb{x}}=\pmb{a},\quad\therefore\frac{\partial}{\partial\pmb{\mu}_1}(\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{x}_i)=\pmb{\Sigma}^{-1}\pmb{x}_i$$
- $$\because\frac{\partial\pmb{x}^\text{T}\pmb{Ax}}{\partial\pmb{x}}=(\pmb{A}+\pmb{A}^{\text{T}})\pmb{x},\quad\therefore\frac{\partial}{\partial\pmb{\mu}_1}\left(-\frac{1}{2}\pmb{\mu}_1^{\text{T}}\pmb{\Sigma}^{-1}\pmb{\mu}_1\right)=-\frac{1}{2}(\pmb{\Sigma}^{-1}+(\pmb{\Sigma}^{-1})^{\text{T}})\pmb{\mu}_1=-\pmb{\Sigma}^{-1}\pmb{\mu}_1$$

令 $$\frac{\partial{\text{log}L}}{\partial\pmb{\mu}_1}=0$$ ，则得：
$$
\pmb{m}_1=\hat{\pmb{\mu}_1}=\frac{1}{N}_1\sum_{i\in C_1}\pmb{x}_i\tag{30}
$$
同样方法，可以求得：
$$
\pmb{m}_2=\hat{\pmb{\mu}_2}=\frac{1}{N}_2\sum_{i\in C_2}\pmb{x}_i\tag{31}
$$
**3. 计算 $$\pmb{\Sigma}$$**

将（15）式正态分布的具体函数代入到（26）式，并根据对数展开，将含有 $$\pmb{\Sigma}$$ 的项都写出来，其余的项用 $$\beta$$ 表示。

另外，还是用了矩阵迹的性质：$$\text{trace}(\pmb{ABC})=\text{trace}(\pmb{BCA})=\text{trace}(\pmb{CBA})$$
$$
\begin{split}
\text{log}L &= && -\frac{1}{2}\sum_{i=1}^ny_i\text{log}|\pmb{\Sigma}|-\frac{1}{2}\sum_{i=1}^ny_i(\pmb{x}_i-\pmb{\mu}_1)^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_1)
\\
& &&-\frac{1}{2}\sum_{i=1}^n(1-y_i)\text{log}|\pmb{\Sigma}|-\frac{1}{2}\sum_{i=1}^n(1-y_i)(\pmb{x}_i-\pmb{\mu}_2)^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_2)+\beta
\\
&=&&-\frac{1}{2}\sum_{i\in C_1}\text{log}|\pmb{\Sigma}|-\frac{1}{2}\sum_{i\in C_1}\text{trace}\left((\pmb{x}_i-\pmb{\mu}_1)^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_1)\right)
\\
& &&-\frac{1}{2}\sum_{i\in C_2}\text{log}|\pmb{\Sigma}|-\frac{1}{2}\sum_{i\in C_2}\text{trace}\left((\pmb{x}_i-\pmb{\mu}_2)^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_2)\right)+\beta
\\
&=&&-\frac{n}{2}\text{log}|\pmb{\Sigma}|-\frac{1}{2}\sum_{i\in C_1}\text{trace}\left(\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_1)(\pmb{x}_i-\pmb{\mu}_1)^{\text{T}}\right)
\\
& &&-\frac{1}{2}\sum_{i\in C_2}\text{trace}\left(\pmb{\Sigma}^{-1}(\pmb{x}_i-\pmb{\mu}_2)(\pmb{x}_i-\pmb{\mu}_2)^{\text{T}}\right)+\beta
\\
&=&&-\frac{n}{2}\text{log}|\pmb{\Sigma}|-\frac{1}{2}\text{trace}\left(\pmb{\Sigma}^{-1}\sum_{i\in C_1}(\pmb{x}_i-\pmb{\mu}_1)(\pmb{x}_i-\pmb{\mu}_1)^{\text{T}}+\pmb{\Sigma}^{-1}\sum_{i\in C_1}(\pmb{x}_i-\pmb{\mu}_2)(\pmb{x}_i-\pmb{\mu}_2)^{\text{T}}\right)+\beta
\\
&=&&-\frac{n}{2}\text{log}|\pmb{\Sigma}|-\frac{n}{2}\text{trace}(\pmb{\Sigma}^{-1}\pmb{S})+\beta
\end{split}\tag{32}
$$
其中：
$$
\pmb{S}=\frac{n_1}{n}\pmb{S}_1+\frac{n_2}{n}\pmb{S}_2\\\pmb{S}_j=\frac{1}{n_j}\sum_{i\in C_j}(\pmb{x}_i-\pmb{\mu}_j)(\pmb{x}_i-\pmb{\mu}_j)^{\text{T}}, \quad(j=1,2)\tag{33}
$$
根据迹与行列式导数公式：

- $$\frac{\partial\text{log}(det\pmb{X})}{\partial\pmb{X}}=(\pmb{X}^{-1})^{\text{T}}$$
- $$\frac{\partial\text{trace}(\pmb{AX}^{-1}\pmb{B})}{\partial\pmb{X}}=-(\pmb{X}^{-1}\pmb{BAX}^{-1})^{\text{T}}$$

计算（32）式对 $$\pmb{\Sigma}$$ 的导数：
$$
\frac{\partial\text{L}}{\partial\pmb{\Sigma}}=-\frac{n}{2}\pmb{\Sigma}^{-1}+\frac{n}{2}\pmb{\Sigma}^{-1}\pmb{S\Sigma}^{-1}\tag{34}
$$
令（34）式等于零，则得到协方差矩阵的最大似然估计：
$$
\hat{\pmb{\Sigma}}=\pmb{S}\tag{35}
$$
也就是样本（有偏差的）协方差矩阵 $$\pmb{S}_j,(j=1,2)$$ 的加权平均。

## 多分类的线性判别分析

将上述二分类下的公式推广到多分类，即：$$j=1,\cdots,k$$

$$n_j$$ 表示类别 $$C_j$$ 的数据样本数量，$$n=\sum_{j=1}^kn_k$$ ，则：
$$
\begin{split}
\hat{\pi}_j&=\hat{P}(C_j)=\frac{n_j}{n}
\\
\pmb{m}_j&=\hat{\pmb{\mu}}_j=\frac{1}{n_j}\sum_{i\in C_j}\pmb{x}_i
\\
\pmb{S}_j&=\frac{1}{n_j}\sum_{i\in C_j}(\pmb{x}_i-\pmb{m}_j)(\pmb{x}_i-\pmb{m}_j)^{\text{T}}
\\
\pmb{S}&=\hat{\pmb{\Sigma}}=\frac{n_1}{n}\pmb{S}_1+\cdots+\frac{n_k}{n}\pmb{S}_k
\end{split}\tag{36}
$$

## 注意

LDA 要求数据样本服从正态分布，且所有类别有相同协方差矩阵。

但对离群值敏感，若有离群值，则会造成较大的参数估计偏差。  

## 实践案例

在 scikit-learn 中，提供了实现线性判别分析的专门模型：https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

这是一个分类器模型，要求数据必须符合正态分布，并且各类有相同的协方差矩阵。这个模型，也能够实现对数据的降维。

**1. 导入数据**

```python
# 使用鸢尾花数据集
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris(as_frame=True)  # 载入鸢尾花数据集
df = iris.data
df['target'] = iris.target   # 将数字表示的标签列添加到 df
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)  # 增加样本类别名称列
df.head()
```

**2. 划分数据集**

```python
from sklearn.model_selection import train_test_split
X = df[iris.feature_names]   # 得到样本数据 sepal length\sepal width\petal length\petal width (cm)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

**3. 训练模型**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()
lda.fit(X_train, y_train)
```

**4.简单评估**

```python
lda.score(X_test, y_test)
```

**5. 可视化显示结果**

```python
lda = LDA(n_components=1)      # 降维
X_1d = lda.fit(X,y).transform(X)
```

```python
print(X_1d[:5])    # 降维之后
print(X)           # 降维之前
```

```python
df['X'] = X_1d
df.head()
```

```python
import seaborn as sns
sns.scatterplot(data=df, x="X", y="target", hue="species")
```

## 总结

优点：

- 使用类别的先验知识
- 以标签、类别衡量差异性的有监督降维方法，相对于 PCA 的模糊性，其目的更明确，更能反应样本间的差异

缺点：

- 不适合对非高斯分布样本进行降维
- 降维最多到 $$k-1$$ 维
- 在样本分类信息依赖方差而不是均值的时候，降维效果不好
- 有过拟合的可能

## 参考资料

[1]. [费雪的线性判别分析](./fisher-lda.html)

[2]. 谢文睿等. 机器学习公式详解. 北京：人民邮电出版社

[3]. 齐伟. 机器学习数学基础. 北京：电子工业出版社

[4]. [贝叶斯定理](http://math.itdiffer.com/bayes.html)



