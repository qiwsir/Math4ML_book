# 不用行列式的特征分析

*打开本页，如果没有显示公式，请刷新页面。*

在传统的线性代数教材中，行列式占据重要地位，其原因可能在于历史发展顺序。历史上，数学家们先研究怎么用行列式解线性方程组，而后才提出了矩阵等概念。

现在，有数学家提出了不同观点，认为在现代的线性代数中，可以抛弃行列式，最典型代表就是参考文献[1]的作者，美国数学家Sheldon Axler，参考文献[1]就是他的这种思想的集中体现。

下面参考文献[2]，按照Sheldon Axler的思想，不用行列式，演示相关定理的证明。

设向量空间 $$\mathbb{V}$$ 中的线性变换 $$\pmb{A}$$ ，并 $$\pmb{X} \subseteq \mathbb{V}$$ 是一个子空间，$$\pmb{A(X)}$$ 表示子空间向量经 $$\pmb{A}$$ 映射后的像所成的集合，即 $$\pmb{A(X)}=\{\pmb{Ax}|\pmb{x}\in\pmb{X}\}$$ 。

如果 $$\pmb{A(X)}\subseteq\pmb{X}$$ ，则称 $$\pmb{X}$$ 为线性变换 $$\pmb{A}$$ 的一个**不变子空间**（invariant subspace）。

## 定理

### 定理1

对于 $$n\times n$$ 矩阵 $$\pmb{A}$$ ，若 $$\pmb{X}\subseteq \mathbb{C}^n$$ 为 $$\pmb{A}$$ 的一个不变子空间，且 $$\pmb{X}\ne\{\pmb{0}\}$$ ，则存在非零特征向量 $$\pmb{x}\in\pmb{X}$$ ，使得 $$\pmb{Ax}=\lambda\pmb{x}$$ 。

**证明**

设 $$\dim{\mathbb{X}}=r,0\lt r \le n$$ ，非零向量 $$\pmb{x}\in\mathbb{X}$$ ，向量集 $$\{\pmb{x},\pmb{Ax},\pmb{A^2x},\cdots,\pmb{A^rx}\}$$ 属于 $$\pmb{X}$$ 且线性相关。$$r$$ 维子空间不可能有 $$r+1$$ 个线性无关的向量。所以，存在不全为零的数 $$c_0,c_1,\cdots,c_r$$ ，使得：

$$c_0\pmb{x}+c_1\pmb{Ax}+\cdots+c_r\pmb{A}^r\pmb{x}=\pmb{0}\tag{1.1}$$

成立。

设系数中最大值是 $$c_s\ne0$$ ，显然 $$0\le s\le r$$ ，则可以将 $$r$$ 次多项式分解为：

$$c_0+c_1t+\cdots+c_rt^r=c_s(t-\mu_1)\cdots(t-\mu_s)\tag{1.2}$$

其中 $$\mu_j\in\mathbb{C}$$ 。

于是，（1.1）式的左侧多项式可以参考（1.2）式，分解为：

$$\pmb{0}=(c_0+c_1\pmb{A}+\cdots+c_r\pmb{A}^r)\pmb{x}=c_s(\pmb{A}-\mu_1\pmb{I})\cdots(\pmb{A}-\mu_s\pmb{I})\pmb{x} \tag{1.3}$$

（1.3）式等号右边的乘法中，至少有一个 $$\mu_j$$ 和向量 $$\pmb{v}\ne0$$ 使得：

$$(\pmb{A}-\mu_j\pmb{I})\pmb{v}=\pmb{0}$$

成立。

即 $$\pmb{A}$$ 必定有一个特征向量 $$\pmb{v}\in\mathbb{X}$$ 对应的特征值是 $$\mu_j$$ 。

证毕。

### 定理2

对应相异特征值 $$\lambda_1,\cdots,\lambda_m$$ 的特征向量 $$\pmb{x}_1,\cdots,\pmb{x}_m$$ 组成一个线性无关的向量集合。

*此定理在文献[3]中已经证明，并且没有使用行列式，下面的证明即来自文献[3]*

**证明1**

设 $$\lambda_1,\cdots,\lambda_k$$ 为相异特征值，$$2\le k\le n$$ ，对应特征向量集合 $$\{\pmb{x}_1,\cdots,\pmb{x}_k\}$$ ，考虑：

$$c_1\pmb{x}_1+c_2\pmb{x}_2+\cdots+c_k\pmb{x}_k=\pmb{0}\tag{1.3}$$

将（1.3）式等号两侧左乘 $$(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})$$ ，并且 $$\pmb{Ax}_i=\lambda_i\pmb{x}_i,(1\le i\le k)$$ ，得：

$$\begin{split}\pmb{0} &= (\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})(c_1\pmb{x}_1+c_2\pmb{x}_2+\cdots+c_k\pmb{x}_k)\\&=c_1(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})(\pmb{A}-\lambda_1\pmb{I})\pmb{x}_1 \\&\quad+ c_2(\pmb{A}-\lambda_1\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\pmb{x}_2\\&\quad+\cdots\\&\quad+c_k(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})\pmb{x}_k\\&= c_k(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-2}\pmb{I})(\lambda_k-\lambda_{k-1})\pmb{x}_k\\&=c_k(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\lambda_k-\lambda_{k-2})(\lambda_k-\lambda_{k-1})\pmb{x}_k\\&=\cdots\\&=c_k(\lambda_k-\lambda_1)(\lambda_k-\lambda_2)\cdots(\lambda_k-\lambda_{k-2})(\lambda_k-\lambda_{k-1})\pmb{x}_k\end{split}$$

因为 $$\lambda_k\ne\lambda_i,1\le i \le{k-1}$$ ，且 $$\pmb{x}_k\ne0$$ ，所以：$$c_k=0$$ 。

同理，可得：$$c_{k-1}=\cdots=c_2=c_1=0$$ 。

故 $$\{\pmb{x}_1,\cdots,\pmb{x}_k\}$$ 是一个完整的线性无关集合。

证毕。

**证明2**（反证法）

设 $$\{\pmb{x}_1,\cdots,\pmb{x}_k\}$$ 是线性相关集合，在不失一般性的原则下，设 $$\{\pmb{x}_1,\cdots,\pmb{x}_{p-1}\}$$ 是最大的线性无关集，则：

$$\pmb{x}_p=c_1\pmb{x}_1+c_2\pmb{x}_2+\cdots+c_{p-1}\pmb{x}_{p-1} \tag{1.4}$$

其中 $$c_1,\cdots,c_{p-1}$$ 不全为零（因为 $$\pmb{x}_p\ne 0$$ ）。

（1.4）式等号两侧分别左乘 $$\pmb{A}$$ ，可得：

$$\begin{split}\pmb{Ax}_p &= c_1\pmb{A}\pmb{x}_1+c_2\pmb{Ax}_2+\cdots+c_{p-1}\pmb{Ax}_{p-1}\\&=c_1\lambda_1\pmb{x}_1+c_2\lambda_2\pmb{x}_2+\cdots+c_{p-1}\lambda_{p-1}\pmb{x}_{p-1}\end{split}$$

且：

$$\pmb{Ax}_p=\lambda_p\pmb{x}_p=c_1\lambda_p\pmb{x}_1+\cdots+c_{p-1}\lambda_p\pmb{x}_{p-1}$$

以上两式相减：

$$c_1(\lambda_1-\lambda_p)\pmb{x}_1+\cdots+c_{p-1}(\lambda_{p-1}-\lambda_p)\pmb{x}_{p-1}=\pmb{0}$$

因为 $$\{\pmb{x}_1,\cdots,\pmb{x}_{p-1}\}$$ 是线性无关的向量集，且 $$\lambda_1,\cdots,\lambda_p$$ 两两相异，所以：$$c_i=0,(1\le i \le{p-1})$$ 。与（1.4）式假设中的系数矛盾。故假设不成立。

证毕。

### 定理3

对于 $$\pmb{Ax}=\lambda\pmb{x}$$ 中的特征向量，为了跟下面的（1.5）式进行区分，称为**一般特征向量**。而下面所定义的：

$$(\pmb{A}-\lambda\pmb{I})^k\pmb{x}=\pmb{0}\tag{1.5}$$

$$\pmb{x}\ne0$$ 为特征值 $$\lambda$$ 对应的**广义特征向量**（generalized eigenvector），其中 $$k$$ 是正整数。

广义特征向量所形成集合，以及零向量，也是 $$\mathbb{C}^n$$ 的一个子空间，即 $$N((\pmb{A}-\lambda\pmb{I})^k)$$ ，称之为 **广义特征空间** 。具有如下性质：

若 $$\lambda$$ 是 $$n$$ 阶方阵 $$\pmb{A}$$ 的一个特征值，以 $$k$$ 为指数，则：

$$N((\pmb{A}-\lambda\pmb{I})^k)=N((\pmb{A}-\lambda\pmb{I})^n)$$

**证明**

采用类似定理1的证明方法。

对线性组合：

$$c_0\pmb{x}+c_1(\pmb{A}-\lambda\pmb{I})\pmb{x}+\cdots+c_{k-1}(\pmb{A}-\lambda\pmb{I})^{k-1}\pmb{x}=0$$

两侧同乘：$$(\pmb{A}-\lambda\pmb{I})^{k-1}$$ ，根据（1.5）可得：

$$c_0(\pmb{A}-\lambda\pmb{I})^{k-1}\pmb{x}=\pmb{0}$$

所以：$$c_0=0$$ 。

如果两侧同乘以 $$(\pmb{A}-\lambda\pmb{I})^{k-2}$$ ，同理可得 $$c_1=0$$ 。

最终得到 $$c_j=0,j=0,1,\cdots,k-1$$ 。

证毕。

### 定理4

某一特征值 $$\lambda_j$$ 对应的代数重数 $$\beta_j$$ 为广义特征向量集所张成的子空间维数，即 $$\beta_j=\dim N((\pmb{A}-\lambda_j\pmb{I})^{n})$$ 。

向量空间 $$\mathbb{C}^n$$ 可分为两个不相交的集合：广义特征空间 $$N((\pmb{A}-\lambda\pmb{I})^{n})$$ 和值域 $$R((\pmb{A}-\lambda\pmb{I})^n)$$ 。

若 $$\lambda$$ 为 $$n$$ 阶方阵 $$\pmb{A}$$ 的一个特征值，则：

$$N((\pmb{A}-\lambda\pmb{I})^{n})\oplus R((\pmb{A}-\lambda\pmb{I})^n)=\mathbb{C}^n$$

**证明**

由[秩—零化度定理](./basetheory.html)可知：

$$\dim N((\pmb{A}-\lambda\pmb{I})^{n}) + \dim R((\pmb{A}-\lambda\pmb{I})^n) = n$$

设 $$\pmb{x}\in N((\pmb{A}-\lambda\pmb{I})^{n}) \cap R((\pmb{A}-\lambda\pmb{I})^n)$$ ，

则 $$(\pmb{A}-\lambda\pmb{I})^n\pmb{x}=\pmb{0}$$ 且存在 $$y$$ 使得 $$\pmb{x}=(\pmb{A}-\lambda\pmb{I})^n\pmb{y}$$ ，由此二式可得：

$$(\pmb{A}-\lambda\pmb{I})^{2n}\pmb{y}=\pmb{0}$$

所以 $$\pmb{y}\in N((\pmb{A}-\lambda\pmb{I})^{2n})$$ 。

根据定理3，$$N((\pmb{A}-\lambda\pmb{I})^{2n})=N((\pmb{A}-\lambda\pmb{I})^{n})$$ ，所以：

$$\pmb{x}=(\pmb{A}-\lambda\pmb{I})^n\pmb{y}=\pmb{0}$$

证毕。

### 定理5

所有广义特征向量可张成 $$\mathbb{C}^n$$ 。

**证明**

将 $$\mathbb{C}^n$$ 分解为广义特征空间 $$N((\pmb{A}-\lambda_1\pmb{I})^{n})$$ 和值域 $$R((\pmb{A}-\lambda_1\pmb{I})^n)$$ 。

$$\begin{split}\pmb{A}(\pmb{A}-\lambda_1\pmb{I})^n&=\pmb{A}(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_1\pmb{I})^{n-1}\\&=(\pmb{A}-\lambda_1\pmb{I})\pmb{A}(\pmb{A}-\lambda_1\pmb{I})^{n-1}\\&=\cdots\\&=(\pmb{A}-\lambda_1\pmb{I})^n\pmb{A}\end{split}$$

对任意 $$y\in R((\pmb{A}-\lambda_1\pmb{I})^n)$$ ，$$\pmb{y}$$ 可写为 $$\pmb{y}=(\pmb{A}-\lambda_1\pmb{I})^n\pmb{z}$$ ，所以：

$$\pmb{Ay}=\pmb{A}(\pmb{A}-\lambda_1\pmb{I})^n\pmb{z}=(\pmb{A}-\lambda_1\pmb{I})^n\pmb{Az}$$

即 $$\pmb{Ay}\in R((\pmb{A}-\lambda_1\pmb{I})^n)$$ ，也就是说 $$R((\pmb{A}-\lambda_1\pmb{I})^n)$$ 是 $$\pmb{A}$$ 的一个不变子空间。

因为 $$\dim N((\pmb{A}-\lambda_1\pmb{I})^n)\ge 1, \dim R((\pmb{A}-\lambda_1\pmb{I})^n)\le n$$ 。根据定理1，不变子空间必有一特征值，所以子空间 $$R((\pmb{A}-\lambda_1\pmb{I})^n)$$ 也可以分解为广义特征空间和另外一个不变子空间的直和。

继续按照上述方式分割不变子空间，直到整个 $$\mathbb{C}^n$$ 都被分解为广义特征空间为止。

所以，广义特征向量足以张成 $$\mathbb{C}^n$$ 。

### 定理6

子空间 $$N((\pmb{A}-\lambda\pmb{I})^n)$$ 仅有唯一特征值 $$\lambda$$ 。

**证明**

对于非零向量 $$\pmb{x}\in N((\pmb{A}-\lambda\pmb{I})^n)$$ ，设 $$\lambda\ne\lambda'$$ 且 $$\pmb{Ax}=\lambda\pmb{x}$$ ，则：

$$(\pmb{A}-\lambda\pmb{I})\pmb{x}=(\lambda'-\lambda)\pmb{x}$$

故：$$(\pmb{A}-\lambda\pmb{I})^n\pmb{x}=(\lambda'-\lambda)^n\pmb{x}$$

但，已知 $$(\pmb{A}-\lambda\pmb{I})^n\pmb{x}=\pmb{0}, \lambda'-\lambda\ne0$$ ，故 $$\pmb{x}=0$$ ，这与假设矛盾。所以：$$\lambda=\lambda'$$ 

证毕。

根据定理5，方阵 $$\pmb{A}$$ 所有的广义特征向量可张成 $$\mathbb{C}^n$$ ，而且对应相异特征值的广义特征向量是线性无关的，故 $$\mathbb{C}^n$$ 可表示为所有特征向量空间的直和：

$$\mathbb{C}^n = N((\pmb{A}-\lambda_1\pmb{I})^n)\oplus\cdots\oplus N((\pmb{A}-\lambda_m\pmb{I})^n)$$

即：

$$n = \dim N((\pmb{A}-\lambda_1\pmb{I})^n)+\cdots+\dim N((\pmb{A}-\lambda_m\pmb{I})^n)$$

又因为：$$\beta_j = \dim N((\pmb{A}-\lambda_j\pmb{I})^n)$$ ，所以：$$n=\beta_1+\cdots+\beta_m$$ 。

这说明特征值 $$\lambda_j$$ 的代数重数是 $$\beta_j$$ 。

因为 $$N(\pmb{A}-\lambda\pmb{I})\subseteq N((\pmb{A}-\lambda\pmb{I})^n)$$ ，对应特征值 $$\lambda$$ 的线性无关特征向量个数必定不大于线性无关的广义特征向量数。对应的几何重数就是线性无关的特征向量个数，而代数重数等于线性无关的广义特征向量重数。

故 $$\lambda_j$$ 对应的几何重数不大于代数重数。

## 参考文献

[1]. Sheldon Axler. 线性代数应该这样学. 北京：人民邮电出版社

[2]. [线代启示录：拒绝行列式的特征分析](https://ccjou.wordpress.com/2010/05/26/%e6%8b%92%e7%b5%95%e8%a1%8c%e5%88%97%e5%bc%8f%e7%9a%84%e7%89%b9%e5%be%b5%e5%88%86%e6%9e%90/)

[3]. [机器学习数学基础：矩阵对角化](./fibonacii.html)

[4]. [机器学习数学基础：秩—零化度定理](./basetheory.html)

