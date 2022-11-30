# 方差

## 定义

**方差**（variance）度量分布的“伸展”程度，通常记作： $$\sigma^2$$​

$$\begin{split}{\rm{Var}}[X]&=E[(X-\mu)^2]=\int(x-\mu)^2p(x)dx\\&=\int x^2p(x)dx+\mu^2\int p(x)dx-2\mu\int xp(x)dx\\&=E[X^2]+\mu^2-2\mu\cdot\mu\\&=E[X^2]-\mu^2\end{split}$$​​

得到一个重要结论：$$E[X^2]=\sigma^2+\mu^2$$

**从测量的角度理解方差：**

设 $$X$$ 是对长度为 $$\mu$$ 的物体的测量值，则 $$X-\mu$$ 是测量误差，$$(X-\mu)^2$$ 是测量误差的平方。如果测量仪器无系统偏差（即 $$E(X)=\mu$$ ），则 $$E(X-\mu)^2$$ 是测量误差平方的平均，正是方差。用测量误差平方的平均，即均方差，衡量测量值和真实值之间的差异。

当 $$X$$ 有离散分布 $$p_j=P(X=x_j),j=1,2,\cdots$$ 时，有：

$${\rm{Var}}(X)=\sum_{j=1}^{\infty}(x_j-\mu)^2p_j$$

当 $$X$$ 有概率密度 $$f(x)$$ 时，有：

$${\rm{Var}}(X)=\int_{-\infty}^{\infty}(x-\mu)^2f(x)dx$$

**标准差**（standard deviation）定义为：$$\rm{std}[X]=\sqrt{\rm{Var}[X]}=\sigma$$​

## 定理

**定理1：** 如果 $$X,Y$$ 有相同的概率分布，则它们有相同的数学期望和方差。

## 性质

- 性质1：$${\rm{Var}}[aX+b]=a^2{\rm{Var}}[X]$$

**证明：** 根据定义：$${\rm{Var}}[X]=E[X^2]-\mu^2=E[X^2]-(E[X])^2$$ 得：

$$\begin{split}{\rm{Var}}[aX+b] &= E[(aX+b)^2]-(E[(aX+b)])^2\\&=E[a^2X^2+2abX+b^2]-(aE[X]+b)^2\\&=a^2E[X^2]+2abE[X]+b^2-a^2(E[X])^2-2abE[X]-b^2\\&=a^2E[X^2]-a^2(E[X])^2\\&=a^2(E[X^2]-(E[X])^2)\\&=a^2{\rm{Var}}[X]\end{split}$$

- 性质2：$$X$$ 和 $$Y$$ 是独立随机变量，则 $${\rm{Var}}[X+Y]={\rm{Var}}[X]+{\rm{Var}}[Y]$$

**证明：** 使用方差的定义

$$\begin{split}{\rm{Var}}[X+Y] &= E[(X+Y)^2]-(E[X+Y])^2\\&=E[X^2+Y^2+2XY]-(E[X]+E[Y])^2\\&=E[X^2]+E[Y^2]+2E[XY]-(E[X])^2-2E[X]E[Y]-(E[Y])^2\end{split}$$

因为 $$X、Y$$ 是独立随机变量，有 $$E[XY]=E[X]E[Y]$$ ，所以：

$$\begin{split}{\rm{Var}}[X+Y] &= E[X^2]-(E[X])^2+E[Y^2]-(E[Y])^2\\&={\rm{Var}}[X]+{\rm{Var}}[Y]\end{split}$$

- 推论1：独立随机变量 $$X_1,X_2,\cdots,X_n$$ ：$${\rm{Var}}[X_1+X_2+\cdots+X_n]={\rm{Var}}[X_1]+{\rm{Var}}[X_2]+\cdots+{\rm{Var}}[X_n]$$ ，即：$${\rm{Var}}[\sum_{i=1}nX_i]=\sum_{i=1}^n{\rm{Var}}[X_i]$$

- 推论2：独立随机变量 $$X_1,X_2,\cdots,X_n$$ ，常数 $$a_1,a_2,\cdots,a_n$$ ：$$\begin{split}{\rm{Var}}[a_1X_1+a_2X_2+\cdots+a_nX_n]&={\rm{Var}}[a_1X_1]+{\rm{Var}}[a_2X_2]+\cdots+{\rm{Var}}[a_nX_n]\\&=a_1^2{\rm{Var}}[X_1]+a_2^2{\rm{Var}}[X_2]+\cdots+a_n^2{\rm{Var}}[X_n]\end{split}$$ ，即 $${\rm{Var}}[\sum_{i=1}^na_iX_i]=\sum_{i=1}^na_i{\rm{Var}}[X_i]$$

- 性质3：独立随机变量 $$X_i,i=1,\cdots,n$$ ：

  $$\begin{split}{\rm{Var}}\left[\prod_{i=1}^nX_i\right]&= E\left[(\prod_{i=1}^nX_i)^2\right]-\left(E\left[\prod_{i=1}^nX_i\right]\right)^2\\&=E\left[\prod_{i=1}^nX_i^2\right]-\left(\prod_{i=1}^nE[X_i]\right)^2\\&=\prod_{i=1}^nE[X_i^2]-\prod_{i=1}^n(E[X_i])^2\\&=\prod_{i=1}^n({\rm{Var}}[X_i]+(E[X_i])^2)-\prod_{i=1}^n(E[X_i])^2\\&=\prod_{i=1}^n(\sigma_i^2+\mu_i^2)-\prod_{i=1}^n\mu_i^2\end{split}$$​



## 条件方差公式

条件方差公式（conditional variance formula）也称为**总方差定理**（law of total variance）：

$${\rm{Var}}[X]=E[{\rm{Var}}[X|Y]]+{\rm{Var}}[E[X|Y]]$$

令 $$\mu_{X|Y}=E[X|Y],s^2_{X|Y}=E[X^2|Y],\sigma^2_{X|Y}={\rm{Var}}[X|Y]=s_{X|Y}-\mu^2_{X|Y}$$ ，则：

$$\begin{split}{\rm{Var}}[X] &= E[X^2]-(E[X])^2\\ &= E[s_{X|Y}|Y]-(E[\mu_{X|Y}|Y])^2\\ &= E[\sigma^2_{X|Y}|Y]+E[\mu^2_{X|Y}|Y]-(E[\mu_{X|Y}|Y])^2\\&=E_Y[{\rm{Var}}[X|Y]]+{\rm{Var}}_Y[\mu_{X|Y}]\end{split}$$​

## 常用的方差

**1. 伯努利分布** $$\mathcal{B}(1,p)$$

设 $$P(X=1)=p,P(x=0)=1-p$$ ，则 $${\rm{Var}}(X)=pq,(q=1-p)$$

**证明**

因为是伯努利分布，故 $$X^2=X$$ 。

由参考资料 [3] 中关于伯努利分布的期望可知：$$E(X)=p$$

$$\begin{split}{\rm{Var}}(X)&=E(X^2)-[E(X)]^2\\&=E(X)-p^2\\&=p-p^2\quad(令：q=1-p)\\&=pq\end{split}$$

**2. 二项分布** $$\mathcal{B}(n,p)$$

设 $$q=1-p$$ ，$$P(X=j)=C_n^jp^jq^{n-j},(0\le j\le n)$$ ，则：$${\rm{Var}}(X)=npq$$

**证明**

首先，计算一个二项分布的结论公式：$$E[X(X-1)]=n(n-1)p^2$$

$$\begin{split}E[X(X-1)]&=\sum_{j=0}^nC_n^jj(j-1)p^jq^{n-j}\\&=p^2\left(\frac{d^2}{dx^2}\sum_{j=0}^nC_n^jx^jq^{n-j}\right)\Bigg|_{x=p}\\&=p^2\frac{d^2}{dx^2}(x+q)^n\Bigg|_{x=p}\\&=n(n-1)p^2\end{split}$$

又：$$E[X(X-1)]=E[X^2-X]=E[X^2]-E[X]$$ （此结论在后续还会用到）

所以：$$E[X^2]=n(n-1)p^2+E[X]$$

根据参考资料 [3] 可知，二项分布均值 $$E(X)=np$$ 。

得：$$E[X^2]=n(n-1)p^2+np$$

根据 $${\rm{Var}}(X)=E(X^2)-[E(X)]^2$$ 得：

$${\rm{Var}}(X)=n(n-1)p^2+np-(np)^2=np(1-p)=npq$$

**3. 泊松分布** $$\mathcal{P}(\lambda)$$

设 $$P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda},k=0,1,\cdots$$ ，则 $${\rm{Var}}(X)=\lambda$$

**证明** 根据参考资料 [3] 知泊松分布期望 $$E(X)=\lambda$$ ，有：

$$\begin{split}E(X^2)&=E[X(X-1)]+E(X)\quad(见上面的二项分布证明)\\&=\sum_{k=0}^{\infty}k(k-1)\frac{\lambda^k}{k!}e^{-\lambda}+\lambda\quad(\because\frac{k(k-1)}{k!}=\frac{1}{(k-2)!})\\&=\sum_{k=2}^{\infty}\frac{\lambda^k}{(k-2)!}e^{-\lambda}+\lambda\\&=\lambda^2e^{-\lambda}\sum_{k=2}^{\infty}\frac{\lambda^{k-2}}{(k-2)!}+\lambda\quad(\because e^{\lambda}=\sum_{k=1}^{\infty}\frac{\lambda^{k-2}}{(k-2)!})\\&=\lambda^2+\lambda\end{split}$$

根据 $${\rm{Var}}(X)=E(X^2)-[E(X)]^2$$ 得：

$${\rm{Var}}(X)=\lambda^2+\lambda-(\lambda)^2=\lambda$$

**4. 几何分布**

设 $$X$$ 有概率分布 $$P(X=j)=pq^{j-1},j=1,2,\cdots,q=1-p$$ ，则 $${\rm{Var}}(X)=\frac{q}{p^2}$$

**证明**

由参考资料 [3] 知几何分布的期望 $$E(X)=\frac{1}{p}$$

$$\begin{split}E(X^2)&=E[X(X-1)]+E(X)\quad(见上面的二项分布证明)\\&=\sum_{j=1}^{\infty}j(j-1)pq^{j-1}+\frac{1}{p}\\&=pq\left(\sum_{j=0}^{\infty}q^j\right)''+\frac{1}{p}\\&=pq\left(\frac{1}{1-q}\right)''+\frac{1}{p}\\&=\frac{2pq}{(1-q)^3}+\frac{1}{q}=\frac{2q}{p^2}+\frac{1}{p}\quad(\because1-q=p)\end{split}$$

根据 $${\rm{Var}}(X)=E(X^2)-[E(X)]^2$$ 得：

$${\rm{Var}}(X)=\frac{2q}{p^2}+\frac{1}{p}-\left(\frac{1}{p}\right)^2=\frac{q}{p^2}$$



## 参考资料

[1]. Kevin P. Murphy. *Probabilistic Machine Learning An Introduction*[M]:43-44. The MIT Press.

[2]. 概率引论. 何书元. 北京：高等教育出版社. 2012.1，第1版

[3]. [期望](./mean.html)