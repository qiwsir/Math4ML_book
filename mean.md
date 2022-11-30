# 期望

## 定义

一个分布的均值（mean），也称为期望值（expected value），通常记作：$$\mu$$

连续型随机变量：$$E[X]=\int_{\mathcal{X}}xp(x)dx$$​

离散型随机变量：$$E[X]=\sum_{x\in\mathcal{X}}xp(x)$$​

## 定理

**定理1：**总期望定理（law of total expectation）也称为**迭代期望定理**（law of iterated expectation），即多个随机变量的期望。

$$E[X]=E[E[X|Y]]$$

**证明：**

不妨设 $$X$$ 、$$Y$$ 是离散随机变量，

$$\begin{split}E[E[X|Y]]&=E\left[\sum_xxp(X=x|Y)\right]\\&=\sum_y\left[\sum_xxp(X=x|Y)\right]p(Y=y)\\&=\sum_{xy}xp(X=x,Y=y)\\&=E[X]\end{split}$$​​

**定理2：** 设 $$X$$ 的数学期望有限，概率密度 $$f(x)$$ 关于 $$\mu$$ 对称：$$f(\mu+x)=f(\mu-x)$$ ，则 $$E(X)=\mu$$ 。$$^{[2]}$$

**证明：**这时 $$g(t)=tf(t+\mu)$$ 是奇函数：$$g(-t)=g(t)$$ 。因为 $$g(t)$$ 在 $$(-\infty,\infty)$$ 中的积分等于 0 ，所以有：

$$\begin{split}E(X) &= \int_{-\infty}^{\infty}xf(x)dx\\&=\int_{-\infty}^{\infty}\mu f(x)dx+\int_{-\infty}^{\infty}(x-\mu)f(x-\mu+\mu)dx\\&=\mu+\int_{-\infty}^{\infty}tf(t+\mu)dt\\&=\mu+\int_{-\infty}^{\infty}g(t)dt=\mu\end{split}$$

**推论：** 正态分布 $$N(\mu,\sigma^2)$$ 的数学期望是 $$\mu$$ ，均匀分布 $$\mathcal{U}(a,b)$$ 的数学期望是 $$(a+b)/2$$ 。

## 计算$$^{[2]}$$

**定理3：** 设 $$g(x)$$ 是 $$x$$ 的函数，$$h(x,y)$$ 是 $$x,y$$ 的函数，

（1）若 $$X$$ 有概率密度 $$f(x)$$ ，则：

$$E(|g(X)|)=\int_{-\infty}^{\infty}|g(x)|f(x)dx$$

当 $$E(|g(X)|)\lt\infty$$ 时，有：

$$E(g(X))=\int_{-\infty}^{\infty}g(x)f(x)dx$$

（2）若 $$[X,Y]$$ 有联合密度 $$f(x,y)$$ ，则：

$$E(|h(X,Y)|)=\int\int_{\mathbb{R}^2}|h(x,y)|f(x,y)dxdy$$

当 $$E(|h(X,Y)|)\lt\infty$$ 时，有：

$$E(h(X,Y))=\int\int_{\mathbb{R}^2}h(x,y)f(x,y)dxdy$$

（3）若 $$X$$ 是非负随机变量，则：

$$E(X)=\int_0^{\infty}P(X\gt x)dx$$

**证明**

（3）对于 $$x\ge 0$$ ，有：

$$x=\int_0^xdy=\int_0^{\infty}I[y\lt x]dy$$

因为对 $$x\lt0$$ ，有 $$f(x)=0$$ ，所以通过上式可得：

$$\begin{split}E(X)&=\int_0^\infty xf(x)dx=\int_0^{\infty}\int_0^{\infty}I[y\lt x]dyf(x)dx\\&=\int_0^{\infty}\left(\int_0^{\infty}f(x)I[y\lt x]dx\right)dy\\&=\int_0^{\infty}P(X\gt y)dy\end{split}$$

证毕。

**定理4：** 设 $$g(x)$$ 是 $$x$$ 的函数，$$h(x,y)$$ 是 $$x,y$$ 的函数，

（1）若 $$X$$ 有离散概率密度 $$p_j=P(X=x_j),j\ge1$$ ，则：

$$E(|g(X)|)=\sum_{j=1}^{\infty}|g(x_j)p_j$$

当 $$E(|g(X)|)\lt\infty$$ 时，有：

$$E(g(X))=\sum_{j=1}^{\infty}g(x_j)p_j$$

（2）若 $$[X,Y]$$ 有离散概率分布 $$p_{ij}=P(X=x_i,Y=y_j),i,j\ge1$$ ，则：

$$E(|h(X,Y)|)=\sum_{i=1}^{\infty}\sum_{j=1}^{\infty}|h(x,y)|p_{ij}$$

当 $$E(|h(X,Y)|)\lt\infty$$ 时，有：

$$E(h(X,Y))=\sum_{j=1}^{\infty}\sum_{j=1}^{\infty}h(x,y)p_{ij}$$

（3）若 $$X$$ 是非负随机变量，则：

$$E(X)=\sum_{k=1}^{\infty}P(X\ge k)=\sum_{k=0}^{\infty}P(X\gt k)$$

**证明**

（3）设 $$p_j=P(X=j)$$ ，则：

$$\begin{split}E(X)&=\sum_{j=1}^{\infty}jp_j=\sum_{j=1}^{\infty}\sum_{k=1}^jp_j\\&=\sum_{k=1}^{\infty}\sum_{j=k}^{\infty}p_j=\sum_{k=1}^{\infty}P(X\ge k)\end{split}$$

又：

$$\sum_{k=1}^{\infty}P(X\ge k)=\sum_{k=1}^{\infty}P(X\gt k-1)=\sum_{k=0}^{\infty}P(X\gt k)$$

## 性质

- $$E[aX+b]=aE[X]+b$$ （线性）
- 对于 $$n$$ 个独立随机变量：
  - $$E\left[\sum_{i=1}^nX_i\right]=\sum_{i=1}^nE[X_i]$$
  - $$E\left[\prod_{i=1}^nX_i\right]=\prod_{i=1}^nE[X_i]$$

**定理5：**$$^{[2]}$$ 设 $$E(|X_j|)\lt\infty(1\le j\le n)$$ ，$$c_0,c_1,\cdots,c_n$$ 是常数，则有以下结果：

（1）线性组合 $$Y=c_0+c_1X_1+c_2X_2+\cdots+c_nX_n$$ 的数学期望存在，而且：

$$E(Y)=c_0+c_1E(X_1)+\cdots+c_nE(X_n)$$

（2）如果 $$X_1,X_2,\cdots,X_n$$ 相互独立，则乘积 $$Z=X_1\cdots X_n$$ 的数学期望存在，并且：

$$E(Z)=E(X_1)\cdots E(X_n)$$

（3）如果 $$P(X_1\le X_2)=1$$ ，则 $$E(X_1)\le E(X_2)$$

**证明**

不妨设 $$n=2$$ 和 $$[X_1,X_2]$$ 有联合密度 $$f(x_1,x_2)$$ 。

（1）由【定理3】得：

$$\begin{split}E(Y)&=\int\int_{\mathbb{R}^2}|c_0+\sum_{j=1}^2c_jx_j|f(x_1,x_2)dx_1dx_2\\&\le|c_0|+\sum_{j=1}^2|c_j|\int\int_{\mathbb{R}^2}|x_j|f(x_1,x_2)dx_1dx_2\\&=c_0+\sum_{j=1}^2|c_j|E(|X_j|)\lt\infty\end{split}$$

所以：

$$\begin{split}E(Y)&=\int\int_{\mathbb{R}^2}\left(c_0+\sum_{j=1}^2c_jx_j\right)f(x_1,x_2)dx_1dx_2\\&=c_0+\sum_{j=1}^2c_j\int\int_{\mathbb{R}^2}x_jf(x_1,x_2)dx_1dx_2\\&=c_0+\sum_{j=1}^2c_jE(X_j)\end{split}$$

（2）因为有 $$f(x_1x_2)=f(x_1)f(x_2)$$ ，其中 $$f_j(x_j)$$ 是 $$X_j$$ 的概率密度，则：

$$\begin{split}E(X_1X_2)&=\int\int_{\mathbb{R}^2}|x_1x_2|f(x_1,x_2)dx_1dx_2\\&=\int_{-\infty}^{\infty}|x_1|f_1(x_1)dx_1\int_{-\infty}^{\infty}|x_2|f(x_2)dx_2\\&=E(|X_1|)E(|X_2|)\lt\infty\end{split}$$

所以：

$$\begin{split}E(X_1X_2)&=\int\int_{\mathbb{R}^2}x_1x_2f(x_1,x_2)dx_1dx_2\\&=\int_{-\infty}^{\infty}x_1f_1(x_1)dx_1\int_{-\infty}^{\infty}x_2f(x_2)dx_2\\&=E(|X_1|)E(|X_2|)\end{split}$$

（3）定义 $$Y=X_2-X_1$$ ，则有 $$P(Y\ge0)=P(X_2\ge X_1)=1$$ ，所以：

$$E(X_2)-E(X_1)=E(Y)=\int_0^{\infty}P(Y\gt y)dy\ge0$$

## 常用的数学期望$$^{[2]}$$

**1. 伯努利分布** $$\mathcal{B}(1, p)$$ 

设 $$X\sim\mathcal{B}(1, p)$$ ，则

$$E(X)=1\cdot p+0\cdot(1-p)=p$$

**2. 二项分布** $$\mathcal{B}(n,p)$$

设 $$X\sim\mathcal{B}(n, p)$$ ，则 $$E(X)=np$$

**证明：** 设 $$q=1-p$$ ，由

$$p_j=P(X=j)=C_n^jp^jq^{n-j}$$ ，$$0\le j\le n$$

得到：

$$\begin{split}E(X)&=\sum_{j=0}^njC_n^jp_jq^{n-j}\\&=np\sum_{j=1}^nC_{n-1}^{j-1}p^{j-1}q^{n-j}\quad(\because jC_n^j=nC_{n-1}^{j-1})\\&=np\sum_{k=0}^{n-1}C_{n-1}^kp^kq^{n-1-k}\quad(令k=j-1)\\&=np(p+q)^{n-1}=np\end{split}$$

单次试验成功的概率 $$p$$ 越大，则在 $$n$$ 次独立重复试验中，平均成功的次数越多。

**3. 泊松分布** $$\mathcal{P}(\lambda)$$

设 $$X\sim\mathcal{P}(\lambda)$$ ，则 $$E(X)=\lambda$$

**证明**：由

$$P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda}$$ ，$$k=0,1,\cdots$$

得到：

$$E(X)=\sum_{k=0}^{\infty}k\frac{\lambda^k}{k!}e^{-\lambda}$$

当 $$k=0$$ 时， $$k\frac{\lambda^k}{k!}e^{-\lambda}=0$$ ，所以：

$$\begin{split}E(X)&=\sum_{k=1}^{\infty}k\frac{\lambda^k}{k!}e^{-\lambda}\\&=\sum_{k=1}^{\infty}\frac{\lambda^k}{(k-1)!}e^{-\lambda}\quad(\because\frac{k}{k!}=\frac{1}{(k-1)!})\\&=\sum_{k=1}^{\infty}\frac{\lambda^{k-1}\lambda}{(k-1)!}e^{-\lambda}\\&=\lambda e^{-\lambda}\sum_{k=1}^{\infty}\frac{\lambda^{k-1}}{(k-1)!}\end{split}$$

又因为 $$e^x=1+x+\frac{x^2}{2!}+\cdots+\frac{x^n}{n!}+\cdots=\sum_{k=1}^{\infty}\frac{x^{k-1}}{(k-1)!}$$

所以：

$$E(X)=\lambda e^{-\lambda}\sum_{k=1}^{\infty}\frac{\lambda^{k-1}}{(k-1)!}=\lambda e^{-\lambda}e^{\lambda}=\lambda$$

参数 $$\lambda$$ 是泊松分布 $$\mathcal{P}(\lambda)$$ 的数学期望。

**4. 几何分布** 

设 $$X$$ 服从参数为 $$p$$ 的几何分布，则 $$E(X)=1/p$$

**证明：**由

$$P(X=j)=pq^{j-1}$$ ，$$j=1,2\cdots$$

得到：

$$\begin{split}E(X)&=\sum_{j=1}^{\infty}jpq^{j-1}\\&=p\left(\sum_{j=0}^{\infty}q^j\right)'\\&=p\left(\frac{1}{1-q}\right)'=\frac{1}{p}\end{split}$$

说明：单次试验中的成功概率 $$p$$ 越小，首次成功所需要的平均试验次数就越多。

**5. 指数分布** $$\mathcal{E}(\lambda)$$

设 $$X\sim\mathcal{E}(\lambda)$$ ，则 $$E(X)=1/\lambda$$

**证明**：因为 $$X$$ 的概率密度：$$f(x)=\lambda e^{-\lambda x}$$ ，$$x\ge0$$ ，所以：

$$E(X)=\int_{-\infty}^{\infty}xf(x)dx=\int_0^{\infty}x\lambda e^{-\lambda x}dx=\frac{1}{\lambda}$$

## 参考资料

[1]. Kevin P. Murphy. *Probabilistic Machine Learning An Introduction*[M]:43-44. The MIT Press.

[2]. 概率引论. 何书元. 北京：高等教育出版社. 2012.1，第1版