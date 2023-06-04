# 连续正态分布随机变量的熵

《机器学习数学基础》第 416 页给出了连续型随机变量的熵的定义，并且在第 417 页以正态分布为例，给出了符合 $N(0,\sigma^2)$ 的随机变量的熵。

**注意：在第 4 次印刷以及之前的版本中，此处有误，具体请阅读[勘误表](https://lqlab.readthedocs.io/en/latest/math4ML/corrigendum.html#id6)说明**

## 1. 推导（7.6.6）式

假设随机变量服从正态分布 $X\sim N(\mu,\sigma^2)$ （《机器学习数学基础》中是以标准正态分布为例，即 $X\sim N(0,\sigma^2)$ ）。

根据《机器学习数学基础》的（7.6.1）式熵的定义：
$$
H(X)=-\int f(x)\log f(x)\text{d}x\tag{7.6.1}
$$
其中，$f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ ，是概率密度函数。根据均值的定义，（7.6.1）式可以写成：
$$
H(X)=-E[\log f(x)]
$$
将 $f(x)$ 代入上式，可得：
$$
\begin{split}
H(X)&=-E\left[\log(\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}})\right]
\\&=-E\left[\log(\frac{1}{\sqrt{2\pi}\sigma})+\log(e^{-\frac{(x-\mu)^2}{2\sigma^2}})\right]
\\&=-E\left[\log(\frac{1}{\sqrt{2\pi}\sigma})\right]-E\left[\log(e^{-\frac{(x-\mu)^2}{2\sigma^2}})\right]
\\&=\frac{1}{2}\log(2\pi\sigma^2)-E\left[-\frac{1}{2\sigma^2}(x-\mu)^2\log e\right]
\\&=\frac{1}{2}\log(2\pi\sigma^2)+\frac{\log e}{2\sigma^2}E\left[(x-\mu)^2\right]
\\&=\frac{1}{2}\log(2\pi\sigma^2)+\frac{\log e}{2\sigma^2}\sigma^2\quad(\because E\left[(x-\mu)^2\right]=\sigma^2,参阅 332 页 (G2)式)
\\&=\frac{1}{2}\log(2\pi\sigma^2)+\frac{1}{2}\log e
\\&=\frac{1}{2}\log(2\pi e\sigma^2)
\end{split}
$$
从而得到第 417 页（7.6.6）式。

## 2. 推导多维正态分布的熵

对于服从正态分布的多维随机变量，《机器学习数学基础》中也假设服从标准正态分布，即 $\pmb{X}\sim N(0,\pmb{\Sigma})$ 。此处不失一般性，以 $\pmb{X}\sim N(\mu,\pmb{\Sigma})$ 为例进行推导。

注意：《机器学习数学基础》第 417 页是以二维随机变量为例，书中明确指出：不妨假设 $\pmb{X}=\begin{bmatrix}\pmb{X}_1\\\pmb{X}_2\end{bmatrix}$ ，因此使用的概率密度函数是第 345 页的（5.5.18）式。

下面的推导，则考虑 $n$ 维随机变量，即使用 345 页（5.5.19）式的概率密度函数：
$$
f(\pmb{X})=\frac{1}{\sqrt{(2\pi)^n|\pmb{\Sigma}|}}\text{exp}\left(-\frac{1}{2}(\pmb{X}-\pmb{\mu})^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})\right)
$$
根据熵的定义（第 416 页（7.6.2）式）得：
$$
\begin{split}
H(\pmb{X})&=-\int f(\pmb{X})\log(f(\pmb{X}))\text{d}\pmb{x}
\\&=-E\left[\log N(\mu,\pmb{\Sigma})\right]
\\&=-E\left[\log\left((2\pi)^{-n/2}|\pmb{\Sigma}|^{-1/2}\text{exp}\left(-\frac{1}{2}(\pmb{X}-\pmb{\mu})^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})\right)\right)\right]
\\&=-E\left[-\frac{n}{2}\log(2\pi)-\frac{1}{2}\log(|\pmb{\Sigma}|)+\log\text{exp}\left(-\frac{1}{2}(\pmb{X}-\pmb{\mu})^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})\right)\right]
\\&=\frac{n}{2}\log(2\pi)+\frac{1}{2}\log(|\pmb{\Sigma}|)+\frac{\log e}{2}E\left[(\pmb{X}-\pmb{\mu})^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})\right]
\end{split}
$$
下面单独推导：$E\left[(\pmb{X}-\pmb{\mu})^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})\right]$ 的值：
$$
\begin{split}
E\left[(\pmb{X}-\pmb{\mu})^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})\right]&=E\left[\text{tr}\left((\pmb{X}-\pmb{\mu})^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})\right)\right]
\\&=E\left[\text{tr}\left(\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})(\pmb{X}-\pmb{\mu})^{\text{T}}\right)\right]
\\&=\text{tr}\left(\pmb{\Sigma^{-1}}E\left[(\pmb{X}-\pmb{\mu})(\pmb{X}-\pmb{\mu})^{\text{T}}\right]\right)
\\&=\text{tr}(\pmb{\Sigma}^{-1}\pmb{\Sigma})
\\&=\text{tr}(\pmb{I}_n)
\\&=n
\end{split}
$$
所以：
$$
\begin{split}
H(\pmb{X})&=\frac{n}{2}\log(2\pi)+\frac{1}{2}\log(|\pmb{\Sigma}|)+\frac{\log e}{2}E\left[(\pmb{X}-\pmb{\mu})^{\text{T}}\pmb{\Sigma}^{-1}(\pmb{X}-\pmb{\mu})\right]
\\&=\frac{n}{2}\log(2\pi)+\frac{1}{2}\log(|\pmb{\Sigma}|)+\frac{\log e}{2}n
\\&=\frac{n}{2}\left(\log(2\pi)+\log e\right)+\frac{1}{2}\log(|\pmb{\Sigma}|)
\\&=\frac{n}{2}\log(2\pi e)+\frac{1}{2}\log(|\pmb{\Sigma}|)
\end{split}
$$
当 $n=2$ 时，即得到《机器学习数学基础》第 417 页推导结果：
$$
H(\pmb{X})=\log(2\pi e)+\frac{1}{2}\log(|\pmb{\Sigma}|)=\frac{1}{2}\log\left((2\pi e)^2|\pmb{\Sigma|}\right)
$$

## 参考资料

[1]. Entropy of the Gaussian[DB/OL]. https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/ , 2023.6.4

[2]. Entropy and Mutual Information[DB/OL]. https://gtas.unican.es/files/docencia/TICC/apuntes/tema1bwp_0.pdf ,2023.6.4

[3]. Fan Cheng. CS258: Information Theory[DB/OL]. http://qiniu.swarma.org/course/document/lec-7-Differential-Entropy-Part1.pdf , 2023.6.4.

[4]. Keith Conrad. PROBABILITY DISTRIBUTIONS AND MAXIMUM ENTROPY[DB/OL]. https://kconrad.math.uconn.edu/blurbs/analysis/entropypost.pdf, 2023.6.4.