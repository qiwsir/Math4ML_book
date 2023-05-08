# 矩阵运算

在《机器学习数学基础》第2章的2.1.3节、2.1.4节和2.1.5节分别介绍了矩阵的加（减）法、数量乘法和矩阵乘法，这些构成了矩阵的基本运算，并且列出了矩阵的所有运算性质。在手工计算或者原理证明中，这些计算性质会经常用到。

此处再补充两项。

## 1. 运算技巧

若 $\pmb{A}$ 和 $\pmb{B}$ 是 $n\times n$ 阶矩阵，且 $\pmb{A} + \pmb{B}$ 是可逆的，则：

$$
\pmb{A}(\pmb{A}+\pmb{B})^{-1}\pmb{B}=\pmb{B}(\pmb{A}+\pmb{B})^{-1}\pmb{A}
$$


上述运算技巧来自参考文献 [1]。

**证明：**

因为 $\pmb{A}+\pmb{B}$ 可逆，所以 $(\pmb{A}+\pmb{B})(\pmb{A}+\pmb{B})^{-1}=\pmb{I}$ ，即：

$$
\pmb{A}(\pmb{A}+\pmb{B})^{-1}+\pmb{B}(\pmb{A}+\pmb{B})^{-1}=\pmb{I}
$$

$$
\pmb{A}(\pmb{A}+\pmb{B})^{-1}=\pmb{I}-\pmb{B}(\pmb{A}+\pmb{B})^{-1}
$$


计算：

$$
\begin{split}\pmb{A}(\pmb{A}+\pmb{B})^{-1}\pmb{B}-\pmb{B}(\pmb{A}+\pmb{B})^{-1}\pmb{A}&=(\pmb{I}-\pmb{B}(\pmb{A}+\pmb{B})^{-1})\pmb{B}-\pmb{B}(\pmb{A}+\pmb{B})^{-1}\pmb{A}\\&=\pmb{B}-\pmb{B}(\pmb{A}+\pmb{B})^{-1}\pmb{B}-\pmb{B}(\pmb{A}+\pmb{B})^{-1}\pmb{A}\\&=\pmb{B}-\pmb{B}(\pmb{A}+\pmb{B})^{-1}(\pmb{B}+\pmb{A})\\&=\pmb{B}-\pmb{B}\quad(\because\pmb{A}+\pmb{B}可逆)\\&=0\end{split}
$$


所以：$\pmb{A}(\pmb{A}+\pmb{B})^{-1}\pmb{B}=\pmb{B}(\pmb{A}+\pmb{B})^{-1}\pmb{A}$

证毕。

## 2. 矩阵指数

### 2.1 定义和性质

对于 $n\times n$ 矩阵 $\pmb{A}$ 可以定义**矩阵指数**（matrix exponential）。

设指数函数：$e^x = 1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots$

若将 $x$ 替换为矩阵 $\pmb{A}$ ，常数 $1$ 用单位矩阵 $\pmb{I}$ 代替，则：

$$
e^{\pmb{A}}=\pmb{I}+\pmb{A}+\frac{\pmb{A}^2}{2!}+\frac{\pmb{A}^3}{3!}+\cdots
$$


上述指数矩阵也收敛。

性质：

- 若 $\pmb{AB}=\pmb{BA}$ ，则 $e^{\pmb{A}}e^{\pmb{B}}=e^{\pmb{B}}e^{\pmb{A}}=e^{\pmb{A}+\pmb{B}}$

  根据假设：

  $$
  \begin{split}e^{\pmb{A}+\pmb{B}} &= \sum_{k=0}^{\infty}\frac{(\pmb{A}+\pmb{B})^k}{k!}\\&= \sum_{k=0}^{\infty}\frac{\sum_{j=0}^k\binom{k}{j}\pmb{A}^j\pmb{B}^{k-j}}{k!}\\&=\sum_{k=0}^{\infty}\sum_{j=0}^k\frac{k!}{(k-j)!j!}\frac{1}{k!}\pmb{A}^j\pmb{B}^{k-j}\\&=\left(\sum_{j=0}^{\infty}\frac{\pmb{A}^j}{j!}\right)\left(\sum_{l=0}^{\infty}\frac{\pmb{B}^l}{l!}\right)\\&=e^{\pmb{A}}e^{\pmb{B}}=e^{\pmb{B}}e^{\pmb{A}}\end{split}
  $$
  

- $e^{\pmb{A}^T} = (e^{\pmb{A}})^T$

  $$
  e^{\pmb{A}^T}=\sum_{k=0}^{\infty}\frac{(\pmb{A}^T)^k}{k!}=\left(\sum_{k=0}^{\infty}\frac{\pmb{A}^k}{k!}\right)^T=(e^{\pmb{A}})^T
  $$
  

### 2.2 特征值

设 $\pmb{Ax}=\lambda\pmb{x}$ ，则 $\pmb{A}^k\pmb{x}=\lambda^k\pmb{x}$ ，有：

$$
e^{\pmb{A}}\pmb{x}=\left(1+\lambda+\frac{\lambda^2}{2!}+\frac{\lambda^3}{3!}+\cdots\right)\pmb{x}=e^{\lambda}\pmb{x}
$$


令 $n$ 阶矩阵 $\pmb{A}$ 的特征值为 $\lambda_i$ ，对应的特征向量 $\pmb{x}_i$ ，故 $e^{\pmb{A}}$ 特征值为 $e^{\lambda_i}$ ，对应特征向量仍然是 $\pmb{x}_i$ 。

又因为：

行列式：$\det(\pmb{A})=\lambda_1\lambda_2\cdots\lambda_n$

迹：$tr(\pmb{A})=\lambda_1+\lambda_2+\cdots+\lambda_n$

所以：$\det(e^{\pmb{A}})=e^{\lambda_1}e^{\lambda_2}\cdots e^{\lambda_n}=e^{\lambda_1+\lambda_2+\cdots+\lambda_n}=e^{tr(\pmb{A})}$

因为 $e^x\ne0$ ，所以矩阵指数必定可逆。

### 2.3 对角化

若 $\pmb{A}$ 可对角化，$\pmb{A} = \pmb{SDS}^{-1}$ ，则：

$$
\begin{split}e^{\pmb{A}} &= e^{\pmb{SDS}^{-1}}\\&=\pmb{I}+\pmb{SDS}^{-1}+\frac{\pmb{SD^2S}^{-1}}{2!}+\frac{\pmb{SD^3S}^{-1}}{3!}+\cdots\\&=\pmb{S}\left(\pmb{I}+\pmb{D}+\frac{\pmb{D}^2}{2!}+\frac{\pmb{D}^3}{3!}+\cdots\right)\pmb{S}^{-1}\\&=\pmb{S}e^{\pmb{D}}\pmb{S}^{-1}\end{split}
$$


其中，$e^{\pmb{D}}$ 也是对角矩阵：

$$
e^{\pmb{D}}=\begin{bmatrix}e^{\lambda_1}&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&e^{\lambda_n}\end{bmatrix}
$$


### 2.4 应用举例

对于：$e^{\pmb{A}t}=\pmb{I}+t\pmb{A}+\frac{t^2\pmb{A}^2}{2!}+\cdots$

求导数：

$$
\begin{split}\frac{d}{dt}e^{\pmb{A}t}&=\pmb{A}+t\pmb{A}^2+\frac{t^2}{2!}\pmb{A}^3\cdots\\&=\pmb{A}\left(\pmb{I}+t\pmb{A}+\frac{t^2\pmb{A}^2}{2!}+\cdots\right)\\&=\pmb{A}e^{\pmb{A}t}\end{split}
$$


上述结果用于求解微分方程：$\frac{d\pmb{u}}{dt}=\pmb{Au}$ ，令 $\pmb{u}(0)=\pmb{c}$ ，一般解是：$\pmb{u}(t)=e^{\pmb{A}t}\pmb{c}$ 

## 参考文献

[1]. 矩阵运算的基本技巧[DB/OL]. https://ccjou.wordpress.com/2010/10/04/矩陣運算的基本技巧/

[2]. 矩阵指数[DB/OL]. https://ccjou.wordpress.com/2009/08/20/矩陣指數/