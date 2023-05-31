# 求解线性方程组的克拉默法则

《机器学习数学基础》中并没有将解线性方程组作为重点，只是在第2章2.4.2节做了比较完整的概述。这是因为，如果用程序求解线性方程组，相对于高等数学教材中强调的手工求解，要简单得多了。

本文是关于线性方程组的拓展，供对此有兴趣的读者阅读。

## 1. 线性方程组的解位于一条直线

不失一般性，这里讨论三维空间的情况，对于多维空间，可以由此外推，毕竟三维空间便于想象和作图说明。

设矩阵 $\pmb{A}=\begin{bmatrix}1&2&4\\1&3&5\end{bmatrix}$ ，线性方程

$$
\begin{bmatrix}1&2&4\\1&3&5\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}0\\0\end{bmatrix} \tag{1.1}
$$


的解是：

$$
\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}0\\0\\0\end{bmatrix},\begin{bmatrix}2\\1\\-1\end{bmatrix},\begin{bmatrix}4\\2\\-2\end{bmatrix},\cdots
$$


可以将上述解写成：

$$
\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\alpha\begin{bmatrix}2\\1\\-1\end{bmatrix} \tag{1.2}
$$


其中 $\alpha$ 为任意数。

很显然，（1.1）式是一条通过坐标系原点的直线。推而广之，可以说 $\pmb{Ax}=\pmb{0}$ 的解集是**一条过原点的直线**（记作：$l_1$ ）。

如果是非齐次线性方程组，例如：

$$
\begin{bmatrix}1&2&4\\1&3&5\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}4\\5\end{bmatrix} \tag{1.3}
$$


解为：

$$
\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}2\\1\\0\end{bmatrix},\begin{bmatrix}0\\0\\1\end{bmatrix},\begin{bmatrix}4\\2\\-1\end{bmatrix},\cdots
$$


这些点的集合是一条不过原点的直线。即 $\pmb{Ax}=\pmb{b}$ 的解集是**一条不过原点的直线**（记作：$l_2$ ）。并且，这条直线与 $\pmb{Ax}=\pmb{0}$ 的解集所在直线平行。对此结论证明如下：

设 $\pmb{u}$ 和 $\pmb{v}$ 是 $\pmb{Ax}=\pmb{b}$ 的两个解，则：

$$
\begin{split}&\pmb{Au}=\pmb{b}\\&\pmb{Av}=\pmb{b}\end{split}
$$


上面二式相减，得：

$$
\pmb{A}(\pmb{u}-\pmb{v})=\pmb{0}
$$


即 $\pmb{u}-\pmb{v}$ 是 $\pmb{Ax}=\pmb{0}$ 的一个解。

$\pmb{u}$ 和 $\pmb{v}$ 是 $\pmb{Ax}=\pmb{b}$ 解集对应的直线上（ $l_2$ ）的两个点，则 $\pmb{u}-\pmb{v}$ 的方向必然在直线 $l_2$ 的方向上（或者在直线 $l_2$ 上，或者在于 $l_2$ 平行的直线上）。

又因为 $\pmb{u}-\pmb{v}$ 也是 $\pmb{Ax}=\pmb{0}$ 的解，所以 $\pmb{u}-\pmb{v}$ 在过原点的直线 $l_1$ 上。

因此，$l_1$ 平行于 $l_2$ ，即 $\pmb{Ax}=\pmb{b}$ 的解集所在直线不过原点，且平行于过原点的 $\pmb{Ax}=\pmb{0}$ 的解集所在直线。



## 2. 克拉默法则

对《机器学习数学基础》第2章2.4.2节中克拉默法则进行证明。

克拉默法则（Cramer's rule）利用行列式计算 $\pmb{Ax}=\pmb{b}$ 的解，其中 $\pmb{A}$ 是 $n\times n$ 方阵。

由于克拉默法则的运行效率不如高斯消元法，所以不能用于大数量方程的线性方程组，通常只用于理论推导$^{[2]}$ ，从这个角度看，**此法则除了具有理论意义之外，在计算上完全可以不用**。

下面的证明来自于参考文献[2]，根据需要做了适当修改。

**克拉默法则**

设 $n$ 阶方阵 $\pmb{A}$ ，$n$ 维向量 $\pmb{b}$ ，将 $\pmb{A}$ 的第 $i$ 列以 $\pmb{b}$ 替换，并记作 $\pmb{A}_i(\pmb{b})$ ，用列向量表示为：

$$
\pmb{A}_i(\pmb{b})=\begin{bmatrix}\pmb{a}_1&\cdots&\pmb{a}_{i-1}&\pmb{b}&\pmb{a}_{i+1}&\cdots&\pmb{a}_n\end{bmatrix}
$$


若 $\pmb{A}$ 可逆，即 $|\pmb{A}|\ne0$ ，则 $\pmb{Ax}=\pmb{b}$ 的解：

$$
\pmb{x_i}=\frac{|\pmb{A}_i(\pmb{b})|}{|\pmb{A}|},(i=1,2,\cdots,n)
$$


**证明**

将原方程 $\pmb{Ax}=\pmb{b}$ 转化为等价的 $\pmb{AX}=\pmb{B}$ ，其中 $\pmb{X},\pmb{B}$ 都是 $n\times n$ 矩阵，将单位矩阵以列向量的形式表示为：$\pmb{I}=\begin{bmatrix}\pmb{e}_1&\cdots&\pmb{e}_n\end{bmatrix}$ 。

以列向量 $\pmb{x}$ 取代 $\pmb{I}$ 的第 $i$ 列，再左乘 $\pmb{A}$ ：

$$
\pmb{AI}_i(\pmb{x})=\pmb{A}\begin{bmatrix}\pmb{e}_1&\cdots&\pmb{x}&\cdots&\pmb{e}_n\end{bmatrix}
$$


参考“[对矩阵乘法深入理解](https://lqlab.readthedocs.io/en/latest/math4ML/linearalgebra/multiplication.html)”中以列为单元进行矩阵乘法，上式可以进一步变换：

$$
\begin{split}\pmb{AI}_i(\pmb{x})&=\begin{bmatrix}\pmb{A}\pmb{e}_1&\cdots&\pmb{A}\pmb{x}&\cdots&\pmb{A}\pmb{e}_n\end{bmatrix}\\&=\begin{bmatrix}\pmb{a}_1&\cdots&\pmb{b}&\cdots&\pmb{a}_n\end{bmatrix}\\&=\pmb{A}_i(\pmb{b})\end{split}
$$


上式即为 $\pmb{AX}=\pmb{B}$ ，其中 $\pmb{X}=\pmb{I}_i(\pmb{x}), \pmb{B}=\pmb{A}_i(\pmb{b})$

利用矩阵乘积的行列式性质，得：

$$
|\pmb{AX}|=|\pmb{A}||\pmb{X}|=|\pmb{A}||\pmb{I}_i(\pmb{x})|=|\pmb{A}_i(\pmb{b})|
$$


以余子式展开计算行列式，得：$|\pmb{I}_i(\pmb{x})|=x_i$ （参阅[3]） ，所以，$|\pmb{A}|x_i=|\pmb{A}_i(\pmb{b})|$ 。

若 $|\pmb{A}|\ne0$ ，则：

$$
x_i=\frac{|\pmb{A}_i(\pmb{b})|}{|\pmb{A}|}
$$


## 3. 存在性与唯一性

矩阵 $\pmb{A}$ 是 $m\times n$ ，对于任意 $m$ 维的非零向量 $\pmb{b}$ ，线性方程组 $\pmb{Ax}=\pmb{b}$ 解的唯一性和存在性讨论$^{[4]}$。

### 存在性

$\pmb{Ax}=\pmb{b}$ 有解，当且仅当 $\pmb{b}^T\pmb{y}=0$ ，其中 $\pmb{y}$ 为满足 $\pmb{A}^T\pmb{y}=\pmb0$ 的任何向量。

或曰：

若 $\pmb{b}$ 正交于左零空间 $N(\pmb{A}^T)$ ，则 $\pmb{Ax}=\pmb{b}$ 有解，反之亦然。

### 唯一性

$\pmb{Ax}=\pmb{b}$ 有唯一解（若解存在），当且仅当 $\pmb{Ax}=\pmb{0}$ 有唯一解 $\pmb{x}=\pmb{0}$ 。

或曰：

若矩阵 $\pmb{A}$ 零空间 $N(\pmb{A})$ 仅含零向量，则 $\pmb{Ax}=\pmb{b}$ 有唯一解，反之亦然。



## 参考文献

[1]. [https://ccjou.wordpress.com/2009/03/20/axb-和-ax0-的解集合有什麼關係？/](https://ccjou.wordpress.com/2009/03/20/axb-%e5%92%8c-ax0-%e7%9a%84%e8%a7%a3%e9%9b%86%e5%90%88%e6%9c%89%e4%bb%80%e9%ba%bc%e9%97%9c%e4%bf%82%ef%bc%9f/)

[2]. [https://ccjou.wordpress.com/2009/11/10/克拉瑪公式的證明/](https://ccjou.wordpress.com/2009/11/10/%E5%85%8B%E6%8B%89%E7%91%AA%E5%85%AC%E5%BC%8F%E7%9A%84%E8%AD%89%E6%98%8E/)

[3]. 对 $|\pmb{I}_i(\pmb{x})|=x_i$ ，以 $4\times4$ 矩阵为例，当 $i=2$ 时：

$$
\begin{vmatrix}1&x_1&0&0\\1&x_2&0&0\\1&x_3&0&0\\1&x_4&0&0\end{vmatrix}=x_2\begin{vmatrix}1&0&0\\0&1&0\\0&)&1\end{vmatrix}=x_1\cdot1=x_2
$$


[4]. [https://ccjou.wordpress.com/2011/06/07/線性方程解的存在性與唯一性/](https://ccjou.wordpress.com/2011/06/07/%e7%b7%9a%e6%80%a7%e6%96%b9%e7%a8%8b%e8%a7%a3%e7%9a%84%e5%ad%98%e5%9c%a8%e6%80%a7%e8%88%87%e5%94%af%e4%b8%80%e6%80%a7/)


