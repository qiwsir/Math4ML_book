# 不变子空间与特征值

*打开本页，如果没有显示公式，请刷新页面。*

## 不变子空间

设 $$\pmb{A}$$ 为向量空间 $$\mathbb{V}$$ 的一个线性变换，对于子空间 $$\mathbb{X}\subseteq\mathbb{V}$$ ，令 $$\pmb{A}(\mathbb{X})$$ 表示子空间 $$\mathbb{X}$$ 中所有下向量经过 $$\pmb{A}$$ 映射得到的像（image）所成的集合，即：

$$\pmb{A}(\mathbb{X})=\{\pmb{Ax}|\pmb{x}\in\mathbb{X}\} \tag{1.1}$$

向量空间 $$\mathbb{V}$$ 经 $$\pmb{A}$$ 映射所成的集合 $$\pmb{A}(\mathbb{V})\subseteq\mathbb{V}$$ 称为 $$\pmb{A}$$ 的**值域**（range），记作：$$R(\pmb{A})$$ 。

若 $$\pmb{A}$$ 是一个线性变换参考某个基的表示矩阵，值域 $$R(\pmb{A})$$ 即为 $$\pmb{A}$$ 的列空间 $$C(\pmb{A})$$ 。

显然，$$\pmb{A}(\mathbb{X})$$ 也是 $$\mathbb{V}$$ 的子空间。

如果 $$\pmb{A}(\mathbb{X})\subseteq\mathbb{X}$$ ，称 $$\mathbb{X}$$ 是线性变换 $$\pmb{A}$$ 的一个**不变子空间**（invariant subspace）。

因为 $$\pmb{A0}=\pmb{0}$$ ，所以 $$\{\pmb{0}\}$$ 是一个平凡的不变子空间。

例如：

矩阵 $$\pmb{A}=\begin{bmatrix}4&2&1\\-3&-1&-2\\2&2&4\end{bmatrix}$$ ，且向量集 $$\pmb{\beta}=\{\pmb{x}_1,\pmb{x}_2,\pmb{x}_3\}$$ 为 $$\mathbb{R}^3$$ 的一组基：

$$\pmb{x}_1=\begin{bmatrix}1\\-1\\0\end{bmatrix}, \pmb{x}_2=\begin{bmatrix}-1\\2\\-1\end{bmatrix},\pmb{x}_3=\begin{bmatrix}1\\1\\-1\end{bmatrix}$$

令 $$\mathbb{X}=span\{\pmb{x}_1, \pmb{x}_2\}, \mathbb{Y}=span\{\pmb{x}_3\}$$ 。分别计算 $$\pmb{Ax}_i, i=1,2,3$$ ，如下：

$$\begin{split}\pmb{Ax}_1 &= \begin{bmatrix}2\\-2\\0\end{bmatrix}=2\pmb{x}_1\in\mathbb{X}\\\pmb{Ax}_2 &= \begin{bmatrix}-1\\3\\-2\end{bmatrix}=\pmb{x}_1+2\pmb{x}_2\in\mathbb{X}\\\pmb{Ax}_3 &= \begin{bmatrix}5\\-2\\0\end{bmatrix}=-\pmb{x}_1-3\pmb{x}_2+3\pmb{x}_3\notin\mathbb{Y}\end{split}$$

对于任意 $$\pmb{x}=c_1\pmb{x}_1+c_2\pmb{x}_2$$ ，利用上述计算结果，有：

$$\begin{split}\pmb{Ax} &= \pmb{A}(c_1\pmb{x}_1+c_2\pmb{x}_2)\\&=c_1\pmb{Ax}_1+c_2\pmb{Ax}_2\\&=2c_1\pmb{x}_1+c_2\pmb{x}_1+2c_2\pmb{x}_2\\&=(2c_1+c_2)\pmb{x}_1+2c_2\pmb{x}_2\end{split}$$

所以：$$\pmb{Ax}\in\mathbb{X}$$ ，即 $$\mathbb{X}$$ 是一个不变子空间，但 $$\mathbb{Y}$$ 不是。

将上述结果三个式子，可以用矩阵表示：

$$\pmb{A}\begin{bmatrix}\pmb{x}_1&\pmb{x}_2&\pmb{x}_3\end{bmatrix}=\begin{bmatrix}\pmb{x}_1&\pmb{x}_2&\pmb{x}_3\end{bmatrix}\begin{bmatrix}2&1&-1\\0&2&-3\\0&0&3\end{bmatrix}$$

令矩阵 $$\pmb{B}=\begin{bmatrix}\pmb{x}_1&\pmb{x}_2&\pmb{x}_3\end{bmatrix}$$ （为基），则：

$$[\pmb{A}]_{\pmb\beta}=\pmb{B}^{-1}\pmb{A}\pmb{B}=\begin{bmatrix}2&1&-1\\0&2&-3\\0&0&3\end{bmatrix}$$

其中 $$[\pmb{A}]_{\pmb\beta}$$ 是线性变换 $$\pmb{A}$$ 参考基底 $$\pmb{\beta}$$ 的表示矩阵。

如果换一另外一组基：$$\pmb{\beta}'=\{\pmb{x}_1,\pmb{x}_2,\pmb{x}_3'\}$$ ，其中 $$\pmb{x}_3'=\begin{bmatrix}0\\-1\\2\end{bmatrix}$$ 与上述讨论中的 $$\pmb{x}_3$$ 不同，即有 $$\mathbb{Y}'=spn\{\pmb{x}_3'\}$$ ，则：

$$\pmb{Ax}_3'=\begin{bmatrix}0\\-3\\6\end{bmatrix}=3\pmb{x}+3'\in\mathbb{Y}'$$

故 $$\mathbb{Y}'$$ 是一个不变子空间。

线性变换 $$\pmb{A}$$ 参考基底 $$\pmb\beta'$$ 的表示矩阵为：

$$[\pmb{A}]_{\pmb\beta'}=\begin{bmatrix}2&1&0\\0&2&0\\0&0&3\end{bmatrix}$$

是一个分块主对角形式。

考虑一般情况。

假设 $$n$$ 阶方阵 $$\pmb{A}$$ ，设 $$\mathbb{X}_1,\cdots,\mathbb{X}_k$$ 为不相交的不变子空间，即 $$\mathbb{X}_i\cap\mathbb{X}_j=\{\pmb{0}\},i\ne j$$ 。

令 $$r_j=\dim{\mathbb{X}_j}$$ ，满足 $$\sum_{j=1}^kr_j=n$$ 。

各个子空间 $$\mathbb{X}_j$$ 的基向量可以组成 $$\mathbb{R}^n$$ 的一个基底 $$\pmb\beta$$ 。若 $$\pmb{B}$$ 的列向量依序由这些基向量构成，则 $$\pmb{B}$$ 是可逆矩阵，且：

$$[\pmb{A}]_{\pmb\beta}=\pmb{B}^{-1}\pmb{AB}=\begin{bmatrix}\pmb{D}_1&0&\cdots&0\\0&\pmb{D}_2&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&\pmb{D}_k\end{bmatrix}$$

其中 $$\pmb{D}_j$$ 是 $$r_j\times r_j$$ 阶分块矩阵。

如果每个不变子空间的维数都等于 $$1$$ ，即 $$r_1=\cdots=r_n=1$$ ，则：

$$[\pmb{A}]_{\pmb\beta}=\pmb{B}^{-1}\pmb{AB}=\begin{bmatrix}d_1&0&\cdots&0\\0&d_2&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&d_k\end{bmatrix}\tag{1.2}$$

此时称 $$\pmb{A}$$ 为可对角化矩阵。

根据（1.2）式，可以有如下陈述：

设 $$\mathbb{X}$$ 为 $$\pmb{x}\ne\{\pmb{0}\}$$ 张成的子空间，且 $$\pmb{A}(\mathbb{X})\subseteq\mathbb{X}$$ 。若 $$\pmb{Ax}\in\mathbb{X}$$ ，则必有标量 $$\lambda$$ 使得 $$\pmb{Ax}=\lambda\pmb{x}$$ 成立，其中 $$\lambda$$ 称为 $$\pmb{A}$$ 的特征值，$$\pmb{x}$$ 为对应 $$\lambda$$ 的特征向量。



## 参考文献

[1]. [线代启示录：从不变子空间切入特征值](https://ccjou.wordpress.com/2010/06/01/%e5%be%9e%e4%b8%8d%e8%ae%8a%e5%ad%90%e7%a9%ba%e9%96%93%e5%88%87%e5%85%a5%e7%89%b9%e5%be%b5%e5%80%bc%e5%95%8f%e9%a1%8c/)