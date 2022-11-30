# 超平面

在探讨线性判别分析$$^{[1],[2]}$$ 的时候，遇到了形如 $$\pmb{w}^\text{T}\pmb{x}+w_0=0$$ 的直线，如果针对多维空间，则是超平面（hyperlane）。

## 超平面的另种定义方式$$^{[3]}$$

### 1. 代数定义

对于三维空间中平面，如果推广到 $$\mathbb{R}^n$$ 空间，即有线性方程组：
$$
\pmb{a}^{\text{T}}\pmb{x}=d\tag{1}
$$
的解所形成的集合（其中 $$\pmb{a}=\begin{bmatrix}a_1\\\vdots\\a_n\end{bmatrix},\pmb{x}=\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix}$$ ，$$d$$  是实数）就构成了超平面，其向量表达式可以写成：
$$
{H}=\{\pmb{x}\in\mathbb{R}^n|\pmb{a}^{\text{T}}\pmb{x}=d\}\tag{2}
$$

### 2. 几何定义

设 $$W$$ 是 $$\mathbb{R}^n$$ 的一个子空间，$$W$$ 自原点平移 $$\pmb{q}$$ 之后所得到的集合 $$S$$ 称为仿射空间$$^{[4]}$$，如下图所示。记作：
$$
S=W+\pmb{q}=\{\pmb{w}+\pmb{q} \mid \pmb{w} \in W\}\tag{3}
$$
 ![](./images/hyperplane01.png)

在 $$\mathbb{R}^n$$ 中，超平面是一个维数等于 $$n-1$$ 的仿射空间，或者说，除了 $$\mathbb{R}^n$$ 本身，超平面是具有最大维数的仿射空间。

以上两个定义具有等价性。



## 参考资料

[1]. [费雪的线性判别分析](./fisher-lda.html)

[2]. [线性判别分析](./bayes-lda.html)

[3]. [超平面](https://ccjou.wordpress.com/2013/05/14/%e8%b6%85%e5%b9%b3%e9%9d%a2/)

[4]. [仿射变换](./affine.html)