# 特殊矩阵

## 幂零矩阵

令 $$\pmb{A}$$ 为 $$n\times n$$ 级矩阵，若存在一正整数 $$q$$ 使得：

$$\pmb{A}^{\rm{q}}=0$$

则矩阵 $$\pmb{A}$$ 称为幂零矩阵（nilpotent matrix），意思是幂矩阵为零矩阵，如何此条件的最小正整数 $$q$$​ 称为度数或指数（index）。例如：

$$\pmb{A}=\begin{bmatrix}5&-3&2\\15&-9&6\\10&-6&4\end{bmatrix}$$

$$\pmb{A}^2=\pmb{A}\pmb{A}=\begin{bmatrix}5&-3&2\\15&-9&6\\10&-6&4\end{bmatrix}\begin{bmatrix}5&-3&2\\15&-9&6\\10&-6&4\end{bmatrix}=\begin{bmatrix}0&0&0\\0&0&0\\0&0&0\end{bmatrix}=0$$​​

### 定理 1$$^{[1]}$$​

幂零矩阵的特征值全部为 $$0$$ ，反之，若任一方阵的特征值皆为 $$0$$ ，则该矩阵是幂零矩阵。

或者说：若矩阵 $$\pmb{A}$$ 的一个幂矩阵为零矩阵，则 $$\pmb{A}$$ 的特征值全都是零；若 $$\pmb{A}$$ 至少有一个非零特征值，则 $$\pmb{A}$$​ 的所有幂矩阵都不为零矩阵。

即：

矩阵 $$\pmb{A}$$ 是幂零矩阵 $$\Longleftrightarrow$$​ 矩阵 $$\pmb{A}$$ 的特征值都是 $$0$$

**证明**

【充分性】$$\Longrightarrow$$

设 $$\pmb{A}$$ 是 $$n\times n$$ 幂零矩阵，指数为 $$q$$ ，$$\lambda$$ 为 $$\pmb{A}$$ 的一个特征值，其对应特征向量为 $$\pmb{x}$$ 。则：

$$\pmb{A}^{\rm{q}}\pmb{x}=\pmb{A}^{\rm{q-1}}\pmb{Ax}=\lambda\pmb{A}^{\rm{q-1}}\pmb{x}=\cdots=\lambda^{\rm{q}}\pmb{x}$$

因为 $$\pmb{A}^{\rm{q}}=0$$ ，所以 $$\lambda^{\rm{q}}=0$$ 或 $$\pmb{x}=0$$ 。

又由于特征向量是非零向量，所以：$$\lambda=0$$ 。

【必要性】$$\Longleftarrow$$

设 $$n\times n$$ 矩阵 $$\pmb{A}$$ 的特征值都是 $$0$$ ，则 $$\pmb{A}$$ 的特征多项式为：

$$p_{\pmb{A}}(t)=t^n$$

根据凯莱—哈密顿定理$$^{[2]}$$​ ，$$\pmb{A}^n=0$$ ，故存在最小正整数 $$q\le n$$ 使得 $$\pmb{A}^q=0$$ 。

### 推论1

幂零矩阵不可逆。

**证明**

*方法1*：

因为可逆矩阵的特征值不为 $$0$$ $$^{[3]}$$​，由【定理1】可知，幂零矩阵不可逆。

*方法2*：使用行列式证明。

设密令矩阵的指数 $$q$$ ，使用行列式可乘公式：

$$\det(\pmb{A}^q)=\det(\underbrace{\pmb{A}\cdots\pmb{A}}_{q})=\underbrace{(\det\pmb{A})\cdots(\det\pmb{A})}_{q}=(\det\pmb{A})^q$$

因为 $$\det(\pmb{A}^q)=\det0=0$$ ，所以 $$\det\pmb{A}=0$$ ，就有 $$\operatorname{rank} \pmb{A}\lt n$$ ，则幂零矩阵不可逆。

### 性质1

若 $$\pmb{A}$$ 是幂零矩阵，则 $$\pmb{I}-\pmb{A}$$​ 是可逆矩阵。

**证明**

因为 $$\pmb{A}$$ 是幂零矩阵，所以 $$\pmb{A}^q=0$$ ，则：

$$\pmb{I}=\pmb{I}-\pmb{A}^q$$

根据矩阵多项式的分解：

$$\pmb{I}-\pmb{A}^q=(\pmb{I}-\pmb{A})(\pmb{I}+\pmb{A}+\pmb{A}^2+\cdots+\pmb{A}^{q-1})=\pmb{I}$$

所以：

$$(\pmb{I}-\pmb{A})^{-1}=\pmb{I}+\pmb{A}+\pmb{A}^2+\cdots+\pmb{A}^{q-1}$$​

计算 $$\det(\pmb{I}-\pmb{A})=\det((\pmb{I}-\pmb{A})^{-1})=1\ne0$$ ，（参阅 [3]）

从而 $$\pmb{I}-\pmb{A}$$ 可逆。

### 定理2

对一个 $$n\times n$$​ 的矩阵 $$\pmb{A}$$​ ，使得 $$\pmb{A}^n=0$$​ 的一个充要条件（ $$\Longleftrightarrow$$​ ）为 $$\operatorname{trace}(\pmb{A}^k)=0,k=1,\cdots,n$$​ 。

**证明**

【充分性】$$\Longrightarrow$$

设 $$\pmb{A}$$ 的特征值是 $$\lambda_1,\cdots,\lambda_n$$ 。已知 $$\pmb{A}^n=0$$ ，由【定理1】得到：$$\lambda_1=\cdots=\lambda_n=0$$​ ，所以 $$\pmb{A}^k$$ 的特征值 $$\lambda_i^k$$ 全部是零，从而：

$$\operatorname{trace}(\pmb{A}^k)=\sum_{i=1}^n\lambda_i^k=0, k\ge1$$ 

【必要性】$$\Longleftarrow$$

将已知条件 $$\operatorname{trace}(\pmb{A}^k)=\sum_{i=1}^n\lambda_i^k=0, k\ge1$$ 写成矩阵形式：

$$\begin{bmatrix}1&1&\cdots&1\\\lambda_1&\lambda_2&\cdots&\lambda_n\\\vdots&\vdots&\ddots&\vdots\\\lambda_1^{n-1}&\lambda_2^{n-1}&\cdots&\lambda_n^{n-1}\end{bmatrix}\begin{bmatrix}\lambda_1\\\lambda_2\\\vdots\\\lambda_n\end{bmatrix}=\begin{bmatrix}0\\0\\\vdots\\0\end{bmatrix}$$

上式的系数矩阵为范德蒙矩阵$$^{[4]}$$​ 。

假如所有的 $$\lambda_i$$ 相异，则范德蒙矩阵可逆，故上式仅有零解，$$\lambda_1=\cdots=\lambda_n=0$$​ 。这与刚才的假设矛盾。所以，$$\pmb{A}$$ 有相重的特征值。不妨假设 $$\lambda_1=\lambda_2$$​ ，且其余特征值彼此相异，于是有：

$$\begin{bmatrix}1&1&\cdots&1\\\lambda_2&\lambda_3&\cdots&\lambda_n\\\vdots&\vdots&\ddots&\vdots\\\lambda_2^{n-2}&\lambda_3^{n-2}&\cdots&\lambda_n^{n-2}\end{bmatrix}\begin{bmatrix}2\lambda_2\\\lambda_3\\\vdots\\\lambda_n\end{bmatrix}=\begin{bmatrix}0\\0\\\vdots\\0\end{bmatrix}$$

用上面的方法，仍然可以得知 $$\lambda_2,\lambda_3,\cdots,\lambda_n$$ 中必然含有相重的特征值。

如此持续下去，最终可以归纳所有特征值都相等，即：$$\lambda_1=\cdots=\lambda_n=0$$​​ 。再根据【定理1】可知 $$\pmb{A}$$ 是幂零矩阵，即 $$\pmb{A}^n=0$$​ 。

### 定理3

若 $$\pmb{A}$$ 与 $$\pmb{B}$$ 是同阶幂零矩阵，且 $$\pmb{AB}=\pmb{BA}$$ ，则 $$\pmb{AB}$$ 和 $$\pmb{A}+\pmb{B}$$ 是幂零矩阵。

**证明**

因为 $$\pmb{A}$$ 和 $$\pmb{B}$$ 是幂零矩阵，存在正整数 $$p,q$$ ，使得 $$\pmb{A}^p=\pmb{B}^q=0$$ 。令 $$m=\max\{p,q\}$$ 。因此 $$\pmb{A}^m=\pmb{B}^m=0$$ ，再根据 $$\pmb{AB}=\pmb{BA}$$​ ，得：

$$\begin{split}(\pmb{AB})^m&=(\pmb{ABAB})(\pmb{AB})^{m-2}=\pmb{AABB}(\pmb{AB})^{m-2}=\pmb{A}^2\pmb{B}^2(\pmb{AB})^{m-2}\\&=\cdots\\&=\pmb{A}^m\pmb{B}^m=0\end{split}$$

考虑：

$$(\pmb{A}+\pmb{B})^{2m}=\sum_{k=0}^{2m}\binom{2m}{k}\pmb{A}^k\pmb{B}^{2m-k}$$

对于 $$0\le k\le 2m$$ ，$$\max\{k, 2m-k\}\ge m$$ 致使 $$\pmb{A}^k=0$$ 或 $$\pmb{B}^{2m-k}=0$$ ，故 $$(\pmb{A}+\pmb{B})^{2m}=0$$



## 酉矩阵

**酉矩阵**（unitary matrix），也称为**幺正矩阵**、**么正矩阵**，是一个 $$n\times n$$ 复数矩阵，常用字母 $$\pmb{U}$$ 表示。

酉矩阵满足：$$\pmb{U}^{\ast}\pmb{U}=\pmb{UU}^{\ast}=\pmb{I}_n$$

其中 $$\pmb{U}^{\ast}$$ 是 $$\pmb{U}$$ 的共轭转置。

显然，酉矩阵的逆矩阵，就是它的共轭转置矩阵：$$\pmb{U}^{-1}=\pmb{U}^{\ast}$$

其他性质：

- 酉矩阵的所有特征值，都是绝对值等于 1 的复数，即 $$|\lambda_n|=1$$
- 酉矩阵的行列式的绝对值等于 1，$$|\det(\pmb{U})|=1$$
- 酉矩阵不会改变两个复向量 $$\pmb{x}$$ 和 $$\pmb{y}$$ 的点积：$$(\pmb{Ux})\cdot(\pmb{Uy})\pmb{x}\cdot\pmb{y}$$ ，或者更一般化为内积：$$\langle\pmb{Ux},\pmb{Uy}\rangle=\langle\pmb{x},\pmb{y}\rangle$$​
- 若 $$\pmb{U},\pmb{V}$$ 都是酉矩阵，则 $$\pmb{UV}$$ 也是酉矩阵。
- 对于 $$n\times n$$ 的酉矩阵，以下结论等价：
  - $$\pmb{U}$$​ 是酉矩阵
  - $$\pmb{U}^{\ast}$$ 是酉矩阵
  - $$\pmb{U}$$​ 的列向量是在 $$\mathbb{C}^{\rm{n}}$$​ 上的一组标准正交基
  - $$\pmb{U}$$ 的行向量是在 $$\mathbb{C}^{\rm{n}}$$ 上的一组标准正交基

## 参考文献

[1]. [线代启示录——特殊矩阵（1）：幂零矩阵](https://ccjou.wordpress.com/2009/07/29/%e7%89%b9%e6%ae%8a%e7%9f%a9%e9%99%a3-%e4%b8%80%ef%bc%9a%e5%86%aa%e9%9b%b6%e7%9f%a9%e9%99%a3/)

[2]. [维基百科：凯莱—哈密顿定理](https://zh.wikipedia.org/wiki/%E5%87%B1%E8%90%8A%E2%80%93%E5%93%88%E5%AF%86%E9%A0%93%E5%AE%9A%E7%90%86)

[3]. [可逆矩阵](./invertiblematrix.html)

[4]. [维基百科：范德蒙矩阵](https://zh.wikipedia.org/wiki/%E8%8C%83%E5%BE%B7%E8%92%99%E7%9F%A9%E9%99%A3)

