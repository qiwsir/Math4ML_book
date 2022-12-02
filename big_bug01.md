# 重要更正第1号：过渡矩阵和坐标变换推导

> 尽管《机器学习数学基础》这本书，耗费了比较长的时间和精力，怎奈学识有限，错误难免。因此，除了在专门的网页（ [勘误和修订](./corrigendum.md) ）中发布勘误和修订内容之外，对于重大错误，我还会以专题的形式发布，并做出更多的相关解释。
>
> 更欢迎有识之士、广大读者朋友，指出其中的错误。非常感谢大家的帮助。

在《机器学习数学基础》第29页到第30页，推导过渡矩阵和坐标变换的时候，原文有一些错误。下面将推导过程重新编写如下，并且增加一些更详细的说明。此说明没有写入原文，是为了协助理解这段推导而作。

针对性的修改，请参阅：[勘误与修订](./corrigendum.md)

----

设 $\{\pmb{\alpha}_1, \cdots, \pmb{\alpha}_n\}$（ $\pmb{\alpha}_i$ 表示列向量） 是某个向量空间的一个基，则该空间中一个向量 $\overrightarrow{OA}$ 可以描述为：

$$
\overrightarrow{OA} = x_1\pmb{\alpha}_1 + \cdots + x_n\pmb{\alpha}_n\tag{1.3.4}
$$
其中的 $(x_1, \cdots, x_n)$ 即为向量 $\overrightarrow{OA}$ 在基 $\{\pmb{\alpha}_1, \cdots, \pmb{\alpha}_n\}$ 的**坐标**。

如果有另外一个基 $\{\pmb{\beta}_1, \cdots, \pmb{\beta}_n\}$（ $\pmb{\beta}_i$ 表示列向量），向量 $\overrightarrow{OA}$ 又描述为：

$$
\overrightarrow{OA} = x_1'\pmb{\beta}_1 + \cdots + x_n'\pmb{\beta}_n\tag{1.3.5}
$$
那么，同一个向量空间的这两个基有没有关系呢？有。不要忘记，基是一个向量组，例如基 $\{\pmb{\beta}_1, \cdots, \pmb{\beta}_n\}$ 中的每个向量也在此向量空间，所以可以用基 $\{\pmb{\alpha}_1, \cdots, \pmb{\alpha}_n\}$ 线性表出，即：

$$
\begin{cases}\begin{split}\pmb{\beta}_1 &= b_{11}\pmb{\alpha}_1 + \cdots + b_{n1}\pmb{\alpha}_n \\ \vdots  \\\pmb{\beta}_n &= b_{1n}\pmb{\alpha}_1 + \cdots + b_{nn}\pmb{\alpha}_n \end{split}\end{cases}
$$
以矩阵（这里提前使用了矩阵的概念，是因为本书已经在前言中声明，不假定读者完全没有学过高等数学。关于矩阵的更详细内容，请参阅第2章）的方式，可以表示为：

$$
\begin{bmatrix}\pmb{\beta}_1&\cdots&\pmb{\beta}_n\end{bmatrix} = \begin{bmatrix}\pmb{\alpha}_1&\cdots&\pmb{\alpha}_n\end{bmatrix}\begin{bmatrix}b_{11} & \cdots & b_{1n}\\\vdots\\b_{n1} & \cdots &b_{nn}\end{bmatrix}\tag{1.3.6}
$$
其中：

$$
\pmb P = \begin{bmatrix}b_{11} & \cdots & b_{1n}\\\vdots\\b_{n1} & \cdots &b_{nn}\end{bmatrix}
$$
称为基 $\{\pmb{\alpha}_1, \cdots, \pmb{\alpha}_n\}$ 向基 $\{\pmb{\beta}_1, \cdots, \pmb{\beta}_n\}$ 的**过渡矩阵**。显然，过渡矩阵实现了一个基向另一个基的变换。

> **定义** 在同一个向量空间，由基 $\{\pmb{\alpha}_1\quad\cdots\quad\pmb{\alpha}_n\}$ 向基 $\{\pmb{\beta}_1\quad\cdots\quad\pmb{\beta}_n\}$ 的过渡矩阵是 $\pmb{P}$ ，则：
>$$
> [\pmb{\beta}_1\quad\cdots\quad\pmb{\beta}_n] = [\pmb{\alpha}_1\quad\cdots\quad\pmb{\alpha}_n]\pmb P
>$$

根据（1.3.5）式，可得：

$$
\begin{split}x_1'\pmb{\beta}_1 + \cdots + x_n'\pmb{\beta}_n &= x_1'b_{11}\pmb{\alpha}_1 + \cdots + x_1'b_{n1}\pmb{\alpha}_n \\ & \quad + \cdots \\ & \quad + x_n'b_{1n}\pmb{\alpha}_1 + \cdots + x_n'b_{nn}\pmb{\alpha}_n \\ &=(x_1'b_{11}+ \cdots + x_n'b_{1n})\pmb{\alpha}_1 \\ & \quad + \cdots \\ &\quad+(x_1'b_{n1} + \cdots + x_n'b_{nn})\pmb{\alpha}_n\end{split}
$$
（1.3.4）式 和（1.3.5）式描述的是同一个向量，所以：

$$
\begin{cases}\begin{split}x_1 &= x_1'b_{11} + \cdots + x_n'b_{1n}\\&\vdots\\x_n &= x_1'b_{n1} + \cdots + x_n'b_{nn}\end{split}\end{cases}
$$
如果写成矩阵形式，即：

$$
\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix} = \begin{bmatrix}b_{11} & \cdots & b_{1n}\\\vdots\\b_{n1} & \cdots &b_{nn}\end{bmatrix}\begin{bmatrix}x_1'\\\vdots\\x_n'\end{bmatrix}\tag{1.3.7}
$$
表示了在同一个向量空间中，向量在不同基下的坐标之间的变换关系，我们称为**坐标变换公式**。

> **定义** 在某个向量空间中，由基 $\{\pmb{\alpha}_1\quad\cdots\quad\pmb{\alpha}_n\}$ 向基 $\{\pmb{\beta}_1\quad\cdots\quad\pmb{\beta}_n\}$ 的过渡矩阵是 $\pmb{P}$ 。某向量在基 $\{\pmb{\alpha}_1\quad\cdots\quad\pmb{\alpha}_n\}$ 的坐标是 $\pmb{x}=\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix} $，在基 $\{\pmb{\beta}_1\quad\cdots\quad\pmb{\beta}_n\}$ 的坐标是 $\pmb x'=\begin{bmatrix}x_1'\\\vdots \\x_n'\end{bmatrix}$，这两组坐标之间的关系是：
> $$
> \pmb x = \pmb P \pmb x'
> $$

----

以上错误，是我在录制《机器学习数学基础》的视频课程时候，讲到了这里，发现的。现在深刻体会到：**教，然后知不足**。教学相长，认真地研究教学，也是自我提升。著名物理学家费恩曼有一种非常好的学习方法，就是将要学的东西，讲给别人听，看看是否能讲明白。

