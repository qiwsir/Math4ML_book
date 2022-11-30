# 直和与投影

*打开本页后，若未显示公式，请刷新页面，之后即可显示。*

## 容斥定理

### 子空间的和

向量空间 $$\mathbb{V}$$ 的两个特殊子空间：$$\mathbb{O}=\{\pmb{0}\}$$ ，另外一个是 $$\mathbb{V}$$ 自身。

$$\mathbb{O}$$ 是任何向量空间的子空间。

设 $$\mathbb{X}$$ 、$$\mathbb{Y}$$ 分别是 $$\mathbb{V}$$ 的两个子空间，若 $$\mathbb{X}\cap\mathbb{Y}=\mathbb{O}$$ ，则称这两个子空间无交集。

**命题1：** $$\mathbb{X}\cap\mathbb{Y}$$ 是 $$\mathbb{V}$$ 的一个子空间。

**证明**

因为 $$\pmb{0}\in \mathbb{X},\mathbb{Y}$$ ，所以 $$\pmb{0}\in\mathbb{X}\cap\mathbb{Y}$$ 。

设 $$\pmb{x,y}\in\mathbb{X}\cap\mathbb{Y}$$ ，根据子空间性质，可知：$$c\pmb{x}+d\pmb{y}\in\mathbb{X}\cap\mathbb{Y}$$

所以 $$\mathbb{X}\cap\mathbb{Y}$$ 满足向量加法和数量乘法封闭，则为一个子空间。

但要注意 $$\mathbb{X}\cup\mathbb{Y}$$ 不一定是子空间。

**定义** $$\mathbb{X}$$ 与 $$\mathbb{Y}$$ 的子空间的和：

$$\mathbb{X}+\mathbb{Y}\overset{def}{=}span\{\mathbb{X}\cup\mathbb{Y}\}=\{\pmb{x}+\pmb{y}|\pmb{x}\in\mathbb{X}, \pmb{y}\in\mathbb{Y}\}$$

### 容斥定理

子空间的交集 $$\mathbb{X}\cap\mathbb{Y}$$ 与子空间的和 $$\mathbb{X}+\mathbb{Y}$$ 和子空间 $$\mathbb{X}、\mathbb{Y}$$ 的维度关系：

$$\dim\mathbb{X} + \dim\mathbb{Y} = \dim(\mathbb{X}+\mathbb{Y}) + \dim(\mathbb{X}\cap\mathbb{Y}) \tag{1.1}$$

（1.1）式称为**容斥定理**$$^{[1]}$$。

**证明**

设 $$\{\pmb{x}_1,\cdots,\pmb{x}_m\}$$ 为 $$\mathbb{X}$$ 的一组基，$$\{\pmb{y}_1,\cdots,\pmb{y}_n\}$$ 为 $$\mathbb{Y}$$ 的一组基。令矩阵 $$\pmb{A}$$ 的列向量：

$$\pmb{A}=\begin{bmatrix}\pmb{x}_1&\cdots&\pmb{x}_m&\pmb{y}_1&\cdots&\pmb{y}_n\end{bmatrix}$$

显然，子空间的和 $$\mathbb{X}+\mathbb{Y}$$ 即为 $$\pmb{A}$$ 的列空间，所以：

$$\dim(\mathbb{X}+\mathbb{Y})=rank\pmb{A} \tag{1.3}$$

对于 $$\pmb{A}$$ 的零空间 $$N(\pmb{A})$$ 中的向量 $$\pmb{c}$$ ，$$\pmb{Ac}=\pmb{0}$$ ，$$^{[2]}$$ 即：

$$c_1\pmb{x}_1+\cdots+c_m\pmb{x}_m+c_{m+1}\pmb{y}_1+\cdots+c_{m+n}\pmb{y}_n=\pmb{0}$$

则有：

$$\pmb{z}=c_1\pmb{x}_1+\cdots+c_m\pmb{x}_m=-c_{m+1}\pmb{y}_1-\cdots-c_{m+n}\pmb{y}_n \tag{1.2}$$

（1.2）式说明，$$\pmb{z}\in\mathbb{X}$$ 且 $$\pmb{z}\in\mathbb{Y}$$ ，故 $$\pmb{z}\in\mathbb{X}\cap\mathbb{Y}$$ 。

将上述推理反过来，也成立。

所以，子空间的交集 $$\mathbb{X}\cap\mathbb{Y}$$ 是 $$\pmb{A}$$ 的零空间 $$N(\pmb{A})$$ ，于是有：

$$\dim(\mathbb{X}\cap\mathbb{Y})=\dim N(\pmb{A})\tag{1.4}$$ 。

根据“[秩—零化度定理](./basetheory.html)”，可得：

$$m+n=rank\pmb{A}+\dim N(\pmb{A})$$

其中，$$m=\dim\mathbb{X},n=\dim\mathbb{Y}$$ 。

在结合（1.3）和（1.4）式，（1.1）式得证。

## 直和

### 补子空间

**定义**

设 $$\mathbb{X}$$ 是向量空间 $$\mathbb{V}$$ 的子空间，如果 $$\mathbb{X}\cap\mathbb{Y}=\mathbb{O}$$ 且 $$\mathbb{X}+\mathbb{Y}=\mathbb{V}$$ ，则称 $$\mathbb{Y}$$ 是 $$\mathbb{X}$$ 的补子空间（complementary subspace），简称“补空间”。

根据上述定义，可知：

$$\dim(\mathbb{X}\cap\mathbb{Y})=0$$ 

根据（1.1）式可知：

$$\dim\mathbb{X} + \dim\mathbb{Y} = \dim(\mathbb{X}+\mathbb{Y}) \tag{2.1}$$

又因为 $$\mathbb{X}+\mathbb{Y}=\mathbb{V}$$ ，所以：

$$\dim(\mathbb{X}+\mathbb{Y})=\dim\mathbb{V} \tag{2.2}$$

**举例**$$^{[1]}$$

如下图所示，$$\mathbb{P}$$ 是一个过原点的平面，$$\mathbb{L}$$ 是一条过原点的直线，它们构成了 $$\mathbb{R}^3$$ 的子空间，且 $$\mathbb{P}\cap\mathbb{L}=\mathbb{O}$$ 。

![](./images/images/raw/master/2021-3-16/1615873286390-sum.png)

设 $$\pmb{x}\in\mathbb{P},\pmb{y}\in\mathbb{L}$$ ，根据平行四边形法则，可以计算 $$\pmb{x}+\pmb{y}$$ ，则必能充满整个 $$\mathbb{R}^3$$ ，即 $$\mathbb{P}+\mathbb{L}=\mathbb{R}^3$$ 。

由此可知，向量空间 $$\mathbb{R}^3$$ 可由两个不相交的子空间构成。

### 定义

如果 $$\mathbb{Y}$$ 是向量空间 $$\mathbb{V}$$ 的子空间 $$\mathbb{X}$$ 的补子空间，则称 $$\mathbb{V}$$ 是 $$\mathbb{X}$$ 与 $$\mathbb{Y}$$ 的**直和**（direct sum），记作：$$\mathbb{V}=\mathbb{X}\oplus\mathbb{Y}$$ 。

### 性质

**性质1：** $$\mathbb{X}\cap\mathbb{Y}=\mathbb{O}$$ 且 $$\mathbb{X}+\mathbb{Y}=\mathbb{V}$$ 

根据定义可得此性质。

**性质2：** 对于任意 $$\pmb{z}\in\mathbb{V}$$ ，存在唯一向量 $$\pmb{x}\in\mathbb{X},y\in\mathbb{Y}$$ ，使得 $$\pmb{z}=\pmb{x}+\pmb{y}$$

**证明**

根据性质1，得：$$\dim\mathbb{V}=\dim\mathbb{X}+\dim\mathbb{Y}$$ 。

假设对于 $$\pmb{z}\in\mathbb{V}$$ 可以表示为 $$\pmb{z}=\pmb{u}_1+\pmb{v}_1=\pmb{u}_2+\pmb{v}_2$$ ，其中 $$\pmb{u}_1,\pmb{u}_2\in\mathbb{X};\pmb{v}_1,\pmb{v}_2\in\mathbb{X}$$ ，则：

$$\pmb{u}_1-{u}_2=\pmb{v}_2-\pmb{v}_1$$

即 $$\mathbb{u}_1-\mathbb{u}_2\in\mathbb{X}\cap\mathbb{Y}$$ 

又因为 $$\mathbb{X}\cap\mathbb{Y}=\mathbb{O}$$

所以 $$\pmb{u}_1=\pmb{u}_2$$ 且 $$\pmb{v}_1=\pmb{v}_2$$

证毕。

**性质3：** $$\{\pmb{x}_i\}、\{\pmb{y}_i\}$$ 分别是 $$\mathbb{X}$$ 和 $$\mathbb{Y}$$ 的一组基， $$\{\pmb{x}_i\}\cap\{\pmb{y}_i\}=\phi$$ （表示空集合），$$\{\pmb{x}_i\}\cup\{\pmb{y}_i\}=\phi$$ 为 $$\mathbb{V}$$ 的一组基

**证明**

根据 $$\mathbb{X}+\mathbb{Y}=\mathbb{V}$$ ，则 $$\{\pmb{x}_i\}\cup\{\pmb{y}_i\}$$ 生成 $$\mathbb{X}+\mathbb{Y}$$ ，也必定生成 $$\mathbb{V}$$ 。

设：$$\pmb{0}=\sum_{i}c_i\pmb{x}_i+\sum_jd_j\pmb{y}_j$$ 

根据性质2，可得：$$\pmb{0}=\sum_ic_i\pmb{x}_i$$ 且 $$\pmb{0}=\sum_jd_j\pmb{y}_j$$

所以 $$c_i=0,d_j=0$$

故 $$\{\pmb{x}_i\}\cup\{\pmb{y}_i\}$$ 中向量为线性无关，是 $$\mathbb{V}$$ 的一组基。

### 维数关系

$$\dim(\mathbb{X}\oplus\mathbb{Y})=\dim(\mathbb{X})+\dim\mathbb{Y}$$

## 投影$$^{[4]}$$

在《机器学习数学基础》第3章3.4.4节专门介绍了正交投影，此处用直和的概念，将“投影”概念一般化，并不仅仅局限于“正交”的情况。

### 定义

设 $$\mathbb{V}=\mathbb{X}\oplus\mathbb{Y}$$ ，对于 $$\pmb{z}\in\mathbb{V}$$ ，根据性质2，有唯一向量 $$\pmb{x}\in\mathbb{X},y\in\mathbb{Y}$$ ，使得 $$\pmb{z}=\pmb{x}+\pmb{y}$$ 。则称 $$\pmb{x}$$ 为向量 $$\pmb{z}$$ 沿着 $$\mathbb{Y}$$ 至 $$\mathbb{X}$$ 的**投影**，$$\pmb{y}$$ 为向量 $$\pmb{z}$$ 沿着 $$\mathbb{X}$$ 至 $$\mathbb{Y}$$ 的投影。

如果子空间 $$\mathbb{X}$$ 正交于 $$\mathbb{Y}$$ ，则称之为正交投影（orthogonal projection）。

### 投影矩阵

令 $$n$$ 阶方阵 $$\pmb{P}$$ 为投影矩阵$$^{[3]}$$ ，或称为投影算子（projector）。

设 $$\mathbb{V}=\mathbb{X}\oplus\mathbb{Y}$$ ，$$\mathbb{X}$$ 的一组基为 $$\{\pmb{x}_1,\cdots,\pmb{x}_k\}$$ ，$$\mathbb{Y}$$ 的一组基 $$\{\pmb{y}_1,\cdots,\pmb{y}_{n-k}\}$$ 。

令 $$n\times k$$ 阶矩阵 $$\pmb{X}=\begin{bmatrix}\pmb{x}_1&\cdots&\pmb{x}_k\end{bmatrix}$$ ，$$n\times(n-k)$$ 矩阵 $$\pmb{Y}=\begin{bmatrix}\pmb{y}_1&\cdots&\pmb{y}_{n-k}\end{bmatrix}$$ ，$$\pmb{A}=\begin{bmatrix}\pmb{X}&\pmb{Y}\end{bmatrix}$$ 为 $$n$$ 阶可逆矩阵。则沿着 $$\mathbb{Y}$$ 向 $$\mathbb{X}$$ 的投影矩阵 $$\pmb{P}$$ 的计算公式：

$$\pmb{P}=\begin{bmatrix}\pmb{X}&0\end{bmatrix}\pmb{A}^{-1}=\pmb{A}\begin{bmatrix}\pmb{I}_k&0\\0&0\end{bmatrix}\pmb{A}^{-1} \tag{3.1}$$

**命题1：** 在 $$\mathbb{R}^n$$ 中，沿着 $$\mathbb{Y}$$ 至 $$\mathbb{X}$$ 的投影矩阵具有唯一性（沿着其他子空间亦然，此处仅以这两个空间为例）

**证明**

设两个投影矩阵 $$\pmb{P}_1、\pmb{P}_2$$ ，根据（3.1）式，有：

$$\pmb{P}_i\pmb{A}=\pmb{P}_i\begin{bmatrix}\pmb{X}&\pmb{Y}\end{bmatrix}=\begin{bmatrix}\pmb{P}_i\pmb{X}&\pmb{P}_i\pmb{Y}\end{bmatrix}=\begin{bmatrix}\pmb{X}&0\end{bmatrix}$$

当 $$i=1,2$$ 时上式均成立，所以，$$\pmb{P}_1\pmb{A}=\pmb{P}_2\pmb{A}$$

两边右乘 $$\pmb{A}^{-1}$$ 得：$$\pmb{P}_1=\pmb{P}_2$$

### 投影矩阵的判定

$$\pmb{P}$$ 是幂等（idempotent）矩阵，$$\pmb{P}^2=\pmb{P}\quad \Longleftrightarrow\quad$$ 线性变换 $$\pmb{P}$$ 是投影矩阵

**证明**

（1）证明 $$\Longleftarrow$$

设 $$\pmb{P}$$ 是沿着 $$\mathbb{Y}$$ 到 $$\mathbb{X}$$ 的投影矩阵，对于任意向量 $$\pmb{z}\in\mathbb{V}$$ ，设 $$\pmb{x}=\pmb{Pz}$$ 。

计算：$$\pmb{P}^2\pmb{z}=\pmb{P}(\pmb{Pz})=\pmb{Px}=\pmb{x}=\pmb{Pz}$$

因为 $$\pmb{z}$$ 是任意向量，所以 $$\pmb{P}^2=\pmb{P}$$ 。

（2）证明 $$\Longrightarrow$$

对任意向量 $$\pmb{z}\in\mathbb{V}$$ ，有：

$$\pmb{z}=(\pmb{P}+\pmb{I}-\pmb{P})\pmb{z}=\pmb{Pz}+(\pmb{I}-\pmb{P})\pmb{z}$$

其中 $$\pmb{Pz}\in C(\pmb{P})$$

因为 $$\pmb{P}^2=\pmb{P}$$ ，所以 $$\pmb{P}-\pmb{P}^2=\pmb{0}$$ ，即：

$$\pmb{P}(\pmb{I}-\pmb{P})=\pmb{0}$$

$$\pmb{P}(\pmb{I}-\pmb{P})\pmb{z}=\pmb{0}$$

所以，$$(\pmb{I}-\pmb{P})\pmb{z}\in N(\pmb{P})$$

于是有：$$\mathbb{V}=C(\pmb{P})+N(\pmb{P})$$ 。

设任意向量 $$\pmb{w}\in C(\pmb{P})\cap N(\pmb{P})$$ ，则 $$\pmb{w}=\pmb{Pw}$$ 且 $$\pmb{Pw}=\pmb{0}$$

因为 $$\pmb{P}^2=\pmb{P}$$ ，可得 $$\pmb{w}=\pmb{Pw}=\pmb{P}^2\pmb{w}$$

又因为 $$\pmb{0}=\pmb{Pw}$$ ，所以 $$\pmb{w}=\pmb0$$ 。

故：$$C(\pmb{P})\cap N(\pmb{P})=\pmb{O}$$ 。

所以： $$\mathbb{V}=C(\pmb{P})\oplus N(\pmb{P})$$

### 性质

直和是一种分解向量空间的方法。设 $$\mathbb{V}=\mathbb{X}\oplus\mathbb{Y}$$ ，$$\pmb{P}$$ 是沿着 $$\mathbb{Y}$$ 至 $$\mathbb{X}$$ 的投影矩阵，相关性质总结如下：

- $$\pmb{P}^2=\pmb{P}$$ ，$$\pmb{P}$$ 是幂等矩阵
- 若 $$\mathbb{V}=\mathbb{R}^n，\dim\mathbb{X}=k$$ ，则 $$\pmb{P}=\begin{bmatrix}\pmb{X}&\pmb{Y}\end{bmatrix}\begin{bmatrix}\pmb{I}_k&0\\0&0\end{bmatrix}\begin{bmatrix}\pmb{X}&\pmb{Y}\end{bmatrix}^{-1}$$ ，即（3.1）式（符号含义见此式）。
- 根据 $$\pmb{P}$$ 可以计算出子空间：$$\mathbb{X}=C(\pmb{P})=N(\pmb{I}-\pmb{P}),\mathbb{Y}=N(\pmb{P})=C(\pmb{I}-\pmb{P})$$
- $$\pmb{I}-\pmb{P}$$ 是 $$\pmb{P}$$ 的补投影矩阵，$$\pmb{P}(\pmb{I}-\pmb{P})=(\pmb{I}-\pmb{P})\pmb{P}=0$$ 。$$\pmb{P}$$ 是沿着子空间 $$N(\pmb{P})=C(\pmb{I}-\pmb{P})$$ 至 $$C(\pmb{P})=N(\pmb{I}-\pmb{P})$$ 的投影矩阵；$$\pmb{I}-\pmb{P}$$ 是沿着子空间 $$N(\pmb{I}-\pmb{P})=C(\pmb{P})$$ 至 $$C(\pmb{I}-\pmb{P})=N(\pmb{P})$$ 的投影矩阵。



## 参考文献

[1]. [https://ccjou.wordpress.com/2010/03/31/互補子空間與直和/](https://ccjou.wordpress.com/2010/03/31/%e4%ba%92%e8%a3%9c%e5%ad%90%e7%a9%ba%e9%96%93%e8%88%87%e7%9b%b4%e5%92%8c/)

[2]. [零空间](./rank.html)

[3]. 关于投影矩阵的详细阐述，请参阅《机器学习数学基础》的第3章3.4.4节的详细内容。

[4]. [https://ccjou.wordpress.com/2010/04/06/直和與投影/](https://ccjou.wordpress.com/2010/04/06/%E7%9B%B4%E5%92%8C%E8%88%87%E6%8A%95%E5%BD%B1/)

