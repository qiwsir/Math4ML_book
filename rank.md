# 矩阵的秩

*打开本页面后，如果不显示公式，请刷新。*

《机器学习数学基础》第2章2.5节“矩阵的秩”，介绍了矩阵的基本概念、性质以及如何用程序计算矩阵的秩。

## 零空间

在数学中，一个算子 $$\pmb{A}$$ 的零空间是方程 $$\pmb{Av}=0$$ 的所有解 $$\pmb{v}$$ 的集合。它也叫做 $$\pmb{A}$$ 的核或核空间。用集合建造符号表示为

$$Null(\pmb{A})=\{\pmb{v}\in\mathbb{V}:\pmb{Av}=\pmb{0}\}$$

如果算子是在向量空间上的线性算子，零空间就是线性子空间。因此零空间是向量空间。 

矩阵 $$\pmb{A}$$ 的零空间就是所有向量的空间的线性子空间。这个线性子空间的维度叫做 $$\pmb{A}$$ 的**零化度**（nullity），其值为矩阵 $$\pmb{A}$$ 的行阶梯形矩阵中不包含支点的纵列数$$^{[1]}$$。

例如矩阵 $$\pmb{A}=\begin{bmatrix}-2&-4&4\\2&-8&0\\8&4&-12\end{bmatrix}$$ ，首先将 $$\pmb{A}$$ 变换为简化行阶梯形矩阵：$$\pmb{E}=\begin{bmatrix}1&0&-4/3\\0&1&-1/3\\0&0&0\end{bmatrix}$$

对所有向量 $$\pmb{v}$$ 有 $$\pmb{Av}=0$$ ，等同于 $$\pmb{Ev}=0$$ ，即：

 $$\begin{bmatrix}1&0&-4/3\\0&1&-1/3\\0&0&0\end{bmatrix}\begin{bmatrix}x\\y\\z\end{bmatrix}$$

解得：$$\begin{cases}x=\frac{4z}{3}\\y=\frac{z}{3}\\0=0\end{cases}$$ ，即 $$\begin{cases}x=\frac{4s}{3}\\y=\frac{s}{3}\\z=s\end{cases}$$

所以，$$\pmb{A}$$ 的零空间是 $$\pmb{v}=\begin{bmatrix}4s/3\\s/3\\s\end{bmatrix}$$

## 行秩等于列秩

《机器学习数学基础》第2章2.5节“矩阵的秩”，定义矩阵的秩时，有结论：矩阵的行秩等于列秩，即为矩阵的秩。下面对“行秩等于列秩”结论给予证明，参考文献[2]。

设 $$m\times n$$ 矩阵 $$\pmb{A}$$ 的列秩为 $$c$$ ，行秩为 $$r$$ 。

由此假设，则矩阵 $$\pmb{A}$$ 中有 $$c$$ 个线性无关的列向量，这些列向量生成了 $$\pmb{A}$$ 的列空间。将这些列向量组成 $$m\times c$$ 的矩阵 $$\pmb{B}=[b_{ij}]$$ 。

设 $$\pmb{A}$$ 的列向量为 $$\pmb{a}_j,(j=1,\cdots,n)$$ ，$$\pmb{B}$$ 的列向量为 $$\pmb{b}_i,(i=1,\cdots,c)$$ ，则 $$\pmb{a}_j$$ 都可以用 $$\pmb{B}$$ 的列向量的线性组合唯一表示：

$$\begin{split}\pmb{a}_j &= d_{1j}\pmb{b}_1+d_{2j}\pmb{b}_2+\cdots+d_{cj}\pmb{b}_c\\&=\begin{bmatrix}\pmb{b}_1&\pmb{b}_2&\cdots&\pmb{b}_c\end{bmatrix}\begin{bmatrix}d_{1j}\\d_{2j}\\\vdots\\d_{cj}\end{bmatrix}\\&=\pmb{Bd}_j\end{split}$$

如果将 $$\pmb{d}_j,(j=1,\cdots,n)$$ 写成矩阵，即 $$\pmb{D}=\begin{bmatrix}d_{11}&\cdots&d_{1n}\\\vdots&\ddots&\vdots\\d_{c1}&\cdots&d_{cn}\end{bmatrix}$$ 是 $$c\times n$$ 的矩阵，其行向量数即为 $$\pmb{A}$$ 的列空间维数。

则矩阵 $$\pmb{A}$$ 以列向量的方式，可以写成：

$$\begin{split}\pmb{A}&=\begin{bmatrix}\pmb{a}_1&\pmb{a}_2&\cdots&\pmb{a}_n\end{bmatrix}\\&=\begin{bmatrix}\pmb{Bd}_1&\pmb{Bd}_2&\cdots&\pmb{Bd}_n\end{bmatrix}\\&=\pmb{B}\begin{bmatrix}\pmb{d}_1&\pmb{d}_2&\cdots&\pmb{d}_n\end{bmatrix}\\&=\pmb{BD}\end{split}$$

以 $$row_i(\pmb{A})$$ 表示 $$\pmb{A}$$ 的第 $$i$$ 行，根据“[以行为单元的矩阵乘法](./multiplication.html)”规则，可得：

$$\begin{split}row_i(\pmb{A})&=row_i(\pmb{BD})=row_i(\pmb{B})\cdot\pmb{D}\\&=\begin{bmatrix}b_{i1}&b_{i2}&\cdots&b_{ic}\end{bmatrix}\begin{bmatrix}row_1(\pmb{D})\\row_2(\pmb{D})\\\vdots\\row_c(\pmb{D})\end{bmatrix}\\&=b_{i1}row_1(\pmb{D})+b_{i2}row_2(\pmb{D})+\cdots+b_{ic}row_c(\pmb{D})\end{split}$$

$$\pmb{A}$$ 的每一行都可以表示为 $$\pmb{D}$$ 的行向量的线性组合，因此 $$\pmb{A}$$ 的行空间维数不大于 $$\pmb{D}$$ 的行向量数，即 $$r\le{c}$$ ，即：

**$$\pmb{A}$$ 的行空间维数不大于 $$\pmb{A}$$ 的列空间维数**。

同样的方法，可以得到 $$\pmb{A}^T$$ 的行空间维数不大于 $$\pmb{A}^T$$ 的列空间维数。

又因为 $$\pmb{A}^T$$ 的行空间即为 $$\pmb{A}$$ 的列空间， $$\pmb{A}^T$$ 的列空间即为 $$\pmb{A}$$ 的行空间，所以：**$$\pmb{A}$$ 的列空间维数不大于 $$\pmb{A}$$ 的行空间维数**，即 $$c\le{r}$$ 。

故：$$r=c$$ 。矩阵的行秩等于列秩。

## 有关概念

矩阵的秩表示了矩阵的“真实尺寸”，即最大的线性无关的列（行）向量的集合所包含的向量数量，通过这些向量，能够生成相应的列空间或者行空间。

如果将一个矩阵化为梯形矩阵，矩阵的秩是：

- 梯形矩阵所含主元的数量
- 非零行的数量
- 主元所在的列向量的数量

如果从矩阵的列向量的线性无关角度阐述矩阵的秩，则：

- 是矩阵最大线性无关列向量的个数
- 是矩阵最大线性无关行向量的个数

如果从空间维度的角度阐述，则：

- 矩阵的秩等于列空间的维度，$$rank\pmb{A}=\dim C(\pmb{A})$$
- 矩阵的值等于行空间的维数，$$rank\pmb{A}=\dim C(\pmb{A}^T)$$

## 有关性质

关于秩的一些等式或者不等式，常用于机器学习、数据挖掘原理的证明，下面列出一些，供使用参考$$^{[3]}$$。

### 性质1

设 $$m\times n$$ 矩阵 $$\pmb{A}$$ ，则：

$$rank\pmb{A}=rank\pmb{A}^T \tag{3.1}$$

**证明**

$$rank\pmb{A}$$ 是矩阵 $$\pmb{A}$$ 的列秩，$$rank\pmb{A}^T$$ 是行秩，根据前述“行秩等于列秩”可知，上述等式成立。

### 性质2

设 $$m\times m$$ 矩阵 $$\pmb{A}$$ 可逆，$$n\times n$$ 矩阵 $$\pmb{C}$$ 可逆，矩阵 $$\pmb{B}$$ 是 $$m\times n$$ 。

$$rank(\pmb{AB})=rank\pmb{B}=rank(\pmb{BC})=rank(\pmb{ABC}) \tag{3.2}$$

等式（3.2）说明一个矩阵（如 $$\pmb{B}$$ ）左乘或者右乘可逆矩阵，它的秩不变。

**证明**（方法1）

假设 $$\pmb{Bx}=0$$ ，则 $$\pmb{ABx}=\pmb{A0}=\pmb{0}$$ ，

又若 $$\pmb{ABx}=\pmb{0}$$ ，等号两边同时左乘 $$\pmb{A}^{-1}$$ ，得：

$$\pmb{A}^{-1}\pmb{ABx}=\pmb{Bx}=\pmb{0}$$ 

所以 $$N(\pmb{AB})=N(\pmb{B})$$ ，$$\pmb{AB}$$ 与 $$\pmb{B}$$ 有相同的零空间。

根据“[秩—零化度定理](./basetheory.html)”，可得：

$$rank(\pmb{AB}) = n-\dim N(\pmb{AB})$$

$$rank\pmb{B}=n-\dim N(\pmb{B})$$

所以 $$rank(\pmb{AB})=rank\pmb{B}$$ 。

并利用（3.1）式，可知：

$$rank(\pmb{ABC})=rank(\pmb{BC})=rank(\pmb{BC})^T=rank(\pmb{C}^T\pmb{B}^T)=rank\pmb{B}^T=rank\pmb{B}$$

**证明**（方法2）

对于矩阵 $$\pmb{A}$$ 和 $$\pmb{B}$$ ，根据性质7，可得：$$rank(\pmb{AB})\le rank\pmb{B}$$ ，

又 $$rank\pmb{B}=rank(\pmb{A}^{-1}\pmb{AB})\le rank(\pmb{AB})$$

所以 $$rank(\pmb{AB})=rank\pmb{B}$$

证毕。

### 性质3

设 $$\pmb{A}、\pmb{B}$$ 为 $$m\times n$$ 矩阵，$$\pmb{X}$$ 为 $$m$$ 阶可逆方阵，$$\pmb{Y}$$ 为 $$n$$ 阶可逆方阵，若 $$\pmb{A}=\pmb{XBY}$$ ，则：$$rank\pmb{A}=rank\pmb{B}$$ 。

**证明**

因为 $$\pmb{X}、\pmb{Y}$$ 可逆，且 $$\pmb{A}=\pmb{XBY}$$ ，根据（3.2）式可得：

$$rank\pmb{A}=rank(\pmb{XBY})=rank\pmb{B}$$

性质3的逆也成立：

设 $$\pmb{A}$$ 为 $$m\times n$$ 矩阵，且 $$rank\pmb{A}=r$$ ，则 $$\pmb{A}$$ 可分解为 $$\pmb{A}=\pmb{XBY}$$ ，其中 $$\pmb{X}$$ 是 $$m\times r$$ 矩阵，$$\pmb{Y}$$ 是 $$r\times n$$ 矩阵， $$\pmb{B}$$ 是 $$r$$ 阶可逆方阵。

### 性质4

设 $$m\times n$$ 矩阵 $$\pmb{A}$$ ，则有：

$$rank(\pmb{A}^T\pmb{A})=rank\pmb{A} \tag{3.4}$$

**证明**

对于任意 $$\pmb{x}\in N(\pmb{A})$$ ，有 $$\pmb{Ax}=\pmb{0}$$ ，两边左乘 $$\pmb{A}^T$$ ，得：

$$\pmb{A}^T\pmb{Ax}=\pmb{0}$$

所以，$$\pmb{x}\in N(\pmb{A}^T\pmb{A})$$ 。

由此可得 $$N(\pmb{A})\subseteq N(\pmb{A}^T\pmb{A})$$ 。

又若有 $$\pmb{A}^T\pmb{Ax}=0$$ ，左乘 $$\pmb{x}^T$$ ，得 $$\pmb{x}^T\pmb{A}^T\pmb{Ax}=(\pmb{Ax})^T\pmb{Ax}=\begin{Vmatrix}\pmb{Ax}\end{Vmatrix}^2=\pmb{0}$$

所以 $$\pmb{Ax}=\pmb{0}$$

即 $$N(\pmb{A}^T\pmb{A})\subseteq N(\pmb{A})$$

故，最终得到 $$N(\pmb{A})= N(\pmb{A}^T\pmb{A})$$

根据“[矩阵子空间的正交补关系](./basetheory.html)”，有 $$C(\pmb{A}^T)=N(\pmb{A})^{\bot}$$ ，$$C(\pmb{A}^T\pmb{A})=N(\pmb{A}^T\pmb{A})^{\bot}$$

所以 $$C(\pmb{A}^T)=C(\pmb{A}^T\pmb{A})$$

因此，$$rank\pmb{A}=\dim C(\pmb{A}^T)=\dim C(\pmb{A}^T\pmb{A})=rank(\pmb{A}^T\pmb{A})$$

### 性质5

设 $$\pmb{A}$$ 为 $$m\times s$$ 矩阵，$$\pmb{B}$$ 为 $$n\times p$$ 矩阵，则：

$$rank(\pmb{AB})=rank\pmb{B}-\dim(N(\pmb{A})\cap C(\pmb{B}))$$

**说明：**将性质5与性质2注意区分，在性质2中，矩阵 $$\pmb{A}$$ 明确说明，是可逆的。在性质5中，并没有说明矩阵 $$\pmb{A}$$ 是否可逆。如果可逆，则 $$N(\pmb{A})\cap C(\pmb{B})=\{\pmb{0}\}$$ ，退回到性质2。

对性质5可以这样理解：$$\pmb{AB}$$ 视为 $$\pmb{B}$$ 的列向量与 $$\pmb{A}$$ 相乘：$$\pmb{AB}=\pmb{A}\begin{bmatrix}\pmb{b}_1&\cdots&\pmb{b}_p\end{bmatrix}=\begin{bmatrix}\pmb{Ab}_1&\cdots&\pmb{Ab}_p\end{bmatrix}$$

若 $$\pmb{Ab}_i=\pmb{0}$$ ，即 $$\pmb{b}_i\in N(\pmb{A})$$ ，这样就会使 $$\pmb{Ab}_i,(i=1,\cdots,p)$$ 中线性无关的向量数建设，即维度（或秩）比 $$\pmb{B}$$ 减少。

**证明**（方法1）

因为 $$N(\pmb{A})$$ 和 $$C(\pmb{B})$$ 都是 $$\mathbb{R}^n$$ 的子空间，设 $$\dim(N(\pmb{A})\cap C(\pmb{B})) = s$$ 且 $$\pmb{\Theta} = \{\pmb{x}_1,\cdots,\pmb{x}_s\}$$ 是 $$N(\pmb{A})\cap C(\pmb{B})$$ 的一组基。

由于 $$N(\pmb{A})\cap C(\pmb{B})\subseteq C(\pmb{B})$$ ，设 $$\dim C(\pmb{B}) = s+t$$ ，于是将 $$\pmb{\Theta}$$ 与 $$t$$ 个列向量一起构成了 $$C(\pmb{B})$$ 的基：$$\pmb{\Beta} = {\pmb{x}_1,\cdots,\pmb{x}_s,\pmb{y}_1,\cdots,\pmb{y}_t}$$ 。

显然 $$rank\pmb{B}=\dim C(\pmb{B}) = s+t$$ 。

接下来要证明 $$\pmb{AB}$$ 的秩是 $$t$$ ，即证明

 $$rank(\pmb{AB})\dim(\pmb{AB})=\dim(C(\pmb{AB}))=t \tag{5.1}$$ 

根据本证明中第一句的假设，可以进一步表示：$$\pmb{x}_i\in N(\pmb{A})$$ ，即 $$\pmb{Ax}_i=\pmb{0},(i=1,\cdots,s)$$ 

对于 $$C(\pmb{AB})$$ 内的任意向量 $$\pmb{b}$$ ，存在向量 $$\pmb{z}$$ ，使 $$\pmb{b}=\pmb{ABz}$$ 成立。由于 $$\pmb{Bz}\in C(\pmb{B})$$ ，则：

$$\pmb{Bz}=c_1\pmb{x}_1+\cdots+c_s\pmb{x}_s+d_1\pmb{y}_1+\cdots+d_t\pmb{y}_t$$

将上式代入 $$\pmb{b}=\pmb{ABz}$$ ，得：

$$\begin{split}\pmb{b}&=\pmb{A}(c_1\pmb{x}_1+\cdots+c_s\pmb{x}_s+d_1\pmb{y}_1+\cdots+d_t\pmb{y}_t)\\&=c_1\pmb{Ax}_1+\cdots+c_s\pmb{Ax}_s+d_1\pmb{Ay}_1+\cdots+d_t\pmb{Ay}_t\\&=d_1\pmb{Ay}_1+\cdots+d_t\pmb{Ay}_t\quad(\because\pmb{Ax}_i=\pmb{0})\end{split}$$  

将 $$\pmb{Ay}_i,(i=1,\cdots,t)$$ 的向量集记作 $$\pmb\Sigma=\{\pmb{Ay}_1,\cdots,\pmb{Ay}_t\}$$ ，则以此向量集为基向量，可生成 $$\pmb{AB}$$ 列空间 ，即 $$C(\pmb{AB})=span\pmb{\Sigma}$$ 。同时也说明 $$\pmb{\Sigma}$$ 也属于 $$\pmb{A}$$ 的列空间，则 $$C(\pmb{AB})\subseteq C(\pmb{A})$$ 。

若：

$$\pmb{0}=d_1\pmb{Ay}_1+\cdots+d_t\pmb{Ay}_t=\pmb{A}(d_1\pmb{y}_1+\cdots+d_t\pmb{y}_t)$$

表明 $$d_1\pmb{y}_1+\cdots+d_t\pmb{y}_t$$ 属于 $$N(\pmb{A})\cap C(\pmb{B})$$ ，则一定存在一组数 $$f_1,\cdots,f_s$$ ，使得：

$$d_1\pmb{y}_1+\cdots+d_t\pmb{y}_t=f_1\pmb{x}_1+\cdots+f_s\pmb{x}_s$$

即：

$$-f_1\pmb{x}_1-\cdots-f_x\pmb{x}_s+d_1\pmb{y}_1+\cdots+d_t\pmb{y}_t=0$$

上式为 $$\pmb{\Beta}$$ 的线性组合，则 $$f_1=\cdots=f_s=d_1=\cdots=d_t=0$$ ，从而说明 $$\pmb\Sigma=\{\pmb{Ay}_1,\cdots,\pmb{Ay}_t\}$$ 各向量线性无关 ，由此基生成 $$\pmb{AB}$$ 列空间，其空间维度为 $$t$$ （5.1）式得证。

**证明**（方法2）$$^{[4]}$$

$$m\times n$$ 的矩阵 $$A$$ 可以看做是线性变换 $$\pmb{A}:\mathbb{R}^n\to\mathbb{R}^m$$ ，其中列空间为值域，即 $$C(\pmb{A})=ran(\pmb{A})$$ ，零空间为 $$N(\pmb{A})=\ker(\pmb{A})$$ 。

两个矩阵相乘 $$\pmb{AB}$$ 可以看做 $$\pmb{A}$$ 对矩阵 $$\pmb{B}$$ 的列空间 $$C({B})$$ 进行变换，记作：$$\pmb{A}_{/C(\pmb{B})}$$ ，此变换所对应的值域为 $$\pmb{AB}$$的列空间，即 $$ran(\pmb{A}_{/C(\pmb{B})})=C(\pmb{AB})$$ 。

用向量集合关系表示：

$$ran(\pmb{A}_{/C(\pmb{B})})=\{\pmb{Ay}|\pmb{y}\in C(\pmb{B})\}=\{\pmb{ABx}|\pmb{x}\in\mathbb{R}^p\}=C(\pmb{AB})$$

并且，$$\ker(\pmb{A}_{/C(\pmb{B})})=N(\pmb{A})\cap C(\pmb{B})$$ 。

根据“[秩—零化度定理](./basetheory.html)”得：

$$\begin{split}\dim C(B) &= \dim\ker(\pmb{A}_{/C(\pmb{B})})+\dim ran(\pmb{A}_{/C(\pmb{B})})\\&=\dim(N(\pmb{A})\cap C(\pmb{B}))+\dim C(\pmb{AB})\end{split}$$

证毕。

### 性质6

$$m\times n$$ 矩阵 $$\pmb{A}$$ ，$$rank\pmb{A}\le\min\{m,n\}$$

**证明**

根据矩阵秩的定义：矩阵的秩等于线性无关的列或者行向量综述，所以，秩不大于列或行的数量。

### 性质7

$$m\times n$$ 的矩阵 $$\pmb{A}$$ 和 $$n\times p$$ 的矩阵 $$\pmb{B}$$ ，

$$rank\pmb{A}+rank\pmb{B}-n\le rank(\pmb{AB})\le\min\{rank\pmb{A},rank\pmb{B}\}$$

**证明**

根据性质5：

$$rank(\pmb{AB})=rank\pmb{B}-\dim(N(\pmb{A})\cap C(\pmb{B}))\le rank\pmb{B}$$

根据性质1：

$$rank(\pmb{AB})=rank(\pmb{AB})^T=rank(\pmb{B}^T\pmb{A}^T)\le rank\pmb{A}^T=rank\pmb{A}$$

因为 $$N(\pmb{A})\cap C(\pmb{B})\subseteq N(\pmb{A})$$ ，根据“[秩—零化度定理](./basetheory.html)”，得：

$$\dim(N(\pmb{A})\cap C(\pmb{B}))\le \dim N(\pmb{A})=n-rank\pmb{A}$$

根据性质5，得：

$$rank\pmb{AB}=rank\pmb{B}-\dim(N(\pmb{A})\cap C(\pmb{B}))\ge rank\pmb{B}+rank\pmb{A}-n$$

对性质7的左侧不等式的另外一种证明：

根据“[秩—零化度定理](./basetheory.html)”，$$\dim N(\pmb{A})=n-rank\pmb{A}$$ ，所以，如果 $$rank\pmb{B}-rank(\pmb{AB})\le\dim N(\pmb{A})$$ 成立，则原不等式成立。

设线性变换 $$\pmb{T}:C(\pmb{B})\to \mathbb{R}^m$$ ，对于 $$\pmb{x}\in C(\pmb{B})$$ ，有 $$\pmb{T}(\pmb{x})=\pmb{Ax}$$ ，或者 $$\pmb{T}(\pmb{y})=\pmb{ABy}$$ ，其中 $$\pmb{y}\in\mathbb{R}^p$$ 。则：

$$\dim ima(\pmb{T})+\dim\ker(\pmb{T})=\dim C(\pmb{B})$$

而 $$\dim ima(\pmb{T})=\dim C(\pmb{AB})=rank(\pmb{AB})$$ ，且 $$\dim C(\pmb{B})=rank\pmb{B}$$ ，则：

$$\dim\ker(\pmb{T})=rank\pmb{B}-rank(\pmb{AB})$$

由于 $$C(\pmb{B})\subseteq\mathbb{R}^n$$ ，线性变换 $$\pmb{T}$$ 的核为 $$\pmb{A}$$ 的零空间 $$N(\pmb{A})$$ 的子空间，故 $$\dim\ker(\pmb{T})\le\dim N(\pmb{A})$$ 。

### 性质8

$$m\times n$$ 的矩阵 $$\pmb{A}$$ 和 $$\pmb{B}$$ ，有：

$$rank(\pmb{A}+\pmb{B})\le rank\pmb{A}+rank\pmb{B}$$

**证明**$$^{[5]}$$

设 $$\mathbb{U}$$ 和 $$\mathbb{W}$$ 是向量空间 $$\mathbb{V}$$ 的两个子空间，令 $$\pmb{u}\in\mathbb{U},\pmb{w}\in\mathbb{W}$$ ，则 $$\pmb{u}+\pmb{w}$$ 也构成了 $$\mathbb{V}$$ 的一个子空间，这个子空间记作 $$\mathbb{U+W}$$ ，并令 $$\pmb{v}\in\mathbb{U+W}$$ ，则：

$$\pmb{v}=c\pmb{u}+d\pmb{w}$$

即 $$\mathbb{U+W}$$ 可由子空间 $$\mathbb{U}$$ 和 $$\mathbb{W}$$ 的并集生成，

$$\mathbb{U+W}=span(\mathbb{U}\cup\mathbb{W})$$

设 $$\mathbb{U}$$ 的一组基 $$\{\pmb{u}_1,\cdots,\pmb{u}_m\}$$ ， $$\mathbb{W}$$ 的一组基 $$\{\pmb{w}_1,\cdots,\pmb{w}_n\}$$ ，则 $$\mathbb{U+W}$$ 的一组基

$$\{\pmb{u}_1,\cdots,\pmb{u}_m,\pmb{w}_1,\cdots,\pmb{w}_n\}$$

可知，$$\mathbb{U+W}$$ 的维数不大于 $$m+n$$ ，即：

$$\dim(\mathbb{U+W})\le\dim\mathbb{U}+\dim\mathbb{W}\tag{8.1}$$

根据上述理解，对于矩阵 $$\pmb{A}$$ 和 $$\pmb{B}$$ ，它们的列空间之和 $$C(\pmb{A})+C(\pmb{B})$$ 包含所有的 $$\pmb{Ax} + \pmb{By}$$ ，其中 $$\pmb{x}、\pmb{y}$$ 是任意向量。显然：

$$C(\pmb{A+B})\subseteq C(\pmb{A})+C(\pmb{B})$$

子空间的维数等于基向量的数量，所以：

$$rank(\pmb{A+B})\le\dim(C(\pmb{A})+C(\pmb{B}))$$

由前述对 $$\mathbb{U+W}$$ 维数的讨论结果（8.1）式可知：

$$\dim(C(\pmb{A})+C(\pmb{B}))\le\dim C(\pmb{A})+\dim C(\pmb{B})$$

所以：

$$rank(\pmb{A+B})\le rank\pmb{A}+rank\pmb{B}$$

### 性质9

设矩阵 $$\pmb{A}$$ 为 $$m\times n$$ ，$$\pmb{B}$$ 为 $$n\times p$$ ，$$\pmb{C}$$ 为 $$p\times q$$ ，

$$rank(\pmb{AB})+rank(\pmb{BC})\le rank\pmb{B} + rank(\pmb{ABC})$$

**证明**

因为 $$C(\pmb{BC})\subseteq C(\pmb{B})$$ ，则：$$N(\pmb{A})\cap C(\pmb{BC})\subseteq N(\pmb{A})\cap C(\pmb{B})$$ ，得：

$$\dim(N(\pmb{A})\cap C(\pmb{BC}))\le \dim(N(\pmb{A})\cap C(\pmb{B}))$$

根据性质5：$$\dim(N(\pmb{A})\cap C(\pmb{B}))=rank\pmb{B}-rank(\pmb{AB})$$

以 $$\pmb{BC}$$ 取代 $$\pmb{B}$$ ：

$$\dim(N(\pmb{A})\cap C(\pmb{BC}))=rank(\pmb{BC})-rank(\pmb{ABC})$$

 将后面的两个等式中结论代入到前面的不等式：

$$rank(\pmb{BC})-rank(\pmb{ABC})\le rank\pmb{B}-rank(\pmb{AB})$$

本性质得证。



## 参考文献

[1]. [https://zh.wikipedia.org/wiki/零空间](https://zh.wikipedia.org/wiki/%E9%9B%B6%E7%A9%BA%E9%97%B4)

[2]. [https://ccjou.wordpress.com/2009/11/13/行秩列秩](https://ccjou.wordpress.com/2009/11/13/%e8%a1%8c%e7%a7%a9%e5%88%97%e7%a7%a9/)

[3]. [https://ccjou.wordpress.com/2010/01/14/破解矩陣秩的等式與不等式證明/](https://ccjou.wordpress.com/2010/01/14/%e7%a0%b4%e8%a7%a3%e7%9f%a9%e9%99%a3%e7%a7%a9%e7%9a%84%e7%ad%89%e5%bc%8f%e8%88%87%e4%b8%8d%e7%ad%89%e5%bc%8f%e8%ad%89%e6%98%8e/)

[4]. [https://ccjou.wordpress.com/2014/02/17/運用輸入輸出模型活化秩─零度定理/](https://ccjou.wordpress.com/2014/02/17/%E9%81%8B%E7%94%A8%E8%BC%B8%E5%85%A5%E8%BC%B8%E5%87%BA%E6%A8%A1%E5%9E%8B%E6%B4%BB%E5%8C%96%E7%A7%A9%E2%94%80%E9%9B%B6%E5%BA%A6%E5%AE%9A%E7%90%86/)

[5]. [https://ccjou.wordpress.com/2009/09/22/利用子空間之和證明-rankab≦rank-arank-b/](https://ccjou.wordpress.com/2009/09/22/%e5%88%a9%e7%94%a8%e5%ad%90%e7%a9%ba%e9%96%93%e4%b9%8b%e5%92%8c%e8%ad%89%e6%98%8e-rankab%e2%89%a6rank-arank-b/)

