## logistic 回归

## 中文名称

对于 *logistic* 的中文翻译，目前在各种资料中常见有如下几种：

- *逻辑* 。此译名或是受到英文单词“logical”影响？事实上，此算法中一定是符合逻辑的，也不仅仅是此算法，其他算法也服从一定的逻辑。所以，用此译名，显然不合适。
- *逻辑斯蒂* 。直接音译。
- *对数几率* 。这是周志华教授提出来的$$^{[1]}$$ ，但是，在数学领域，目前已经很少使用“几率”、“机率”这些词汇，现在通用的词汇是“概率”。是否改为“对数概率”呢？

以上三种译名，不同人有不同的理解，此处不争论。但在有公认的译名之前，我还是直接只用英文。

## 阐释原理的方法

Logistic 回归用于解决二分类问题，虽然常称之为*logistic 回归*。对此算法原理的阐述，通常有两种方法。

**方法1：直接引入 logistic 函数**

以参考资料 [1] 为代表资料，通常用这方方法。认为如果直接使用函数：
$$
y=\begin{cases}0,\quad&(z\lt0)\\0.5,\quad&(z=0)\\1,\quad&(z\gt0)\end{cases}\tag{1}
$$
其问题在于**不连续**，不能直接用于回归的线性模型，故要找一个连续，且接近（1）式的函数，于是找到了 logistic 函数。如下图所示：

![](images/logistic.png)

这种方法，在逻辑上并不严格，不能说（1）式不能，于是就选择了 logistic 函数。当然，从工程角度看，是可以的，毕竟能够实现目标。但是，如果作为一门科学，而不仅仅是技术，那么就应该以逻辑上更严谨的方法来说明，为什么选用了 logistic 函数。

即，如果从算法的原理角度来看，上述方式并没有解决如下问题：

- 是否可以使用其他 S 型函数？
- 根据什么找到此函数的？碰巧？还是按照一定逻辑？

**方法2：用统计学理解 logistic 回归**

这种方法，不会生硬地引入一个函数，而是将二分类问题看做统计学中的伯努利分布，并在该分布基础上，并结合贝叶斯定理，自然而然地使用 logistic 函数。

本文即从这个角度阐述 logistic 回归。

## 伯努利分布

从统计学的角度来看，二分类问题可以看做伯努利分布。

> 《机器学习数学基础》5.3.2节 离散型随机变量的分布。
>
> 或者观看《机器学习数学基础》视频课程的有关章节讲授。

根据伯努利分布或者（0-1）分布
$$
P(X=k)=p^kq^{1-k},(k=0,1)\tag{2}
$$


即：

$$
P(X=1)=p,P(X=0)=1-p\tag{3}
$$
## logistic 回归的推导过程

### 数据集和目标

设数据集 $$D=\{\pmb{x}_1,\cdots,\pmb{x}_n\}$$ ，$$\pmb{x}_i$$ 表示一个样本，行向量。

$$C_1,C_2$$ 分别表示样本的类别

- 目标：设计一个分类器，能根据给定的样本 $$\pmb{x}$$ ，判断它属于那个类别。

  即根据概率 $$P(C_1|\pmb{x}),P(C_2|\pmb{x})$$ 的大小关系，判断所属类别。如果 $$P(C_1|\pmb{x})\gt P(C_2|\pmb{x})$$ ，则 $$\pmb{x}$$ 属于 $$C_1$$ 类。

- 条件：判别的概率分布服从伯努利分布，即：

  $$\begin{split}P(C_1|\pmb{x})=p\\P(C_2|\pmb{x})=1-p\end{split}$$

### 问题转换

由贝叶斯定理得：

$$
P(C_j|\pmb{x})=\frac{p(\pmb{x}|C_j)P(C_j)}{P(\pmb{x})},\quad j=1,2\tag{4}
$$


- $$P(C_j)$$ 是类别 $$C_j$$ 出现的概率，称为先验概率
- $$p(\pmb{x}|C_j)$$ 是条件概率，即给定类别 $$C_j$$ ，样本 $$\pmb{x}$$ 的概率函数，也称为似然
- $$p(\pmb{x})$$ 是样本 $$\pmb{x}$$ 的概率，即：$$p(\pmb{x})=p(\pmb{x}|C_1)P(C_1)+p(\pmb{x}|C_2)P(C_2)$$
- $$P(C_j|\pmb{x})$$ 是给定样本 $$\pmb{x}$$ 的情况下，该样本属于 $$C_j$$ 的概率，称为后验概率。

根据（4）式可得：
$$
\frac{P(C_1|\pmb{x})}{P(C_2|\pmb{x})}=\frac{p(\pmb{x}|C_1)P(C_1)}{p(\pmb{x}|C_2)P(C_2)}\tag{5}
$$
设 $$p(\pmb{x}|C_1),p(\pmb{x}|C_2)$$ ：

- 它们都是正态分布
- 且有相同的协方差矩阵

因为：$$P(C_1|\pmb{x})=1-P(C_2|\pmb{x})$$

对（5）式的结果求对数（自然对数），并利用参考资料 [2] 中（20）式结论，可得：
$$
\text{log}\frac{P(C_1|\pmb{x})}{P(C_2|\pmb{x})}=\text{log}\frac{P(C_1|\pmb{x})}{1-P(C_1|\pmb{x})}=\pmb{w}^{\text{T}}\pmb{x}+w_0\tag{6}
$$
如此转化为线性函数表示，进而得到 $$P(C_1|\pmb{x})$$ （考虑自然对数）：

$$
P(C_1|\pmb{x})=\frac{1}{1+\exp(-(\pmb{w}^{\text{T}}\pmb{x}+w_0))}\qquad\qquad\tag{7}
$$
为了计算 $$P(C_1|\pmb{x})$$ ，由（6）式可知，可以求 $$\pmb{w}^{\text{T}}$$ 和 $$w_0$$ 。

下面用最大似然估计，计算这两个系数。

### 二分类问题：

考虑一个训练集 $$D=\{(\pmb{x}_i, y_i)\}_{i=1}^n$$ ，如果样本属于 $$C_1$$ 类，则 $$y_i=1$$ ；属于 $$C_2$$ 了，则 $$y_i=0$$ 。$$y_i$$ 服从伯努利分布，其概率：$$p_i=P(C_1|\pmb{x}_i)$$ 。

不妨令 $$\pmb{\theta}=\begin{bmatrix}w_0\\\pmb{w}\end{bmatrix},\widetilde{\pmb{x}}=\begin{bmatrix}1\\\pmb{x}\end{bmatrix}$$ ，（7）式可以写成：

$$
p_i=\frac{1}{1+\exp(-\pmb{\theta^{\text{T}}}\widetilde{\pmb{x}}_i)}\tag{8}
$$
$$\widetilde{\pmb{x}}_i$$ 是已知数据集中第 $$i$$ 个样本，$$\pmb{\theta}$$ 是待估计的参数。

写出似然函数：

$$
L(\pmb{\theta}|D)=P(\pmb{\theta}|\pmb{x_1},\cdots,\pmb{x_n})=\prod_{i=1}^n(p_i)^{y_i}(1-p_i)^{1-y_i}\qquad\qquad\tag{9}
$$
目的是使 $$L$$ 最大化，即样本属于某类别的概率越大越好。

若令 $$E=-\text{log}L$$ ，

则最大化 $$L$$ ，即是最小化 $$E$$ 。

按照上面的假设，对（9）式取以 $$e$$ 为底的对数，得：

$$
\begin{split}
E &= -\text{log}\left(\prod_{i=1}^n(p_i)^{y_i}(1-p_i)^{1-y_i}\right)
\\
&=-\sum_{i=1}^n\left[y_i\text{log}p_i+(1-y_i)\text{log}(1-p_i)\right]
\end{split}\tag{10}
$$
其中：$$p_i=\frac{1}{1+\exp(-\pmb{\theta^{\text{T}}}\widetilde{\pmb{x}})}$$

计算（10）式最小值：

$$
\frac{\partial E}{\partial\pmb{\theta}}=\sum_{i=1}^n\frac{\partial E}{\partial p_i}\frac{\partial p_i}{\partial\pmb{\theta}}\tag{11}
$$
分别计算（11）式的两部分：

- 由（10）式（使用 $$\frac{\partial}{\partial x}\text{ln}x=\frac{1}{x}$$ ）

$$
\frac{\partial E}{\partial p_i}=-\sum_{i=1}^n\left(\frac{y_i}{p_i}-\frac{1-y_i}{1-p_i}\right)\tag{12}
$$

- 由 $$p_i=\frac{1}{1+\exp(-\pmb{\theta^{\text{T}}}\widetilde{\pmb{x}}_i)}$$ 得：

$$
\frac{\partial p_i}{\partial\pmb{\theta}}=p_i(1-p_i)\widetilde{\pmb{x}}_i\tag{13}
$$

将（12）（13）代入到（11）式，得：
$$
\frac{\partial E}{\partial\pmb{\theta}}=-\sum_{i=1}^n\left(\frac{y_i}{p_i}-\frac{1-y_i}{1-p_i}\right)p_i(1-p_i)\widetilde{\pmb{x}}_i=-\sum_{i=1}^n(y_i-p_i)\widetilde{\pmb{x}}_i\tag{14}
$$
接下来，就令 $$\frac{\partial E}{\partial\pmb{\theta}}=0$$ ，从而得到 $$\pmb{\theta}$$ ，但是，函数 $$p_i$$ 是 $$\pmb{\theta}$$ 的非线性函数，所以没有代数解，可以通过**梯度下降法**和**牛顿法**求解。

**用梯度下降法：**

假设一个初始值 $$\hat{\pmb{\theta}}$$ ，根据梯度下降有：

$$
\hat{\pmb{\theta}}\leftarrow\hat{\pmb{\theta}}-\eta\frac{\partial E}{\partial\hat{\pmb{\theta}}}=\hat{\pmb{\theta}}+\sum_{i=1}^n(y_i-p_i)\widetilde{\pmb{x}}_i\tag{15}
$$
**牛顿法：**

假设一个初始值 $$\hat{\pmb{\theta}}$$ ，根据牛顿法迭代公式，有：

$$
\hat{\pmb{\theta}}\leftarrow\hat{\pmb{\theta}}-\pmb{H}^{-1}\frac{\partial E}{\partial\hat{\pmb{\theta}}}\tag{16}
$$
其中 $$\pmb{H}$$ 是梯度 $$\frac{\partial E}{\partial\hat{\pmb{\theta}}}$$ 的雅可比矩阵（Jocobian matrix）。

若假设（7）式中的 $$\pmb{w}$$ 是 $$d$$ 维向量（即假设数据集有 $$d$$ 个特征），则 $$\pmb{\theta}=\begin{bmatrix}w_0\\\pmb{w}\end{bmatrix}$$ 是 $$d+1$$ 维。

那么，$$H$$ （雅可比矩阵）也就是 $$E$$ 的 $$(d+1)\times(d+1)$$ 阶 Hessian 矩阵：

$$\pmb{H}(\hat{\pmb{\theta}})=\begin{bmatrix}\frac{\partial^2E}{\partial{\theta_0}\partial{\theta_0}}&\frac{\partial^2E}{\partial{\theta_0}\partial{\theta_1}}&\cdots&\frac{\partial^2E}{\partial{\theta_0}\partial{\theta_d}}\\\frac{\partial^2E}{\partial{\theta_1}\partial{\theta_0}}&\frac{\partial^2E}{\partial{\theta_1}\partial{\theta_1}}&\cdots&\frac{\partial^2E}{\partial{\theta_1}\partial{\theta_d}}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial^2E}{\partial{\theta_d}\partial{\theta_0}}&\frac{\partial^2E}{\partial{\theta_d}\partial{\theta_1}}&\cdots&\frac{\partial^2E}{\partial{\theta_d}\partial{theta_d}}\end{bmatrix}$$

考虑上述矩阵中的第 $$(s,t)$$ 个元（第 $$s$$ 行，第 $$t$$ 列个元素）：

$$h_{st}=\frac{\partial^2E}{\partial{\theta_s}\partial{\theta_t}}=\frac{\partial}{\partial{\theta_s}}\left(\frac{\partial{E}}{\partial{\theta_t}}\right)=\frac{\partial}{\partial{w_s}}\left(-\sum_{i=1}^n(y_i-p_i)x_{it}\right)$$ 

这里的 $$x_{it}$$ 是第 $$i$$ 个数据 $$\widetilde{\pmb{x}}_i$$ 的第 $$t$$ 个特征的数值。

继续上式：

$$
h_{st}=\sum_{i=1}^n\frac{\partial p_i}{\partial\theta_s}=\sum_{i=1}^np_i(1-p_i)x_{is}x_{it}=\sum_{i=1}^nx_{is}p_i(1-p_i)x_{it}\tag{17}
$$
注意这里角标的取值范围：$$1\le i\le n,0\le s,t\le d$$ 。

根据（17）式，可以将 Hessian 矩阵写成：

$$\pmb{H}=\begin{bmatrix}1&1&\cdots&1\\x_{11}&x_{21}&\cdots&x_{n1}\\\vdots&\vdots&\ddots&\vdots\\x_{1d}&x_{2d}&\cdots&x_{nd}\end{bmatrix}\begin{bmatrix}p_1(1-p_1)&0&\cdots&0\\0&p_2(1-p_2)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&p_n(1-p_n)\end{bmatrix}\begin{bmatrix}1&x_{11}&\cdots&x_{1d}\\1&x_{21}&\cdots&x_{2d}\\\vdots&\vdots&\ddots&\vdots\\1&x_{n1}&\cdots&x_{nd}\end{bmatrix}$$

令：

$$\pmb{X}=\begin{bmatrix}1&x_{11}&\cdots&x_{1d}\\1&x_{21}&\cdots&x_{2d}\\\vdots&\vdots&\ddots&\vdots\\1&x_{n1}&\cdots&x_{nd}\end{bmatrix}$$

$$\pmb{\mathcal{P}}=\begin{bmatrix}p_1(1-p_1)&0&\cdots&0\\0&p_2(1-p_2)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&p_n(1-p_n)\end{bmatrix}$$

则：

$$
\pmb{H}=\pmb{X}^{\text{T}}\pmb{\mathcal{P}X}\tag{18}
$$
另外，对于梯度 $$\frac{\partial E}{\partial{\hat{\pmb{\theta}}}}$$ 也可以写成矩阵形式，根据（14）式：

$$
\begin{split}
\frac{\partial E}{\partial{\hat{\pmb{\theta}}}} &= -\sum_{i=1}^n(y_i-p_i)\widetilde{\pmb{x}}_i
\\
&=-\begin{bmatrix}\widetilde{\pmb{x}}_1&\cdots&\widetilde{\pmb{x}}_n\end{bmatrix}\begin{bmatrix}y_1-p_1\\\vdots\\y_n-p_n\end{bmatrix}
\\
&=-\pmb{X}^{\text{T}}(\pmb{y}-\pmb{p})
\end{split}\tag{19}
$$
其中：$$\pmb{y}=\begin{bmatrix}y_1\\\vdots\\y_n\end{bmatrix},\pmb{p}=\begin{bmatrix}p_1\\\vdots\\p_n\end{bmatrix}$$

于是（16）式，牛顿法迭代公式，可以整理成：

$$
\hat{\pmb{\theta}}\leftarrow\hat{\pmb{\theta}}-\pmb{H}^{-1}\frac{\partial E}{\partial\hat{\pmb{\theta}}}=\hat{\pmb{\theta}}+(\pmb{X}^{\text{T}}\pmb{\mathcal{P}X})^{-1}\pmb{X}^{\text{T}}(\pmb{y}-\pmb{p})\tag{20}
$$


### 多分类问题：

logistic 回归是二分类，也可以把它推广到多分类问题中。

目前有两种方法：

1. 将多分类问题转变为 $$k-1$$ 个 logistic 函数所表达的二分类问题（$$k$$ 是分类的类别，在二分类中，$$k=0,1$$ ）。即计算 $$P(C_j|\pmb{x})$$ 的概率。
2. 使用 Softmax 函数$$^{[2]}$$作为后验概率的表达式。

**第一个方法**

以 $$C_k$$ 类别作为比较的基准，则：

$$\text{log}\frac{P(C_j|\pmb{x})}{P(C_k|\pmb{x})}=\pmb{w}_j^{\text{T}}\pmb{x}+w_{j0},\quad(j=1,\cdots,k-1)$$

仍然考虑以 $$e$$ 为底的对数，则：

$$
\frac{P(C_j|\pmb{x})}{P(C_k|\pmb{x})}=\text{exp}(\pmb{w}_j^{\text{T}}\pmb{x}+w_{j0})\tag{21}
$$
又因为：$$\sum_{l=1}^kP(C_l|\pmb{x})=1$$ （概率的性质），则：

$$\frac{1-P(C_k|\pmb{x})}{P(C_k|\pmb{x})}=\sum_{l=1}^{k-1}\frac{P(C_l|\pmb{x})}{P(C_k|\pmb{x})}=\sum_{l=1}^{k-1}\text{exp}(\pmb{w}_l^{\text{T}}\pmb{x}+w_{l0})$$

从而得到：

$$
P(C_k|\pmb{x})=\frac{1}{1+\sum_{l=1}^{k-1}\text{exp}(\pmb{w}_l^{\text{T}}\pmb{x}+w_{l0})}\tag{22}
$$
根据（21）式，并得到：

$$
P(C_j|\pmb{x})=\frac{\text{exp}(\pmb{w}_j^{\text{T}}\pmb{x}+w_{j0})}{1+\sum_{l=1}^{k-1}\text{exp}(\pmb{w}_l^{\text{T}}\pmb{x}+w_{l0})},\quad(j=1,\cdots,k-1)\tag{23}
$$
然后利用前面的二分类的最大似然估计和梯度下降或者牛顿法计算，即可得到 $$\pmb{w}_j$$ 和 $$w_{j0}$$ ，$$1\le j\le k-1$$ 。

- 若 $$j\ne k$$ ，如果 $$C_j$$ 和 $$C_k$$ 在 $$\mathbb{R}^d$$ 是相邻区域，则边界为超平面 $$\pmb{w}^{\text{T}}_j\pmb{x}+w_{j0}=0$$
- 对于 $$1\le j,l\le k-1$ 且 $j\ne l$$ ，如果 $$C_j$$ 和 $$C_l$$ 相邻，则边界为超平面 $$\pmb{w}^{\text{T}}_j\pmb{x}+w_{j0}=\pmb{w}^{\text{T}}_l\pmb{x}+w_{l0}$$

**第二个方法**

令后验概率为：

$$
P(C_j|\pmb{x})=\frac{\text{exp}(\pmb{w}^{\text{T}}_j\pmb{x}+w_{j0})}{\sum_{l=1}^k\text{exp}(\pmb{w}^{\text{T}}_l\pmb{x}+w_{l0})},\quad(j=1,\cdots,k)\tag{24}
$$
若令 $$a_j=\pmb{w}^{\text{T}}_j\pmb{x}+w_{j0}$$ ，则上式等号右边可以写成：

$$\frac{\text{exp}(a_j)}{\sum_{l=1}^k\text{exp}(a_l)}$$

这就是 softmax 函数，它是 max 函数的一种平滑版本。

仿照之前的方法，由（24）式，可得：

$$
\text{log}\frac{P(C_j|\pmb{x})}{P(C_l|\pmb{x})}=(\pmb{w}_j-\pmb{w}_l)^{\text{T}}\pmb{x}+w_{j0}-w_{l0},\quad(j\ne l)\tag{25}
$$
上式说明，若 $$C_j$$ 和 $$C_l$$ 有相邻的区域，则边界为超平面 $$\pmb{w}^{\text{T}}_j\pmb{x}+w_{j0}=\pmb{w}^{\text{T}}_l\pmb{x}+w_{l0}$$ 。

当 $$l\ne j$$ 时，$$a_j=\pmb{w}^{\text{T}}_j\pmb{x}+w_{j0}>>a_l=\pmb{w}^{\text{T}}_l\pmb{x}+w_{l0}$$ ，得到：

$$P(C_j|\pmb{x})\approx1,\quad P(C_l|\pmb{x})=0$$ 

> 对于 $$(a_1+c,\cdots,a_k+c)$$ ，softmax 函数具有相同的返回值，故用 Softmax 函数表达的后验概率并不唯一。但是，由于 Softmax 函数具有对称表达的有点，故受到青睐

下面讨论 Softmax 函数表达的 logistic 回归的最大似然估计和数值计算方法。

设样本数据 $$D=\{\pmb{x}_i, y_{i1},y_{i2},\cdots, y_{ik}\}$$ 。若 $$i\in C_j$$ ，则 $$y_{ij}=1$$ （ $$y_{ij}$$ 表示样本标签）；否则 $$y_{ij}=0$$ 。

对于数据集中的样本 $$\pmb{x}_i$$ ，假设其标签为 $$y_{ij}$$ ，并服从多项式分布，其概率为：

$$p_{ij}=P(C_j|\pmb{x}_i),(1\le i\le n,1\le j\le k)$$

仿照前述二分类的方法，定义 $$(d+1)$$ 维向量：$$\pmb{\theta}_j=\begin{bmatrix}w_0\\\pmb{w}_j\end{bmatrix},(1\le j\le k)$$ ，以及数据向量：$$\widetilde{\pmb{x}}_i=\begin{bmatrix}1\\\pmb{x}_i\end{bmatrix}$$ 。

写出似然函数：

$$
L(\pmb{\theta}_1,\cdots,\pmb{\theta}_k|D)=\prod_{i=1}^n\prod_{j=1}^k(p_{ij})^{y_{ij}}\tag{26}
$$
其中 $$p_{ij}$$ 由 Softmax 函数定义：

$$
p_{ij}=\frac{\text{exp}(a_{ij})}{\sum_{l=1}^k\text{exp}(a_{il})}=\frac{\text{exp}(\pmb{\theta}^{\text{T}}_j\widetilde{\pmb{x}}_i)}{\sum_{l=1}^k\text{exp}(\pmb{\theta}^{\text{T}}_l\widetilde{\pmb{x}}_i)}\tag{27}
$$

$$
a_{ij}=\pmb{\theta}^{\text{T}}_j\widetilde{\pmb{x}}_i\qquad\tag{28}
$$

对似然函数求对数，并增加负号，得到最小化目标函数：

$$
E(\pmb{\theta}_1,\cdots,\pmb{\theta}_k)=-\text{log}L(\pmb{\theta}_1,\cdots,\pmb{\theta}_k|D)=-\sum_{i=1}^n\sum_{j=1}^ky_{ij}\text{log}(p_{ij})\tag{29}
$$
（29）式为多类别交叉熵。

为了后续计算，先预备对 Softmax 函数求导：

> $$p_l=\frac{\text{exp}(a_l)}{\sum_{q=1}^k\text{exp}(a_q)}$$
>
> $$\frac{\partial p_l}{\partial a_j}=p_l(\delta_{lj}-p_j)$$
>
> 若 $$l=j$$ ，则 $$\delta_{lj}=1$$ ；若 $$l\ne j$$ ，则 
> $$
> \delta_{lj}=0\tag{30}
> $$
> 

计算（29）式对 $$\pmb{\theta}_j$$ 的导数，而后令 $$\frac{\partial E}{\partial\pmb{\theta}_j}=\pmb{0}$$ ，解此方程组，得到未知量 $$\pmb{\theta}_j$$ 。当然，此未知量是用梯度下降法或牛顿法求得的。

$$
\begin{split}\frac{\partial E}{\partial\pmb{\theta}_j}&=\sum_{i=1}^n\sum_{l=1}^k\frac{\partial E}{\partial p_{il}}\frac{\partial p_{il}}{\partial a_{ij}}\frac{\partial a_ij}{\partial\pmb{\theta}_j}\\&=-\sum_{i=1}^n\sum_{l=1}^k\frac{y_{ij}}{p_{il}}p_{il}(\delta_{lj}-p_{ij})\widetilde{\pmb{x}}_i\\&=-\sum_{i=1}^n\sum_{l=1}^ky_{il}\delta_{lj}\widetilde{\pmb{x}}_i+\sum_{i=1}^np_{ij}\sum_{l=1}^ky_{il}\widetilde{\pmb{x}}_i \end{split}\tag{31}
$$

- 计算（31）式中的第一项：

  因为（30）式中的条件，只有当 $$l=j$$ 时，$$\delta_{lj}=1$$ ，其余项都是 $$0$$ ，所以：$$\sum_{l=1}^ky_{il}\delta_{lj}\widetilde{\pmb{x}}_i=y_{ij}\widetilde{\pmb{x}}_i$$ 。

  故，第一项计算结果是：

  $$-\sum_{i=1}^ny_{ij}\widetilde{\pmb{x}}_i$$

- 计算（31）式中的第二项：

  在本问题的数据集假设之中有：若 $$i\in C_j$$ ，则 $$y_{ij}=1$$ （ $$y_{ij}$$ 表示样本标签）；否则 $$y_{ij}=0$$ 。则：

  $$y_{i1}+\cdots+y_{ik}=1=\sum_{l=1}^ky_{il}$$

  所以，第二项计算结果是：

  $$\sum_{i=1}^np_{ij}\widetilde{\pmb{x}}_i$$

故：（31）式等于：

$$
\begin{split}(31)&=-\sum_{i=1}^ny_{ij}\widetilde{\pmb{x}}_i+\sum_{i=1}^np_{ij}\widetilde{\pmb{x}}_i\\&=-\sum_{i=1}^n(y_{ij}-p_{ij})\widetilde{\pmb{x}}_i\\&=-\pmb{X}^{\text{T}}(\pmb{y}_j-\pmb{p}_j)\qquad\end{split}\tag{32}
$$


其中：$$\pmb{y}_j=\begin{bmatrix}y_{1j}\\\vdots\\y_{nj}\end{bmatrix},\pmb{p}_j=\begin{bmatrix}p_{1j}\\\vdots\\p_{nj}\end{bmatrix}$$

**利用梯度下降求解：**

$$
\hat{\pmb{\theta}}_j\leftarrow\hat{\pmb{\theta}}_j+\eta\sum_{i=1}^n(y_{ij}-p_{ij})\widetilde{\pmb{x}}_i,(j=1,\cdots,k)\tag{33}
$$
写成矩阵形式：

$$
\hat{\pmb{\theta}}_j\leftarrow\hat{\pmb{\theta}}_j+\eta\pmb{X}^{\text{T}}(\pmb{y}_j-\pmb{p}_j),(j=1,\cdots,k)\tag{34}
$$
**用牛顿法求解：**

将 k 个 $$(d+1)$$ 维 $$\pmb{\theta}_1,\cdots,\pmb{\theta}_k$$ 合并成一个 $$k(d+1)$$ 维向量，则得到 $$k(d+1)\times k(d+1)$$ 阶的 Hessian 矩阵，以下用 $$k\times k$$ 阶分块矩阵表示：

$$\pmb{H}=\begin{bmatrix}\pmb{H}_{11}&\cdots&\pmb{H}_{1k}\\\vdots&\ddots&\vdots\\\pmb{H}_{k1}&\cdots&\pmb{H}_{kk}\end{bmatrix}$$

其中，分块 $$\pmb{H}_{pq},(1\le p,q\le k)$$ 是 $$(d+1)\times(d+1)$$ 阶矩阵。

令 $$\theta_{js}$$ 表示 $$\pmb{\theta}_j$$ 的第 $$s$$ 个元素（ $$1\le j\le k,0\le s\le d$$ ），则 $$\pmb{H}_{pq}$$ 的第（s,t）元素：

$$\begin{split}(\pmb{H}_{pq})_{st}=\frac{\partial}{\partial\theta_{ps}}\left(\frac{\partial E}{\partial\theta_{qt}}\right)&=\frac{\partial}{\partial\theta_{ps}}\left(-\sum_{i=1}^n(y_{iq}-p_{iq})x_{it}\right)\\&=\sum_{i=1}^n\frac{\partial p_{iq}}{\partial\theta_{ps}}x_{it}\\&=\sum_{i=1}^n\frac{\partial p_{iq}}{\partial a_{ip}}\frac{\partial a_{ip}}{\partial\theta_{ps}}x_{it}\\&=\sum_{i=1}^np_{iq}(\delta_{pq}-p_{ip})x_{is}x_{it}\end{split}$$

用矩阵表示：

$$
\pmb{H}_{pq}=\pmb{X}^{\text{T}}\pmb{\mathcal{P}}_{pq}\pmb{X}\tag{35}
$$
其中：$$\pmb{\mathcal{P}}_{pq}=\text{diag}\left(p_{1q}(\delta_{pq}-p_{1p}),\cdots,p_{nq}(\delta_{pq}-p_{np})\right)$$

于是得到用牛顿法计算多类别 Logistic 回归的迭代公式：

$$
\begin{bmatrix}\hat{\pmb{\theta}}_1\\\vdots\\\hat{\pmb{\theta}}_k\end{bmatrix}\leftarrow\begin{bmatrix}\hat{\pmb{\theta}}_1\\\vdots\\\hat{\pmb{\theta}}_k\end{bmatrix}+\pmb{H}^{-1}\begin{bmatrix}\pmb{X}^{\text{T}}(\pmb{y}_1-\pmb{p}_1)\\\vdots\\\pmb{X}^{\text{T}}(\pmb{y}_k-\pmb{p}_k)\end{bmatrix}\tag{36}
$$


## Logistic VS. LDA

-  二者有相同的分类法则和类别边界（超平面）。
-  二者分别采用不同的方法计算模型参数
-  二者适用于数据正态分布且各类别的协方差矩阵相差不大
-  二者都对离群值敏感，对此，改进方法为：先让样本数据 $$\pmb{x}$$ 通过非线性奇函数，将奇函数的返回值作为新数据，再应用于 Logistic 模型。

## Logistic 回归案例

官方网页：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

```python
class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
```

### 简单案例

```python
# 使用鸢尾花数据集
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X,y = load_iris(return_X_y=True)
```

```python
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 9
)
```

```python
# 用训练集训练模型
clf = LogisticRegression(solver='liblinear', random_state=0) 
clf.fit(X_train, y_train)
```

参数说明：

- solver：指定求解最优化问题的算法，
  - 'liblinear'适用于小规模数据，数据量较大时使用 'sag'（Stochastic Average Gradient descent，随机平均梯度下降）
  - 'newton-cg'(牛顿法), 'lbfgs'（使用L-BFGS拟牛顿法），'sag'只处理`penalty='ls'`的情况

```python
# 选两个样本，用于预测
X_test[:2, :]
```

```python
# 预测结果
clf.predict(X_test[:2, :])
```

```python
# 计算模型的拟合优度
clf.score(X_test, y_test)
```

### 稍微复杂的案例：识别手写数字

```python
# 引入
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

```python
# 获得数据
X, y = load_digits(return_X_y=True)
```

X 是1797×64 的多维数组，y 是1797 个整数的一维数组（0~9）。

```python
# 划分数据集
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, random_state=0)
```

```python
# 对数据进行标准差标准化变换
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
```

```python
# 创建并训练模型
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
model.fit(X_train, y_train)
```

系数：

- `C`：惩罚性系数的倒数，值越小，则正则化项所占比重越大
- `multi_class`：对于多分类问题，可以采用的策略：
  - `'ovr'`：采用 one-vs-rest 策略
  - `'multinomial'`：直接采用多分类策略

```python
# 评估模型
model.score(X_test, y_test)
```

还可以用混淆矩阵（confusion matrix）输出对测试集每个样本的判断结果，并输出图示。

```python
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs', color='black')
ax.xaxis.set(ticks=range(10))
ax.yaxis.set(ticks=range(10))
ax.set_ylim(9.5, -0.5)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()
```



## 参考资料

[1]. 周志华. 机器学习. 北京：清华大学出版社

[2]. [用贝叶斯定理理解线性判别分析](http://www.itdiffer.com/bayes-lda.html)