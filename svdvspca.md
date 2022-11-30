# 比较 SVD 和 PCA

*打开本页，如果没有显示公式，请刷新页面。*

在参考资料【1】的第3章3.5.3节对奇异值分解（SVD）做了详细讲解，参考资料【2】中对主成分分析（PCA）做了完整推导。

此处将 SVD 和 PCA 进行比较。

假设样本数量为 $$n$$ 的数据集 $$\{\pmb{x}_1,\cdots,\pmb{x}_n\}$$ ，其中 $$\pmb{x}_k$$ 是维数为 $$p$$ 的向量，即此数据集的维数是 $$p$$ 。

以下是实现 PCA 的过程：

1. 计算每个数据点相对样本平均 $$\pmb{\mu}=\frac{1}{n}\sum_{k=1}^n\pmb{x}_k$$ 的偏差，并以 $$n\times p$$ 的矩阵表示，称为偏差矩阵（deviation matrix）：

   $$\pmb{X}=\begin{bmatrix}(\pmb{x}_1-\pmb{\mu})^{\rm{T}}\\\vdots\\(\pmb{x}_n-\pmb{\mu})^{\rm{T}}\end{bmatrix}$$

   矩阵 $$\pmb{X}$$ 的每一行表示一个数据点，每一列代表一个维度（特征、属性、变量）。

   并假设不存在常数列。

2. 将 $$\pmb{X}$$ 标准化，每一列除以该列的样本标准差。

   - 第 $$j$$ 列特征值的样本平均：$$\mu_j=\frac{1}{n}\sum_{k=1}^nx_{kj}$$
   - 第 $$j$$ 列的样本方差$$^{[1]}$$ ：$$s_j^2 = \frac{1}{n-1}\sum_{k=1}^n(x_{kj}-\mu_j)^2$$ 
   - 第 $$j$$ 列的样本标准差：$$s_j=\sqrt{s_j^2}$$
   - $$\widetilde{\pmb{X}}=\begin{bmatrix}(\pmb{x}_1-\pmb{\mu})^{\rm{T}}\\\vdots\\(\pmb{x}_n-\pmb{\mu})^{\rm{T}}\end{bmatrix}\begin{bmatrix}1/s_1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&1/s_p\end{bmatrix}=\pmb{XD}^{-1}$$ ，其中 $$\pmb{D}={\rm{diag}}(s_1,\cdots,s_p)$$

3. 定义 $$p\times p$$ 样本协方差矩阵$$^{[1]}$$ ：$$\pmb{S}=\frac{1}{n-1}\pmb{X}^{\rm{T}}\pmb{X}$$ ，其中 $$\pmb{X}$$ 即为前述表示数据的矩阵。

   也可以定义样本相关系数矩阵：

   $$\pmb{R}=\frac{1}{n-1}\widetilde{\pmb{X}}^{\rm{T}}\widetilde{\pmb{X}}=\frac{1}{n-1}(\pmb{XD}^{-1})^{\rm{T}}\pmb{XD}^{-1}=\frac{1}{n-1}\pmb{D}^{-1}\pmb{X}^{\rm{T}}\pmb{XD}^{-1}=\pmb{D}^{-1}\pmb{S}\pmb{D}^{-1}$$

   （在下述讨论中，使用样本协方差矩阵 $$\pmb{S}$$ ，如果需要使用样本相关系数矩阵，则将 $$\pmb{S}$$ 替换为 $$\pmb{R}$$ 即可）

4. $$\pmb{S}$$ 是实对称矩阵，可正交对角化：$$\pmb{S}=\pmb{W\Lambda W}^{\rm{T}}$$ ，

   其中

   - $$\pmb{W}=\begin{bmatrix}\pmb{w}_1&\cdots&\pmb{w}_p\end{bmatrix}$$ 是特征向量（单位正交）构成的 $$p\times p$$ 正交矩阵，满足：$$\pmb{W}^{\rm{T}}\pmb{W}=\pmb{W}\pmb{W}^{\rm{T}}=\pmb{I}_p$$
   - $$\pmb{\Lambda}={\rm{diag}}(\lambda_1,\cdots,\lambda_p)$$ 是特征值矩阵，$$\lambda_1\ge\cdots\ge\lambda_p\ge0$$

5. 主成分系数矩阵：$$\pmb{Z}=\pmb{XW}$$

   特征值 $$\lambda_j$$ 是第 $$j$$ 个主成分系数（即 $$\pmb{Z}$$ 的第 $$j$$ 列）的样本方差，代表主成分 $$\pmb{w}_j$$ 的权重。

但是，在上述计算样本协方差矩阵 $$\pmb{S}=\frac{1}{n-1}\pmb{X}^{\rm{T}}\pmb{X}$$ 时，在真实的数值计算中，可能会遇到计算中的舍入误差，从造成破坏性影响。比如：

$$\pmb{A}=\begin{bmatrix}1&1&1\\\epsilon&0&0\\0&\epsilon&0\\0&0&\epsilon\end{bmatrix}$$

其中 $$\epsilon$$ 是很小的数，在计算中会得到 $$1+\epsilon^2\approx1$$ ，于是有：

$$\pmb{A}^{\rm{T}}\pmb{A}=\begin{bmatrix}1+\epsilon^2&1&1\\1&1+\epsilon^2&1\\1&1&1+\epsilon^2\end{bmatrix}$$

此时，$${\rm{rank}}(\pmb{A}^{\rm{T}}\pmb{A})=-1\ne{\rm{rank}}\pmb{A}$$ 。

所以，通常将 $$\frac{1}{n-1}\pmb{X}^{\rm{T}}\pmb{X}$$ 仅用于理论推导，不用于实际的数值计算。

但是，在 SVD 中，则可以绕过上述问题。

对于偏差矩阵 $$\pmb{X}$$ ，有：$$\pmb{X}=\pmb{U\Sigma V}^{\rm{T}}$$

其中：

- $$n\times p$$ 级矩阵 $$\pmb{U}$$ 的列向量是单位正交左奇异向量 $$\begin{bmatrix}\pmb{u}_1&\cdots&\pmb{u}_p\end{bmatrix}$$ ，$$\pmb{U}^{\rm{T}}\pmb{U}=\pmb{I}_p$$ ，但 $$\pmb{UU}^{\rm{T}}$$ 不一定等于 $$\pmb{I}_n$$
- $$\pmb{\Sigma}={\rm{diag}}(\sigma_1,\cdots,\sigma_p)$$ ，$$\sigma_1\ge\cdots\ge\sigma_p\ge0$$ 是奇异值
- $$p\times p$$ 矩阵 $$\pmb{V}$$ 的列向量是单位正交右奇异向量 $$\begin{bmatrix}\pmb{v}_1&\cdots&\pmb{v}_p\end{bmatrix}$$ ，$$\pmb{V}^{\rm{T}}\pmb{V}=\pmb{VV}^{\rm{T}}=\pmb{I}_p$$

注意：$$\pmb{X}$$ 和 $$\widetilde{\pmb{X}}$$ 的奇异值分解结果不同。

从而有：

$$\pmb{X}^{\rm{T}}\pmb{X}=(\pmb{U\Sigma V}^{\rm{T}})^{\rm{T}}(\pmb{U\Sigma V}^{\rm{T}})=\pmb{V\Sigma}^{\rm{T}}\pmb{U}^{\rm{T}}\pmb{U\Sigma V}^{\rm{T}}=\pmb{V\Sigma}^2\pmb{V}^{\rm{T}}$$

代入到 $$\pmb{S}=\frac{1}{n-1}\pmb{X}^{\rm{T}}\pmb{X}$$ 中计算协方差矩阵：

$$\pmb{S}=\frac{1}{n-1}\pmb{X}^{\rm{T}}\pmb{X}=\frac{1}{n-1}\pmb{V\Sigma}^2\pmb{V}^{\rm{T}}=\pmb{V}\left(\frac{1}{n-1}\pmb{\Sigma}^2\right)\pmb{V}^{\rm{T}}$$

将这个对角化的结果与前述对角化结果 $$\pmb{S}=\pmb{W\Lambda W}^{\rm{T}}$$ 比较，可知：

- 特征值矩阵： $$\pmb{\Lambda}=\frac{1}{n-1}\pmb{\Sigma}^2$$ ，奇异值 $$\sigma_j$$ 决定特征值：$$\lambda_j=\frac{1}{n-1}\sigma_j^2$$
- 主成分矩阵： $$\pmb{W}=\pmb{V}$$
- 主成分系数矩阵： $$\pmb{Z}=\pmb{XW}=\pmb{XV}=(\pmb{U\Sigma V}^{\rm{T}})\pmb{V}=\pmb{U\Sigma}$$ ，矩阵 $$\pmb{U}=[u_{kj}]$$ 和奇异值 $$\sigma_j$$ 共同决定主成分系数 $$z_{kj}=u_{kj}\sigma_j$$ 

不妨假设 $$\pmb{X}$$ 的列向量线性无关，即 $${\rm{rank}}\pmb{X}=p$$ ，则 $$\sigma_1\ge\cdots\ge\sigma_p\gt0$$ ，$$\pmb{\Sigma}=\sqrt{(n-1)\pmb{\Lambda}}$$ 可逆，其中 $$\sqrt{\Lambda}={\rm{diag}}(\sqrt{\lambda_1},\cdots,\sqrt{\lambda_p})$$ 。

对 $$\pmb{Z}=\pmb{U\Sigma}$$ 等式两侧右乘 $$\pmb{\Sigma}^{-1}$$ ，得：

$$\begin{split}\pmb{U}&=\pmb{Z\Sigma}^{-1}=\frac{1}{\sqrt{n-1}}\pmb{Z\Lambda}^{-\frac{1}{2}}\\&=\frac{1}{\sqrt{n-1}}\begin{bmatrix}z_{11}&z_{12}&\cdots&z_{1p}\\z_{21}&z_{22}&\cdots&z_{2p}\\\vdots&\vdots&\ddots&\vdots\\z_{n1}&z{n2}&\cdots&z_{np}\end{bmatrix}\begin{bmatrix}\frac{1}{\sqrt{\lambda_1}}&0&\cdots&0\\0&\frac{1}{\sqrt{\lambda_2}}&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&\frac{1}{\sqrt{\lambda_p}}\end{bmatrix}\\&=\frac{1}{\sqrt{n-1}}\begin{bmatrix}\frac{z_{11}}{\sqrt{\lambda_1}}&\frac{z_{12}}{\sqrt{\lambda_2}}&\cdots&\frac{z_{1p}}{\sqrt{\lambda_p}}\\\frac{z_{21}}{\sqrt{\lambda_1}}&\frac{z_{22}}{\sqrt{\lambda_2}}&\cdots&\frac{z_{2p}}{\sqrt{\lambda_p}}\\\vdots&\vdots&\dots&\vdots\\\frac{z_{n1}}{\sqrt{\lambda_1}}&\frac{z_{n2}}{\sqrt{\lambda_2}}&\cdots&\frac{z_{np}}{\sqrt{\lambda_p}}\end{bmatrix}\end{split}$$

第 $$j$$ 个主成分系数的样本方差等于 $$\lambda_j$$ ，

$$\widetilde{\pmb{Z}}=\pmb{Z\Lambda}^{-\frac{1}{2}}=\sqrt{n-1}\pmb{U}$$

即 $$\pmb{Z}$$ 经过标准差标准化的结果，称为**因子分数矩阵**（factor score），它的第 $$j$$ 列 $$\sqrt{n-1}\pmb{u}_j$$ 表示数据集在第 $$j$$ 个主成分的权重。平均值为 0 ，方差为 1 。

下面再引入一个概念：**因子负荷**（factor loading），用以衡量主成分系数与原特征之间的相关性。用符号 $$f_{ij}$$ 表示第 $$i$$ 个特征与第 $$j$$ 个主成分系数的相关系数，即因子负荷（特别注意区别**特征之间的相关系数**，请参阅参考资料【1】第5章5.5.2节内容）。

令 $$\pmb{F}=[f_{ij}]_{p\times p}$$ 为因子负荷矩阵，前述已经计算得到的 $$\widetilde{\pmb{X}}$$ 和 $$\widetilde{\pmb{Z}}$$ 都是经过标准差标准化后的数据，则：

$$\pmb{F}=\frac{1}{n-1}\widetilde{\pmb{X}}^{\rm{T}}\widetilde{\pmb{Z}}$$

因为：$$\widetilde{\pmb{X}}=\pmb{XD}^{-1}$$ ，$$\widetilde{\pmb{Z}}=\pmb{XV\Lambda}^{-\frac{1}{2}}$$ ，$$\pmb{S}=\frac{1}{n-1}\pmb{X}^{\rm{T}}\pmb{X}=\pmb{V\Lambda V}^{\rm{T}}$$ ，$$\pmb{\Lambda}^{\frac{1}{2}}=\frac{1}{\sqrt{n-1}}\pmb{\Sigma}$$ 。代入上式，得到：

$$\begin{split}\pmb{F}&=\frac{1}{n-1}(\pmb{D}^{-1}\pmb{X}^{\rm{T}})(\pmb{XV\Lambda}^{-\frac{1}{2}})\\&=\pmb{D}^{-1}\pmb{S}\pmb{V\Lambda}^{-\frac{1}{2}}\\&=\pmb{D}^{-1}(\pmb{V\Lambda V}^{\rm{T}})\pmb{V\Lambda}^{-\frac{1}{2}}\\&=\pmb{D}^{-1}\pmb{V\Lambda}\pmb{\Lambda}^{-\frac{1}{2}}\\&=\pmb{D}^{-1}\pmb{V\Lambda}\pmb{\Lambda}^{\frac{1}{2}}\\&=\frac{1}{\sqrt{n-1}}\pmb{D}^{-1}\pmb{V}\pmb{\Sigma}\end{split}$$

前面曾经计算过相关系数矩阵 $$\pmb{R}=\pmb{D}^{-1}\pmb{S}\pmb{D}^{-1}$$ ，如果用 $$\pmb{R}$$ 替代 $$\pmb{S}$$ ，则解得因子负荷矩阵：

$$\pmb{F}=\frac{1}{\sqrt{n-1}}\pmb{V\Sigma}$$ 

注意，此处的 $$\pmb{V}$$ 和 $$\pmb{\Sigma}$$ 是由 $$\widetilde{\pmb{X}}$$ 求得，如果使用 SVD 计算，与前述值有所不同。

## 参考资料

1. 齐伟. 机器学习数学基础. 电子工业出版社
2. [主成分分析](./pca.html)