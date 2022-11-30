# 相似矩阵

*打开本页，如果不能显示公式，请刷新页面*

《机器学习数学基础》第3章3.3节专门介绍了相似矩阵的有关内容，包括相似变换、几何理解和对角化，并给出了用程序实现的方法。

## 用特征值判断矩阵是否相似

设 $$\pmb{A}$$ 和 $$\pmb{B}$$ 都是 $$n\times n$$ 矩阵，且 $$\pmb{A}\sim\pmb{B}$$ ，即 $$\pmb{A}=\pmb{PBP}^{-1}$$ 。

> 以 $$\pmb{P}$$ 的列向量为基向量，线性变换 $$\pmb{A}$$ 参考此基底的变换矩阵即为 $$\pmb{B}$$ 。

**命题1：两个相似矩阵包含相同的特征值。**

证明1：

将相似关系代入 $$\pmb{Ax}=\lambda\pmb{x}$$ ，得：

$$\pmb{PBP}^{-1}\pmb{x}=\lambda\pmb{x}$$

等号两侧左乘 $$\pmb{P}^{-1}$$ 得：

$$\pmb{BP}^{-1}\pmb{x}=\lambda\pmb{P}^{-1}\pmb{x}$$

由此可知 $$\pmb{B}$$ 有与 $$\pmb{A}$$ 相同的特征值 $$\lambda$$ ，但 $$\pmb{B}$$ 的特征向量为 $$\pmb{P}^{-1}\pmb{x}$$ 。

证明2：

计算特征多项式：

$$\begin{split}p_{\pmb{A}}(\lambda)&= \det(\pmb{A}-\lambda\pmb{I})=\det(\pmb{PBP}^{-1}-\lambda\pmb{I})\\&=\det(\pmb{P}(\pmb{B}-\lambda\pmb{I})\pmb{P}^{-1})\\&=\det(\pmb{P})\det(\pmb{B}-\lambda\pmb{I})(\det(\pmb{P}))^{-1}\\&=\det(\pmb{B}-\lambda\pmb{I})\\&=p_{\pmb{B}}(\lambda)\end{split}$$

在“两个矩阵有相同的特征值”条件下，讨论一下三种情况：

**情况1：$$\pmb{A}$$ 和 $$\pmb{B}$$ 皆可对角化**

设 $$\pmb{A}=\pmb{SDS}^{-1}，\pmb{B}=\pmb{TDT}^{-1}$$ ，则 $$\pmb{A}\sim\pmb{D}，\pmb{B}\sim\pmb{D}$$ 。

由 $$\pmb{B}=\pmb{TDT}^{-1}$$ 可得：$$\pmb{D}=\pmb{T}^{-1}\pmb{BT}$$ ，则：

$$\pmb{A}=\pmb{S}\pmb{T}^{-1}\pmb{BT}\pmb{S}^{-1}=\pmb{ST}^{-1}\pmb{B}(\pmb{ST}^{-1})^{-1}$$

所以：$$\pmb{A}\sim\pmb{B}$$ 。

**情况2：$$\pmb{B}$$ 可对角化，$$\pmb{A}$$ 不可对角化**

此时 $$\pmb{A}$$ 不相似于 $$\pmb{B}$$ 。

**情况3：都不可对角化**

两个矩阵有可能相似，也有可能不相似。需要进一步论证。

## 相似关系

$$\pmb{A}、\pmb{B}$$ 是 $$n$$ 阶方阵，且 $$\pmb{B}=\pmb{SAS}^{-1}$$  ，即 $$\pmb{B}\sim\pmb{A}$$ 。

- $$\pmb{A}^T\sim\pmb{A}$$

**证明**

下述证明参考了文献[2]的相关内容。

设可逆矩阵 $$\pmb{M}$$ ，使得：

$$\pmb{J}=\pmb{M}^{-1}\pmb{AM}$$

其中 $$\pmb{J}$$ 为 Jordan 矩阵或 Jordan 典型形式，是由 Jordan分块构成的主对角分块矩阵，则：任何一个 Jordan 分块必定与其转置相似，$$\pmb{J}\sim\pmb{J}^T$$ 。

则 $$\pmb{A}=\pmb{MJM}^{-1}$$ ，$$\pmb{A}\sim\pmb{J}\sim\pmb{J}^T$$

又：$$\pmb{A}^T=(\pmb{MJM}^{-1})^T=(\pmb{T}^T)^{-1}\pmb{J}^T\pmb{M}^T$$

所以 $$\pmb{J}^T\sim\pmb{A}^T$$

故 $$\pmb{A}\sim\pmb{A}^T$$

证毕

- 若 $$\pmb{B}\sim\pmb{A}$$ ，则 $$\pmb{B}^2\sim\pmb{A}^2, \pmb{B}^T\sim\pmb{A}^T$$ ；又若 $$\pmb{A}$$ 和 $$\pmb{B}$$ 可逆，则 $$\pmb{B}^{-1}\sim\pmb{A}^{-1}$$ 。

**证明**

根据 $$\pmb{B}\sim\pmb{A}$$ ，设 $$\pmb{B}=\pmb{SAS}^{-1}$$ ，则 $$\pmb{B}^2=\pmb{SAS}^{-1}\pmb{SAS}^{-1}=\pmb{SA}^2\pmb{S}^{-1}\Rightarrow\pmb{B}^2\sim\pmb{A}$$

根据 $$\pmb{A}^T\sim\pmb{A}$$ ，可知 $$\pmb{B}^T\sim\pmb{B}$$ ，又因为 $$\pmb{B}\sim\pmb{A}$$ ，所以 $$\pmb{B}^T\sim\pmb{A}^T$$

$$\pmb{A}$$ 和 $$\pmb{B}$$ 可逆，$$\pmb{B}^{-1}=(\pmb{SAS}^{-1})^{-1}=\pmb{SA}^{-1}\pmb{S}^{-1}$$

证毕

- 若 $$\pmb{A}、\pmb{B}$$ 至少有一个是可逆的，则 $$\pmb{AB}\sim\pmb{BA}$$ 。

**证明**

假设 $$\pmb{A}$$ 可逆，则 $$\pmb{AB}=\pmb{A}\pmb{BAA}^{-1}=\pmb{A}(\pmb{BA})\pmb{A}^{-1}\Rightarrow\pmb{AB}\sim\pmb{BA}$$

证毕

- $$\pmb{A}^T\pmb{A}\sim\pmb{AA}^T$$

**证明**

设 $$n$$ 阶方阵 $$\pmb{A}$$ 的奇异值分解为 $$\pmb{A}=\pmb{U\Sigma V}$$ ，$$\pmb{U}、\pmb{V}$$ 是正交矩阵，$$\pmb{U}^T=\pmb{U}^{-1},\pmb{V}^T=\pmb{V}^{-1},\pmb{\Sigma}=diag(\sigma_1,\cdots,\sigma_n)$$

则：

$$\pmb{A}^T\pmb{A}=(\pmb{U\Sigma V})^T(\pmb{U\Sigma V})=\pmb{V}^T\pmb{\Sigma U}^T\pmb{U\Sigma V}=\pmb{V}^T\pmb{\Sigma}^2\pmb{V}=\pmb{V}^{-1}\pmb{\Sigma}^2\pmb{V}$$

$$\therefore\quad \pmb{A}^T\pmb{A}\sim\pmb\Sigma^2$$

$$\pmb{A}\pmb{A}^T=(\pmb{U\Sigma V})(\pmb{U\Sigma V})^T=\pmb{U}\pmb{\Sigma V}\pmb{V}^T\pmb{\Sigma}^T\pmb{ U}^T=\pmb{U}\pmb{\Sigma}^2\pmb{U}^T$$

$$\therefore\quad\pmb{AA}^T\sim\pmb\Sigma^2$$

故：$$\pmb{A}^T\pmb{A}\sim\pmb{AA}^T$$

证毕







## 参考文献

[1]. [线代启示录：如何检查量矩阵是否相似](https://ccjou.wordpress.com/2009/06/25/%e5%a6%82%e4%bd%95%e6%aa%a2%e6%9f%a5%e4%ba%8c%e7%9f%a9%e9%99%a3%e6%98%af%e5%90%a6%e7%9b%b8%e4%bc%bc/)

[2]. [线代启示录：矩阵与其转置的相似性](https://ccjou.wordpress.com/2009/09/11/%e7%9f%a9%e9%99%a3%e8%88%87%e5%85%b6%e8%bd%89%e7%bd%ae%e7%9a%84%e7%9b%b8%e4%bc%bc%e6%80%a7/)



