# LU分解

*打开本页，如果没有显示公式，请刷新页面。*

本文是对《机器学习数学基础》第2章2.3.3节矩阵LU分解的拓展。

## 判断是否可LU分解

**并非所有矩阵都可以实现LU分解**。

**定理1：**若 $$n$$ 阶可逆矩阵 $$\pmb{A}$$ 可以进行LU分解，则 $$\pmb{A}$$ 的 $$k$$ 阶顺序主子阵（leading principal submatrix）$$\pmb{A}_k$$ 都是可逆的$$^{[1]}$$。

**证明**

将 $$\pmb{A}=\pmb{LU}$$ 用分块矩阵表示：

$$\pmb{A}=\pmb{LU}=\begin{bmatrix}\pmb{L}_{11}&0\\\pmb{L}_{21}&\pmb{L}_{22}\end{bmatrix}\begin{bmatrix}\pmb{U}_{11}&\pmb{U}_{12}\\0&\pmb{U}_{22}\end{bmatrix}=\begin{bmatrix}\pmb{L}_{11}\pmb{U}_{11}&\pmb{L}_{11}\pmb{U}_{12}\\\pmb{L}_{21}\pmb{U}_{11}&\pmb{L}_{21}\pmb{U}_{12}+\pmb{L}_{22}\pmb{U}_{22}\end{bmatrix}$$

其中 $$\pmb{L}_{11}、\pmb{U}_{11}$$ 是 $$k\times k$$ 分块矩阵。

因为 $$\pmb{L}$$ 是单位下三角矩阵，且主对角线元素都是 $$1$$ ，则其分块矩阵 $$\pmb{L}_{11}$$ 亦为三角矩阵，且主对角线元素非零。同理，$$\pmb{U}_{11}$$ 亦然。

所以 $$\pmb{L}_{11}$$ 和 $$\pmb{U}_{11}$$ 可逆，则 $$\pmb{A}_k=\pmb{L}_{11}\pmb{U}_{11}$$ 可以。

证毕。

**例**：$$\pmb{A}=\begin{bmatrix}3&-1&2\\6&-1&5\\-9&7&3\end{bmatrix}$$ 的顺序主子阵依次为：

$$\begin{split}\pmb{A}_1&=[3]=[1][3]\\\pmb{A}_2&=\begin{bmatrix}3&-1\\6&-1\end{bmatrix}=\begin{bmatrix}1&0\\2&1\end{bmatrix}\begin{bmatrix}3&-1\\0&1\end{bmatrix}\\\pmb{A}_3&=\pmb{A}=\pmb{LU}\end{split}$$

**定理2：**（定理1的逆定理）若矩阵 $$\pmb{A}$$ 的所有顺序主子阵 $$\pmb{A}_k$$ 都可逆，则该矩阵存在LU分解。

**证明**（用归纳法）

$$k=1$$ ，$$\pmb{A}_1=[a_{11}]$$ 可逆，则 $$a_{11}\ne 0$$ ，所以有：$$\pmb{A}_1=[1][a_{11}]$$ ，即为LU分解。

设 $$k$$ 阶顺序主子阵 $$\pmb{A}_k$$ 可逆，且可LU分解，$$\pmb{A}_k=\pmb{L}_k\pmb{U}_k$$ 。

$$k+1$$ 阶顺序主子阵 $$\pmb{A}_{k+1}$$ 可以表示为：

$$\pmb{A}_{k+1}=\begin{bmatrix}\pmb{A}_k&\pmb{b}\\\pmb{c}^T&d\end{bmatrix}$$

其中 $$\pmb{b}、\pmb{c}$$ 是 $$k$$ 维向量，$$d$$  是标量。则上式可以进一步写成：

$$\pmb{A}_{k+1}=\begin{bmatrix}\pmb{A}_k&\pmb{b}\\\pmb{c}^T&d\end{bmatrix}=\begin{bmatrix}\pmb{L}_k&\pmb{0}\\\pmb{x}^T&1\end{bmatrix}\begin{bmatrix}\pmb{U}_k&\pmb{y}\\\pmb{0}^T&z\end{bmatrix}$$

通过对应关系，可知：

$$\pmb{b}=\pmb{L}_k\pmb{y},\pmb{c}^T=\pmb{x}^T\pmb{U}_k,d=\pmb{x}^T\pmb{y}+z$$

解得：

$$\pmb{y}=\pmb{L}_k^{-1}\pmb{b},\pmb{x}^T=\pmb{c}^T\pmb{U}_k^{-1},z=d-\pmb{x}^T\pmb{y}=d-\pmb{c}^T(\pmb{U}_k^{-1}\pmb{L}_k^{-1})\pmb{b}=d-\pmb{c}^T\pmb{A}^{-1}\pmb{b}$$

所以：$$\pmb{A}_{k+1}=\pmb{L}_{k+1}\pmb{U}_{k+1}$$

其中 $$\pmb{L}_{k+1}=\begin{bmatrix}\pmb{L}_k&\pmb{0}\\\pmb{c}^T\pmb{U}_k^{-1}&1\end{bmatrix},\pmb{U}_{k+1}=\begin{bmatrix}\pmb{U}_k&\pmb{y}\\\pmb{0}^T&d-\pmb{c}^T\pmb{A}^{-1}\pmb{b}\end{bmatrix}$$

因为 $$\pmb{A}_{k+1}$$ 和 $$\pmb{L}_{k+1}$$ 可逆，所以 $$\pmb{U}_{k+1}$$ 可逆，则 $$d-\pmb{c}^T\pmb{A}^{-1}\pmb{b}\ne0$$ 。即 $$\pmb{A}_{k+1}$$ 可以分解为 $$\pmb{L}_{k+1}\pmb{U}_{k+1}$$ 。

综上，定理得证。

## LU分解的唯一性

对于 $$\pmb{A}=\pmb{LU}$$ 而言，$$\pmb{L}$$ 是单位下三角矩阵，主对角线元素为 $$1$$ 。对于 $$\pmb{U}$$ ，以 $$3\times 3$$ 为例，可以转化为：

$$\pmb{U}=\begin{bmatrix}u_{11}&u_{12}&u_{13}\\0&u_{22}&u_{23}\\0&0&u_{33}\end{bmatrix}=\begin{bmatrix}u_{11}&0&0\\0&u_{22}&0\\0&0&u_{33}\end{bmatrix}\begin{bmatrix}1&\frac{u_{12}}{u_{11}}&\frac{u_{13}}{u_{11}}\\0&1&\frac{u_{23}}{u_{22}}\\0&0&1\end{bmatrix}=\pmb{D}\pmb{U}'$$

所以：$$\pmb{A}=\pmb{LDU}'$$

假设 $$\pmb{A}=\pmb{L}_1\pmb{D}_1\pmb{U}_1'$$ ，$$\pmb{A}=\pmb{L}_2\pmb{D}_2\pmb{U}_2'$$ ，则：

$$\pmb{L}_1\pmb{D}_1\pmb{U}_1'=\pmb{L}_2\pmb{D}_2\pmb{U}_2'$$

由因为 $$\pmb{L}_i$$ 和 $$\pmb{U}'_i$$ 都可逆，所以：

$$\begin{split}\pmb{L}_1^{-1}\pmb{L}_1\pmb{D}_1\pmb{U}_1'\pmb{U}_2'^{-1}&=\pmb{L}_1^{-1}\pmb{L}_2\pmb{D}_2\pmb{U}_2'\pmb{U}_2'^{-1}\\\pmb{D}_1\pmb{U}'_1\pmb{U}_2'^{-1}&=\pmb{L}_1^{-1}\pmb{L}_2\pmb{D}_2\end{split}$$

继续以 $$3$$ 阶方阵为例，将上式等号左右分别用矩阵方式展开，得：

$$\begin{bmatrix}(\pmb{D}_1)_{11}&*&*\\0&(\pmb{D}_1)_{22}&*\\0&0&(\pmb{D}_1)_{33}\end{bmatrix}=\begin{bmatrix}(\pmb{D}_2)_{11}&0&0\\*&(\pmb{D}_2)_{22}&0\\*&*&(\pmb{D}_2)_{33}\end{bmatrix}$$

所以：$$\pmb{D}_1=\pmb{D}_2$$ ，非主元的值 $$* = 0$$ ，故 $$\pmb{U}_1'\pmb{U}_2'^{-1}=\pmb{I}, \pmb{L}_1^{-1}\pmb{L}_2=\pmb{I}$$

所以：$$\pmb{U}_1'=\pmb{U}_2',\pmb{L}_1=\pmb{L}_2$$ ，即 LU 分解具有唯一性。

证毕。

## LU分解的应用

### 求解线性方程组

此应用在《机器学习数学基础》第2章2.3.3节中有详细介绍，请参阅。

### 计算行列式

利用LU分解可以手工计算 $$n$$ 阶行列式。

$$|\pmb{A}|=|\pmb{LU}|=|\pmb{L}||\pmb{U}|$$

三角矩阵的行列式等于主对角元乘积。

所以：$$|\pmb{L}|=1$$ ，则：

$$|\pmb{A}|=|\pmb{U}|=\prod_{i=1}^nu_{ii}$$



## 参考文献

[1]. [https://ccjou.wordpress.com/2010/09/01/lu-分解/](https://ccjou.wordpress.com/2010/09/01/lu-%e5%88%86%e8%a7%a3/)