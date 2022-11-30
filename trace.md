# 矩阵的迹

$$n$$ 阶方阵 $$\pmb{A}$$ 的迹：$$\operatorname{trace}\pmb{A}=\sum_{i=1}^na_{ii}$$

迹与行列式都是方阵的函数。

设 $$\pmb{A}$$ 的特征值 $$\lambda_1,\cdots,\lambda_n$$ ，则：

$$\operatorname{trace}\pmb{A}=\lambda_1+\cdots+\lambda_n$$

$$\det\pmb{A}=\lambda_1\cdots\lambda_n$$

## 性质$$^{[1]}$$

- $$\operatorname{trace}(\pmb{A}+\pmb{B})=\trace\pmb{A}+\operatorname{trace}\pmb{B}$$

  推导

  $$\operatorname{trace}(\pmb{A}+\pmb{B})=\sum_{i=1}^n(a_{ii}+b_{ii})=\sum_{i=1}^na_{ii}+\sum_{i=1}^nb_{ii}=\operatorname{trace}\pmb{A}+\operatorname{trace}\pmb{B}$$​

- $$\operatorname{trace}(c\pmb{A})=c(\operatorname{trace}\pmb{A})$$​

  推导

  $$\operatorname{trace}(c\pmb{A})=\sum_{i=1}^nca_{ii}=c\sum_{i=1}^na_{ii}=c(\operatorname{trace}\pmb{A})$$​​

- $$\operatorname{trace}(\pmb{AB})=\operatorname{trace}(\pmb{BA})$$​​

  推导

  $$\operatorname{trace}(\pmb{AB})=\sum_{i=1}^n(\pmb{AB})_{ii}=\sum_{i=1}^m\left(\sum_{j=1}^na_{ij}b_{ji}\right)$$​​

  $$\operatorname{trace}(\pmb{BA})=\sum_{j=1}^n(\pmb{BA})_{jj}=\sum_{j=1}^n\left(\sum_{i=1}^mb_{ji}a_{ij}\right)$$​​
  
- 转置与共轭的迹：$$\operatorname{trace}\pmb{A}^{\rm{T}}=\operatorname{trace}\pmb{A}$$​​ ，$$\operatorname{trace}\pmb{A}^{\ast}=\overline{\operatorname{trace}{A}}$$​
  
- 相似变换不改变矩阵的迹。若 $$\pmb{M}$$​ 是一个可逆矩阵，则：$$\operatorname{trace}(\pmb{MAM}^{-1})=\operatorname{trace}{\pmb{A}}$$​
  
  证明：
  
  $$\operatorname{trace}(\pmb{MAM}^{-1})=\operatorname{trace}((\pmb{AM}^{-1})\pmb{M})=\operatorname{trace}{\pmb{A}}$$
  
  所以，迹是相似变换下不变的性质之一。
  
## 迹与特征值的关系$$^{[2]}$$​

**式一：** $$\operatorname{trace}(\pmb{A^2})=\sum_{i=1}^n\lambda^2_i$$

幂矩阵 $$\pmb{A}^2$$ 的特征值是 $$\lambda_1^2,\cdots,\lambda_n^2$$​ ，根据前述迹与特征值之间的关系，可以得到上式。

**式二：** $$\operatorname{trace}(\pmb{A}^{\ast}\pmb{A})=\sum_{i=1}^n\sigma^2_i\ge\sum_{i=1}^n|\lambda_i|^2$$

此式称为 **Schur 不等式**。

根据奇异值分解$$^{[3]}$$​​ ：$$\pmb{A}^{\ast}\pmb{A}=(\pmb{V\Sigma U}^{\ast})(\pmb{U\Sigma V}^{\ast})=\pmb{V}\pmb{\Sigma}^2\pmb{U}^{\ast}$$

即知 $$\sigma^2_1,\cdots,\sigma^2_n$$ 是 $$\pmb{A}^{\ast}\pmb{A}$$ 的特征值，有：

$$\operatorname{trace}(\pmb{A}^{\ast}\pmb{A})=\operatorname{trace}(\pmb{V}\pmb{\Sigma}^2\pmb{U}^{\ast})=\operatorname{trace}(\pmb{\Sigma}^2\pmb{V}\pmb{U}^{\ast})=\sum_{i=1}^n\sigma_i^2$$

使用 Schur 分解，$$\pmb{A}^{\ast}\pmb{A}=(\pmb{UT}^{\ast}\pmb{U}^{\ast})(\pmb{UT}\pmb{U}^{\ast})=\pmb{UT}^{\ast}\pmb{TU}^{\ast}$$ ，得到：

$$\operatorname{trace}(\pmb{A}^{\ast}\pmb{A})=\operatorname{trace}(\pmb{UT}^{\ast}\pmb{TU}^{\ast})=\operatorname{trace}(\pmb{T}^{\ast}\pmb{TQ}^{\ast}\pmb{Q})=\operatorname{trace}(\pmb{T}^{\ast}\pmb{T})$$

因为 $$t_{ii}=\lambda_i, i=1,\cdots,n$$ ，

$$\operatorname{trace}(\pmb{T}^{\ast}\pmb{T})=\sum_{i=1}^n\sum_{j=1}^n|t_{ij}|^2=\sum_{i=1}^n|\lambda_i|^2+\sum_{i\lt j}|t_{ij}|^2$$

所以：$$\operatorname{trace}(\pmb{A}^{\ast}\pmb{A})\ge\sum_{i=1}^n|\lambda_i|^2$$

**式三：** $$|\operatorname{trace}{\pmb{A}}|=\left|\sum_{i=1}^n\lambda_i\right|\le\sum_{i=1}^n\sigma_i$$​

$$\begin{split}|\operatorname{trace}\pmb{A}|&=\left|\sum_{i=1}^na_{ii}\right|=\left|\sum_{i=1}^n\sum_{j=1}^nu_{ij}\sigma_j\overline{v_{ij}}\right|\operatorname{trace}\&\le\sum_{j=1}^n\left|\sum_{i=1}^nu_{ij}\overline{v_{ij}}\right|\sigma_j\quad(根据三角不等式)\\&\le\sum_{j=1}^n\left(\sum_{i=1}^n|u_{ij}\overline{v_{ij}}|\right)\sigma_j\quad(根据三角不等式)\\&\le\sum_{j=1}^n\sigma_j\quad(根据 Cauchy 不等式)\end{split}$$

Cauchy 不等式：

$$\sum_{i=1}^n|u_{ij}\overline{v_{ij}}|=\sum_{i=1}^n|u_{ij}|\cdot|\overline{v_{ij}}|\le\sqrt{\sum_{i=1}^n|u_{ij}|^2\sum_{i=1}^n|\overline{v_{ij}}|^2}=1$$​

**式四：** $$|\operatorname{trace}{\pmb{A}}|\le\sum_{i=1}^n|\lambda_i|\le\sum_{i=1}^n\sigma_i$$​​





## 参考文献

[1]. [线代启示录：迹数的性质与应用](https://ccjou.wordpress.com/2010/08/18/%E8%B7%A1%E6%95%B8%E7%9A%84%E6%80%A7%E8%B3%AA%E8%88%87%E6%87%89%E7%94%A8/)

[2]. [线代启示录：矩阵迹数与特征值和奇异值的关系](https://ccjou.wordpress.com/2013/10/30/%e7%9f%a9%e9%99%a3%e8%b7%a1%e6%95%b8%e8%88%87%e7%89%b9%e5%be%b5%e5%80%bc%e5%92%8c%e5%a5%87%e7%95%b0%e5%80%bc%e7%9a%84%e9%97%9c%e4%bf%82/)

[3]. [常用的矩阵分解](./matrix_factorization.html)
