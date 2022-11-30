# 特征值的代数重数与几何重数

*打开本页，如果没有显示公式，请刷新页面。*

## 定义

令 $$\pmb{A}$$ 为一个 $$n\times n$$ 阶矩阵，非零向量 $$\pmb{x}$$ ，有：

$$\pmb{Ax}=\lambda\pmb{x}\tag{1.1}$$

则 $$\lambda$$ 是 $$\pmb{A}$$ 的特征值，$$\pmb{x}$$ 是对应的特征向量。

由（1.1）可得：

$$(\pmb{A}-\lambda\pmb{I})\pmb{x}=\pmb{0}\tag{1.2}$$

故 $$\pmb{A}-\lambda\pmb{I}$$ 的零空间 $$N(\pmb{A}-\lambda\pmb{I})$$ $$^{[2]}$$（或称对应 $$\lambda$$ 的特征空间）包括非零向量 $$\pmb{x}$$ ，所以 $$\pmb{A}-\lambda\pmb{I}$$ 是不可逆矩阵，即 

$$det(\pmb{A}-\lambda\pmb{I})=0\tag{1.3}$$ 

定义 $$\pmb{A}$$ 的特征多项式为：

$$p(t)=det(\pmb{A}-t\pmb{I}) \tag{1.4}$$

$$\lambda$$ 即为 $$p(t)$$ 的根。

设 $$\pmb{A}$$ 有 $$k$$ 个相异的特征值 $$\lambda_1,\lambda_2,\cdots,\lambda_k,1\le k\le n$$ ，特征多项式可以分解为：

$$p(t)=det(\pmb{A}-t\pmb{I})=(\lambda_1-t)^{\beta_1}\cdots(\lambda_k-t)^{\beta_k} \tag{1.5}$$

其中特征值 $$\lambda_i$$ 的重根数 $$\beta_i$$ 称为**代数重数**（algebraic multiplicity）。

$$n$$ 次多项式 $$p(t)$$ 有 $$n$$ 个根（包含重根），则：$$\beta_1+\cdots+\beta_k=n$$ 。

特征空间 $$N(\pmb{A}-\lambda\pmb{I})$$ 的维数 $$\dim N(\pmb{A}-\lambda\pmb{I})$$ 称为 $$\lambda_i$$ 的**几何重数**（geometric multiplicity），也就是对应 $$\lambda_i$$ 的最大线性无关的特征向量数。

## 几何重数不大于代数重数

下面参考文献 [3] 给出此命题的证明方法

令 $$\lambda_1,\cdots,\lambda_k$$ 为 $$n\times n$$ 阶矩阵 $$\pmb{A}$$ 的相异特征值，$$k\le n$$ 。特征值 $$\lambda_i$$ 的代数重数为 $$\beta_i$$ 。则 $$\pmb{A}$$ 的特征多项式为：

$$\begin{split}p(t)&=\det(\pmb{A}-t\pmb{I})\\&=(\lambda_1-t)^{\beta_1}(\lambda_2-t)^{\beta_2}\cdots(\lambda_k-t)^{\beta_k}\end{split}$$​

其中，$$\sum_{i=1}^k\beta_i=n$$ 。

上述所要证明的命题，用数学式表示：$$\dim N(\pmb{A}-\lambda_i\pmb{I})\le\beta_i,i=1,2,\cdots,k$$ 。

$$\pmb{A}-\lambda_i\pmb{I}$$ 的特征多项式：

$$\begin{split}p_{\pmb{A}-\lambda_i\pmb{I}}(t)&=\det((\pmb{A}-\lambda_i\pmb{I})-t\pmb{I})\\&=\det(\pmb{A}-(\lambda_i+t)\pmb{I})\\&=(\lambda_1-(\lambda_i+t))^{\beta_1}(\lambda_2-(\lambda_i+t))^{\beta_2}\cdots(\lambda_k-(\lambda_i+t))^{\beta_k}\\&=((\lambda_1-\lambda_i)-t)^{\beta_1}((\lambda_2-\lambda_i)-t)^{\beta_2}\cdots((\lambda_k-\lambda_i)-t)^{\beta_k}\end{split}$$​​

对于第 $$i$$ 项，$$\lambda_i-\lambda_i=0$$ ，所以该项为 $$(-t)^{\beta_i}$$ 。因此， $$\pmb{A}-\lambda_i\pmb{I}$$ 有特征值 0 ，其代数重数为 $$\beta_i$$ ，以及 $$k-1$$ 个相异非零特征值 $$\lambda_j-\lambda_i$$ ，代数重数为 $$\beta_j$$ ，$$1\le j \le k$$ 且 $$j\ne i$$ ，根据 Schur 定理，$$\pmb{A}-\lambda_i\pmb{I}$$ 可三角化为：

$$\pmb{A}-\lambda_i\pmb{I}=\pmb{UTU}^{\ast}$$

其中：$$\pmb{U}$$ 是一个酉矩阵（unitary matrix，又译作“幺正矩阵”或“么正矩阵”），满足 $$\pmb{U}^{\ast}=\pmb{U}^{-1}$$ 。$$\pmb{T}$$ 是上三角矩阵。因为 $$\pmb{A}-\lambda_i\pmb{I}$$ 相似于 $$\pmb{T}$$ ，可知 $$\rank(\pmb{A}-\lambda_i\pmb{I})=\rank\pmb{T}$$ 而且这两个矩阵有相同的特征值$$^{[4]}$$ 。所以，$$\pmb{T}$$ 的主对角元为 $$\pmb{A}-\lambda_i\pmb{I}$$ 的特征值，也就是说 $$\pmb{T}$$ 的主对角元包含 $$\beta_i$$ 个零元，以及 $$n-\beta_i$$ 个非零元，表明 $$\rank\pmb{T}\ge n-\beta_i$$ 。由“秩—零度定理”$$^{[2]}$$ 可得：

$$\begin{split}\dim N(\pmb{A}-\lambda_i\pmb{I})&=n-\rank(\pmb{A}-\lambda_i\pmb{I})\\&=n-\rank\pmb{T}\\&\le n-(n-\beta_i)=\beta_i\end{split}$$

证毕。

## 参考文献

[1]. [线代启示录：特征值的代数重数与几何重数](https://ccjou.wordpress.com/2015/11/19/%e7%89%b9%e5%be%b5%e5%80%bc%e7%9a%84%e4%bb%a3%e6%95%b8%e9%87%8d%e6%95%b8%e8%88%87%e5%b9%be%e4%bd%95%e9%87%8d%e6%95%b8/)

[2]. [矩阵的秩：零空间](./rank.html)

[3]. [现代启示录：几何重数不大于代数重数的证明](https://ccjou.wordpress.com/2014/11/14/%E5%B9%BE%E4%BD%95%E9%87%8D%E6%95%B8%E4%B8%8D%E5%A4%A7%E6%96%BC%E4%BB%A3%E6%95%B8%E9%87%8D%E6%95%B8%E7%9A%84%E8%AD%89%E6%98%8E/)

[4]. [相似矩阵](./similarity.html)