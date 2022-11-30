# 向量范数

*打开本页，如果不能显示公式，请刷新页面。*

《机器学习数学基础》第1章1.5.3节介绍了向量范数的基本定义。

本文在上述基础上，介绍向量范数的有关性质。

**注意：**以下均在欧几里得空间塔伦，即欧氏范数。

## 性质

- 实（或复）向量 $$\pmb{x}$$ ，范数 $$\begin{Vmatrix}x\end{Vmatrix}$$ 满足：

  - $$\begin{Vmatrix}x\end{Vmatrix}\ge0$$
  - $$\begin{Vmatrix}x\end{Vmatrix}=0 \Leftrightarrow \pmb{x}=\pmb{0}$$
  - $$\begin{Vmatrix}cx\end{Vmatrix}=|c|\begin{Vmatrix}x\end{Vmatrix}$$ ，$$c$$ 是标量

- 设 $$\pmb{x,y}\in\mathbb{C}^n$$ ，根据[施瓦茨不等式](./cauchy-schwarz.html)：$$|\pmb{x}^*\pmb{y}|\le\begin{Vmatrix}x\end{Vmatrix}\begin{Vmatrix}y\end{Vmatrix}$$ 。

  若 $$n=1$$ ，则上式退化为 $$|\overline{x}y|\le|x||y|$$ ，其中 $$x,y\in\mathbb{C}$$ 。因为 $$|\overline{x}|=|x|$$ ，所以 $$|\overline{x}y|\le|\overline{x}||y|$$

- 三角不等式：$$\pmb{x}+\pmb{y}\le \begin{Vmatrix}\pmb{x}\end{Vmatrix}+\begin{Vmatrix}\pmb{y}\end{Vmatrix}$$

  **证明**

  $$\begin{split}\begin{Vmatrix}\pmb{x}+\pmb{y}\end{Vmatrix}^2 &= (\pmb{x}+\pmb{y})^*(\pmb{x}+\pmb{y})\\ &= \pmb{x}^*\pmb{x}+\pmb{x}^*\pmb{y}+\pmb{y}^*\pmb{x}+\pmb{y}^*\pmb{y}\\&=\begin{Vmatrix}\pmb{x}\end{Vmatrix}^2+\pmb{x}^*\pmb{y}+\pmb{y}^*\pmb{x}+\begin{Vmatrix}\pmb{y}\end{Vmatrix}^2\end{split}$$

  根据复数的性质和施瓦茨不等式：

  $$\pmb{x}^*\pmb{y}+\pmb{y}^*\pmb{x}=\pmb{x}^*\pmb{y}+\overline{\pmb{x}^*\pmb{y}}=2Re(\pmb{x}^*\pmb{y})\le 2|\pmb{x}^*\pmb{y}|\le2\begin{Vmatrix}\pmb{x}\end{Vmatrix}\begin{Vmatrix}\pmb{y}\end{Vmatrix}$$

  由上述结果，可得：

  $$\begin{Vmatrix}\pmb{x}+\pmb{y}\end{Vmatrix}^2 \le \begin{Vmatrix}\pmb{x}\end{Vmatrix}^2+2\begin{Vmatrix}\pmb{x}\end{Vmatrix}\begin{Vmatrix}\pmb{y}\end{Vmatrix}+\begin{Vmatrix}\pmb{y}\end{Vmatrix}^2=(\begin{Vmatrix}\pmb{x}\end{Vmatrix}+\begin{Vmatrix}\pmb{y}\end{Vmatrix})^2$$

  证毕。

## 极小范数$$^{[1]}$$

$$m\times n$$ 的矩阵 $$\pmb{A}$$ ，列空间 $$C(\pmb{A})=\{\pmb{Ax}|\pmb{x}\in\mathbb{R}^n\}$$ （ $$C(\pmb{A})$$ 是 $$\mathbb{R}^m$$ 的一个子空间），对任一 $$\pmb{b}\in C(\pmb{A})$$ ，线性方程组 $$\pmb{Ax}=\pmb{b}$$ 有解。在解集合中，有一个特解，在 $$\pmb{A}$$ 的行空间，即 $$\pmb{A}^T$$ 的列空间 $$C(\pmb{A}^T)$$ ，并且具有最小的 $$l_2$$ 范数，称为**极小范数解**（minimum norm solution），记作 $$\pmb{x}^+$$ ，即：

 $$\pmb{x}^+\in C(\pmb{A}^T)$$ 使得 $$\pmb{Ax}^+=\pmb{b}$$

### 定理一

若 $$\pmb{b}\in C(\pmb{A})$$ ，则存在唯一的 $$\pmb{y}\in C(\pmb{A}^T)$$ 使得 $$\pmb{Ay}=\pmb{b}$$ 。

**证明**

设特解 $$\pmb{x}\in \mathbb{R}^n$$ 使得 $$\pmb{Ax}=\pmb{b}$$ 。

在 $$\mathbb{R}^n$$ 中，$$\pmb{A}$$ 的行空间 $$C(\pmb{A}^T)$$ 是零空间 $$N(\pmb{A})$$ 的正交补（参考：矩阵基本子空间$$^{[2]}$$）。则 $$\pmb{x}$$ 可以分解为 $$\pmb{x}=\pmb{y}+\pmb{z}$$ ，其中 $$\pmb{y}\in C(\pmb{A}^T), \pmb{z}\in N(\pmb{A})$$ ，得：

$$\pmb{Ax}=\pmb{A}(\pmb{y}+\pmb{z})=\pmb{Ay}+\pmb{Az}=\pmb{b}$$

这说明 $$\pmb{y}$$ 也是一个特解。

设 $$\pmb{y},\pmb{y}'\in C(\pmb{A}^T)$$ 使得 $$\pmb{Ay}=\pmb{b},\pmb{Ay}'=\pmb{b}$$ 。两式子相减：

$$\pmb{A}(\pmb{y}-\pmb{y}')=\pmb{0}$$

所以 $$\pmb{y}-\pmb{y}'\in N(\pmb{A})$$ 。

又因为 $$\pmb{y}-\pmb{y}'\in C(\pmb{A}^T)$$ ，

合并以上结果，得：

$$\pmb{y}-\pmb{y}'\in N(\pmb{A})\cap C(\pmb{A}^T)=\{\pmb{0}\}$$

即 $$\pmb{y}=\pmb{y}'$$ 。$$\pmb{y}$$ 唯一。

证毕。

### 定理二

若 $$\pmb{b}\in C(\pmb{A})$$ 且 $$\pmb{y}\in \{\pmb{x}|\pmb{Ax}=\pmb{b}\}$$ 具有最小 $$l_2$$ 范数，则 $$\pmb{y}\in C(\pmb{A}^T)$$ 。

**证明**

由定理一，任意特解可以表示为 $$\pmb{x}=\pmb{y}+\pmb{z}$$ ，且 $$\pmb{y}$$ 唯一存在。因为 $$\pmb{y}\bot\pmb{z}$$ ，则：

$$\begin{Vmatrix}\pmb{x}\end{Vmatrix}^2=\begin{Vmatrix}\pmb{y}\end{Vmatrix}^2+\begin{Vmatrix}\pmb{z}\end{Vmatrix}^2\ge\begin{Vmatrix}\pmb{y}\end{Vmatrix}^2$$

当 $$\pmb{z}=\pmb{0}$$ 时，上式等号成立。

证毕。

### 定理三

若 $$rank \pmb{A}=m$$ ，即 $$\pmb{A}$$ 的列向量线性无关，则 $$\pmb{Ax}=\pmb{b}$$ 必有解，且极小范数解为：

$$\pmb{x}^+=\pmb{A}^T(\pmb{AA}^T)^{-1}\pmb{b}$$

**证明**

因为 $$rank \pmb{A}=m$$ ，则 $$\dim C(\pmb{A})=m$$ ，列空间 $$C(\pmb{A})$$ 充满 $$\mathbb{R}^m$$ ，所以任一 $$\pmb{b}\in\mathbb{R}^m$$ 使 $$\pmb{Ax}=\pmb{b}$$ 有解。

*推导方法1*

因为 $$\pmb{A}$$ 的列向量线性无关，所以 $$\pmb{x}^+\in C(\pmb{A}^T)$$ 可唯一表示为列向量的线性组合，即存在唯一的 $$\pmb{c}$$ 使得 $$\pmb{x}^+=\pmb{A}^T\pmb{c}$$ 。代入 $$\pmb{Ax}^+=\pmb{b}$$ ，得：

$$\pmb{AA}^T\pmb{c}=\pmb{b}$$

因为 $$rank(\pmb{AA}^T)=rank(\pmb{A})=m$$ ，所以 $$\pmb{AA}^T$$ 可逆$$^{[3]}$$。

故：$$\pmb{c}=(\pmb{AA}^T)^{-1}\pmb{b}$$

解得：$$\pmb{x}^+=\pmb{A}^T(\pmb{AA}^T)^{-1}\pmb{b}$$

*推导方法2*，使用拉格朗日乘数法$$^{[4]}$$

$$\begin{split}minimize \quad &\begin{Vmatrix}\pmb{x}\end{Vmatrix}\\subject\quad to \quad& \pmb{Ax}=\pmb{b}\end{split}$$

最小化 $$\begin{Vmatrix}\pmb{x}\end{Vmatrix}$$ ，等价于最小化 $$\begin{Vmatrix}\pmb{x}\end{Vmatrix}^2=\pmb{x}^T\pmb{x}$$

拉格朗日函数：$$L(\pmb{x},\pmb{\lambda})=\pmb{x}^T\pmb{x}+\pmb{\lambda}^T(\pmb{Ax}-\pmb{b})$$

其中 $$\pmb{\lambda}$$ 是 $$m$$ 维拉格朗日乘数向量。计算：

$$\begin{split}\frac{\partial L}{\partial\pmb{x}}&=2\pmb{x}+\pmb{A}^T\pmb{\lambda}\\\frac{\partial L}{\partial\pmb{\lambda}}&=\pmb{Ax}-\pmb{b}\end{split}$$

令上述两式等于零，得到最优化条件式。得：$$\pmb{x}^+=-\frac{1}{2}\pmb{A}^T\pmb{\lambda}$$ ，代入 $$\pmb{Ax}^+=\pmb{b}$$ ，得：

$$-\frac{1}{2}\pmb{AA}^T\pmb{\lambda}=\pmb{b}$$

解得：$$\pmb{\lambda}=-2(\pmb{AA}^T)^{-1}\pmb{b}$$

所以：$$\pmb{x}^+=\pmb{A}^T(\pmb{AA}^T)^{-1}\pmb{b}$$

### 计算方法

计算 $$\pmb{x}^+$$ ，可以使用QR分解$$^{[5]}$$ 。

设 $$\pmb{A}^T=\pmb{QR}$$ ，其中 $$\pmb{Q}$$ 是 $$n\times m$$ 矩阵，且 $$\pmb{Q}^T\pmb{Q}=\pmb{I}_m$$ ，$$\pmb{R}$$ 是 $$m$$ 阶上三角矩阵。

$$\begin{split}\pmb{x}^+ &= \pmb{A}^T(\pmb{AA}^T)^{-1}\pmb{b}\\ &= \pmb{QR}(\pmb{R}^T\pmb{Q}^T\pmb{QR})^{-1}\pmb{b}\\&=\pmb{QR}(\pmb{R}^T\pmb{R})^{-1}\pmb{b}\\&=\pmb{QRR}^{-1}(\pmb{R}^T)^{-1}\pmb{b}\\&=\pmb{Q}(\pmb{R}^T)^{-1}\pmb{b}\end{split}$$

最佳值：

$$\begin{split}\begin{Vmatrix}\pmb{x}\end{Vmatrix}^2 &= (\pmb{A}^T(\pmb{AA}^T)^{-1}\pmb{b})^T(\pmb{A}^T(\pmb{AA}^T)^{-1}\pmb{b})\\&=\pmb{b}^T(\pmb{AA}^T)^{-1}\pmb{b}\\&=\pmb{b}^T(\pmb{R}^T\pmb{R})^{-1}\pmb{b}\end{split}$$

**注意：**

- 在上述计算中，使用了矩阵求导等相关计算，请参阅《机器学习数学基础》第4章“向量分析”有关内容，书中的附录中也附有各种计算公式。
- 定理三，仅限于 $$\pmb{A}$$ 的列向量线性无关。若列向量线性相关，即 $$rank\pmb{A}\le m$$ ，则 $$\pmb{AA}^T$$ 不可逆。此时仍有极小范数解，表示为 $$\pmb{x}^+=\pmb{A}^+\pmb{b}$$ ，其中 $$\pmb{A}^+$$ 称为 $$\pmb{A}$$ 的伪逆矩阵（或广义逆矩阵）$$^{[6]}$$。

## 参考文献

[1]. [https://ccjou.wordpress.com/2014/05/21/極小範數解/](https://ccjou.wordpress.com/2014/05/21/%e6%a5%b5%e5%b0%8f%e7%af%84%e6%95%b8%e8%a7%a3/)

[2]. [矩阵基本子空间](./basetheory.html)

[3]. [矩阵的秩](./rank.html)

[4]. [拉格朗日乘数法](./lagrangemulti.html)

[5]. [QR分解](./qr_decomposition.html)

[6]. [维基百科：广义逆矩阵](https://zh.wikipedia.org/wiki/%E5%B9%BF%E4%B9%89%E9%80%86%E9%98%B5)



