# 可逆矩阵

*打开本页，如果没有显示公式，请刷新页面。*

《机器学习数学基础》第2章2.3.1节阐述了可逆矩阵的定义、性质，并演示了Python中的计算函数及其应用。

若一个矩阵存在逆矩阵，则称之为可逆矩阵，或者非奇异矩阵。

并不是所有的矩阵都是可逆矩阵。

如果矩阵可逆，用手工计算，通过高斯法可以得到其逆矩阵。此外，还有其他方法可以求得逆矩阵。此处以常用的2、3阶矩阵为例，说明手工计算逆矩阵的方法。

以下内容主要参考[5]。

## 2阶逆矩阵公式

设 $$\pmb{A}=\begin{bmatrix}a&b\\c&d\end{bmatrix}$$ 是 $$2\times2$$ 可逆矩阵，则其逆矩阵公式：

$$\pmb{A}^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$$

### 方法1：化简增广矩阵

其推导过程如下：

$$\begin{split}\left[\begin{array}{c c|c c}a&b&1&0\\c&d&0&1\end{array}\right] &\to \left[\begin{array}{c c|c c}a&b&1&0\\0&ad-bc&-c&a\end{array}\right]\\&\to\left[\begin{array}{c c|c c}a&b&1&0\\0&1&\frac{-c}{ad-bc}&\frac{a}{ad-bc}\end{array}\right]\\&\to\left[\begin{array}{c c|c c}a&0&\frac{ad}{ad-bc}&\frac{-ab}{ad-bc}\\0&1&\frac{-c}{ad-bc}&\frac{a}{ad-bc}\end{array}\right]\\&\to\left[\begin{array}{c c|c c}1&0&\frac{d}{ad-bc}&\frac{-b}{ad-bc}\\0&1&\frac{-c}{ad-bc}&\frac{a}{ad-bc}\end{array}\right]\end{split}$$

故，矩阵 $$\pmb{A}$$ 可逆的充要条件是 $$|\pmb{A}|=ad-bc\ne0$$ 。

### 方法2：伴随矩阵

设 $$\pmb{X}=\begin{bmatrix}x_{11}&x_{12}\\x_{21}&x_{22}\end{bmatrix}$$ 是矩阵 $$\pmb{A}=\begin{bmatrix}a&b\\c&d\end{bmatrix}$$ 的逆矩阵，则 $$\pmb{AX}=\pmb{I}_2$$ ，即：

$$\begin{bmatrix}a&b\\c&d\end{bmatrix}\begin{bmatrix}x_{11}&x_{12}\\x_{21}&x_{22}\end{bmatrix}=\begin{bmatrix}1&0\\0&1\end{bmatrix}$$

根据矩阵乘法的含义$$^{[1]}$$，可得：

$$\begin{bmatrix}a&b\\c&d\end{bmatrix}\begin{bmatrix}x_{11}\\x_{21}\end{bmatrix}=\begin{bmatrix}1\\0\end{bmatrix}，\begin{bmatrix}a&b\\c&d\end{bmatrix}\begin{bmatrix}x_{12}\\x_{22}\end{bmatrix}=\begin{bmatrix}0\\1\end{bmatrix}$$

利用克拉默法则$$^{[2]}$$ ，可以解得：

$$x_{11}=\frac{\begin{vmatrix}1&b\\0&d\end{vmatrix}}{\begin{vmatrix}a&b\\c&d\end{vmatrix}}=\frac{d}{ad-bc}$$

$$x_{21}=\frac{\begin{vmatrix}a&1\\c&0\end{vmatrix}}{\begin{vmatrix}a&b\\c&d\end{vmatrix}}=\frac{-c}{ad-bc}$$

$$x_{12}=\frac{\begin{vmatrix}0&b\\1&d\end{vmatrix}}{\begin{vmatrix}a&b\\c&d\end{vmatrix}}=\frac{-b}{ad-bc}$$

$$x_{22}=\frac{\begin{vmatrix}a&0\\c&1\end{vmatrix}}{\begin{vmatrix}a&b\\c&d\end{vmatrix}}=\frac{a}{ad-bc}$$

### 方法3：凯莱-哈密顿定理

根据凯莱-哈密顿定理，对 $$n$$ 阶方阵，特征多项式为 $$p(\lambda)=det(\pmb{A}-\lambda\pmb{I})$$ （ $$det(\cdot)$$ 表示行列式，与 $$|\cdot|$$ 含义一样），有 $$p(\pmb{A})=0$$ $$ 。^{[3]}$$

对矩阵 $$\pmb{A}=\begin{bmatrix}a&b\\c&d\end{bmatrix}$$ ，其特征多项式：

$$\begin{split}p(\lambda)&=\begin{vmatrix}a-\lambda&b\\c&d-\lambda\end{vmatrix}\\&=\lambda^2-(a+d)\lambda+(ad-bc)\\&=\lambda^2-Tr(\pmb{A})\lambda+|\pmb{A}|\end{split}$$

根据凯莱-哈密顿定理，有：

$$p(\pmb{A})=\pmb{A}^2-Tr(\pmb{A})\pmb{A}+|\pmb{A}|\pmb{I}_2=0$$

上式乘以 $$\frac{1}{|\pmb{A}|}\pmb{A}^{-1}$$ ，得：

$$\frac{1}{|\pmb{A}|}\pmb{A}^2\pmb{A}^{-1}-\frac{1}{|\pmb{A}|}Tr(\pmb{A})\pmb{A}\pmb{A}^{-1}+\frac{1}{|\pmb{A}|}|\pmb{A}|\pmb{I}_2\pmb{A}^{-1}=0$$

$$\frac{1}{|\pmb{A}|}\pmb{A}-\frac{1}{|\pmb{A}|}Tr(\pmb{A})\pmb{I}_2+\pmb{I}_2\pmb{A}^{-1}=0$$

因为 $$\pmb{I}_2\pmb{A}^{-1}=\pmb{A}^{-1}$$ ，则由上式可得：

$$\begin{split}\pmb{A}^{-1}&=\frac{1}{|\pmb{A}|}(-\pmb{A}+Tr(\pmb{A})\pmb{I}_2)\\&=\frac{1}{|\pmb{A}|}\left(-\begin{bmatrix}a&b\\c&d\end{bmatrix}+(a+d)\begin{bmatrix}1&0\\0&1\end{bmatrix}\right)\\&=\frac{1}{|\pmb{A}|}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}\end{split}$$



## 3阶逆矩阵公式

### 方法1：化简增广矩阵

与前述 $$2$$ 阶方法一样。

### 方法2：伴随矩阵

设 $$\pmb{A}=[a_{ij}]_{m\times n}$$ 的伴随矩阵（adjugate）$$adj\pmb{A}$$ ，各元素为 $$(adj\pmb{A})_{ij}=(-1)^{i+j}det\widetilde{\pmb{A}}_{ji}$$ ，其中 $$\widetilde{\pmb{A}}_{ji}$$ 表示移除 $$\pmb{A}$$ 的第 $$j$$ 行与第 $$i$$ 列之后得到的 $$(n-1)\times(n-1)$$ 子阵，$$det\widetilde{\pmb{A}}_{ji}$$ 为余子式（minor），$$(-1)^{i+j}det\widetilde{\pmb{A}}_{ji}$$ 称为 $$a_{ji}$$ 的余子式。

伴随矩阵的等式$$^{[4]}$$：

$$\pmb{A}(adj\pmb{A})=(det\pmb{A})\pmb{I}_n$$

上式两边乘以 $$\frac{1}{|\pmb{A}|}\pmb{A}^{-1}$$ ，得：

$$\pmb{A}^{-1}=\frac{1}{|\pmb{A}|}adj\pmb{A}$$

以 $$2\times2$$ 矩阵为例，则可以得到 $$\pmb{A}=\begin{bmatrix}a&b\\c&d\end{bmatrix}$$ 的逆矩阵：

$$\pmb{A}^{-1}=\frac{1}{|\pmb{A}|}\begin{bmatrix}\widetilde{\pmb{A}}_{11}&-\widetilde{\pmb{A}}_{21}\\-\widetilde{\pmb{A}}_{12}&\widetilde{\pmb{A}}_{22}\end{bmatrix}=\frac{1}{|\pmb{A}|}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$$

对于 $$3\times3$$ 的矩阵，则为：

$$\pmb{A}^{-1}=\begin{bmatrix}a_{11}&a_{12}&a_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{bmatrix}^{-1}=\frac{1}{|\pmb{A}|}\begin{bmatrix}\begin{vmatrix}a_{22}&a_{23}\\a_{32}&a_{33}\end{vmatrix}-\begin{vmatrix}a_{12}&a_{13}\\a_{32}&a_{33}\end{vmatrix}&\begin{vmatrix}a_{12}&a_{13}\\a_{22}&a_{23}\end{vmatrix}\\-\begin{vmatrix}a_{21}&a_{23}\\a_{31}&a_{33}\end{vmatrix}&\begin{vmatrix}a_{11}&a_{13}\\a_{31}&a_{33}\end{vmatrix}-\begin{vmatrix}a_{11}&a_{13}\\a_{21}&a_{23}\end{vmatrix}\\\begin{vmatrix}a_{21}&a_{22}\\a_{31}&a_{32}\end{vmatrix}-\begin{vmatrix}a_{11}&a_{12}\\a_{31}&a_{32}\end{vmatrix}&\begin{vmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{vmatrix}\end{bmatrix}$$

### 方法3：凯莱-哈密顿定理

参照前述 $$2$$ 阶矩阵的方法，可以写出 $$3$$ 阶矩阵的特征多项式

$$\begin{split}p(\lambda)&=\begin{vmatrix}a_{11}-\lambda&a_{12}&a_{13}\\a_{21}&a_{22}-\lambda&a_{23}\\a_{31}&a_{32}&a_{33}-\lambda\end{vmatrix}\\&=-\lambda^3+(a_{11}+a_{22}+a_{33})\lambda^2-\left(\begin{vmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{vmatrix}+\begin{vmatrix}a_{11}&a_{13}\\a_{31}&a_{33}\end{vmatrix}+\begin{vmatrix}a_{22}&a_{23}\\a_{32}&a_{33}\end{vmatrix}\right)\lambda+\begin{vmatrix}a_{11}&a_{12}&a_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{vmatrix}\\&=-\lambda^3+Tr(\pmb{A})\lambda^2-\left(\begin{vmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{vmatrix}+\begin{vmatrix}a_{11}&a_{13}\\a_{31}&a_{33}\end{vmatrix}+\begin{vmatrix}a_{22}&a_{23}\\a_{32}&a_{33}\end{vmatrix}\right)\lambda+|\pmb{A}|\end{split}$$

令 $$\lambda_1,\lambda_2,\lambda_3$$ 为 $$\pmb{A}$$ 的特征值，因为这些特征值是特征多项式（上式）的跟，所以：

$$\begin{split}p(\lambda)&=-(\lambda-\lambda_1)(\lambda-\lambda_2)(\lambda-\lambda_3)\\&=-\lambda^3+(\lambda_1+\lambda_2+\lambda_3)\lambda^2-(\lambda_1\lambda_2+\lambda_1\lambda_3+\lambda_2\lambda_3)\lambda+\lambda_1\lambda_2\lambda_3\end{split}$$

因为 $$Tr(\pmb{A})=\lambda_1+\lambda_2+\lambda_3$$ ，$$Tr(\pmb{A}^2)=\lambda_1^2+\lambda_2^2+\lambda_3^2$$ （参阅《机器学习数学基础》第3章3.1.2节矩阵的迹），则：

$$\lambda_1\lambda_2+\lambda_1\lambda_3+\lambda_2\lambda_3=\frac{(\lambda_1+\lambda_2+\lambda_3)^2-(\lambda_1^2+\lambda_2^2+\lambda_3^2)}{2}=\frac{[Tr(\pmb{A})]^2-Tr(\pmb{A}^2)}{2}$$

又因为 $$|\pmb{A}|=\lambda_1\lambda_2\lambda_3$$ ，根据凯莱-哈密顿定理：

$$p(\pmb{A})=-\pmb{A}^3+Tr(\pmb{A})\pmb{A}^2-\frac{[Tr(\pmb{A})]^2-Tr(\pmb{A}^2)}{2}\pmb{A}+|\pmb{A}|\pmb{I}_3=0$$

上式乘以 $$\frac{1}{|\pmb{A}|}\pmb{A}^{-1}$$ ，得到 $$3\times3$$ 矩阵的逆矩阵的矩阵多项式：

$$\pmb{A}^{-1}=\frac{1}{|\pmb{A}|}\left(\pmb{A}^2-Tr(\pmb{A})\pmb{A}+\frac{[Tr(\pmb{A})]^2-Tr(\pmb{A}^2)}{2}\pmb{I}_3\right)$$

同时，可以计算 $$\pmb{A}$$ 的伴随矩阵：

$$adj\pmb{A}=\pmb{A}^2-Tr(\pmb{A})+\frac{[Tr(\pmb{A})]^2-Tr(\pmb{A}^2)}{2}\pmb{I}_3$$



##可逆矩阵定理和性质总结

### 定理

令 $$\pmb{A}$$ 是一个 $$n\times n$$ 的实矩阵，则以下表述对于可逆矩阵 $$\pmb{A}$$ 是等价的：

1. $$\pmb{A}$$ 是可逆的，或者说存在 $$\pmb{A}^{-1}$$ 满足 $$\pmb{A}^{-1}\pmb{A}=\pmb{I}_n$$ ，$$\pmb{I}_n$$ 是 $$n$$ 阶单位矩阵。

2. $$\pmb{Ax}=\pmb{0}$$ 仅有平凡解 $$\pmb{x}=\pmb{0}$$ 。

   **证明**：

   若 $$\pmb{A}$$ 可逆，对于 $$\pmb{Ax} = \pmb{0}$$ 两侧左乘 $$\pmb{A}^{-1}$$ 。左侧得：$$\pmb{A}^{-1}\pmb{Ax}=\pmb{Ix}=\pmb{x}$$ ；右侧为：$$\pmb{A}^{-1}\pmb{0}=\pmb{0}$$ 。故 $$\pmb{x}=\pmb0$$ 。 

3. $$\pmb{Ax}=\pmb{b}$$ 有唯一解 $$\pmb{x}=\pmb{A}^{-1}\pmb{b}$$ 。

   **证明**：

   假设此方程有两个相异的解 $$\pmb{u}$$ 和 $$\pmb{v}$$ ，由已知方程式，可知：

   $$\pmb{A}(\pmb{u}-\pmb{v})=\pmb{Au}-\pmb{Av}=\pmb{b}-\pmb{b}=\pmb{0}$$ 

   即非平凡解满足 $$\pmb{Ax}=\pmb{0}$$ 。故 $$\pmb{Ax}=\pmb{b}$$ 有唯一解。

4. $$\pmb{A}$$​ 有 $$n$$​ 个非零主元（pivot）。

5. $$\pmb{A}$$ 的简约行梯形式（reduced row echelon form）为 $$\pmb{I}_n$$ ，或者说 $$\pmb{A}$$ 行等价于 $$\pmb{I}_n$$ 。

   **证明**

   由第3条唯一解知，增广矩阵 $$\begin{bmatrix}\pmb{A}&\pmb{b}\end{bmatrix}$$ 可化简为 $$\begin{bmatrix}\pmb{I}&\pmb{A}^{-1}\pmb{b}\end{bmatrix}$$ ，即：

   $$\pmb{A}^{-1}\begin{bmatrix}\pmb{A}&\pmb{b}\end{bmatrix}=\begin{bmatrix}\pmb{A}^{-1}\pmb{A}&\pmb{A}^{-1}\pmb{b}\end{bmatrix}=\begin{bmatrix}\pmb{I}&\pmb{A}^{-1}\pmb{b}\end{bmatrix}$$

6. $$\pmb{A}$$ 有线性无关的列向量 (column vector)。

   **证明**

   设 $$\begin{bmatrix}\pmb{a}_1&\cdots&\pmb{a}_n\end{bmatrix}$$ 为 $$\pmb{A}$$ 的列向量，则：

   $$\begin{bmatrix}\pmb{a}_1&\cdots&\pmb{a}_n\end{bmatrix}\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix}=x_1\pmb{a}_1+\cdots+x_n\pmb{a}_n=\pmb{0}$$

   如果满足第2条定理，则 $$\begin{bmatrix}\pmb{a}_1&\cdots&\pmb{a}_n\end{bmatrix}$$ 各列向量线性无关。

7. $$\pmb{A}$$ 有线性无关的行向量 (row vector)。

   **证明**

   假设 $$\pmb{A}$$ 的行向量线性相关，其中必定存在至少一行可表示其他行的線性組合，也就是说对 $$\pmb{A}$$ 进行高斯消元法将会至少有一个零行，于是总的轴数就小于 $$n$$ 。这与第4条定理矛盾。假设不成立。

8. $${\rm{rank}}\pmb{A}=n$$​ 。

9. $$\pmb{A}$$ 的列空間 (column space) 为 $$\mathbb R^n$$ 或者说线性变换 $$\pmb{A}:\mathbb{R}^n\to\mathbb{R}^n$$ 是满射（onto，surjective）。

10. $$\pmb{A}$$ 的行空間 (row space) 为 $$\mathbb{R}^n$$ 。

11. $$\pmb{A}$$ 的零空間 (nullspace) 为 $$\{\pmb{0}\}$$ ，或者说线性变换 $$\pmb{A}:\mathbb{R}^n\to\mathbb{R}^n$$ 是一对一（one-to-one，injective）。

12. $$\pmb{A}^{\rm{T}}$$ 的零空間为 $$\{\pmb{0}\}$$ 。

13. $$\det\pmb{A}\ne0$$​ 。

14. $$\pmb{A}$$ 的特征值不为零。

15. $$\pmb{A}^{\rm{T}}\pmb{A}$$ 是实对称正定矩阵。 

16. $$\pmb{A}$$ 的奇异值（sigular value）大于零。  



​    





## 参考文献

[1]. [对矩阵乘法的深入理解](https://qiwsir.gitee.io/mathmetics/multiplication.html)

[2]. [克拉默法则](./equation.html/)

[3]. [https://zh.wikipedia.org/wiki/凱萊–哈密頓定理](https://zh.wikipedia.org/wiki/%E5%87%B1%E8%90%8A%E2%80%93%E5%93%88%E5%AF%86%E9%A0%93%E5%AE%9A%E7%90%86)

[4]. [https://zh.wikipedia.org/wiki/伴随矩阵](https://zh.wikipedia.org/wiki/%E4%BC%B4%E9%9A%8F%E7%9F%A9%E9%98%B5)

[5]. [https://ccjou.wordpress.com/2012/10/04/三階逆矩陣公式/](https://ccjou.wordpress.com/2012/10/04/%E4%B8%89%E9%9A%8E%E9%80%86%E7%9F%A9%E9%99%A3%E5%85%AC%E5%BC%8F/)



