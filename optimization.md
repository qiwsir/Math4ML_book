# 最优化方法

*打开本页，如果不能显示公式，请刷新页面。*

## 无约束优化

### 定义

给定一个目标函数（或称成本函数） $$f:\mathbb{R}^n \to \mathbb{R}$$ ，无约束优化（uncontrained optimization）是指找到 $$\pmb{x}\in\mathbb{R}^n$$ 使得 $$f(\pmb{x})$$ 有最小值，即：

$$\min_{\pmb{x}\in\mathbb{R}^n}f(\pmb{x})$$

若希望找到最大值，将目标函数前面加负号即可。

通常，寻找 $$f(\pmb{x})$$ 的局部最小值，即在某个范围内的最小值。

### 单变量的目标函数

令 $$f:D\to\mathbb{R}$$ 为一个定义于 $$D \subseteq\mathbb{R}$$ 的光滑可导函数，其中 $$D$$ 是一个开集，根据泰勒定理：

$$f(x)=f(y)+f'(y)(x-y)+\frac{f''(y)}{2}(x-y)^2+O(|x-y|^3)\tag{1.1}$$

若 $$f'(y)=0$$ ，则 $$y$$ 为 $$f$$ 的一个驻点（stationary point），或称临界点（critical point）。

当 $$y$$ 是驻点时，若 $$|x-y|$$ 足够小，则（1.1）式近似为：

$$f(x)-f(y)\approx \frac{f''(y)}{2}(x-y)^2 \tag{1.2}$$

- 如果 $$f''(y)=0$$ ，则 $$y$$ 为一个局部最小值（local minimum），即：存在一个 $$\delta\gt0$$ ，对有所 $$x$$ 满足 $$|x-y|\le\delta$$ 都有 $$f(y)\le f(x)$$ 。
- 如果 $$f''(y)\lt0$$ ，则 $$y$$ 为一个局部最大值（local maximum）。
- 如果 $$f''(y)=0$$ ，必须计算 $$f(x)$$ 和 $$f(y)$$ 的值才能决定。

所以，驻点是函数 $$f$$ 的一个局部最小值的必要条件。

### 多变量的目标函数

令 $$\pmb{x} = \begin{bmatrix}x_1&\cdots&x_n\end{bmatrix}^{\rm{T}}$$ 为 $$f$$ 的变量，$$f(\pmb{x})$$ 为定义域 $$\pmb{D}\subseteq\mathbb{R}^{\rm{n}}$$ 的可导实函数，根据泰勒定理，得：

$$\begin{split}f(\pmb{x})=&f(\pmb{y})+\sum_{i=1}^n\frac{\partial f}{\partial x_i}\bigg|_{\pmb{y}}(x_i-y_i)\\&+\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\frac{\partial^2f}{\partial x_i\partial x_j}\bigg|_{\pmb{y}}(x_i-y_i)(x_j-y_j)+O(\begin{Vmatrix}\pmb{x}-\pmb{y}\end{Vmatrix}^3)\end{split}\tag{1.3}$$

函数 $$f$$ 在点 $$\pmb{y}$$ 的梯度$$^{[2]}$$：

$$\nabla f(\pmb{y})=\begin{bmatrix}\frac{\partial f}{\partial x_1}\bigg|_{\pmb{y}}\\\vdots\\\frac{\partial f}{\partial x_n}\bigg|_{\pmb{y}}\end{bmatrix}$$

$$f$$ 在 $$\pmb{y}$$ 点的黑塞矩阵（Hessian）$$^{[2]}$$ ：

$$[H(\pmb{y})]_{ij}=\frac{\partial^2 f}{\partial x_i\partial x_j}\bigg|_{\pmb{y}}$$

则式（1.3）可以写成：

$$f(\pmb{x})=f(\pmb{y})+(\nabla f(\pmb{y}))^{\rm{T}}(\pmb{x}-\pmb{y})+\frac{1}{2}(\pmb{x}-\pmb{y})^{\rm{T}}H(\pmb{y})(\pmb{x}-\pmb{y})+O(\begin{Vmatrix}\pmb{x}-\pmb{y}\end{Vmatrix}^3)\tag{1.4}$$

若 $$\pmb{y}$$ 是一个驻点，即 $$\nabla f(\pmb{y})=\pmb{0}$$ ，当 $$\begin{Vmatrix}\pmb{x}-\pmb{y}\end{Vmatrix}$$ 足够小，（1.4）式化为：

$$f(\pmb{x})-f(\pmb{y})\approx\frac{1}{2}(\pmb{x}-\pmb{y})^{\rm{T}}H(\pmb{y})(\pmb{x}-\pmb{y}) \tag{1.5}$$

因为：$$\frac{\partial^2f}{\partial x_i\partial x_j}=\frac{\partial^2f}{\partial x_j\partial x_i}$$

所以 $$H(\pmb{y})$$ 是一个对称矩阵。

- 若 $$H(\pmb{y})$$ 是正定的，即 $$\pmb{z}^{\rm{T}}H(\pmb{y})\pmb{z}\gt0$$ ，则 $$\pmb{z}\ne0$$ ，$$\pmb{y}$$ 是 $$f$$ 的一个局部最小值。

- 若 $$H(\pmb{y})$$ 是负定的，$$\pmb{y}$$ 是 $$f$$ 的一个局部最大值。
- 若 $$H(\pmb{y})$$ 是未定的，称 $$\pmb{y}$$ 是鞍点（saddle point）。

梯度下降法是寻找函数局部最小值的常用方法，具体参阅参考文献[2]。



## 参考文献

[1]. [线代启示录：最佳化理论与正定矩阵](https://ccjou.wordpress.com/2009/10/06/%e6%9c%80%e4%bd%b3%e5%8c%96%e5%95%8f%e9%a1%8c%e8%88%87%e6%ad%a3%e5%ae%9a%e7%9f%a9%e9%99%a3/)

[2]. 齐伟. 机器学习数学基础. 北京：电子工业出版社. 