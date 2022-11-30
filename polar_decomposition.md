# 极分解

*打开本页，如果不能显示公式，请刷新页面。*

任一 $$n\times n$$ 实数矩阵 $$\pmb{A}$$ 都可以分解为：

$$\pmb{A}=\pmb{QS}\tag{1}$$

其中 $$\pmb{Q}$$ 是实正交矩阵，$$\pmb{S}$$ 是实对称半正定矩阵。这就是**极分解**（polar decomposition）。

**推导过程**

将 $$\pmb{A}$$ 奇异值分解$$^{[2]}$$为：

$$\pmb{A}=\pmb{U\Sigma V}^T\tag{2}$$

因为 $$\pmb{A}$$ 是 $$n$$ 阶矩阵，$$\pmb{U}$$、$$\pmb{V}$$ 和 $$\pmb{\Sigma}$$ 都是 $$n$$ 阶，其中 $$\pmb{V、U}$$ 是正交矩阵，$$\pmb{\Sigma}$$ 是对角矩阵且主对角元素都不为负。

将 $$\pmb{V}^T\pmb{V}=\pmb{I}$$ 代入（2）：

$$\pmb{A}=\pmb{U}(\pmb{V}^T\pmb{V})\pmb{\Sigma V}^T=(\pmb{U}\pmb{V}^T)(\pmb{V}\pmb{\Sigma V}^T)\tag{3}$$

令 $$\pmb{Q}=\pmb{UV}^T$$ 且 $$\pmb{S}=\pmb{V}\pmb{\Sigma V}^T$$ ，则（3）式化为：

$$\pmb{A}=\pmb{QS}$$

又因为：

$$\pmb{Q}^T\pmb{Q}=(\pmb{UV}^T)^T(\pmb{UV}^T)=\pmb{VU}^T\pmb{UV}^T=\pmb{VV}^T=\pmb{I}$$

故 $$\pmb{Q}$$ 是实正交矩阵.

因为 $$\pmb{\Sigma}$$ 是对称半正定矩阵，对任一 $$\pmb{x}\in\mathbb{R}^n$$ ，有：

$$\pmb{x}^T\pmb{Sx}=\pmb{x}^T\pmb{V}\pmb{\Sigma V}^T\pmb{x}=(\pmb{V}^T\pmb{x})^T\pmb{\Sigma}(\pmb{V}^T\pmb{x})\ge 0$$

当 $$\pmb{A}$$ 可逆是，$$\pmb{\Sigma}$$ 的主对角元都不为零，$$\pmb{S}$$ 是正定矩阵。

## 参考文献

[1]. [线代启示录：极分解](https://ccjou.wordpress.com/2009/09/09/%e6%a5%b5%e5%88%86%e8%a7%a3/)

[2]. 齐伟. 机器学习数学基础. 北京：电子工业出版社

