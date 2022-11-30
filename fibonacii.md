# 矩阵对角化

*打开本页，如果没有显示公式，请刷新页面。*

## 可对角化矩阵

### 定义

令 $$\pmb{A}$$ 是 $$n\times n$$ 矩阵，若存在一个 $$n$$ 阶可逆矩阵 $$\pmb{S}$$ ，使得 $$\pmb{D}=\pmb{S}^{-1}\pmb{AS}$$ ，其中 $$\pmb{D}$$ 是对角矩阵，则称 $$\pmb{A}$$ 是**可对角化矩阵**（diagonalizable matrix），$$\pmb{S}$$ 是 $$\pmb{A}$$ 的**对角化矩阵**（diagonalizing matrix）。

如果不存在满足上述条件的 $$\pmb{S}$$ ，则称 $$\pmb{A}$$ 是不可对角化矩阵。

### 可对角化矩阵的判定$$^{[5]}$$

设 $$\lambda_i$$ 是 $$\pmb{A}$$ 的特征值（含相同的特征值），对应的特征向量 $$\pmb{x}_i$$ ，$$i=1,\cdots,n$$ 。可以写出 $$n$$ 个特征方程：

$$\pmb{Ax}_i = \lambda_i\pmb{x}_i,\quad i=1,\cdots,n \tag{1.1}$$

将（1.1）式写成矩阵形式：

$$\begin{split}\pmb{A}\begin{bmatrix}\pmb{x}_1&\cdots&\pmb{x}_n\end{bmatrix} &= \begin{bmatrix}\pmb{A}\pmb{x}_1&\cdots&\pmb{A}\pmb{x}_n\end{bmatrix}\\&=\begin{bmatrix}\lambda_1\pmb{x}_1&\cdots&\lambda_n\pmb{x}_n\end{bmatrix}\\&=\begin{bmatrix}\pmb{x}_1&\cdots&\pmb{x}_n\end{bmatrix}\begin{bmatrix}\lambda_1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&\lambda_n\end{bmatrix}\end{split} $$

设 $$\pmb{S}=\begin{bmatrix}\pmb{x}_1&\cdots&\pmb{x}_n\end{bmatrix}$$ ，$$\pmb{D}$$ 是特征值构成的对角矩阵，即 $$\pmb{D}=diag(\lambda_1,\cdots,\lambda_n)$$ ，则上式化为：

$$\pmb{AS}=\pmb{SD} \tag{1.2}$$

- 若 $$\pmb{A}$$ 有 $$n$$ 个线性无关的特征向量，则 $$\pmb{S}$$ 是可逆的，由（1.2）式得：

  - $$\pmb{A}$$ 可分解为：$$\pmb{A} = \pmb{SDS}^{-1}$$
  - $$\pmb{S}$$ 对角化 $$\pmb{A}$$ ：$$\pmb{S}^{-1}\pmb{AS}=\pmb{D}$$

- 若 $$\pmb{A}$$ 不拥有 $$n$$ 个完整的线性无关的特征向量，则 $$\pmb{S}$$ 不可逆。故无法对角化 $$\pmb{A}$$ 。

所以，可以通过判断 $$\pmb{A}$$ 的特征向量是否完全线性无关，进而确定其是否可对角化。

那么，如何判断特征向量完全线性无关？可以通过特征值部分判断。

### 特征值两两相异

**定理1：**若 $$\pmb{A}$$ 的特征值两两相异，则 $$\pmb{A}$$ 有完整的线性无关特征向量。

**证明1**

设 $$\lambda_1,\cdots,\lambda_k$$ 为相异特征值，$$2\le k\le n$$ ，对应特征向量集合 $$\{\pmb{x}_1,\cdots,\pmb{x}_k\}$$ ，考虑：

$$c_1\pmb{x}_1+c_2\pmb{x}_2+\cdots+c_k\pmb{x}_k=\pmb{0}\tag{1.3}$$

将（1.3）式等号两侧左乘 $$(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})$$ ，并且 $$\pmb{Ax}_i=\lambda_i\pmb{x}_i,(1\le i\le k)$$ ，得：

$$\begin{split}\pmb{0} &= (\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})(c_1\pmb{x}_1+c_2\pmb{x}_2+\cdots+c_k\pmb{x}_k)\\&=c_1(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})(\pmb{A}-\lambda_1\pmb{I})\pmb{x}_1 \\&\quad+ c_2(\pmb{A}-\lambda_1\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\pmb{x}_2\\&\quad+\cdots\\&\quad+c_k(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-1}\pmb{I})\pmb{x}_k\\&= c_k(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\pmb{A}-\lambda_{k-2}\pmb{I})(\lambda_k-\lambda_{k-1})\pmb{x}_k\\&=c_k(\pmb{A}-\lambda_1\pmb{I})(\pmb{A}-\lambda_2\pmb{I})\cdots(\lambda_k-\lambda_{k-2})(\lambda_k-\lambda_{k-1})\pmb{x}_k\\&=\cdots\\&=c_k(\lambda_k-\lambda_1)(\lambda_k-\lambda_2)\cdots(\lambda_k-\lambda_{k-2})(\lambda_k-\lambda_{k-1})\pmb{x}_k\end{split}$$

因为 $$\lambda_k\ne\lambda_i,1\le i \le{k-1}$$ ，且 $$\pmb{x}_k\ne0$$ ，所以：$$c_k=0$$ 。

同理，可得：$$c_{k-1}=\cdots=c_2=c_1=0$$ 。

故 $$\{\pmb{x}_1,\cdots,\pmb{x}_k\}$$ 是一个完整的线性无关集合。

证毕。

**证明2**（反证法）

设 $$\{\pmb{x}_1,\cdots,\pmb{x}_k\}$$ 是线性相关集合，在不失一般性的原则下，设 $$\{\pmb{x}_1,\cdots,\pmb{x}_{p-1}\}$$ 是最大的线性无关集，则：

$$\pmb{x}_p=c_1\pmb{x}_1+c_2\pmb{x}_2+\cdots+c_{p-1}\pmb{x}_{p-1} \tag{1.4}$$

其中 $$c_1,\cdots,c_{p-1}$$ 不全为零（因为 $$\pmb{x}_p\ne 0$$ ）。

（1.4）式等号两侧分别左乘 $$\pmb{A}$$ ，可得：

$$\begin{split}\pmb{Ax}_p &= c_1\pmb{A}\pmb{x}_1+c_2\pmb{Ax}_2+\cdots+c_{p-1}\pmb{Ax}_{p-1}\\&=c_1\lambda_1\pmb{x}_1+c_2\lambda_2\pmb{x}_2+\cdots+c_{p-1}\lambda_{p-1}\pmb{x}_{p-1}\end{split}$$

且：

$$\pmb{Ax}_p=\lambda_p\pmb{x}_p=c_1\lambda_p\pmb{x}_1+\cdots+c_{p-1}\lambda_p\pmb{x}_{p-1}$$

以上两式相减：

$$c_1(\lambda_1-\lambda_p)\pmb{x}_1+\cdots+c_{p-1}(\lambda_{p-1}-\lambda_p)\pmb{x}_{p-1}=\pmb{0}$$

因为 $$\{\pmb{x}_1,\cdots,\pmb{x}_{p-1}\}$$ 是线性无关的向量集，且 $$\lambda_1,\cdots,\lambda_p$$ 两两相异，所以：$$c_i=0,(1\le i \le{p-1})$$ 。与（1.4）式假设中的系数矛盾。故假设不成立。

证毕。

**若 $$\pmb{A}$$ 有相同的特征值，在可能可对角化，也可能不可对角化。**

对于 $$n$$ 阶方阵 $$\pmb{A}$$ ，特征多项式：

$$p(t)=\det(\pmb{A}-t\pmb{I})\tag{1.5}$$

特征值 $$\lambda$$ 即为 $$p(t)$$ 的根。

将 $$n$$ 阶行列式展开，（1.5）式即为 $$n$$ 次多项式，故有 $$n$$ 个根。

- 将（1.5）式的根中相重的根的数量，称为**代数重数**（algebraic multiplicity）。

由 $$\pmb{Ax}=\lambda\pmb{x}$$ 得 $$(\pmb{A}-\lambda\pmb{I})\pmb{x}=\pmb{0}$$ ，根据[零空间](./rank.html)的概念，可知，每个特征值对应的特征向量必定属于 $$N(\pmb{A}-\lambda\pmb{I})$$ ，称之为对应 $$\lambda$$ 的**特征空间**（eigenspace）：

- 特征空间中的非零向量都是特征向量。
- 对应特征值 $$\lambda$$ 的特征空间维数，$$\dim N(\pmb{A}-\lambda\pmb{I})$$ ，即所能找到的最大线性无关向量个数，称为 $$\lambda$$ 的**几何重数**（geometric multiplicity）

### 可对角化充要条件

**定理2：** $$n$$ 阶方阵 $$\pmb{A}$$ 可对角化的一个充要条件是：每个特征值 $$\lambda$$ ，代数重数等于集合重数。或者说，不可对角化矩阵同义语缺陷矩阵。

**证明**

证明 $$\to$$

假设 $$\pmb{A}$$ 可对角化，则有 $$n$$ 个线性无关的特征向量。设 $$\lambda$$ 的代数重数为 $$k$$ ，一定有可逆矩阵 $$\pmb{S}$$ 将 $$\pmb{A}$$ 对角化为：

$$\pmb{S}^{-1}\pmb{AS}=\begin{bmatrix}\lambda\pmb{I}_k&0\\0&\pmb{D}\end{bmatrix}$$

且 $$\lambda$$ 不为 $$(n-k)$$ 阶对角矩阵 $$\pmb{D}$$ 的特征值，故 $$\pmb{D}-\lambda\pmb{I}$$ 是可逆矩阵。因为：

$$\pmb{A}-\lambda\pmb{I}=\pmb{S}\begin{bmatrix}0&0\\0&\pmb{D}-\lambda\pmb{I}\end{bmatrix}\pmb{S}^{-1}$$

因为可逆矩阵相乘，不改变矩阵的秩，所以：

$$rank(\pmb{A}-\lambda\pmb{I})=rank(\begin{bmatrix}0&0\\0&\pmb{D}-\lambda\pmb{I}\end{bmatrix})=rank(\pmb{D}-\lambda\pmb{I})=n-k$$

根据“[秩—零花度原理](./basetheory.html)”：

$$\dim N(\pmb{A}-\lambda\pmb{I})=n-rank(\pmb{A}-\lambda\pmb{I})=n-(n-k)=k$$

故：特征值 $$\lambda$$ 的集合重数等于代数重数。

证明 $$\leftarrow$$

设方阵 $$\pmb{A}$$ 有 $$m$$ 个相异特征值 $$\lambda_i,i=1,\cdots,m$$ 。其几何重数等于代数重数，记作 $$\beta_i$$ 。

则特征向量总数为：$$\sum_{i=1}^n\beta_i=n$$ 。

因为对应特征值 $$\lambda_i$$ 的 $$\beta_i$$ 个特征向量线性无关，这些向量为特征空间 $$N(\pmb{A}-\lambda_i\pmb{I})$$ 的基；又因为两个相异特征值的特征空间无交集，$$\lambda_i\ne\lambda_j,N(\pmb{A}-\lambda_i\pmb{I})\cap N(\pmb{A}-\lambda_j\pmb{I})={\pmb{0}}$$ ，则两个特征值对应的特征向量线性无关。

所以，$$\pmb{A}$$ 有完整的 $$n$$ 个线性无关的特征向量。

## 斐波那契数列

神奇的斐波那契数列。公元1150年印度数学家首先描述了斐波那契数列，此后，意大利人斐波那契（Leonardo Fibonacci）在描述兔子繁殖的数量，这样描述：

- 第一个月初有一对刚诞生的兔子
- 第二个月之后（第三个月初）它们可以生育
- 每月每对可生育的兔子会繁殖一对新兔子
- 兔子永不死

假设在 $$n$$ 月有兔子共 $$a$$ 对，$$n+1$$ 月共有 $$b$$ 对，在 $$n+2$$ 月必定共有 $$a+b$$ 对：因为在 $$n+2$$ 月的时候，前一月（n+1月）的 $$b$$ 对兔子可以存留至第 $$n+2$$ 月（在当月属于新诞生的兔子尚不能生育）。而新生育出的兔子对数等于所有在 $$n$$ 月就已存在的 $$a$$ 对。

即设 $$a_1=1,a_2=1$$ ，$$a_n = a_{n-1}+a_{n-2}$$

按照表达式，可知斐波那契数是：$$1,1,2,3,5,8,13,21,34,55,89,144,233,\cdots$$

编程序，计算斐波那契数。可以有多种方法。

### 计算斐波那契数

**递归**

这里编写了一个用递归方法计算斐波那契数的函数，关于函数的创建方法，请参阅参考文献[1]。

```python
# 递归
def fib_recu(n):
    if n <= 1:
        return n
    else:
        return fib_recu(n-1) + fib_recu(n-2)

print("fibonacci number:")
for i in range(10):
    print(f"{fib_recu(i)}", end=" ")
    
# 输出：
fibonacci number:
0 1 1 2 3 5 8 13 21 34 
```

递归方法，思路简单，但是执行效率比较低。

**循环**$$^{[1]}$$

```python
def fib_loop(n):
    fibs = [0, 1]
    if n <= 1:
        return n
    else:
        for i in range(n-2):
            fibs.append(fibs[-2] + fibs[-1])
    return fibs

print(f"fibonacci number: {fib_loop(10)}")

# 输出：
fibonacci number: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

循环方法，也称为递推法，即递增。随着数量的增大，执行速度会越来越慢。

**迭代器对象**$$^{[1]}$$

```python
class Fibs:
    def __init__(self, max):
        self.max = max
        self.a = 0
        self.b = 1

    def __iter__(self):
        return self

    def __next__(self):
        fib = self.a
        if fib > self.max:
            raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib

fibs = Fibs(100000)
lst = [ fibs.__next__() for i in range(10)]
print(lst)

# 输出
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**生成器对象**$$^{[1]}$$

```python
def fibs():
    prev, curr = 0, 1
    while True:
        yield prev
        prev, curr = curr, prev + curr


import itertools
print(list(itertools.islice(fibs(), 10)))

# 输出
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

通过`fibs()`得到了含有无限项斐波那契数的对象，只不过在上面只向内存中读入了10个。

**矩阵**$$^{[2]}$$

```python
import numpy as np
def fib_matrix(n):
    F = np.mat([[1,1], [1,0]])
    F_m = pow(F, n)
    return F_m[0, 0]

print("fibonacci numbers:")
for i in range(10):
    print(fib_matrix(i), end=" ")
    
# 输出
fibonacci numbers:
1 1 2 3 5 8 13 21 34 55
```

### 黄金分割

如果用斐波那契数列中的后面的数字除以前面的数字，结果会如何？

```python
def fib_iter():
    prev, curr = 0, 1
    while True:
        yield prev
        prev, curr = curr, prev + curr

import itertools
fibs = list(itertools.islice(fib_iter(), 100))
for i in range(2, 30):
    print(f"{fibs[i+1]}/{fibs[i]}={fibs[i+1]/fibs[i]}")
    
# 输出
2/1=2.0
3/2=1.5
5/3=1.6666666666666667
8/5=1.6
13/8=1.625
21/13=1.6153846153846154
34/21=1.619047619047619
55/34=1.6176470588235294
89/55=1.6181818181818182
144/89=1.6179775280898876
233/144=1.6180555555555556
377/233=1.6180257510729614
610/377=1.6180371352785146
987/610=1.618032786885246
1597/987=1.618034447821682
2584/1597=1.6180338134001253
4181/2584=1.618034055727554
6765/4181=1.6180339631667064
10946/6765=1.6180339985218033
17711/10946=1.618033985017358
28657/17711=1.6180339901755971
46368/28657=1.618033988205325
75025/46368=1.618033988957902
121393/75025=1.6180339886704431
196418/121393=1.6180339887802426
317811/196418=1.618033988738303
514229/317811=1.6180339887543225
832040/514229=1.6180339887482036
```

从上面的输出结果发现：

$$\frac{f_{n+1}}{f_n}\approx 1.618\cdots$$

这就是我们熟悉的黄金分割——黄金比例$$^{[3]}$$，准确值是 $$\varphi = \frac{1+\sqrt{5}}{2}$$ ，这是一个无理数，小数点后20位的近似值是：

$$\varphi=1.61803398874989484820$$

一般取 $$1.618$$ 即可。

在历史上，伟大的天文学家开普勒$$^{[4]}$$，最早发现了这个结论。

![](./images/images/2021-3-3/1614731513624-kepler.jpg)

<center>开普勒（Johannes Kepler），德国天文学家、数学家。提出了著名的开普勒三定律，被誉为“为天文立法的人”</center>

设 $$F_k$$ 为斐波那契数列中的第 $$k$$ 项，则：

$$F_{k+2} = F_{k+1} + F_k,(k=0,1,2,\cdots) \tag{5.1}$$

根据斐波那契数列的特点，设定初始条件：$$F_0=0, F_1=1$$ 。

将（5.1）式写成：

$$\begin{cases}F_{k+2}=F_{k+1}+F_k\\F_{k+1}=F_{k+1}\end{cases}$$

以矩阵方式表示为：

$$\begin{bmatrix}F_{k+2}\\F_{k+1}\end{bmatrix}=\begin{bmatrix}1&1\\1&0\end{bmatrix}\begin{bmatrix}F_{k+1}\\F_{k}\end{bmatrix} \tag{5.2}$$

**推导通项表达式 $$F_k$$ 的第一种方法**

令 $$\pmb{u}_k=\begin{bmatrix}F_{k+1}\\F_{k}\end{bmatrix}$$ ，由（5.2）式得到：

$$\pmb{u}_{k+1}=\pmb{Au}_k,(k=0,1,\cdots) \tag{5.3}$$

其中，$$\pmb{A}=\begin{bmatrix}1&1\\1&0\end{bmatrix}$$ ，初始条件 $$\pmb{u}_0=\begin{bmatrix}1\\0\end{bmatrix}$$ 。

（5.3）式即为《机器学习数学基础》第3章3.2.2节中所介绍的差分方程。又因为：

$$\pmb{u}_k=\pmb{Au}_{k-1}=\pmb{A}(\pmb{Au}_{k-2})=\pmb{A}^2\pmb{u}_{k-2}=\cdots=\pmb{A}^k\pmb{u}_0\tag{5.4}$$

所以，只要计算出 $$\pmb{A}^k$$ ，然后将它与初始条件 $$\pmb{u}_0$$ 相乘，即得到 $$\pmb{u}_k$$ 。

对于矩阵 $$\pmb{A}=\begin{bmatrix}1&1\\1&0\end{bmatrix}$$ ，其特征多项式：

$$|\pmb{A}-\lambda\pmb{I}|=\begin{vmatrix}1-\lambda & 1\\1&-\lambda\end{vmatrix}=\lambda^2-\lambda-1$$

根据特征方程：$$|\pmb{A}-\lambda\pmb{I}|=0$$ ，得：

$$\lambda^2-\lambda-1=0\tag{5.5}$$

解得：

$$\begin{cases}\lambda_1=\frac{1+\sqrt{5}}{2}\\\lambda_2=\frac{1-\sqrt{5}}{2}\end{cases}\tag{5.6}$$

显然，$$\lambda_1+\lambda_2=1,\lambda_1\lambda_2=-1$$ 。

再根据 $$\pmb{Ax}=\lambda\pmb{x}$$ 计算（5.6）中每个特征值所对应的特征向量。

当 $$\lambda=\lambda_1$$ 时，设特征向量为 $$\begin{bmatrix}x_1\\y_1\end{bmatrix}$$ ，则：

$$\begin{bmatrix}1&1\\1&0\end{bmatrix}\begin{bmatrix}x_1\\y_1\end{bmatrix}=\lambda_1\begin{bmatrix}x_1\\y_1\end{bmatrix}$$

可得：$$\begin{bmatrix}x_1\\y_1\end{bmatrix}=\begin{bmatrix}\lambda_1\\1\end{bmatrix}$$ 是满足条件的一个向量，即特征向量 $$\pmb{x}_1=\begin{bmatrix}\lambda_1\\1\end{bmatrix}$$ （将 $$\begin{bmatrix}\lambda_1\\1\end{bmatrix}$$ 代入到上式，并利用（5.5）式，上式成立，即说明 $$\begin{bmatrix}\lambda_1\\1\end{bmatrix}$$ 是一个基向量）。

同理，求得另外一个特征值 $$\lambda_2$$ 所对应的特征向量 $$\pmb{x}_2=\begin{bmatrix}\lambda_2\\1\end{bmatrix}$$ 。

$$\begin{cases}\pmb{x}_1=\begin{bmatrix}\lambda_1\\1\end{bmatrix}\\\pmb{x}_2=\begin{bmatrix}\lambda_2\\1\end{bmatrix}\end{cases}\tag{5.7}$$

很显然，特征向量 $$\pmb{x}_1$$ 和 $$\pmb{x}_2$$ 线性无关，则矩阵 $$\pmb{A}=\begin{bmatrix}1&1\\1&0\end{bmatrix}$$ 可以对角化（参阅《机器学习数学基础》第3章3.3.3节）。

设 $$\pmb{A}=\pmb{CDC}^{-1}$$ ，其中 $$\pmb{C}$$ 的列向量即为 $$\pmb{A}$$ 的特征向量，$$\pmb{D}$$ 为特征向量构成的对角矩阵（参阅《机器学习数学基础》第3章3.3.3节（3.3.10）式和（3.3.11）式）。

因为：$$\pmb{A}^k = \pmb{C}\pmb{D}^k\pmb{C}^{-1}$$ 

于是（5.4）式可以转化为：

$$\pmb{u}_k=\pmb{C}\pmb{D}^k\pmb{C}^{-1}\pmb{u}_0 \tag{5.8}$$

由于 $$\pmb{u}_0$$ 是一个列向量，则 $$\pmb{C}^{-1}\pmb{u}_0$$ 结果也是列向量，故令 $$\pmb{c}=\pmb{C}^{-1}\pmb{u}_0$$ ，（5.8）式变换为：

$$\pmb{u}_k=\pmb{C}\pmb{D}^k\pmb{c}\tag{5.9}$$

若用一般化的式子表示，$$\pmb{C} = \begin{bmatrix}\pmb{x}_1&\cdots\pmb{x}_n\end{bmatrix}$$ （由 $$\pmb{A}$$ 的 $$n$$ 个特征向量组成），$$\pmb{D}^k=\begin{bmatrix}\lambda_1^k & \cdots &0\\0&\ddots&0\\0&\cdots&\lambda_n^k\end{bmatrix}$$ ，代入（5.9）式：

$$\begin{split}\pmb{u}_k &= \begin{bmatrix}\pmb{x}_1&\cdots\pmb{x}_n\end{bmatrix}\begin{bmatrix}\lambda_1^k & \cdots &0\\0&\ddots&0\\0&\cdots&\lambda_n^k\end{bmatrix}\begin{bmatrix}c_1\\\vdots\\c_n\end{bmatrix}\\&=\begin{bmatrix}\lambda_1^k\pmb{x}_1&\cdots&\lambda_n^k\pmb{x}_n\end{bmatrix}\begin{bmatrix}c_1\\\vdots\\c_n\end{bmatrix}\\&=c_1\lambda_1^k\pmb{x}_1+\cdots+c_n\lambda_n^k\pmb{x}_n\end{split}$$

即：

$$\pmb{u}_k=c_1\lambda_1^k\pmb{x}_1+\cdots+c_n\lambda_n^k\pmb{x}_n\tag{5.10}$$

若 $$k=0$$ ，则（5.9）式即为：$$\pmb{u}_0=\pmb{CD}^0\pmb{c}=\pmb{CIc}=\pmb{Cc}$$ ，从而得到（5.10）式中的系数 $$c_1,\cdots,c_n$$ 。（5.10）式即为差分方程的通解（《机器学习数学基础》第3章3.2.2节，给出了差分方程通解的另外一种计算方法，在该方法中没有使用对矩阵 $$\pmb{A}$$ 可对角化的假设）。

现在将 $$\pmb{A}=\begin{bmatrix}1&1\\1&0\end{bmatrix}$$ 的特征向量（5.7）和特征值（5.6）式分别代入（5.10）式，即得到（5.3）差分方程的通解：

$$\pmb{u}_k=c_1\lambda_1^k\pmb{x}_1+c_2\lambda_2^k\pmb{x}_2 \tag{5.11}$$

当 $$k=0$$ 时，$$\pmb{u}_0=\begin{bmatrix}1\\0\end{bmatrix}$$ ，$$\lambda_1^0=\lambda_2^0=1$$，代入（5.11）式：

$$\begin{bmatrix}1\\0\end{bmatrix}=c_1\pmb{x}_1+c_2\pmb{x}_2=c_1\begin{bmatrix}\frac{1+\sqrt{5}}{2}\\1\end{bmatrix}+c_2\begin{bmatrix}\frac{1-\sqrt{5}}{2}\\1\end{bmatrix}$$

解得：

$$\begin{cases}c_1=\frac{1}{\sqrt{5}}\\c_2=-\frac{1}{\sqrt{5}}\end{cases}$$

代入（5.11），同时将特征值和特征向量也代入，得：

$$\begin{split}\pmb{u}_k&=\frac{1}{\sqrt{5}}\left(\frac{1+\sqrt{5}}{2}\right)^k\begin{bmatrix}\frac{1+\sqrt{5}}{2}\\1\end{bmatrix}-\frac{1}{\sqrt{5}}\left(\frac{1-\sqrt{5}}{2}\right)^k\begin{bmatrix}\frac{1-\sqrt{5}}{2}\\1\end{bmatrix}\\&=\frac{1}{\sqrt{5}}\begin{bmatrix}\left(\frac{1+\sqrt{5}}{2}\right)^{k+1}\\\left(\frac{1+\sqrt{5}}{2}\right)^{k}\end{bmatrix}-\frac{1}{\sqrt{5}}\begin{bmatrix}\left(\frac{1-\sqrt{5}}{2}\right)^{k+1}\\\left(\frac{1-\sqrt{5}}{2}\right)^{k}\end{bmatrix}\\&=\frac{1}{\sqrt{5}}\begin{bmatrix}\left(\frac{1+\sqrt{5}}{2}\right)^{k+1}-\left(\frac{1-\sqrt{5}}{2}\right)^{k+1}\\\left(\frac{1+\sqrt{5}}{2}\right)^{k}-\left(\frac{1-\sqrt{5}}{2}\right)^{k}\end{bmatrix}\end{split}$$

在（5.3）式中假设 $$\pmb{u}_k=\begin{bmatrix}F_{k+1}\\F_{k}\end{bmatrix}$$ ，对比上式，则：

$$F_k=\frac{1}{\sqrt{5}}\left[\left(\frac{1+\sqrt{5}}{2}\right)^{k}-\left(\frac{1-\sqrt{5}}{2}\right)^{k}\right] ,(k=0,1,2,\cdots)\tag{5.12}$$

这样就得到了斐波那契数列通项表达式。

**推导通项表达式 $$F_k$$ 的第二种方法**

由于 $$\pmb{A}=\begin{bmatrix}1&1\\1&0\end{bmatrix}$$ 可对角化，根据 $$\pmb{A}=\pmb{CDC}^{-1}\Rightarrow\pmb{A}^k = \pmb{C}\pmb{D}^k\pmb{C}^{-1}$$ 得：

$$\pmb{A}^k=\pmb{C}\begin{bmatrix}\lambda_1^k&0\\0&\lambda_2^k\end{bmatrix}\pmb{C}^{-1}\tag{5.13}$$

其中 $$\pmb{C}=\begin{bmatrix}\lambda_1&\lambda_2\\1&1\end{bmatrix}$$ （ $$\pmb{C}$$ 由特征向量 $$\pmb{x}_1$$、$$\pmb{x}_2$$ 构成），则 $$\pmb{C}^{-1}=\begin{bmatrix}\frac{1}{\lambda_1-\lambda_2}&\frac{\lambda_2}{\lambda_2-\lambda_1}\\\frac{1}{\lambda_2-\lambda_1}&\frac{\lambda_1}{\lambda_1-\lambda_2}\end{bmatrix}=\frac{1}{\lambda_1-\lambda_2}\begin{bmatrix}1&-\lambda_2\\-1&\lambda_1\end{bmatrix}$$

由（5.4）式可得：

$$\begin{bmatrix}F_{k+1}\\F_k\end{bmatrix}=\pmb{A}^k\begin{bmatrix}1\\0\end{bmatrix}$$

 将（5.13）代入上式，可得：

$$\begin{split}\begin{bmatrix}F_{k+1}\\F_k\end{bmatrix}&=\pmb{C}\begin{bmatrix}\lambda_1^k&0\\0&\lambda_2^k\end{bmatrix}\pmb{C}^{-1}\begin{bmatrix}1\\0\end{bmatrix}\\&=\frac{1}{\lambda_1-\lambda_2}\begin{bmatrix}\lambda_1&\lambda_2\\1&1\end{bmatrix}\begin{bmatrix}\lambda_1^k&0\\0&\lambda_2^k\end{bmatrix}\begin{bmatrix}1&-\lambda_2\\-1&\lambda_1\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}\\&=\frac{1}{\lambda_1-\lambda_2}\begin{bmatrix}\lambda_1^{k+1}-\lambda_1^{k+1}\\\lambda_1^k-\lambda_2^k\end{bmatrix}\end{split}$$

所以，通项表达式：

$$\begin{split}F_k&=\frac{\lambda_1^k-\lambda^k_2}{\lambda_1-\lambda_2}=\frac{\left(\frac{1+\sqrt{5}}{2}\right)^k-\left(\frac{1-\sqrt{5}}{2}\right)^k}{\left(\frac{1+\sqrt{5}}{2}\right)-\left(\frac{1-\sqrt{5}}{2}\right)}\\&=\frac{1}{\sqrt{5}}\left[\left(\frac{1+\sqrt{5}}{2}\right)^{k}-\left(\frac{1-\sqrt{5}}{2}\right)^{k}\right] ,(k=0,1,2,\cdots)\end{split}$$

与（5.12）式一样的结果。

**性质1：** $$\lim_{k\to\infty}\frac{F_{k+1}}{F_k}=\frac{1+\sqrt{5}}{2}$$

**证明**

对于 $$k\ge 1$$ ，由（5.12）式得：

$$\begin{split}\lim_{k\to\infty}\frac{F_{k+1}}{F_k}&=\lim_{k\to\infty}\frac{(1+\sqrt{5})^{k+1}-(1-\sqrt{5})^{k+1}}{2^{k+1}\sqrt{5}}\cdot\frac{2^k\sqrt{5}}{(1+\sqrt{5})^{k}-(1-\sqrt{5})^{k}}\\&=\frac{1}{2}\lim_{k\to\infty}\frac{(1+\sqrt{5})^{k+1}-(1-\sqrt{5})^{k+1}}{(1+\sqrt{5})^{k}-(1-\sqrt{5})^{k}}\\&=\frac{1}{2}\lim_{k\to\infty}\frac{(1+\sqrt{5})-\left(\frac{1-\sqrt{5}}{1+\sqrt{5}}\right)^k(1-\sqrt{5})}{1-\left(\frac{1-\sqrt{5}}{1+\sqrt{5}}\right)^k}\\&=\frac{1+\sqrt{5}}{2}\end{split}$$

（ $$\because \quad \begin{vmatrix}\frac{1-\sqrt{5}}{1+\sqrt{5}}\end{vmatrix}\lt 1, \quad \lim_{k\to\infty}\left(\frac{1-\sqrt{5}}{1+\sqrt{5}}\right)^k=0$$ ）

证毕。

所以，在前面的程序中，显示后项除以前项的结果趋近于黄金分割。

**性质2：** 幂矩阵 $$\pmb{A}^k$$

因为 $$\pmb{A}=\begin{bmatrix}1&1\\1&0\end{bmatrix}$$ ，又因为斐波那契数列 $$F_0=0,F_1=1,F_2=1$$ ，所以：

$$\pmb{A}=\begin{bmatrix}1&1\\1&0\end{bmatrix}=\begin{bmatrix}F_2&F_1\\F_1&F_0\end{bmatrix}$$

$$\pmb{A}^2=\begin{bmatrix}1&1\\1&0\end{bmatrix}\begin{bmatrix}F_2&F_1\\F_1&F_0\end{bmatrix}=\begin{bmatrix}F_2+F_1&F_1+F_0\\F_2&F_1\end{bmatrix}=\begin{bmatrix}F_3&F_2\\F_2&F_1\end{bmatrix}$$

以此类推，可得：

$$\pmb{A}^k=\begin{bmatrix}F_{k+1}&F_k\\F_k&F_{k-1}\end{bmatrix}\tag{5.14} $$

其中 $$\pmb{A}=\begin{bmatrix}1&1\\1&0\end{bmatrix}$$

这就是矩阵法计算斐波那契数列的程序编写依据。

**性质3：**Cassini等式

在《机器学习数学基础》第2章2.4节中曾经介绍过行列式的性质，下面应用其中一条：$$|\pmb{AB}|=|\pmb{A}||\pmb{B}|$$ 。

由（5.14）可得：

$$|\pmb{A}^k|=\begin{vmatrix}F_{k+1}&F_k\\F_k&F_{k-1}\end{vmatrix}=F_{k+1}F_{k-1}-F_n^2$$

又因为：

$$|\pmb{A}^k|=|\pmb{A}|^k=\begin{vmatrix}1&1\\1&0\end{vmatrix}^k=(-1)^k$$

所以：

$$F_{k+1}F_{k-1}-F_n^2=(-1)^k\tag{5.15}$$

（5.15）式即为Cassini等式。表明了斐波那契数列的数之间的一种关系。如：

$$\begin{split}2\times5-3^2&=1\\3\times8-5^2&=-1\\5\times13-8^2&=1\\8\times21-13^2&=-1\end{split}$$

**性质4：**前 $$n$$ 项的和

由（5.1）式得：

$$\begin{split}F_n &= F_{n+2}-F_{n+1}\\F_{n-1}&=F_{n+1}-F_n\\F_{n-2}&=F_{n}-F_{n-1}\\&\vdots\\F_2&=F_4-F_3\\F_1&=F_3-F_2\end{split}$$

上面各式等号相加，得：

$$F_1+\cdots+F_n=(F_3+\cdots+F_{n+1}+F_{n+2})-(F_2+F_3+\cdots+F_{n+1})=F_{n+2}-F_2$$

因为 $$F_{2}=1$$ ，所以，前 $$n$$ 项的和是：

$$\sum_{i=1}^nF_i=F_{n+2}-1\tag{5.16}$$

**性质5：**每项的平方和

$$\begin{split}F_nF_{n+1}&=F_n(F_n+F_{n-1})\\&=F^2_n+F_{n-1}F_n\\&=F_n^2+F_{n-1}(F_{n-1}+F_{n-2})\\&=F_n^2+F_{n-1}^2+F_{n-2}F_{n-1}\\&=\cdots\\&=F_n^2+F_{n-1}^2+\cdots+F_2^2+F_2F_1\end{split}$$

又因为：$$F_2=F_1=1$$ ，所以：

$$\sum_{i=1}^nF_i^2=F_nF_{n+1} \tag{5.17}$$



## 参考文献

[1]. 齐伟. Python大学实用教程[M]. 北京：电子工业出版社. 2019年3月，第1版

[2]. 齐伟. 跟老齐学Python：数据分析[M]. 北京：电子工业出版社. 2018年6月，第1版

[3]. [https://zh.wikipedia.org/wiki/黄金分割率](https://zh.wikipedia.org/wiki/%E9%BB%84%E9%87%91%E5%88%86%E5%89%B2%E7%8E%87)

[4]. 开普勒的墓志铭：*Mensus eram coelos, nunc terrae metior umbras*；*Mens coelestis erat, corporis umbra iacet.*（“我曾测天高，今欲量地深。”“我的灵魂来自上天，凡俗肉体归于此地。”）[https://zh.wikipedia.org/wiki/约翰内斯·开普勒](https://zh.wikipedia.org/wiki/%E7%BA%A6%E7%BF%B0%E5%86%85%E6%96%AF%C2%B7%E5%BC%80%E6%99%AE%E5%8B%92)

[5]. [线代启示录：可对角化矩阵与缺陷矩阵的判定](https://ccjou.wordpress.com/2010/05/13/%e5%8f%af%e5%b0%8d%e8%a7%92%e5%8c%96%e7%9f%a9%e9%99%a3%e8%88%87%e7%bc%ba%e9%99%b7%e7%9f%a9%e9%99%a3%e7%9a%84%e5%88%a4%e5%ae%9a/)

