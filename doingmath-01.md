# 基础运算

## 四则运算和指数运算

### 加法、减法和四舍五入

```python
>>> 3 - 9
-6
>>> 6 + 1
7
```

整数加法和减法，等同于小学算术的内容。

但是，要注意，以下关于浮点数的计算。

```python
>>> 3.14 + 2.1
5.24
>>> 2.3 - 4.1
-1.7999999999999998
>>> 4.1 - 2.3
1.7999999999999998
```

在`2.3 - 4.1`的计算中，所得的结果并不是`-1.8`——显然应该是这个。这是因为，在利用Python进行计算的时候，要将上面示例中所使用的十进制数字，转换为二进制的数字。在这个转换中，并不是所有的浮点数都对应有限个数的二进制数字，如果没有有限个数的二进制数字，那么计算的时候就产生了误差。这并不是Python的Bug。

要解决这个问题，可以使用四舍五入的方法。

```python
>>> a = 2.3 - 4.1
>>> round(a,1)
-1.8
```

四舍五入，已经能够解决绝大部分问题了。

这里使用了Python的内置函数`round()`，用它实现四舍五入计算。可以用`help(round)`的方式查看它的使用方法。

```python
round(number, ndigits=None)
    Round a number to a given precision in decimal digits.

    The return value is an integer if ndigits is omitted or None.  Otherwise
    the return value has the same type as the number.  ndigits may be negative.
```

参数`ndigits`表示要保留的有理数位数。

如果参数`number`是浮点数，当`ndigits`是正数的时候，就表示所保留的小数位数。例如上面的示例。

看上面文档，`ndigits`还可以是负数，其含义为有理数的整数部分所不保留的位数。例如：

```python
>>> round(163, -2)
200
>>> round(126.78, -1)
130.0
```

### 乘法和乘方

乘法使用一个 `*`实现。

```python
>>> 2 * 3
6
>>> 2.3 * 4.5
10.35
```

乘方运算，可以使用 `**`实现：

```python
>>> 2 ** 3
8
>>> 2.3 ** 2
5.289999999999999
```

根据数学中的知识，指数还可以不是整数，例如 $$4^{\frac{1}{2}}$$ ，其含义就是 $$\sqrt{4}=2$$ 。

```python
>>> 4 ** (1/2)
2.0
```

但注意，此时返回的是浮点数。

当然，指数变成小数，与分数是同样的含义。

```python
>>> 4 ** 0.5
2.0
```

指数也可以是负数，如 $$2^{-2}=\frac{1}{4}$$

```
>>> 2 ** -2
0.25
```

除了使用 `**`实现乘方运算，还可以用内置函数实现。

```python
>>> pow(2, 3)
8
>>> pow(4, -2)
0.0625
>>> pow(4, 1/2)
2.0
```

`pow()`的第一个参数是底，第二个是幂。用上面的方法，与使用 `**`完全一致，即`pow(n, m)`等于` n ** m`。

如果查看`pow()`的帮助文档：`help(pow)`：

```python
pow(base, exp, mod=None)
    Equivalent to base**exp with 2 arguments or base**exp % mod with 3 arguments
```

`pow()`函数有三个参数，如果要给`mod`参数赋值，则表示`base ** exp % mod`，其含义是`base`的`exp`次方，然后除以`mod`，返回其余数。例如：

```python
>>> pow(2, 3, 5)
3
```

$$2^3$$ 是 $$8$$ ，除以 $$5$$ 的余数是 $$3$$ 。关于余数和符号`%`，见后面的除法计算。

### 科学计数法

在数学中，对于很大或者很小的数，习惯使用科学计数法，例如 $$200000$$ 可以记作 $$2\times10^5$$ ，用Python实现科学记数法：

```python
>>> 2e2
200.0
>>> 2e-2
0.02
>>> e3
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'e3' is not defined
>>> 1e3
1000.0
```

必须要在`e`前面有数字，可以是整数，也可以是浮点数。

```python
>>> 23e2
2300.0
>>> 23e-2
0.23
>>> 12.34e2
1234.0
```

### 除法

Python3.x与Python2.x对除法的定义有所不同，这里所演示的是Python3.x中的除法。

```
>>> 3 / 2
1.5
>>> 4 / 2
2.0
>>> 1 / 5
0.2
```

如果用`//`，结果返回商，俗称“地板除”。

```python
>>> 5 // 2
2
>>> 3 // 4
0
>>> 3.4 // 2
1.0
>>> 6.4 // 3
2.0
```

“地板除”能够得到商，用`%`则得到余数。

```python
>>> 5 % 2
1
```

在内置函数中，有`divmod()`，可以同时返回商和余数。

```python
>>> divmod(5, 2)
(2, 1)
```



## 分数

Python中表示分数，实用标准库中的一个模块`fractions`。

```python
>>> from fractions import Fraction
>>> a = Fraction(3, 4)
>>> b = Fraction(2, 6)
>>> a
Fraction(3, 4)
>>> b
Fraction(1, 3)
```

其中，`a`是分数 $$\frac{3}{4}$$ ，`b`在定义的时候，是分数 $$\frac{2}{6}$$ ，Python会自动将这个分数最简化，即 $$\frac{1}{3}$$ 。

分数之间可以进行运算。

```python
>>> a + b
Fraction(13, 12)
>>> a * b
Fraction(1, 4)
>>> a - b
Fraction(5, 12)
>>> a / b
Fraction(9, 4)
>>> a ** b
0.9085602964160698
>>> pow(a, b)
0.9085602964160698
```

分数也可以跟整数、浮点数进行运算。

```python
>>> a + 2
Fraction(11, 4)
>>> a * 2.1
1.5750000000000002
>>> b / 4
Fraction(1, 12)
>>> b / 2.5
0.13333333333333333
>>> a ** 2
Fraction(9, 16)
>>> pow(a, 2)
Fraction(9, 16)
```

## 复数

前面讨论中，所涉及到的都是实数。Python中也支持定义复数（complex number）。

```python
# 定义1，注意表示虚数必须用j，不能用数学中习惯的i
>>> a = 2 + 3j
>>> type(a)
<class 'complex'>
# 定义2
>>> b = complex(4, 5)
>>> b
(4+5j)
# 只定义虚部
>>> 2j
2j
>>> type(2j)
<class 'complex'>
```

同时，支持复数的有关基本运算。

```python
>>> a + b
(6+8j)
>>> a - b
(-2-2j)
>>> a * b
(-7+22j)
>>> a / b
(0.5609756097560976+0.0487804878048781j)
>>> a ** b
(-0.7530458367485596-0.9864287886477445j)
```

通过复数的如下两个方法，分别得到其实部和虚部。

```python
# 实部
>>> a.real
2.0
# 虚部
>>> a.imag
3.0
```

复数的共轭。数学中定义 $$z=a+ib$$ 的共轭复数是 $$\overline{z} = a-ib$$

```python
>>> a
(2+3j)
>>> a.conjugate()
(2-3j)
```

在数学中，因为 $$i^2=-1$$ ，如果用Python表示：

```
# 这样做不行，因为这样做会认为 j 是变量
>>> j ** 2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'j' is not defined
# 必须这样做
>>> 1j ** 2
(-1+0j)
>>> 2j ** 2
(-4+0j)
```

对于复数 $$z=a+ib$$ ，$$|z|=\sqrt{a^2+b^2}$$ ，称为“绝对值”或者“模”。

利用前面已经学过的运算，可以计算：

```python
>>> (a.real ** 2 + a.imag ** 2) ** 0.5
3.605551275463989
```

最简捷的方法，是使用内置函数`abs()`直接计算 $$|z|$$ 。

```python
>>> abs(a)
3.605551275463989
```

函数`abs()`中的参数也可以是实数，即得到了该实数的绝对值。

```python
>>> abs(-2)
2
```

## 初等函数

Python标准库有一个模块`math`，其中包括了常用的初等函数。

```python
>>> import math
>>> dir(math)
['__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'comb', 'copysign', 'cos', 'cosh', 'degrees', 'dist', 'e', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 'gcd', 'hypot', 'inf', 'isclose', 'isfinite', 'isinf', 'isnan', 'isqrt', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'modf', 'nan', 'perm', 'pi', 'pow', 'prod', 'radians', 'remainder', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'tau', 'trunc']
```

例如，$$\pi$$ 可以是：

```python
>>> math.pi
3.141592653589793
```

计算 $$\sin(2\pi)$$ ，数学上应该等于 $$0$$ ，用Python计算结果：

```python
>>> math.sin(2*math.pi)
-2.4492935982947064e-16
```

$$\sin(\frac{\pi}{2}) = 1$$

```python
>>> math.sin(math.pi/2)
1.0
```

有了`math`模块，常见函数计算就容易了。

## 综合实践

### 斐波那契数列简介

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





斐波纳契数是[帕斯卡三角形](https://zh.wikipedia.org/wiki/帕斯卡三角形)的每一条红色对角线上数字的和。



## 参考文献

[1]. Python大学实用教程. 齐伟. 北京：电子工业出版社. 2019年3月，第1版

[2]. 跟老齐学Python：数据分析. 齐伟. 北京：电子工业出版社. 2018年6月，第1版

[3]. [https://zh.wikipedia.org/wiki/黄金分割率](https://zh.wikipedia.org/wiki/%E9%BB%84%E9%87%91%E5%88%86%E5%89%B2%E7%8E%87)

[4]. 开普勒的墓志铭：*Mensus eram coelos, nunc terrae metior umbras*；*Mens coelestis erat, corporis umbra iacet.*（“我曾测天高，今欲量地深。”“我的灵魂来自上天，凡俗肉体归于此地。”）[https://zh.wikipedia.org/wiki/约翰内斯·开普勒](https://zh.wikipedia.org/wiki/%E7%BA%A6%E7%BF%B0%E5%86%85%E6%96%AF%C2%B7%E5%BC%80%E6%99%AE%E5%8B%92)

