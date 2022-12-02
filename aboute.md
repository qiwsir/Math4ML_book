# 关于自然常数

自然常数 $e=2.718 28 \cdots$ 在数学、自然科学中都是一个非常重要的常数。用级数表示为：

$$
e=1+\frac{1}{1!}+\frac{1}{2!}+\cdots+\frac{1}{n!}+\cdots=\sum_{n=0}^{\infty}\frac{1}{n!}\tag{1}
$$

## 极限证明

**求证**   $e=\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n \tag{1}$

**证明**

设 $e_n=\left(1+\frac{1}{n}\right)^n$ ，根据二项式定理，可得：

$$e_n=1+\frac{n}{1!}\frac{1}{n}+\frac{n(n-1)}{2!}\frac{1}{n^2}+\frac{n(n-1)(n-2)}{3!}\frac{1}{n^3}+\cdots+\frac{1}{n^n}$$

令 $a_{n,k}=\frac{n(n-1)(n-2)\cdots(n-k+1)}{k!}\frac{1}{n^k}$

即 $e_n=1+\sum_{k=1}^na_{n,k}$

因为 $a_{n,k}=\frac{1}{k!}\left(1-\frac{1}{n}\right)\left(1-\frac{2}{n}\right)\cdots\left(1-\frac{k-1}{n}\right)$

$$a_{n,k}\lt{a_{n+1,k}}\lt\frac{1}{k!}$$

所以：

$$e_n\lt{e_{n+1}}\lt 1+\sum_{k=1}^{\infty}\frac{1}{k!}=e$$

即 $\{e_n\}$ 是单调递增数列，且 $\lim_{n\to\infty}e_n\le e$ 。

又因为 $\lim_{n\to\infty}a_{n,k}=\frac{1}{k!}$ ，对任意 $m$ ，有：

$$\lim_{n\to\infty}e_n\ge\lim_{n\to\infty}\left(1+\sum_{k=1}^ma_{n,k}\right)=1+\sum_{k=1}^m\frac{1}{k!}$$

因此，$\lim_{n\to\infty}e_n\ge 1+\sum_{k=1}^{\infty}\frac{1}{k!}=e$ ，所以：

$\lim_{n\to\infty}e_n=e$ ，即 $e=\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n$ 成立

证毕。

## 指数函数 $e^x$

指数函数 $e^x$ 是一个重要函数，表示为

$$
e^x=\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^n \tag{2}
$$


**证明**

首先证明：

$$
e=\lim_{n\to+\infty}\left(1+\frac{1}{t}\right)^t \tag{3}
$$


根据（1）式，对于 $t$ ，取满足 $n\le t \lt n+1$ 的自然数 $n$ ，则：

$$\left(1+\frac{1}{n+1}\right)^n\lt\left(1+\frac{1}{t}\right)^t\lt\left(1+\frac{1}{n}\right)^{n+1}$$

当 $t\to+\infty$ 时，$n\to+\infty$ ，并且 $\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^{n+1}=\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n\left(1+\frac{1}{n}\right)=e$

同理，$\lim{n\to\infty}\left(1+\frac{1}{n+1}\right)^n=e$

所以（3）式成立。

再证明：

$$
e=\lim_{n\to+\infty}\left(1-\frac{1}{t}\right)^{-t} \tag{4}
$$


设 $\frac{1}{1-\frac{1}{t}}=1+\frac{1}{s}$ ，则 $s=t-1$ 。当 $t\to+\infty$ 时，$s\to+\infty$ ，得：

$$
\lim_{n\to+\infty}\left(1-\frac{1}{t}\right)^{-t}=\lim_{s\to+\infty}\left(1+\frac{1}{s}\right)^{s+1}=e
$$


所以（4）式成立

当 $x\gt0$ 时，令 $s=tx$ ，

$$e^x=\lim_{t\to+\infty}\left(1+\frac{1}{t}\right)^{tx}=\lim_{s\to+\infty}\left(1+\frac{x}{s}\right)^s=\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^n$$

当 $x\lt0$ 时，令 $x=-y, s=ty$ ，则 $\frac{1}{t}=\frac{y}{s}$ ，

 $$e^x=\lim_{t\to+\infty}\left(1-\frac{1}{t}\right)^{-tx}=\lim_{s\to+\infty}\left(1-\frac{y}{s}\right)^s=\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^n$$

所以（2）式成立。

证毕。

根据 $\lim_{n\to\infty}\left(1+\frac{z}{n}\right)^n=\sum_{n=0}^{\infty}\frac{z^n}{n!}$ （见 [1] 的56页）和（2）式，可得：

$$e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!} \tag{5}$$

如果以自然常数 $e$ 作为对数的底，即 $\log_ex$ ，称为**自然对数**，一般记作 $\ln x$ ，$\ln x$ 是 $x$ 的单调递增函数。

## 对数函数的导数

设对数函数 $\log_ax,(a\ne1)$ ，定义域是 $(0,+\infty)$ 。

$$\frac{1}{h}(\log_a(x+h)-\log_ax)=\frac{1}{h}\log_a\left(\frac{x+h}{x}\right)=\log_a\left(\frac{x+h}{x}\right)^{1/h}$$

令 $s=\frac{h}{x}$ ，则上式变化为：

$$\frac{1}{h}(\log_a(x+h)-\log_ax)=\log_a(1+s)^{\frac{1}{sx}}=\frac{1}{x}\log_a(1+s)^{\frac{1}{s}}$$

根据（1）式，可得：$e=\lim_{s\to0}\log_a(1+s)^{1/s}$ 。结合 $\log_ax$ 的连续性，可得：

$$\lim_{h\to0}\frac{1}{h}(\log_a(x+h)-\log_ax)=\frac{1}{x}\lim_{s\to0}\log_a(1+s)^{\frac{1}{s}}=\frac{1}{x}\log_ae$$

即

$$\frac{d}{dx}\log_ax=(\log_ae)\frac{1}{x}  \tag{6}$$

对于自然对数，$a=e$ ，则：

$$\frac{d}{dx}\ln x=\frac{1}{x} \tag{7}$$




## 参考文献

1. 微积分入门(I)一元微积分. [日]小平邦彦. 北京：人民邮电出版社，2008.4.第1版