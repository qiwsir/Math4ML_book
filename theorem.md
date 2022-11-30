# 高等数学中的定理

*打开本页，如果不能显示公式，请刷新页面。*

## 最值定理

**定理：**连续函数的函数值必定有最值

设 $$f(x)\in C[a, b]$$ ，则 $$f(x)$$ 在 $$[a, b]$$ 上取到最小值 $$m$$ 与最大值 $$M$$ ，即 $$\exists x_1,x_2\in[a, b]$$ ，使 $$f(x_1)=m$$ ，$$f(x_2)=M$$

## 有界定理

**定理：**连续函数必定有界

设 $$f(x)\in C[a, b]$$ ，且 $$\exists k\gt0$$ ，使 $$\forall x\in[a,b]$$ ，有 $$|f(x)|\le k$$

**证明**

$$\because f(x)\in\C[a, b]$$

根据【最值定理】，$$f(x)$$ 在 $$[a, b]$$ 上取最小值 $$m$$ 与最大值 $$M$$ ，即：

$$f(x)\ge m$$ ， $$f(x)\le M$$

故：$$f(x)$$ 在 $$[a,b]$$ 上有上下界，从而：

 $$\exists k\gt0$$ ，使 $$\forall x\in[a,b]$$ ，有 $$|f(x)|\le k$$

## 零点定理

**定理：**连续函数横跨 $$x$$ 轴两侧，必定与 $$x$$ 轴有交点

设 $$f(x)\in C[a, b]$$ ，若 $$f(a)f(b)\lt0$$ ，则 $$\exists c\in(a,b)$$ ，使 $$f(c)=0$$

## 介值定理

**定理：**介于 $$m$$ 和 $$M$$ 之间的值，$$f(x)$$ 皆可取到

设 $$f(x)\in C[a, b]$$ ，则 $$\forall\eta\in[m,M]$$ ，$$\exists\xi\in[a,b]$$ ，使 $$f(\xi)=\eta$$

## 罗尔中值定理

**定理：**连续函数两端点函数值相同，至少存在一个切线平行于 $$x$$ 轴的点

设 $$f(x)\in C[a, b]$$ ，$$f(x)$$ 在 $$(a,b)$$ 内可导，$$f(a)=f(b)$$ ，则 $$\exists\xi\in(a,b)$$ ，使 $$f'(\xi)=0$$

**证明**

$$\because f(x)\in C[a, b]$$

根据【最值定理】，$$f(x)$$ 在 $$[a, b]$$ 上取最小值 $$m$$ 与最大值 $$M$$ 

若 $$m=M$$ ，则 $$f(x)=C_0$$ ，故 $$\forall\xi(a,b)$$ ，有 $$f'(\xi)=0$$

若 $$m\lt M$$ ，$$\because f(a)=f(b)$$

若 $$f(a)=m$$ ，则 $$f(b)=m$$ ，$$\Rightarrow$$ $$M$$ 在 $$(a,b)$$ 内取到

若 $$f(a)=M$$ ，则 $$f(b)=M$$ ，$$\Rightarrow$$ $$m$$ 在 $$(a,b)$$ 内取到

所以，$$m$$ 和 $$M$$ 至少有一个在 $$(a,b)$$ 内取到

设 $$\exists\xi\in(a,b)$$ ，使 $$f(\xi)=m$$ $$\Rightarrow$$ $$f'(\xi)=0$$ 或 $$f'(\xi)$$ 不存在

又因为 $$f(x)$$ 在 $$(a,b)$$ 内可导，所以 $$f'(\xi)=0$$

## 拉格朗日中值定理

**定理：**连续函数至少存在一点的切线斜率等于两端点的斜率

设 $$f(x)\in C[a, b]$$ ，$$f(x)$$ 在 $$(a,b)$$ 内可导。$$\exists\xi\in[a,b]$$ ，使 $$f(\xi)=\frac{f(b)-f(a)}{b-a}$$

**证明**

由已知条件，可得过端点的直线：$$L_{ab}=f(a)+\frac{f(b)-f(a)}{b-a}(x-a)$$

令 $$\phi(x)=f(x)-L_{ab}=f(x)-f(a)-\frac{f(b)-f(a)}{b-a}(x-a)$$

$$\phi(x)\in C[a,b]$$ ，$$\phi(x)$$ 在 $$(a,b)$$ 内可导，

且 $$\phi(a)=\phi(b)=0$$ ，则 $$\exists\xi\in(a,b)$$ ，使 $$\phi'(\xi)=0$$

$$\because\phi'(x)=f'(x)-\frac{f(b)-f(a)}{b-a}$$

$$\therefore f'(\xi)=\phi'(\xi)+\frac{f(b)-f(a)}{b-a}=\frac{f(b)-f(a)}{b-a}$$

## 柯西中值定理

**定理：** 设 $$f(x)\in C[a, b]$$ ，$$f(x)，g(x)$$ 在 $$(a,b)$$ 内可导，$$g'(x)\ne0，(a\lt x\lt b)$$  。则 $$\exists\xi\in[a,b]$$ ，使 $$f(\xi)=\frac{f(b)-f(a)}{g(b)-g(a)}=\frac{f'(\xi)}{g'(\xi)}$$

**证明**

当 $$g(x)=x$$ 时，令 $$\phi(x)=f(x)-f(a)-\frac{f(b)-f(a)}{g(b)-g(a)}[g(x)-g(a)]$$

$$\phi(x)\in C[a,b]$$ ，$$\phi(x)$$ 在 $$(a,b)$$ 内可导，

且 $$\phi(a)=\phi(b)=0$$ ，则 $$\exists\xi\in(a,b)$$ ，使 $$\phi'(\xi)=0$$

$$\because\phi'(x)=f'(x)-\frac{f(b)-f(a)}{g(b)-g(a)}g'(x)$$

$$\therefore f'(\xi)=\phi'(\xi)+\frac{f(b)-f(a)}{g(b)-g(a)}g'(\xi)$$

$$\because g'(\xi)\ne0，\phi'(\xi)=0$$

故 $$\frac{f'(\xi)}{g'(\xi)}==\frac{f(b)-f(a)}{g(b)-g(a)}$$

## 积分中值定理

**定理：**设 $$f(x)\in C[a, b]$$ ，则 $$\exists\xi\in[a,b]$$ ，使 $$\int_a^bf(x)dx=f(\xi)(b-a)$$

**证明**

因为 $$f(x)\in C[a, b]$$ ，所以 $$f(x)$$ 在 $$[a,b]$$ 上存在最值，$$m\le f(x)\le M$$ ，则：

$$\int_a^bmdx\le\int_a^bf(x)dx\le\int_a^bMdx$$

$$m(b-a)\le\int_a^bf(x)dx\le M(b-a)$$

$$m\le\frac{1}{b-a}\int_a^bf(x)dx\le M$$

$$\therefore\exists\xi\in[a,b]$$ ，使 $$f(\xi)=\frac{1}{b-a}\int_a^bf(x)dx$$ （介值定理）

故 $$\int_a^bf(x)dx=f(\xi)(b-a)$$

### 推理

设 $$f(x)\in C[a, b]$$ ，则 $$\exists\xi\in(a,b)$$ ，使 $$\int_a^bf(x)dx=f(\xi)(b-a)$$

（不取两端点）

**证明**

令 $$F(x)=\int_a^xf(t)dt$$ ，$$F'(x)=f(x)$$

则 $$\int_a^bf(x)dx=F(b)-0=F(b)-\int_a^af(x)dx=F(b)-F(a)$$

$$\therefore\exists\xi\in(a,b)$$ ，使 $$\frac{F(b)-F(a)}{b-a}=F'(\xi)$$ （拉格朗日中值定理）

$$F(b)-F(a)=F'(\xi)(b-a)=f(\xi)(b-a)$$



## 参考资料

[1]. https://zhuanlan.zhihu.com/p/363817029