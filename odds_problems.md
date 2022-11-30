# 概率的典型问题和解答

## 取球问题

**题目 1**

一袋中有 6 个白球 4 个红球，随机从中抽取 3 球，若 P(A) 表抽中 2 白球 1 红球的
概率 ，P(B) 表至少抽中 1 个白球的概率 。 试P(A) 与 P(B) 。

*解*

$$\begin{split}P(A)&=\frac{C_6^2C_4^1}{C_{10}^3}=\frac{15\times4}{\frac{10\times9\times8}{1\times2\times3}}=\frac{1}{2}\\P(B)&=\frac{C_6^1C_4^2+C_6^2C_4^1+C_6^3}{C_{10}^3}=\frac{6\times6+15\times4+\times3}{120}=\frac{29}{30}\end{split}$$

或者，先计算抽取 3 个红球的概率 $$P(\overline{B})$$，然后计算 P(B)

$$P(B)=1-P(\overline{B})=1-\frac{C_4^3}{C_{10}^3}=1-\frac{4}{120}=\frac{29}{30}$$



http://w2.tpsh.tp.edu.tw/math0128/Example/MOOCs_Ball/Learn_Blue_0528.pdf