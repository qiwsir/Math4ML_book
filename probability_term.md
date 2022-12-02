# 概率论的基本概念

*打开本页，如果不能显示公式，请刷新页面。*

本文内容来自参考资料 [1] ，并针对我们的语言习惯进行了适当修改。

> 我发现数学很容易，但我还是喜欢探索事物。你必须有必要的信息。例如，平均数与中位数的差异是甚么？概率论让我着迷。你必须非常仔细地考虑事情，这也正是我心灵运作的方式。
>
> ——───英国作家丹尼尔·谭米特 (Daniel Tammet)

概率论)是一个研究机会与运气的数学领域。概率虽然经常出现于日常语言，但学者往往对于概率论的概念有所误解或认为难以捉摸。我想到的原因有两个层面。

- 第一是观念面的原因。概率论有它自己的奥术语汇，掌握这些必要的语汇是建立模型与精准推理的前提。一些看似简单的概率问题，答案常常违反直觉甚至可能让专业数学家跌破眼镜，原因即在于误入思考的陷阱。

- 第二是技术面的原因。许多概率问题无法用简单的排列组合计算，必须引进较为复杂的方法。譬如，投掷一枚公正的硬币 20 次，求连续 4 次正面出现的概率涉及解递归关系式，而连续概率分布问题则难以免除技巧性的积分运算。

我们研习任何一门学科总要先从观念面下手，本文列举一些基本的概率论词汇并解释它们的意义（其他重要的词汇参考本站其他内容或者参考资料 [2] ）。

## 样本空间

一个样本空间（sample space）是某个特定的实验（experiment）所有可能出现的结果(outcome）形成的集合。我们将样本空间记为 $$\Omega$$ ，元素则以 $$\omega$$ 表示。

概率与统计意义下的实验是指数据产生的任何过程，例如，投掷一枚硬币、测量每日的累积降雨量、从群体中抽出一个人并记录他的生日等。

- 在概率论中，样本空间是所有可能的实验结果的集合。
- 在统计学中，样本空间是指能被抽样的个体或项目的集合，也就是从中抽样的总体（population）。

因此，统计学所称的样本是从总体选出的一组个体或项目，譬如，一项政府政策支持度的抽样调查。

**例1.** 投掷一枚硬币两次并记录可能的结果，样本空间为 $$\Omega=\{HH, HT, TH, TT\}$$ ，其中 $$H$$ （head） 表示正面，$$T$$ （tail）表示反面。注意，$$HT$$ 与 $$TH$$ 是两个不同的结果，$$HT$$ 表示第一次掷出 $$H$$ 且第二次掷出 $$T$$ ，而 $$TH$$ 表示第一次掷出 $$T$$ 且第二次掷出 $$H$$ 。

**例2.** 假设一个长度为 10 的 DNA 序列由四种核甘酸 (nucleotide) $$\text{A, C, G, T}$$ 组成，例如，$$w=(\text{G, A, T, T, G, C, A, C, T, C})$$ ，则样本空间 $$\Omega$$ 包含 $$4^{10}$$ 个元素，记为 $$\vert\Omega\vert=4^{10}$$ 。

**例3.** 假设在平年出生的人的生日以 1 至 365 的整数表示。考虑随机抽选两个在平年出生的人并记录他们的生日，样本空间为 $$\Omega=\{(i,j)\,\vert\,1\le i,j\le 365\}$$ 。

**例4.** 测量某一天的日累积降雨量 (单位为毫米)，样本空间为 $$\Omega=\{0,1,\ldots,1403\}$$ 。2009年8月8日至9日，莫拉克 (Morakot) 台风期间，屏东尾寮山测得降雨量 1,403 毫米，创台湾所有气象站中单日最大雨量纪录。理想上，样本空间不要少于或多于所有可能的实验结果，但在真实世界，我们往往不知道样本空间为何 (谁知道累积降雨量是否会更创新高)。一个解决方式是设 $$\Omega=\{0,1,2,\ldots\}$$ ，日后我们再介绍如何让统计学帮助建立一个模型。

在掷币实验中，即便绝少发生，或许仍有人坚持应该纳入硬币垂直站立的情况，于是将样本空间设为 $$\{H,T,E\}$$ ，其中 $$E$$ 表示硬币垂直站立。应用概率学于现实问题时，样本空间并不是唯一的，我们必须根据所考虑的问题现象与情境决定样本空间。如果在沙地上掷币，那么 $$\{H,T,E\}$$ 可能是一个恰当的样本空间，但如果在水泥地板上掷币，$$\{H,T\}$$ 应该是比较合乎实况的选择。针对眼前的问题，我们选择的样本空间即为对思考模型所作的理论假设。

## 事件

一个事件 (event) 是样本空间的一个子集合，也就是实验可能出现所有结果的子集合。事件常用集合符号或大写英文字母表示。


**例5.** 投掷一枚硬币两次，$$\Omega=\{HH,HT,TH,TT\}$$ 是一个小样本空间，总共有 $$2^4=16$$ 个可能的事件：

- 不包含任何元素的事件，称为空集合：$$\emptyset$$ ，
- 包含一个元素的事件：$$\{HH\},\{HT\},\{TH\},\{TT\}$$ ，
- 包含两个元素的事件：$$\{HH,HT\},\{HH,TH\},\{HH,TT\},\{HT,TH\},\{HT,TT\},\{TH,TT\}$$ ，
- 包含三个元素的事件： $$\{HH,HT,TH\},\{HH,HT,TT\},\{HH,TH,TT\},\{HT,TH,TT\}$$ ，
- 包含所有元素的事件：$$\{HH,HT,TH,TT\}$$ 。

**例6.** 在例4，我们将日累积降雨量予以分级，定义下列事件：

- 小雨 (light rain)：$$L=\{0<\omega\le 80\}$$ ，
- 大雨 (heavy rain)：$$H=\{80<\omega\le 200\}$$ ，
- 暴雨 (extremely heavy rain)：$$EH=\{200<\omega\le 350\}$$ ，
- 大暴雨 (torrential rain)：$$T=\{350<\omega\le 500\}$$ ，
- 超大暴雨 (extremely torrential rain)：$$ET=\{500<\omega\}$$ 。

当然，$$\{0\}$$ 表示没有降雨。明显地，$$\Omega=\{0\}\cup L\cup H\cup EH\cup T\cup ET$$ 。上述事件的集合运算可产生其他事件，例如，$$L\cup H$$ 是小雨或大雨出现的事件，$$\{0\}^c$$ (即事件 \{0\} 的补集) 表示该日下雨。

## 试验

我们进行的每一次实验，称为一个试验 (trial)。每一个试验必定可观察到一个结果 $$\omega\in\Omega$$ 。若 $$\omega\in A$$ ，我们说事件 $$A$$ 发生；若 $$\omega\notin A$$ ，则事件 $$A$$ 未发生。

譬如，投掷一枚硬币两次是一个试验，假设结果是 $$HH$$ 。事件 $$\{HH,HT\}$$ 表示第一次出现正面，事件 $$\{HT,TH,TT\}$$ 表示至少出现一次反面。在这一个试验，我们说事件 $$\{HH,HT\}$$ 发生，因为 $$HH\in \{HH,HT\}$$ ，但事件 $$\{HT,TH,TT\}$$ 未发生，因为 $$HH\notin \{HT,TH,TT\}$$ 。事实上，定义于样本空间 $$\Omega=\{HH,HT,TH,TT\}$$ 的 16 个事件 (见例5) 共有 8 个事件发生。

## 概率函数

对于一个样本空间，一个概率函数给定每一事件一个概率值。概率函数的制定是为了量化「随机」概念，直白地说，概率函数回答这个问题：某件事情发生的可能性有多大？在不造成混淆的情况下，概率值经常简称为概率。

**例7.** 投掷一颗公正的六面骰子，出现点数小于 $$3$$ 的概率是 $$\frac{1}{3}$$，出现点数大于 $$6$$ 的概率是 $$0$$ 。

**例8.** 投掷一枚公正的硬币两次，出现两次正面 $$\{HH\}$$ 的概率是 $$\frac{1}{4}$$ ，两次结果相异 $$\{HT,TH\}$$ 的概率是 $$\frac{1}{2}$$ 。


上面两个例子与我们的直觉吻合，但一个事件的概率究竟是怎么得出的？答案在于概率函数具备甚么性质。为了符合日常经验，我们要求定义于样本空间 $$\Omega$$ 的概率函数 $$P$$ 满足下面三个条件 (公理)：

1. $$P(A)\ge 0$$
2. $$P(\Omega)=1$$
3. 若 $$A\cap B=\emptyset$$ ，则 $$P(A\cup B)=P(A)+P(B)$$ 。

条件 1 是自明的真理：任何事件的概率不允许是负值。一个事件发生的最小概率值为 $$0$$ 表示 $$0\%$$，即不可能发生。

那么最大值呢？如果一个事件必然发生，概率值是多少？每一次试验会出现一个结果，而这个结果必定属于样本空间，条件 2 说整个样本空间的概率是 $$1$$ ，对应 100% 。

条件 3 讲的是一个事件的概率等于它所包含的元素的概率之和，也就是概率的计算方法。在例7，投掷一颗骰子的样本空间为 $$\Omega=\{1,2,3,4,5,6\}$$ ，公正骰子意味 $$A=\{1\}$$ 的概率为 $$P(A)=\frac{1}{6}$$ ，$$B=\{2\}$$ 的概率为 $$P(B)=\frac{1}{6}$$ ，$$C=\{3\}$$ 的概率为 $$P(C)=\frac{1}{6}$$ ，余此类推。因为 $$A\cap B=\emptyset$$ ，也就是说事件 $$A$$ 与 $$B$$ 互斥，直观经验告诉我们出现骰子点数小于 $$3$$ 的事件 $$A\cup B=\{1,2\}$$ 的概率 $$P(A\cup B)$$ 等于 $$P(A)+P(B)=\frac{1}{6}+\frac{1}{6}=\frac{1}{3}$$ 。若要计算骰子点数小于 $$4$$ 的事件 $$A\cup B\cup C=\{1,2,3\}$$ 的概率，因为事件 $$A, B, C$$ 两两互斥，重复使用条件3，

$$\begin{aligned} P(A\cup B\cup C)&=P(A\cup B)+P(C)\\ &=P(A)+P(B)+P(C)\\ &=\frac{1}{6}+\frac{1}{6}+\frac{1}{6}=\frac{3}{6}=\frac{1}{2}. \end{aligned}$$

条件 3 可以推广至多个两两互斥事件的并集。若 $$A_1,\ldots,A_k$$ 满足 $$A_i\cap A_j=\emptyset，i\neq j$$ ，则

$$P(A_1\cup\cdots\cup A_k)=P(A_1)+\cdots+P(A_k)$$

我们可以从一个事件发生的频率来解释概率的意义：在相同的条件下，如果一个实验重复 $$n$$ 次，或者说进行 $$n$$ 次试验，$$P(A)$$ 近似实验结果 $$\omega_1,\ldots,\omega_n$$ 属于 $$A$$ 的次数 (即事件 $$A$$ 发生的次数)，记为 $$n_A$$ ，与试验总数 $$n$$ 的比值，即 $$P(A)\simeq \frac{n_A}{n}$$ 。概率的频率观点解释符合前述三个条件，说明于下：

1. $$P(A)\ge 0$$ ，因为 $$n_A\ge 0$$ 且 $$n>0$$ 。
2. P(\Omega)=1，因为事件 \Omega 每次都发生，即 n_\Omega=n。
3. 若 $$A\cap B=\emptyset$$ ，则 $$P(A\cup B)=P(A)+P(B)$$ 。因为若 $$A\cup B$$ 发生，则 $$A$$ 或 $$B$$ 发生，但不会同时发生，故 $$P(A\cup B)\simeq\frac{n_{A\cup B}}{n}=\frac{n_A}{n}+\frac{n_B}{n}\simeq P(A)+P(B)$$ 。

条件3之所以要求 $$A\cap B=\emptyset$$ 是为了避免重复计数，看这个极端的例子 $$A=\{1\}$$ 且 $$B=\{1\}$$ ，显然 $$P(A\cup B)=\frac{1}{6}$$ 不等于 $$P(A)+P(B)=\frac{1}{3}$$ 。再看 $$A\cap B\neq\emptyset$$ 的另一个例子，$$A=\{1,2\}$$ 且 $$B=\{2,3\}$$ ，如果不先找出 $$A\cup B$$ 的元素，要如何计算 $$P(A\cup B)$$ ？前述概率函数的定义条件足以回答这个问题吗？可以的。以下是概率函数的三个条件的推论。设 $$A$$ 和 $$B$$ 为定义于样本空间 $$\Omega$$ 的任何事件。

(a) $$P(\emptyset)=0$$

因为 $$A\cap \emptyset=\emptyset$$ 且 $$A\cup \emptyset=A$$ ，条件 3 说 $$P(A)=P(A\cup \emptyset)=P(A)+P(\emptyset)$$ 。

(b) $$P(A)=1-P(A^c)\le 1$$

因为 $$A\cap A^c=\emptyset$$ 且 $$A\cup A^c=\Omega$$ ，故 $$1=P(\Omega)=P(A\cup A^c)=P(A)+P(A^c)$$ 。

(c) $$P(A\cup B)=P(A)+P(B)-P(A\cap B)$$

写出 $$A\cup B=A\cup (A^c\cap B)$$ 且 $$B=(A\cap B)\cup (A^c\cap B)$$ ，则 $$P(A\cup B)=P(A)+P(A^c\cap B)$$ 且 $$P(B)=P(A\cap B)+P(A^c\cap B)$$ ，合并即得证。

(d) 若 $$B\subseteq A$$ ，则 $$P(B)\le P(A)$$ 。

若 $$B\subseteq A$$ ，则 $$A=B\cup (A\cap B^c)$$ ，故 $$P(A)=P(B)+P(A\cap B^c)\ge P(B)$$ 。


使用 (c)，$$\{1,2\}\cup\{2,3\}$$ 的概率计算如下：

$$\begin{aligned} P(\{1,2\}\cup\{2,3\})&=P(\{1,2\})+P(\{2,3\})-P(\{1,2\}\cap\{2,3\})\\ &=P(\{1\})+P(\{2\})+P(\{2\})+P(\{3\})-P(\{2\})\\ &=P(\{1\})+P(\{2\})+P(\{3\})\\ &=\frac{1}{6}+\frac{1}{6}+\frac{1}{6}=\frac{1}{2}. \end{aligned}$$


如果样本空间 $$\Omega$$ 包含 $$n$$ 个有限元素 $$\omega_i$$ ，则任何一个事件皆可用基本事件 $$\{\omega_i\}$$ 的概率 $$P(\{\omega_i\})=p_i$$ 表示。从概率函数满足的三个条件可推论 $$p_i\ge 0$$ 且 $$p_1+\cdots+p_n=1$$ 。若 $$A=\{\omega_{i_1},\ldots,\omega_{i_k}\}$$ ，如上例使用条件三可得

$$P(A)=P(\{\omega_{i_1}\})+\cdots+P(\{\omega_{i_k}\})=p_{i_1}+\cdots+p_{i_k}$$

当样本空间 $$\Omega$$ 为一无限可数集 $$\{\omega_1,\omega_2,\ldots\}$$ (包含无穷多个元素的集合，其中每一个元素唯一对应一个自然数)，我们可以另加入一个概率函数的条件，称为无限可加性：若 $$A_1,A_2,\ldots$$ 两两互斥，则

$$P(A_1\cup A_2\cup\cdots)=P(A_1)+P(A_2)+\cdots$$

因此，若 $$A=\{\omega_{i_1},\omega_{i_2},\ldots\}$$ ，算式 $$P(A)=P(\{\omega_{i_1}\})+P(\{\omega_{i_2}\})+\cdots=p_{i_1}+p_{i_2}+\cdots$$ 仍成立。


如果样本空间 $$\Omega$$ 包含无限多个不可数的元素，譬如，$$\Omega=\mathbb{R}$$ 或 $$\Omega=\{(x,y)|0\le x,y\le 1\}$$ ，则 $$\Omega$$ 上的一些事件，如包含单一点的事件，不存在满足前述三个条件的概率函数。为了建立概率函数，我们要求所有的事件必须定义为一个区间，譬如，$$\{x_1\le x\le x_2\}$$ 或 $$\{x_1\le x\le x_2,y_1\le y\le y_2\}$$ ，以及它们的可数的并集与交集。以 $$\Omega=\mathbb{R}$$ 为例，设定事件的形式为 $$\{x\le x_i\}$$ ，其中 $$x_i$$ 是任何数，通过集合运算便足以衍生其他的事件。考虑函数 $$f(x)$$ 满足 $$f(x)\ge 0$$ 且

$$\displaystyle \int_{-\infty}^{\infty}f(x)dx=1$$

事件 $$\{x\le x_i\}$$ 的概率定义为

$$\displaystyle P(\{x\le x_i\})=\int_{-\infty}^{x_i}f(x)dx$$

不难确认此式满足概率函数的三个条件。

最后还有一个实际问题需要厘清：谁决定或该怎么决定概率函数 $$P$$ ？考虑投掷一枚硬币，样本空间为 $$\Omega=\{H,T\}$$ ，设 $$P(\{H\})=p$$ ，则 $$P(\{T\})=1-P(\{H\})=1-p$$ 。因此，任何一个参数 $$p\in[0,1]$$ 皆可定义合法的概率函数，存在无穷多个概率函数，我们应该挑选那一个？概率论没有提供标准答案，这里是数学与物理世界的交界点。我们知道概率函数制定的目的是为了准确预测未来事件发生的可能性。如果投掷一枚硬币非常多次，我们希望挑选出来的 $$p$$ 等于正面出现的次数与总投掷次数的比值。确定了这个目标后，至少有两个办法可找出合适的概率函数。

- 第一个办法，我们可以进行多次掷币试验。假如投掷一枚硬币100次，共出现54次正面，可设 $$p=0.54$$ ，从此便使用这个概率函数来预测未来的掷币实验结果。采用实验方式决定概率函数的方法就是大家常讲的「根据经验」。
- 第二个办法，我们可以研究硬币的型态、构造材质等，再根据这些知识推出 $$p$$ 的「理论值」。假设我们发现硬币的正反两面其实没有甚么差异，于是设 $$p=0.5$$ ，并用它来预测日后的掷币实验结果。现在我们有两个概率函数，但到底 $$p=0.54$$ 还是 $$p=0.5$$ 的准确性较高呢？这个问题属于统计学的研究范围，留待日后讨论。



## 参考资料

[1]. [线代启示录：概率学的基本语汇](https://ccjou.wordpress.com/2016/01/29/%e6%a9%9f%e7%8e%87%e5%ad%b8%e7%9a%84%e5%9f%ba%e6%9c%ac%e8%aa%9e%e5%bd%99/)

[2]. 齐伟，机器学习数学基础，北京：电子工业出版社