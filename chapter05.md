# 抽样技术与样本平均数的抽样分布

## 抽样技术与统计推断

### 抽样调查

- 全面调查（complete survey）：针对总体全部成员进行观察或实验。
- 抽样调查（sampling survey）：从总体中抽取部分个体组成样本，对该样本进行观察或试验，获得样本信息，进而推断未知总体的情况。
- 抽样误差（sampling error）：根据样本信息来推断总体信息时产生的随机误差。

**补充**

幸存者偏差（英语：survivorship bias），另译为「生存者偏差 」，是一种逻辑谬误，选择偏差的一种。过度关注「幸存了某些经历」的人事物，忽略那些没有幸存的（可能因为无法观察到），造成错误的结论。$$^{[1]}$$

### 抽样方法

- 简单随机抽样（simple random sampling）：从总体中完全以随机形式抽取若干个个体组成一个样本。在抽取的过程中，总体中的每个个体被抽到的概率是均等的，并且在任何一个个体被抽取之后总体内成分不变。
- 分层随机抽样（group sampling,stratification sampling）：按有关的因素或指标将总体划分为互不重叠的几个层，再从各层中独立地抽取一定数量的个体，最后将从各层中抽取的个体合在一起，组成一个样本。
- 机械抽样（systematic sampling）：先将总体中的所有个体按顺序编号，然后每隔一定的间隔抽取个体，组成样本。
- 整群抽样（cluster sampling）：以整群为单位的抽样方法，即从总体中抽出来的是某个群体的所有个体。
- 放回抽样（sampling with replacement）：个体被抽取以后即放回总体，同一个个体可以被重复抽取的抽样方式。亦称重复抽样。
- 不放回抽样（sampling without replacement）：个体被抽取以后不放回总体，同一个体能被重复抽取的抽样方式。亦称不重复抽样。

## 抽样分布

### 抽样分布

- 总体分布（population distribution）：总体内个体观察值的次数分布或概率分布。
- 样本分布（sample distribution）：样本内个体观察值的次数分布或概率分布。
- 抽样分布（sampling distribution）：统计量的概率分布，是根据样本的所有可能的观察值计算出来的某个统计量的观察值的分布。
- 标准误（standard error）：样本统计量的标准差，例如，样本平均数的标准差 $$\sigma_{\overline{X}}=\frac{\sigma}{\sqrt{n}}$$ 。
- 方差齐性（homogeneity of variance）：两个或多个总体的方差无显著差异。

**补充**

参阅：[中心极限定理](./central_limit.html)

- 放回抽样（sampling with replacement）：个体被抽取以后即放回总体，同一个个体可以被重复抽取的抽样方式。亦称重复抽样。
- 不放回抽样（sampling without replacement）：个体被抽取以后不放回总体，同一个体能被重复抽取的抽样方式。亦称不重复抽样。

## 参考资料

[1]. [维基百科：幸存者偏差](https://zh.wikipedia.org/wiki/%E5%80%96%E5%AD%98%E8%80%85%E5%81%8F%E8%AA%A4)











