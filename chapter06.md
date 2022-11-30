# 平均数的参数估计

## 参数估计

- 待估参数（parameter to be estimated）：在参数估计中需要估计的参数。
- 估计量（estimator）：用来估计参数的统计量。
- 估计值（estimated value）：作为估计量的统计量的值。
- 参数估计（parameter estimation）：设总体 $$X$$ 服从分布 $$F(x)$$ ，该总体有参数 $$\theta$$ ，根据抽自该总体的样本 $$X_1,X_2,\cdots,X_n$$ 构造出一个估计量 $$\hat{\theta}$$ 去估计 $$\theta$$ ，估计值 $$\hat{\theta}$$ 由样本的一组观察值计算得出。
- 点估计（point estimation）：根据样本的观察值计算出一个与 $$\theta$$ 相应的估计值，用这个估计值直接作为对参数 $$\theta$$ 的估计。
- 区间估计（interval estimation）：根据样本的观察值计算出两个估计值 $$\hat{\theta}_1$$ 和 $$\hat{\theta}_2$$ ，用区间 $$(\hat{\theta}_1, \hat{\theta}_2)$$ 作为参数 $${\theta}$$ 可能的取值范围，并指出参数 $$\theta$$ 落在这一区间的概率。
- 无偏估计量（unbiased estimator）：设 $$\hat{\theta}$$ 为待估参数 $$\theta$$ 的估计量，若 $$E(\hat{\theta})=\theta$$ ，则称 $$\hat{\theta}$$ 为 $$\theta$$ 的无偏估计量。
- 有效估计量（effective estimator）：设 $$\hat{\theta}_1$$ 和 $$\hat{\theta}_2$$ 为待估参数 $$\theta$$ 的两个无偏估计量，若 $$\sigma_{\theta_1}^2\lt\sigma_{\theta_2}^2$$ ，则称 $$\hat{\theta}_1$$是较 $$\hat{\theta}_2$$ 有效的估计量。
- 一致估计量（consistent estimator）：设 $$\hat{\theta}$$ 为待估参数 $$\theta$$ 的估计量，若 $$n\to\infty$$ 时，$$\hat{\theta}$$ 收敛于 $$\theta$$ ，即 $$\text{lim}_{n\to\infty}\hat{\theta}=\theta$$ ，则称 $$\hat{\theta}$$ 为 $$\theta$$ 的一致估计量。
- 充分估计量（fully estimator）：充分地利用了样本提供的所有有关待估参数的信息的估计量。
- 置信水平（confidence level）：参数真值 $$\theta$$ 落在置信区间 $$(\hat{\theta}_1,\hat{\theta}_2)$$ 里的概率。
- 置信区间（confidence interval）：在区间估计中，若关系式 $$P(\hat{\theta}_1\lt\theta\lt\hat{\theta}_2)=1-\alpha$$ 成立，则称区间 $$(\hat{\theta}_1,\hat{\theta}_2)$$ 为参数 $$\theta$$ 在置信水平 $$1-\alpha$$ 下的置信区间。
