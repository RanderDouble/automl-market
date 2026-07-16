# Optimal Pricing for Data-Augmented AutoML Marketplaces

课程大作业的可复现实验与报告工程。当前版本覆盖：

- 论文 Proposition 4.1 / Algorithm 3 的买方最优停止动态规划；
- 论文式 (7)/(8) 的经验价格曲线目标的小规模精确网格复现；
- Appendix D.1 的 independent pricing 基线；
- 论文 RQ3 的“由停止时间学习买家类型先验”实验；
- 改进机制：给定价问题加入“不购买”的外部选项（个体理性，IR）；
- 进一步的分布鲁棒 IR 定价，在多个候选买家先验上最大化最坏收入。

## 一键运行

环境只依赖 Python、NumPy、Matplotlib 和 XeLaTeX。

```bash
make all
```

主要输出：

- `results/summary.json`：全部可复现实验参数和指标；
- `results/tables/pricing_results.csv`：定价实验原始表格；
- `results/figures/*.png|pdf`：报告图；
- `report/main.pdf`：阶段性实验报告。

随机种子固定为 `20260716`。测试可单独运行：

```bash
make test
```

## 复现边界

论文的大规模发现实验使用 69K NYC Open Data 数据集池、Metam/Exp3 与多种 AutoML
系统；这些数据与原作者实现没有随论文公开。因此当前工程首先复现理论机制和可验证的
RQ2/RQ3 合成实验，明确不把它冒充为论文 Figure 3--5 的逐点重现。后续版本将补充公开
表格数据上的 augmentation-model bandit 实验和更大规模 MILP 求解器接口。

