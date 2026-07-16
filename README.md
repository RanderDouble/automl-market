# Optimal Pricing for Data-Augmented AutoML Marketplaces

课程大作业 C.3 的中文讲解、实验复现和报告工程。当前工作重点是忠实解释论文并复现 RQ1/RQ2/RQ3；已有的机制修正和 MILP 求解边界作为补充实验记录，不继续做大规模调参探索。

建议先阅读 [`docs/paper_walkthrough.md`](docs/paper_walkthrough.md)。它按“市场流程 → 最优停止 → MILP 定价 → 先验学习 → Data-Bandit → 三个研究问题”的顺序讲解论文，并标出公式、代码和结果之间的对应关系。

## 已实现内容

- Proposition 4.1 / Algorithm 3：买家最优停止动态规划；
- Theorem 4.2 对应的样本平均定价目标和 HiGHS MILP；
- Appendix D.1：Independent、Shift、Jiggle 定价基线；
- Algorithm 1 / RQ3：根据停止时间学习买家类型先验；
- Algorithm 2 / RQ1：公开 UCI Wine Quality 上的 Data-Bandit 缩减复现；
- RQ2：IS/OOS、多随机任务、论文强制购买指标和自愿购买实现指标；
- 补充记录：外部选项（IR）、鲁棒 IR、有限价格上界和连续 MILP/估值网格差异。

## Conda 环境

推荐使用 Conda，因为论文定价复现的可选 MILP 后端依赖 `highspy`：

```bash
conda env create -f environment.yml
conda activate automl-market
```

当前机器已创建环境，也可以直接使用解释器：

```bash
PYTHON=/home/rander/miniforge3/envs/automl-market/bin/python
```

基础代码只依赖 NumPy 和 Matplotlib；没有安装 `highspy` 时，三个 MILP 单元测试会被跳过，其他测试仍可运行。

## 运行方式

```bash
make test
make rq1
make paper-experiments
make report
```

若未激活环境：

```bash
make all PYTHON=/home/rander/miniforge3/envs/automl-market/bin/python
```

各目标含义：

- `make experiment`：原有小型定价、IR/鲁棒 IR 和先验学习记录；
- `make rq1`：60 个公开数据重复任务；
- `make paper-experiments`：10 个 RQ2 任务与 5 种先验、3 种学习率的 RQ3；
- `make report`：先刷新全部实验，再生成中文 PDF；
- `make slides`：生成已有的中期展示；
- `make all`：测试、报告和展示。

随机种子固定，XeLaTeX 构建使用固定 `SOURCE_DATE_EPOCH`。

## 主要输出

- `results/rq1_summary.json`：RQ1 端点、发现曲线 AUC、样本效率和 bootstrap 区间；
- `results/tables/rq1_*.csv`：RQ1 逐任务、汇总和配对比较；
- `results/paper_experiments_summary.json`：RQ2/RQ3 完整参数和结果；
- `results/tables/rq2_paper_*.csv`：RQ2 IS/OOS 收入；
- `results/tables/rq3_paper_*.csv`：RQ3 多先验、学习率和批量结果；
- `results/figures/rq*_*.pdf|png`：报告图；
- `report/main.pdf`：中文复现报告；
- `slides/midterm.pdf`：已有中期展示和 `slides/speaker_notes.md` 讲稿。

## 当前核心结果

RQ1 中，Data-Bandit 相比 Data-All 的归一化发现曲线 AUC 优势为：分类 `+0.00794`，95% bootstrap 区间 `[0.00340,0.01220]`；回归 `+0.00335`，区间 `[0.00185,0.00481]`。达到 95% oracle 增益所需训练数分别减少 30.3% 和 27.5%。完整预算终点仍是 Data-All 最好；Data-Bandit 与 Data-Alt 的区间跨 0，因此只报告表现相当。

RQ2 中，论文强制购买指标下 IS MILP 捕获 `0.7926±0.0098` 的归一化福利；加入自愿购买后，本缩减设置的 OOS 实现收入由 Independent 取得最高均值 `0.5468±0.0119`。这个差异揭示了论文优化目标和真实购买行为之间的语义边界。

RQ3 中，大批量下较激进学习率收敛更快；常数 `1/2` 波动最大，批量 10 时尤其不稳定；`1/sqrt(t)` 是较好的速度—稳定性折中。

## 复现边界

原文 RQ1 使用约 69K NYC Open Data 数据集池、Metam/Exp3 与多种 AutoML 系统；原始任务、估值和完整代码没有随论文公开。本项目的 RQ1 使用公开 UCI 数据，RQ2/RQ3 使用透明的缩减合成设置，目标是复现算法语义和定性结论，不冒充原文 Figure 3--5 的逐点重现。

论文式强制购买 MILP 若只有 `x(q)>=0` 而没有有限价格上界会无界；本项目显式加入价格上界。估值诱导的有限价格网格只是基线，不是一般多质量连续定价的精确解。较大 `|Q|=20,|Theta|=20,m=100` 的 HiGHS 运行在 60 秒内仍有很大最优性间隙，因此只记录为求解边界，不宣称完成论文规模最优求解。
