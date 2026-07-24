# Optimal Pricing for Data-Augmented AutoML Marketplaces

课程大作业 C.3 的中文讲解、实验复现、机制改进和报告工程。项目主线是论文提出的“按数据增强带来的模型性能提升定价”：平台不按原始数据表数量、数据体量或计算搜索成本收费，而是对可测模型质量提升发布价格曲线。当前项目在解释并复现 RQ1/RQ2/RQ3 的基础上，把 forced-choice 指标与可退出市场的语义差距、先验冷启动、搜索成本忽略和发现--定价脱节四个部署边界做成了可运行、可测试、可出图的补充实验。

建议先阅读 [`docs/paper_walkthrough.md`](docs/paper_walkthrough.md)。它按“市场流程 → 最优停止 → MILP 定价 → 先验学习 → Data-Bandit → 三个研究问题”的顺序讲解论文，并标出公式、代码和结果之间的对应关系。准备实际运行时，再按 [`docs/reproduction_guide.md`](docs/reproduction_guide.md) 逐项核对实验协议、命令、指标和输出文件。机制改进部分见 [`docs/improvement_results.md`](docs/improvement_results.md)。

## 已实现内容

- Proposition 4.1 / Algorithm 3：买家最优停止动态规划；
- Theorem 4.2 对应的样本平均定价目标和 HiGHS MILP；
- Appendix D.1：Independent、Shift、Jiggle 定价基线；
- Algorithm 1 / RQ3：根据停止时间学习买家类型先验；
- Algorithm 2 / RQ1：公开 UCI Wine Quality 上的 Data-Bandit 缩减复现；
- RQ2：IS/OOS、多随机任务、论文强制购买指标和自愿购买实现指标；
- 机制改进：外部选项修正、有限场景鲁棒定价、成本感知 Markov 定价、warm-start 定价引导 Data-Bandit；
- 补充记录：有限价格上界和连续 MILP/估值网格差异。

## Conda 环境

推荐使用 Conda，因为论文定价复现的可选 MILP 后端依赖 `highspy`：

```bash
conda env create -f environment.yml
conda activate automl-market
```

完整复现请优先使用 `environment.yml`。`requirements.txt` 只覆盖非 MILP 基础模块；若用 `pip` 环境复现 RQ2 的连续 MILP，请额外安装 `requirements-milp.txt`。

## 运行方式

```bash
make test
make improvements
make rq1
make paper-experiments
make report
```

若希望通过 `conda run` 直接指定环境：

```bash
conda run -n automl-market make all
```

各目标含义：

- `make experiment` / `make legacy-experiment`：旧版小型定价、IR/鲁棒 IR 和先验学习记录，保留作兼容入口；正式报告不再调用它；
- `make improvements`：四个机制改进的表格、JSON 和图；
- `make rq1`：60 个公开数据重复任务；
- `make paper-experiments`：10 个 RQ2 任务与 5 种先验、3 种学习率的 RQ3；
- `make report`：使用现有结果生成中文 PDF；
- `make slides` / `make rq-handout`：生成统一展示与 RQ1/RQ3 讲义；
- `make paper-summary`：更新论文中文摘要 PDF；
- `make all`：测试、刷新全部实验并生成报告、展示和讲义。

随机种子和 Conda 中影响数值/图形的依赖版本均已固定，XeLaTeX 构建使用固定 `SOURCE_DATE_EPOCH`。

第一次理解代码时，可以先运行缩小次数的教学版本；它只验证端到端数据流，不替代报告中的完整统计：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src \
  python scripts/run_rq1.py --output /tmp/automl-market-demo \
  --repeats-per-color 1 --budget 12
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src \
  python scripts/run_paper_experiments.py --output /tmp/automl-market-demo \
  --rq2-repeats 2 --rq3-seeds 1 --rq3-rounds 100
```

## 主要输出

- `results/rq1_summary.json`：RQ1 端点、发现曲线 AUC、样本效率和 bootstrap 区间；
- `results/tables/rq1_*.csv`：RQ1 逐任务、汇总和配对比较；
- `results/paper_experiments_summary.json`：RQ2/RQ3 完整参数和结果；
- `results/tables/rq2_paper_*.csv`：RQ2 IS/OOS 收入；
- `results/tables/rq3_paper_*.csv`：RQ3 多先验、学习率和批量结果；
- `results/figures/rq*_*.pdf|png`：报告图；
- `results/improvement_experiments_summary.json`：机制改进实验结果；
- `results/tables/improvement_*.csv`：机制改进表格；
- `results/figures/improvement_*.pdf|png`：机制改进图；
- `deliverables/final_report.pdf`：当前可提交的完整中文实验报告；
- `deliverables/project_slides.pdf`：统一项目展示；
- `deliverables/rq1_rq3_handout.pdf`：RQ1/RQ3 详细讲义；
- `report/main.tex`：报告源码；
- `papers/Han2023_Optimal_Pricing_Data-Augmented_AutoML.pdf`：参考论文。

## 当前核心结果

RQ1 中，Data-Bandit 相比 Data-All 的归一化发现曲线 AUC 优势为：分类 `+0.00794`，95% bootstrap 区间 `[0.00340,0.01220]`；回归 `+0.00335`，区间 `[0.00185,0.00481]`。达到 95% oracle 增益所需训练数分别减少 30.3% 和 27.5%。完整预算终点仍是 Data-All 最好；Data-Bandit 与 Data-Alt 的区间跨 0，因此只报告表现相当。

RQ2 中，论文强制购买指标下 IS MILP 捕获 `0.7926±0.0098` 的归一化福利；这里的 IS 集合是完整质量菜单的 price-fitting 构造，不是真实 Markov 搜索轨迹。加入自愿购买后，本缩减设置的 OOS 实现收入由 Independent 取得最高均值 `0.5468±0.0119`。这个差异揭示了论文优化目标和真实购买行为之间的语义边界。

RQ3 使用 `(历史最佳质量, 当前质量)` 状态上的精确前向 DP 计算停止分布，不再受有限轨迹 Monte Carlo 噪声影响。在项目 Conda 环境中，随机先验、批量 100 时，三种学习率的最终 KL 分别为 `0.12083`、`0.01926` 和 `0.00495`；常数 `1/2` 波动最大，`1/sqrt(t)` 是较好的速度—稳定性折中。该缩减实验仍采用零价格停止分布，相当于免费探索阶段，不是完整在线“边定价边学习”闭环。

机制改进实验中，Cost-aware Markov 在主实例名义搜索成本 `c=0.03` 下取得最高精确 Markov 收入 `0.3082`；Robust cost-aware Markov 在主实例多个先验和搜索成本场景的最坏情况下取得最高收入 `0.1857`，但名义收入相比 Cost-aware Markov 下降约 23.9%，适合更保守的冷启动/分布偏移场景。定价引导发现的平均发现曲线 AUC 为 `0.7583`，高于 Data-Bandit 的 `0.6949`；12 个扰动分数表上的价格信号消融显示，Pricing-guided Data-Bandit 的 AUC 为 `0.7764±0.0070`，高于随机信号引导的 `0.7588±0.0081`，但随机信号本身也带来明显排序收益，因此价格信号应理解为 warm-start 信息的有限增量。

## 复现边界

原文 RQ1 使用约 69K NYC Open Data 数据集池、Metam/Exp3 与多种 AutoML 系统；原始任务、估值和完整代码没有随论文公开。本项目的 RQ1 使用公开 UCI 数据，RQ2/RQ3 使用透明的缩减合成设置，目标是复现算法语义和定性结论，不冒充原文 Figure 3--5 的逐点重现。

论文式强制购买 MILP 若只有 `x(q)>=0` 而没有有限价格上界会无界；本项目显式加入价格上界。估值诱导的有限价格网格只是基线，不是一般多质量连续定价的精确解。有限场景鲁棒定价不是 Wasserstein/KL DRO；Pricing-guided Data-Bandit 使用历史/元数据式 warm-start 信号，不假设冷启动第一单就知道高收入增强方向。较大 `|Q|=20,|Theta|=20,m=100` 的 HiGHS 运行在 60 秒内仍有很大最优性间隙，因此只记录为求解边界，不宣称完成论文规模最优求解。
