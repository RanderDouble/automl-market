# RQ1/RQ2/RQ3 复现实验手册

这份手册回答一个最实际的问题：从一个干净环境开始，怎样运行本项目，并知道输出是否与报告一致。

本项目是“算法语义与定性结论的缩减复现”。原论文的 NYC 数据池、School 数据任务估值和完整实验代码没有公开，因此不能声称逐点重现论文 Figure 3--5。这里把每个替代设置、随机种子、评价指标和输出文件全部写明，保证项目内部可以重复。

## 1. 先理解三个研究问题

| 研究问题 | 论文想验证什么 | 本项目如何验证 | 主要图 |
|---|---|---|---|
| RQ1 | Data-Bandit 能否用较少模型训练发现好的“数据增强 + 模型”组合 | 在 UCI Wine Quality 上构造可连接外部特征，比较 Data-Bandit、Data-All、Data-Alt、AutoML | `results/figures/rq1_discovery.pdf` |
| RQ2 | 联合定价 MILP 相比 Independent、Shift、Jiggle 能获得多少收入，IS 价格能否泛化到 OOS | 在 10 个透明合成任务上比较五种价格，并同时报告论文强制购买指标和自愿购买实现指标 | `results/figures/rq2_paper_reproduction.pdf` |
| RQ3 | 平台能否从停止时间学习买家类型先验；学习率和批量如何影响收敛 | 构造 5 类可区分的停止分布，比较 3 种学习率、2 种批量、5 个先验和 5 个种子 | `results/figures/rq3_paper_reproduction.pdf` |

理论和代码的阅读顺序建议是：

1. `src/automl_market/market.py`：买家为什么停止；
2. `src/automl_market/pricing.py` 与 `milp.py`：平台怎样定价；
3. `src/automl_market/learning.py`：怎样从停止轮次更新先验；
4. `src/automl_market/discovery.py`：Data-Bandit 怎样减少模型训练；
5. `scripts/run_rq1.py` 与 `scripts/run_paper_experiments.py`：怎样把算法组成实验。

## 2. 环境与完整复现

### 2.1 创建环境

从项目根目录开始：

```bash
# 1. 用 environment.yml 创建 Conda 环境（包含 Python、NumPy、Matplotlib、HiGHS 等）
conda env create -f environment.yml

# 2. 激活环境
conda activate automl-market

# 3. 验证环境：跑通单元测试
make test
```

测试全部通过（或因缺少 `highspy` 跳过 3 项 MILP 测试）即为环境准备完毕。

如果不用 Conda，也可以 pip 安装：

```bash
pip install -r requirements.txt
pip install -r requirements-milp.txt   # MILP 可选
```

### 2.2 快速试跑（可选）

完整实验耗时较长，建议第一次先跑缩小版，确认端到端能通：

```bash
rm -rf /tmp/automl-market-demo
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src \
  python scripts/run_rq1.py --output /tmp/automl-market-demo \
  --repeats-per-color 1 --budget 12

MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src \
  python scripts/run_paper_experiments.py --output /tmp/automl-market-demo \
  --rq2-repeats 2 --rq3-seeds 1 --rq3-rounds 100
```

### 2.3 完整复现

激活环境后，按需执行：

```bash
make rq1                # RQ1：60 个公开数据重复任务
make paper-experiments  # RQ2/RQ3：10 个定价任务 + 先验学习
make report             # 编译中文 PDF 报告（依赖前面的 JSON/CSV）
```

或者一键全跑：

```bash
make all
```

执行顺序：测试 → 补充实验 → RQ1 → RQ2/RQ3 → 中文报告 → 展示。随机种子已固定，同一软件环境下每次跑出来的 CSV/JSON 和图完全一致。

如果只想复现论文三组实验，不重新编译文档：

```bash
make rq1
make paper-experiments
```

成功时，终端会分别打印 `Completed RQ1 ...` 和 `Completed RQ2 ... and RQ3 ...`，并在输出目录下生成 JSON、CSV 和图。

## 3. RQ1：数据与发现算法

### 3.1 实验单位

- 数据：UCI Wine Quality 红葡萄酒与白葡萄酒数据；压缩包保存在 `data/raw/wine-quality.zip`。
- 任务：分类与回归各 30 个重复任务，共 60 个任务。
- 特征：2 个基础特征，9 个特征分别视为可连接的外部增强表。
- 模型：6 个自包含模型族。
- 预算：每个方法最多 60 次模型训练。
- 随机种子：由任务类型、数据颜色和重复编号确定，起始基数为 31000。

### 3.2 四种方法到底差在哪里

- **Data-Bandit**：每轮选择一个增强，并按 Exp3 概率只训练一个模型；奖励更新该模型的权重。
- **Data-All**：对当前增强训练全部模型，单轮更全面，但消耗预算更快。
- **Data-Alt**：轮换模型的低成本对照，用于判断收益是否真的来自 bandit 自适应。
- **AutoML**：只在买家的基础数据上选模型，不使用外部增强。

验证集用于搜索和选择，测试集只用于报告 incumbent 曲线，避免直接用测试分数指导搜索。

### 3.3 指标怎样读

- `final_utility`：预算用完时的测试效用；回答“最后谁最好”。
- `normalized_auc`：预算 1 到 60 的 incumbent 测试效用平均值；回答“整个搜索过程谁更早找到好方案”。这里的 normalized 指按预算长度取平均，不是再除以 oracle。
- `calls_to_95pct_oracle_gain`：达到从基础模型到 oracle 的 95% 增益需要多少次训练；回答“谁更省训练”。未达到记为 61。
- 配对 bootstrap：在相同任务上比较方法的 AUC 差，避免只看两组独立均值。

完整结果应复现以下结论：Data-Bandit 相比 Data-All 的 AUC 差在分类/回归上为 `+0.00794`/`+0.00335`，95% 区间均不跨 0；达到 95% oracle 增益所需训练减少 30.3%/27.5%。但预算终点仍是 Data-All 最好，Data-Bandit 与 Data-Alt 的区间跨 0。

核对文件：

- `results/rq1_summary.json`：参数与核心统计；
- `results/tables/rq1_task_results.csv`：每个任务、每种方法的原始结果；
- `results/tables/rq1_paired_comparisons.csv`：配对区间；
- `results/figures/rq1_discovery.pdf`：报告图。

## 4. RQ2：定价收入的 IS/OOS 比较

### 4.1 实验单位

- 10 个随机任务；固定总种子 20260716。
- 每个任务有 4 个质量状态、6 个买家类型。
- IS 训练集合有 100 条轨迹；OOS 测试集合有 4000 条轨迹。
- 比较 MILP (IS)、Independent、Shift、Jiggle、OOS-informed MILP。
- 收入除以零价格下的可获得福利，使不同任务可以汇总。

IS 训练集合让所有质量都可达，用于拟合论文式价格；OOS 轨迹来自随机发现过程，用于检查价格换到真实可达集合后是否仍有效。OOS-informed MILP 看到了 OOS 训练轨迹，只作为参考上界式对照，不是实际可部署方法。

### 4.2 为什么报告两种收入

- `forced_*`：严格遵循论文 MILP 的强制选择语义，每个买家都购买一个可达质量。
- `realized_*`：给买家零效用的不购买选项；如果所有模型净效用都为负，收入为零。

因此 RQ2 的正确结论分两层：论文指标下 MILP 最好，IS 归一化收入为 `0.7926±0.0098`，并复现 Shift ≥ Independent、Jiggle ≥ Shift；自愿购买指标下，本缩减设置的 Independent OOS 收入最高，为 `0.5468±0.0119`。后一结果说明评价语义会改变排序，不代表 Independent 在一般市场中最优。

核对文件：

- `results/tables/rq2_paper_reproduction.csv`：10 个任务的逐任务结果；
- `results/tables/rq2_paper_summary.csv`：均值与标准误；
- `results/figures/rq2_paper_reproduction.pdf`：强制/实现收入的 IS/OOS 图。

## 5. RQ3：从停止时间学习类型先验

### 5.1 实验单位

- 10 个质量状态、5 个买家类型、最长 15 轮。
- 通过最优停止 DP 和 12000 条 Markov 轨迹估计每类买家的停止时间似然。
- 五类平均停止时间约为 1.0、4.0、7.2、10.2、12.6，保证行为可区分。
- 真实先验：random、uniform、slightly/highly/extremely skewed。
- 学习率：`1/(t+1)`、`1/sqrt(t)`、常数 `1/2`。
- 批量：10、100；5 个种子；1000 轮。

每轮先根据该批停止时间做 Bayes 更新，再用学习率把后验平滑进当前先验。评价使用真实先验到估计先验的 KL 散度。

### 5.2 怎样区分速度和稳定性

- `final_kl`：第 1000 轮误差；越小越好。
- `tail_mean_kl`：最后 100 轮的平均误差；避免只看偶然的最后一点。
- `tail_std_kl`：最后 100 轮的波动；越小越稳定。

随机先验、批量 100 时，最终 KL 应约为 `0.03752`、`0.00684`、`0.00519`；对应尾部波动约为 `0.00008`、`0.00019`、`0.00353`。因此常数 `1/2` 快但持续振荡，`1/(t+1)` 很稳但早期衰减过快，`1/sqrt(t)` 是较均衡的折中。

核对文件：

- `results/tables/rq3_paper_reproduction.csv`：每个先验、学习率、批量和种子的结果；
- `results/tables/rq3_paper_summary.csv`：分组统计；
- `results/figures/rq3_paper_reproduction.pdf`：KL 曲线与最终 KL 图。

## 6. 结果不一致时怎样排查

1. 先运行 `make test`；若缺少 `highspy`，三个 MILP 测试会跳过，此时 RQ2 的连续 MILP 不能完整核验。
2. 确认使用 `environment.yml` 对应环境，而不是系统 Python。
3. 确认没有改动脚本中的默认随机种子、重复次数或预算。
4. 查看 JSON 顶层参数，确认本次实验规模与报告一致。
5. 图的字体或抗锯齿可能因系统不同而略有差异；CSV/JSON 数值才是核验依据。

## 7. 当前不继续扩展的实验

IR-aware、鲁棒 IR、连续 MILP 与估值网格差异，以及较大 MILP 的 60 秒求解间隙都保留在 `results/summary.json`、`results/tables/pricing_results.csv` 和报告补充章节中。它们用于说明机制边界，不属于当前 RQ1/RQ2/RQ3 复现主线。本阶段不继续增加求解器调参、更多鲁棒模型或转移误差消融。
