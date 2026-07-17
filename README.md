# Optimal Pricing for Data-Augmented AutoML Marketplaces

课程大作业 C.3 的中文讲解、实验复现和报告工程。当前工作重点是忠实解释论文并复现 RQ1/RQ2/RQ3；已有的机制修正和 MILP 求解边界作为补充实验记录，不继续做大规模调参探索。

建议先阅读 [`docs/paper_walkthrough.md`](docs/paper_walkthrough.md)。它按“市场流程 → 最优停止 → MILP 定价 → 先验学习 → Data-Bandit → 三个研究问题”的顺序讲解论文，并标出公式、代码和结果之间的对应关系。准备实际运行时，再按 [`docs/reproduction_guide.md`](docs/reproduction_guide.md) 逐项核对实验协议、命令、指标和输出文件。

## 已实现内容

- Proposition 4.1 / Algorithm 3：买家最优停止动态规划；
- Theorem 4.2 对应的样本平均定价目标和 HiGHS MILP；
- Appendix D.1：Independent、Shift、Jiggle 定价基线；
- Algorithm 1 / RQ3：根据停止时间学习买家类型先验；
- Algorithm 2 / RQ1：公开 UCI Wine Quality 上的 Data-Bandit 缩减复现；
- RQ2：IS/OOS、多随机任务、论文强制购买指标和自愿购买实现指标；
- 补充记录：外部选项（IR）、鲁棒 IR、有限价格上界和连续 MILP/估值网格差异。

## 环境搭建与运行

下面从一个刚 clone 完项目的干净环境开始，一步步走到完整复现。

### 1. 准备 Python 环境

项目需要 Python 3.10+。推荐使用 Conda（论文定价复现的 MILP 后端依赖 `highspy`，Conda 安装最省事）：

```bash
conda env create -f environment.yml
```

这会创建一个名为 `automl-market` 的环境，包含 NumPy、Matplotlib、HiGHS 等全部依赖。

如果不用 Conda，也可以 pip 安装：

```bash
pip install -r requirements.txt
# MILP 可选，没有 highspy 时相关单元测试会自动跳过
pip install -r requirements-milp.txt
```

### 2. 激活环境并跑通测试

```bash
conda activate automl-market
make test
```

看到 14 项测试全部通过（或 11 项通过 + 3 项 MILP 跳过）就说明环境正确。

### 3. 快速试跑（可选）

在跑完整实验之前，可以先用缩小参数跑一遍，确认端到端数据流没问题：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src \
  python scripts/run_rq1.py --output /tmp/automl-market-demo \
  --repeats-per-color 1 --budget 12

MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src \
  python scripts/run_paper_experiments.py --output /tmp/automl-market-demo \
  --rq2-repeats 2 --rq3-seeds 1 --rq3-rounds 100
```

终端打印 `Completed RQ1 ...` 和 `Completed RQ2 ... and RQ3 ...` 即成功。

### 4. 完整复现

```bash
make rq1                # RQ1：60 个公开数据重复任务
make paper-experiments  # RQ2/RQ3：10 个定价任务 + 先验学习
make report             # 编译中文 PDF 报告（依赖前面实验的 JSON/CSV）
make slides             # 编译中期展示 PDF
```

或者一键全跑：

```bash
make all
```

执行顺序：测试 → 补充实验 → RQ1 → RQ2/RQ3 → 中文报告 → 两套展示 → RQ1/RQ3 讲义 → 论文摘要。随机种子已固定，同一软件环境下每次跑出来的 CSV/JSON 和图完全一致。LaTeX 中间文件统一写入 `/tmp/automl-market-latex`，仓库中只保留最终 PDF。

如果不想一直激活 Conda 环境，也可以临时用 `conda run`：

```bash
conda run -n automl-market make all
```

## 主要输出

- `results/rq1_summary.json`：RQ1 端点、发现曲线 AUC、样本效率和 bootstrap 区间；
- `results/tables/rq1_*.csv`：RQ1 逐任务、汇总和配对比较；
- `results/paper_experiments_summary.json`：RQ2/RQ3 完整参数和结果；
- `results/tables/rq2_paper_*.csv`：RQ2 IS/OOS 收入；
- `results/tables/rq3_paper_*.csv`：RQ3 多先验、学习率和批量结果；
- `results/figures/rq*_*.pdf|png`：报告图；
- `deliverables/final_report.pdf`：中文复现报告；
- `deliverables/project_overview_slides.pdf`：完整项目展示；
- `deliverables/rq1_rq3_handout.pdf`：RQ1/RQ3 详细讲义；
- `deliverables/rq1_rq3_slides.pdf`：RQ1/RQ3 正式汇报；
- `deliverables/zby_theory_slides.pdf`：ZBY 的论文理论部分。

## 项目结构

```text
automl-market/
├── papers/                 # 原论文及中文摘要
├── deliverables/           # 可直接查看或提交的最终 PDF
├── docs/                   # 论文讲解、复现指南、讲义源码
├── slides/
│   ├── rq1_rq3/            # 本项目的 RQ1/RQ3 页面与讲稿
│   ├── project_overview/   # 完整项目展示与讲稿
│   └── zby/                # 仅保留 ZBY 的理论页、模板和素材
├── src/                    # Python 实现
├── scripts/                # 实验入口
├── tests/                  # 单元测试
└── results/                # CSV/JSON 与实验图
```

原论文位于 `papers/Han2023_Optimal_Pricing_Data-Augmented_AutoML.pdf`。目录职责详见 [`slides/README.md`](slides/README.md) 和 [`deliverables/README.md`](deliverables/README.md)。

## RQ1 / RQ3 中期汇报材料

负责 RQ1 与 RQ3 的同学可直接使用：

- `deliverables/rq1_rq3_handout.pdf`：13 页详细中文讲义，覆盖研究问题、Algorithm 1/2、原论文实验、缩减复现、讲述稿和问答；
- `deliverables/rq1_rq3_slides.pdf`：采用 ZBY 的修改版 ZJU Beamer 风格，前 11 页为正式汇报，后 3 页为备查；
- `slides/rq1_rq3/frames.tex`：可插入 ZBY 主文件的 frame 源码；
- `slides/rq1_rq3/main.tex`：同风格独立编译入口；
- `slides/rq1_rq3/speaker_notes.md`：约 7--8 分钟逐页讲稿。

重新编译：

```bash
make rq-handout
make rq-slides
```

两条命令只在 `/tmp` 产生中间文件，并分别更新 `deliverables/rq1_rq3_handout.pdf` 和 `deliverables/rq1_rq3_slides.pdf`。

## 当前核心结果

RQ1 中，Data-Bandit 相比 Data-All 的归一化发现曲线 AUC 优势为：分类 `+0.00794`，95% bootstrap 区间 `[0.00340,0.01220]`；回归 `+0.00335`，区间 `[0.00185,0.00481]`。达到 95% oracle 增益所需训练数分别减少 30.3% 和 27.5%。完整预算终点仍是 Data-All 最好；Data-Bandit 与 Data-Alt 的区间跨 0，因此只报告表现相当。

RQ2 中，论文强制购买指标下 IS MILP 捕获 `0.7926±0.0098` 的归一化福利；加入自愿购买后，本缩减设置的 OOS 实现收入由 Independent 取得最高均值 `0.5468±0.0119`。这个差异揭示了论文优化目标和真实购买行为之间的语义边界。

RQ3 中，大批量下较激进学习率收敛更快；常数 `1/2` 波动最大，批量 10 时尤其不稳定；`1/sqrt(t)` 是较好的速度—稳定性折中。

## 复现边界

原文 RQ1 使用约 69K NYC Open Data 数据集池、Metam/Exp3 与多种 AutoML 系统；原始任务、估值和完整代码没有随论文公开。本项目的 RQ1 使用公开 UCI 数据，RQ2/RQ3 使用透明的缩减合成设置，目标是复现算法语义和定性结论，不冒充原文 Figure 3--5 的逐点重现。

论文式强制购买 MILP 若只有 `x(q)>=0` 而没有有限价格上界会无界；本项目显式加入价格上界。估值诱导的有限价格网格只是基线，不是一般多质量连续定价的精确解。较大 `|Q|=20,|Theta|=20,m=100` 的 HiGHS 运行在 60 秒内仍有很大最优性间隙，因此只记录为求解边界，不宣称完成论文规模最优求解。
