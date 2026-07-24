# 机制改进实验结果

本页汇总四个补充机制实验结果：外部选项修正、有限场景鲁棒定价、成本感知定价和 warm-start 定价引导发现。实验入口为：

```bash
make improvements
```

输出文件：

- `results/improvement_experiments_summary.json`
- `results/tables/improvement_pricing_results.csv`
- `results/tables/improvement_pricing_multi_instance.csv`
- `results/tables/improvement_pricing_scenarios.csv`
- `results/tables/improvement_discovery_results.csv`
- `results/tables/improvement_discovery_signal_ablation.csv`
- `results/tables/improvement_summary.csv`
- `results/figures/improvement_pricing_revenue.pdf`
- `results/figures/improvement_pricing_multi_instance.pdf`
- `results/figures/improvement_discovery_curves.pdf`
- `results/figures/improvement_discovery_signal_ablation.pdf`

## 1. 定价机制改进

定价实验使用项目已有透明合成市场：4 个质量状态、6 个买家类型、Markov 发现过程和固定随机种子。主实例用于画图和解释机制；另有 5 个扰动市场用于报告均值与标准误。

比较方法：

- **Paper forced-choice**：论文强制购买目标。
- **Independent**：每个质量单独垄断定价。
- **IR-aware**：加入不购买选项，用于修正 forced-choice 指标与真实可退出市场之间的语义差距。
- **Robust IR**：IR 目标上对多个先验场景取最坏情况。
- **Cost-aware Markov**：用精确 Markov 最优停止收入优化非零搜索成本。
- **Robust cost-aware Markov**：同时对多个先验和多个搜索成本场景取最坏情况；这是有限场景 max-min，不是完整 DRO。

关键结果如下。

| 方法 | 名义 Markov 收入 `c=0.03` | OOS 先验收入 `c=0.03` | 最坏先验/成本收入 |
|---|---:|---:|---:|
| Paper forced-choice | 0.1658 | 0.0622 | 0.0296 |
| Independent | 0.2838 | 0.1927 | 0.1490 |
| IR-aware | 0.2634 | 0.1702 | 0.1366 |
| Robust IR | 0.2478 | 0.1729 | 0.1317 |
| Cost-aware Markov | **0.3082** | **0.2231** | 0.1592 |
| Robust cost-aware Markov | 0.2347 | 0.2049 | **0.1857** |

5 个扰动市场上的均值如下。

| 方法 | 名义 `c=0.03` | OOS 先验 `c=0.03` | 最坏先验/成本 |
|---|---:|---:|---:|
| Paper forced-choice | 0.1451±0.0273 | 0.0673±0.0137 | 0.0417±0.0094 |
| Independent | 0.2600±0.0107 | 0.1698±0.0053 | 0.1527±0.0026 |
| IR-aware | 0.2217±0.0250 | 0.1503±0.0131 | 0.1257±0.0072 |
| Robust IR | 0.2644±0.0158 | 0.1814±0.0130 | 0.1411±0.0112 |
| Cost-aware Markov | **0.3072±0.0116** | **0.2079±0.0056** | 0.1614±0.0034 |
| Robust cost-aware Markov | 0.2420±0.0131 | 0.2041±0.0045 | **0.1843±0.0033** |

结论：

- 强制购买价格在真实可退出市场中最脆弱，最坏情况收入只有 0.0296。
- 成本感知 Markov 定价在名义搜索成本和 OOS 先验下收入最高，说明非零搜索成本会改变最优价格。
- IR-aware 的主要价值是修正 forced-choice 目标与自愿购买市场之间的语义偏差，不保证在所有实例上收入支配 Independent。
- Robust cost-aware Markov 在最坏先验/成本场景下最好，主实例相比 Independent 的最坏情况收入提高约 24.6%，扰动市场均值提高约 20.7%；代价是主实例名义收入相比 Cost-aware Markov 下降约 23.9%，因此它更适合冷启动或分布偏移下的保守部署。

对应图为 `results/figures/improvement_pricing_revenue.pdf` 和 `results/figures/improvement_pricing_multi_instance.pdf`。鲁棒场景细节见 `results/tables/improvement_pricing_scenarios.csv`，包括训练先验、OOS 先验、均匀冷启动先验、高估值偏移先验，以及搜索成本 `0.00/0.03/0.08`。

## 2. 定价引导发现

发现实验使用一个透明的增强--模型 score table。价格信号代表平台从历史交易、元数据或相似任务中得到的 warm-start 收入方向先验；冷启动第一单不能假设该信号已经可得。`Pricing-guided Data-Bandit` 先探索高价格信号增强，再对最有潜力的增强做全模型确认。

| 方法 | 最终测试效用 | 平均发现曲线 AUC | 达到 95% oracle 增益训练数 | 前 4 次训练平均效用 |
|---|---:|---:|---:|---:|
| Data-Bandit | 0.7460 | 0.6949 | 17 | 0.5415 |
| Data-All | 0.5960 | 0.5470 | 17 | 0.5015 |
| Data-Alt | **0.8060** | 0.6678 | 12 | 0.5345 |
| AutoML | 0.5060 | 0.5049 | 17 | 0.5015 |
| Pricing-guided Data-Bandit | **0.8060** | **0.7583** | 13 | **0.7350** |

这里预算为 16，因此 `calls_to_95pct_oracle_gain = 17` 表示预算内没有达到阈值。

结论：

- Pricing-guided Data-Bandit 的平均发现曲线 AUC 最高，说明它更早找到高价值增强。
- 前 4 次训练平均效用从 Data-Bandit 的 0.5415 提升到 0.7350，提升约 35.7%。
- 最终测试效用达到 0.8060，与 Data-Alt 并列最高，但发现过程更早受益。

12 个扰动分数表上的价格信号消融如下。

| 方法 | AUC | 前 4 次训练平均效用 |
|---|---:|---:|
| Data-Bandit | 0.7191±0.0042 | 0.5418±0.0009 |
| Random-signal guided | 0.7588±0.0081 | 0.7096±0.0147 |
| Noisy-pricing guided | 0.7756±0.0052 | 0.7566±0.0071 |
| Pricing-guided Data-Bandit | **0.7764±0.0070** | **0.7630±0.0094** |

消融说明随机探索顺序已经能带来一部分收益；真实 warm-start 价格信号和噪声价格信号进一步提高 AUC 和早期效用，但增量有限。因此结论应表述为“历史/元数据价格信号在排序收益之外提供额外信息”，而不是价格信号无条件显著优越。

对应图为 `results/figures/improvement_discovery_curves.pdf` 和 `results/figures/improvement_discovery_signal_ablation.pdf`。

## 3. 可上交结论

本项目不再只停留在“提出改进想法”。四个改进已有可运行实现、单元测试和实验产物：

- IR-aware 外部选项修正 forced-choice 指标与自愿购买市场之间的语义差距；
- Robust IR 和 Robust cost-aware Markov 解决先验冷启动和分布偏移问题，其中联合鲁棒成本感知定价在最坏场景指标上最稳定，但需要接受名义收入损失；
- Cost-aware Markov 定价显式考虑非零搜索成本对停止行为的影响；
- Pricing-guided Data-Bandit 把 warm-start 价格信号反馈给发现过程，优先探索高收入潜力增强。

这些实验仍是透明缩减设置，不宣称替代原论文未公开数据上的完整大规模实验；但作为课程项目的机制改进部分，已经具备代码、命令、表格、图像和结论闭环。
