# 项目进度

更新日期：2026-07-16

## 已完成

- [x] 阅读并整理论文模型、关键定理和证明思路。
- [x] 买家最优停止 DP（Proposition 4.1 / Algorithm 3）。
- [x] 经验定价目标的小规模精确求解与 independent pricing 基线。
- [x] RQ3 风格的先验学习实验。
- [x] 发现并验证“缺少不购买外部选项”的机制问题。
- [x] IR-aware 与 distributionally robust IR 两种改进。
- [x] 自动化测试、实验图表和阶段性 PDF 报告。

## 后续里程碑

- [ ] 接入开源表格数据，复现 augmentation--model discovery 的 Exp3 部分。
- [ ] 增加 Gurobi/SCIP 可选接口，在 $|Q|=20,|\Theta|=20,m=100$ 设置下求 MILP。
- [ ] 进行 20--50 个随机种子的误差条/显著性检验，而非只报告单种子结果。
- [ ] 加入转移矩阵估计误差实验，直接验证 CEE 对 $\|P-P^*\|$ 的敏感性。
- [ ] 将阶段报告扩写为最终报告，并补充中期展示幻灯片。

