# Context relevance V1：三种子实验结论

状态：**三种子训练与汇总收据已产出；本次完成结果文档化，不晋级、不部署。独立实现/证据审核仍待进行。**

本页记录 2026-09-19 实验产物。事实依据为 [冻结协议](../research/context_relevance_v1_protocol.json)、[机器可读报告](../results/context_relevance_oracle_v1_report.json) 和本地 dataset/run 文件。下一步执行入口见 [进度与 AI 交接计划](CURRENT_PROGRESS_AND_HANDOFF.md)。文档完成不代表工作区变更已提交或 Phase 0 已通过。

## 1. 实验问题与边界

检验从 `local_atomic_seed17` 出发，能否学会合成的“最新记录依赖/无关片段”判断，并同时测量固定阈值误删和原迷宫能力回归。这是监督依赖学习，不是通用语义压缩、金融模型或新的 RLCD 实验。

- 数据：`data/context_relevance_oracle_v1_seed20260919`；train/dev/calibration/test/OOD 分别为 1200/240/240/360/360 条，150/30/30/45/45 个 source group。
- 每组含 8 种变体：required_field、two_fields、latest_correction、limit_check、wrong_entity、wrong_field、superseded、unrelated；同组不得跨 split。
- OOD 留出 robotics 与中文领域，但仍共享 oracle 和渲染结构，**不是未见推理任务族**。
- 三个训练种子 17/18/19 均使用同一冻结初始化；不是三份独立训练的 baseline。不要跨种子重复累加相同组来扩大样本量。
- MPS FP32，CE / gold_distribution，8 步 head warmup + 60 步训练；按 dev target CE 选择，含初始化 step 0，平局保留更早 checkpoint。三种子均选择 step 68。
- 不拟合 temperature，不调阈值；`P(irrelevant) >= 0.99` 是固定诊断规则，不是安全认证。
- 无效前身数据及 run 的路径、split 哈希问题和矛盾标签已记录在冻结协议；保留审计，不恢复训练，不覆盖为有效产物。

## 2. 相关性质量与固定阈值误删

准确率是全部二分类问题的准确率，不是“无关类”的 precision。提议/误删数是逐题阈值诊断；**不等于完整 shadow 依赖闭包与请求级回退之后的删除计划，更不是实际删除或 token 节省**。

| 模型 | split | Accuracy | NLL | ECE | 提议删除 / 误删 |
|---|---|---:|---:|---:|---:|
| 初始化 | calibration | 45.42% | 0.74381 | 0.12704 | — |
| 初始化 | test | 46.11% | 0.72847 | 0.11038 | — |
| 初始化 | OOD | 45.56% | 0.73290 | 0.11054 | — |
| seed 17 | calibration | 94.58% | 0.16472 | 0.05214 | 24 / 0 |
| seed 17 | test | 93.61% | 0.18223 | 0.04294 | 42 / 0 |
| seed 17 | OOD | 91.94% | 0.19853 | 0.05935 | 12 / 0 |
| seed 18 | calibration | 97.08% | 0.22016 | 0.02848 | 124 / 7 |
| seed 18 | test | 96.39% | 0.25978 | 0.03533 | 188 / 12 |
| seed 18 | OOD | 95.00% | 0.34860 | 0.05013 | 183 / 13 |
| seed 19 | calibration | 96.67% | 0.10141 | 0.01934 | 64 / 0 |
| seed 19 | test | 95.83% | 0.09979 | 0.02144 | 102 / 0 |
| seed 19 | OOD | 94.44% | 0.15016 | 0.02993 | 81 / 0 |

报告另含 Brier、逐 family/kind/wire 分解、误删 sample ID、source-group bootstrap 及 paired delta 区间。表中“—”表示本表未列初始化阈值诊断，不表示零提议。

观察到零误删不能证明风险为零。报告的零错误单侧组级上界对 30 组约为 9.50%，45 组约为 6.44%，且依赖 iid source-group 假设，只描述合成人群。全局 ECE 较低也不能证明 0.99 尾部的删除安全。

## 3. 原迷宫回归

初始化 test 77.8409%（176 题），OOD 76.5625%（64 题）。Δ Accuracy 单位为百分点；0.5pp 预算仅检查准确率点估计，不代替完整概率质量与不确定性审核。

| 种子 | Test accuracy / Δ | OOD accuracy / Δ | 准确率预算 | Test / OOD Δ NLL |
|---|---:|---:|---|---:|
| 17 | 76.7045% / −1.1364pp | 71.8750% / −4.6875pp | 两项均超预算 | +0.07844 / +0.02966 |
| 18 | 77.8409% / 0pp | 76.5625% / 0pp | 点估计通过 | +0.20583 / +0.12776 |
| 19 | 80.1136% / +2.2727pp | 78.1250% / +1.5625pp | 点估计通过 | +0.07469 / +0.08098 |

**三种子的迷宫 NLL 与 Brier 点估计均恶化**。因此不能把 seed 18/19 的准确率预算通过解释为“无回归”。显著性与区间需逐项查报告，不能由点估计推断。

## 4. 决策：不晋级

课程有明显可学习性，但不足以部署：seed 17 有迷宫准确率回归；seed 18 有高置信度误删；seed 19 的有限样本零误删不构成安全证明，且概率质量仍有回归。**不得看完 test/OOD 后挑 seed 19 当生产赢家**。

生产参考 checkpoint 保持 `checkpoints/local_atomic_seed17/variants/local_atomic_seed17`；实际删除及主模型 token 节省均为零。没有三种主模型的配对任务质量/净 token 成本证据。Phase 0 的人工挑战与三份匹配 baseline/candidate 训练种子要求仍开放。

下一轮若改变训练、数据或校准策略，必须单独预注册，使用新的训练/开发设计及适当的新确认性留出集；当前 test/OOD 只能保留为历史诊断，不能反复据其调参并声称是未见验证。当前协议不回写修改。

## 5. 复核方法与限制

本次文档更新时重新执行：

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
.venv/bin/python -m unittest discover -s integrations/codex-skill/nanojev-local-decider/scripts -p 'test_*.py'
```

结果：主套件运行 165 项，OK（2 项跳过，即 163 项执行通过）；skill 套件 3 项通过。这不是新 checkpoint 推理复现或独立训练审计。

接手者可在原始数据、初始化、三个 run 和 benchmark 收据均可用时重建报告，使用全新输出路径，例如：

```bash
.venv/bin/python scripts/report_context_relevance_v1.py \
  --output results/context_relevance_oracle_v1_report_recheck_001.json
```

若路径已存在请换新编号，不覆盖冻结报告。该命令读取既有预测，不重新训练、不重新运行模型；它检查数据/配置/收据和部分哈希，不证明全部训练过程无泄漏。原始路径包含本机绝对路径；迁移机器需记录显式路径映射与内容哈希核对，不篡改原收据来伪装原位复现。报告已注明源码哈希是在报告时捕获，不是训练时的追溯签名。

待审核：数据 oracle 与划分正确性、训练选择流程、报告 provenance 的覆盖边界、运行时修改及其回归测试。所有检查失败应如实登记，不修饰指标或替换不利种子。
