# NanoJev V4 下一阶段范式路线图 V1

**日期：2026-09-20（Asia/Shanghai）**  
**状态：设计稿；不是能力晋级、训练批准或部署批准。**

**W39 最新增量**：T11 已完成专用 protocol/preflight，数值估计器、feature-only boundary、
固定 instrument holdout 和 30 项测试完成，尚未真实拟合；旧压力窗口排除诊断保留为历史收据，
当前协议把窗口作为描述性 strata，PIT regime masks 延至 T12。本注记覆盖
下文旧快照中“T11–T14 尚未开始”的 T11 部分；不构成 V4 模型能力进展。

这份路线图把用户提出的方向——“本地部署、体积小、成本低廉、反应速度快、准确率和任务处理质量逐步提高”——转成可测量、可回滚、可审计的工程计划。它建立在 [V3 本地决策 substrate](NANOJEV_V3_ROADMAP.md) 之上，但不把 V3 的合成收据或 V2 的未关闭任务当成已通过门禁。

## 1. 先给结论

1. **旧 V2 roadmap 目前不能诚实地标记为全部完成。** T1–T4、T8b–T8f 等已有范围内交付；T5/T8/T9a/T16 已由 W32/W33 独立审核接受（其中 T9a/T16 为负面“不晋级”结论）；T7 已由 owner 决定停用 Aster；T6 的 protocol 与真实 11-feature PIT gate 已完成（Binance-only V1，仍无训练/测量授权）；T9d v2 获 W35 `ACCEPT_WITH_CONDITIONS`，仍需 adapter/heldout/G4 条件；T8g/T9 的 V2 内部隔离已通过，但 fresh heldout/OOD 与 adapter contract 阻塞；T10 已完成协议/预检但真实 paired measurement 尚未开始，T11–T14 尚未开始，T15 仍是 ongoing 维护项。完整矩阵见 [V2 completion audit](ROADMAP_V2_COMPLETION_AUDIT_V1.md)。
2. **V3 已完成的是基础设施和控制路径。** N1/N2/N3/N4 的合约、validator 和 synthetic-only controls 已通过相应测试；N3-S readiness 仍为 `n3_s_readiness_blocked`。这些收据没有证明准确率、速度、体积、成本或部署能力。
3. **V4 的产品单元不是小型聊天模型，而是本地 typed decision substrate：**

   ```text
   local input → typed probability/rank/route/abstain
               → content-free receipt → deterministic caller action
   ```

4. **所有“更小、更快、更便宜、更准确”都必须由同一 workload 的 paired receipt 证明。** 在收据出现之前，只能称为候选目标；不能用模型大小、单次 demo、star 数或测试数量替代证据。

## 2. 当前基线和证据边界

| 面向 | 当前可核对事实 | 不能外推的结论 |
|---|---|---|
| 本地模型 | `local_atomic_seed17`（Qwen3-0.6B），MPS/FP32，离线推理，`network_model_calls=0` | 不是通用工程模型，也不是生产安全决策器 |
| N1 workload | [`n1_smoke`](../results/nanojev_v3_n1_smoke_20260920.json) 与独立进程收据一致；该 workload 的 cold 约 10.5 s（含加载）、warm 约 3.2 s、峰值 RSS 约 5.27 GB | 不是 V4 延迟目标达标，也不是所有 workload 的 p95/p99；V2 B7 的模型 compute 与端到端 paper-decision 目标仍按各自 scope 单独测量 |
| 域内质量 | 迷宫域中 `>=0.9` 置信度样本的观测准确率 97.8% | 不能外推到工程、金融或新的任务包 |
| 域外行为 | 13 个工程判断题在 0.9 门限下 13/13 弃权；最高置信错误仍在安全方向 | 不能通过降低门限换取 coverage |
| 工作流网关 | T8 分批路径、可逆 restore、fail-open 已验证；真实 checkpoint 仍为 0 删除/0 实际 token 节省 | 不能宣称 provider 成本下降或 active pruning 已就绪 |
| 工程回归 | W24 后主仓库 893 OK / 2 skipped；W27 边界测试后 932 OK / 2 skipped；W28 scope-binding 覆盖后 935 OK / 2 skipped；安装版 skill 17 OK | 测试通过不等于模型质量或发布批准 |

当前 V2/N3-S 的外部门禁必须先按原记录处理。这里的 **R1** 特指 V2 金融决策包门禁，不是本文件的性能目标名；V4 不绕过 R1、数据许可、独立审核、协议变更或生产门禁。

## 3. 新范式的五个不变量

### 3.1 Local-first

- 默认只访问本地 loopback 服务和本地权重/tokenizer；推理收据必须记录零网络模型调用。
- 断网仍可启动、健康检查、推理和回滚；任何远程 provider 都不是隐式 fallback。
- 日志默认只存计数、哈希、延迟、设备、版本和事件 ID，不存原文、凭据或完整私有输入。

### 3.2 Typed decision，而非自由文本

- 只暴露版本化的 `choice`、`boolean`、`score`、`route`、`rank`、`abstain` 契约。
- 每个答案都带 confidence、概率归一化、scope assessment、`authorizes_execution=false` 和可关联 receipt。
- 工程判断、授权、删除、部署和交易决定继续由确定性门禁与人类 owner 处理。

### 3.3 Quality-first compression

- 先修数据隔离、标签和校准，再做蒸馏/量化/低比特化。
- 任一压缩候选出现 protected confident error、排列位置伪影、概率不归一、不可复现或恢复失败，立即停止晋级。
- “更小”只有在质量、校准、OOD 和安全约束不劣时才是改进。

### 3.4 Measurement-native cost

- provider 成本只接受同一请求的 unfiltered/filtered provider-reported usage；字符数、tokenizer 估算和 shadow 只能叫 estimate。
- 本地成本报告 wall time、硬件/能源 proxy、模型加载、重试、重建和工具重执行；不伪造绝对电费或碳排。
- 所有速度都分开报告 cold、warm、queue、p50/p95/p99 和吞吐，不能用吞吐除以 batch 大小冒充单决策延迟。

### 3.5 Versioned improvement loop

```text
failure receipt
  → 去标识化 taxonomy
  → hard-negative / contrastive pair
  → frozen calibration
  → candidate train/quantize
  → paired quality/footprint/latency replay
  → reviewer promotion or rollback
```

测试集和 OOD 集永不回流；每个候选保留输入、代码、权重、tokenizer、协议和报告哈希。

## 4. 指标阶梯（候选目标，不是当前成绩）

下列数值是 V4 的**预注册候选目标**，必须在 N3-S/N4-S 真实测量前由 owner/reviewer 固定 workload、硬件和容差；它们不是当前 NanoJev 已达到的指标。

| 指标 | V4-M1 研究目标（建议） | V4-M2 发布目标（建议） | 必须同时满足 |
|---|---:|---:|---|
| package bytes | ≤ 1.5 GB | ≤ 800 MB | 权重、tokenizer、运行时、SBOM 全计入；hash 可复现 |
| peak RSS / unified memory | ≤ 2.0 GB | ≤ 1.25 GB | 同一设备、同一 workload，含服务进程 |
| warm single-decision p95 | ≤ 250 ms | ≤ 100 ms | p99、queue、timeout、取消和 fail-open 另列 |
| cold start p95 | ≤ 5 s | ≤ 2 s | 不把常驻进程结果当 cold |
| quality delta vs FP32 | accuracy 非劣 margin 暂定 2 pp；NLL/Brier/ECE 的 **95% paired CI** 不劣 | 同左且 task-family worst case 不劣 | protected error = 0，OOD/abstain 语义不退化；最终 margin 由 protocol 预注册 |
| local cost proxy | 记录 wall-time、设备和能源 proxy | 在相同硬件、相同 workload 下按预注册的 **95% paired CI/margin** 低于 baseline | 不把未测电价或云价写成事实 |
| reliability | deterministic、zero-network、可回滚 | 新机断网安装/损坏拒绝/kill switch 全通过 | active 默认关闭，scope guard 不可关闭 |

若 V4-M1/V4-M2 被硬件或 workload 证明不现实，应修改协议并保留失败收据，不得调低保护门禁来“达标”。

V4 的端到端目标不替代 V2 B7 的两个独立 scope：准备好 tensor 的 model-compute 目标（p50 < 5 ms、p99 < 20 ms）和包含特征/校验的 paper-decision p99 < 50 ms。每个 scope 都要在同一硬件、同一 workload 下单独出具 p50/p95/p99 收据。

## 5. 分阶段路线

| 阶段 | 目标 | 主要交付 | 退出门禁 | 当前关系 |
|---|---|---|---|---|
| V4-S0 证据与授权收口 | 让旧 roadmap 的事实、权限和 reviewer 状态可追溯 | V2 audit、owner decision log、许可证 verdict、独立 review | T6/T7/T16 由相应权威记录；T8g/T9/T9d 有新审核结论 | W25 已完成文档账本收口；外部门禁仍开放，不能由主模型代签 |
| V4-S1 任务包与数据隔离 | 把“域外弃权”变成少数明确任务域的可测 coverage | N2 domain pack、source-group/lineage、labels、hard negatives、protected/OOD | N2 clean receipt + 四项 review；V1 泄漏集不得训练 | N3-S 当前被此阻塞 |
| V4-S2 架构/readout ladder | 比较 trained head、schema readout、encoder/router 和共享 prefill | N3-S paired receipts、排列/标签/OOD/protected/baseline | protected confident error=0；语义排列稳定；质量/校准 CI 达标 | synthetic 已有，真实 arm 未运行 |
| V4-S3 体积阶梯 | 在质量不退化前提下降低 footprint | FP32、FP16、INT8、INT4、distilled student 的 artifact/内存/延迟收据 | 同 workload paired Pareto；artifact/hash 可复现 | N4 contract/synthetic 已有，真实量化未开始 |
| V4-S4 本地 serving | 把“快”变成可复现的 p95/p99 | persistent load、CPU/MPS、bounded batch、cache、compile、并发/取消 benchmark | V4-M1/M2 延迟与内存候选目标；异常仍 fail-open | V3 N5，需 N3/N4 证据 |
| V4-S5 工作流净价值 | 证明小模型作为前置层有净收益 | ≥3 主模型族 paired unfiltered/filtered receipts、provider usage、下游成功率 | protected 零删除、成功率不降超 margin、真实成本可核对 | V3 N6，active 默认关闭 |
| V4-S6 离线发布 | 形成可安装、可升级、可回滚的本地包 | SBOM/license、hash manifest、health/doctor、offline install、migration/rollback | 新机断网安装、损坏/版本不匹配拒绝、kill switch | V3 N7，尚无发布批准 |
| V4-S7 持续提升 | 用失败驱动质量和任务覆盖逐步增加 | taxonomy、数据版本、候选 registry、paired replay、promotion/rollback log | 每候选可回滚；test/OOD 不回流；reviewer 决策有签名 | V3 N8，持续运行 |

阶段顺序是硬约束：S1 未通过不运行真实 S2；S2 没有质量收据不运行 S3；S3 没有 Pareto 收据不宣称 S4；S4/S5/S6 只能在 shadow 或明确人工批准范围内推进。

### V4-S0 当前执行板（W27 后）

| 子项 | 状态 | 可交付物 | 不能替代的门禁 |
|---|---|---|---|
| S0.1 文档/证据账本 | **完成** | W25 已统一计数、状态词、spread scope、撤回表述和 receipt 边界 | 不等于 owner/legal/reviewer 签字 |
| S0.2 owner 决策包 | **部分完成** | T6 D1–D3/C1–C9、protocol freeze 与真实 PIT receipt 已记录；T7 已决定停用 Aster；T9d amendment 已批准并登记 v2 | T6 仅为 Binance-only V1 且收据本身不授权训练/测量；T9d 已有条件接受意见，adapter/heldout/G4 条件仍未完成 |
| S0.3 独立审核 | **部分完成** | T5/T8/T9a/T16 已有 W32/W33 独立记录；T9d v2 为 W35 ACCEPT_WITH_CONDITIONS；T8g/T9、T10、N2/N3-S 仍待相应 review | synthetic/preflight 不能冒充独立 review |
| S0.4 V4 指标预注册 | **contract/preflight 完成；官方只读复核通过；owner/reviewer 冻结待定** | `NANOJEV_V4_S0_METRICS_CONTRACT_V1.md`、5 候选、3 latency scopes、M1/M2 target profiles、paired quality/cost/reliability fields、canonical protocol hash、Worker `1789857734-79419c9726da` | 不能用一次 demo、模型参数量或测试数达标；真实 measurement 仍未授权 |
| S1 N2 domain pack | **部分完成 / heldout 阻塞** | V2 内部隔离专用审计通过（0 cross-split、0 provenance hit），但 cross-seed 123 common inputs；fresh provenance/source-group/holdout/protected/OOD pack 尚缺 | V1 不得合并、训练或回流；reseed 不能替代独立 heldout |

S0.1 文档账本与 S0.4 指标 contract/preflight（含官方只读复核）是本轮已交付的 V4 子项；S0.2–S1 未关闭前，V4 仍是条件式研究/产品化设计，
不宣称“下一范式”已经达成。安全的并行工作仅限于只读审计、validator 修正、synthetic
controls、决策包和本地 skill lifecycle 记录。

## 6. 现在可做与不可做

### 可以继续做

- 修正文档、收据、validator、回归测试和 benchmark harness；
- 生成不含真实语料/权重推理的 synthetic controls；
- 在本地 NanoJev 上运行 development/testing/optimization/deployment lifecycle，记录 event ID 和 feedback；其中 deployment lifecycle 只记录 readiness advisory，不执行真实部署或改变授权状态；
- 在独立临时目录做只读审计，使用官方 DeepSeek V4.1 Flash Worker 做 bounded review；
- 为下一轮 owner/reviewer 准备决策包和证据索引。

### 在门禁关闭前禁止做

- 把冻结 V1 语料合并、改 split/label、训练或生成真实 N2 clean receipt；
- 运行真实 N3 arm、量化、蒸馏、active pruning 或部署；
- 把 synthetic/preflight/test-count 当成模型准确率、速度、体积、成本或任务质量证据；
- 启用生产 active context removal、真实交易或任何 live capital；
- 为降低弃权率而调低 0.9 门限，或以更强基座的高置信错误替代现有模型。

## 7. 90 天执行建议（条件式）

| 时间 | 重点 | 必须留下的证据 |
|---|---|---|
| 0–2 周 | S0：完成 owner/legal/reviewer 决策包；修正 N2 receipt/文档漂移；冻结 V4-M1/M2 候选指标 | 决策记录、许可证 verdict、portable receipt、基线复现 |
| 2–5 周 | S1：重新构建独立任务包，只做 provenance/label/holdout 审计 | N2 report/hash、source-group 图、人工审核记录 |
| 5–8 周（S1 通过后） | S2：运行真实 paired readout；先做 permutation/OOD/protected，再看平均分 | N3-S raw/summary receipt、质量/校准/延迟/内存 |
| 8–10 周（S2 通过后） | S3/S4：在通过的 arm 上做 footprint 与 serving 对照 | 每候选 artifact/hash、资源 p50/p95/p99、回滚测试 |
| 10–12 周（S3/S4 通过后） | S5/S6：三主模型族 shadow paired replay，准备离线包 | provider usage、下游质量、SBOM、断网安装/回滚 |
| 持续 | S7：失败闭环和 T15 生态账本 | candidate registry、hard-negative 版本、promotion/rollback 记录 |

时间窗不是自动授权；任何阶段若出现 protected regression、数据许可冲突、证据 hash 漂移或 reviewer 拒绝，都停在当前阶段并保留失败收据。

## 8. 成功定义

只有同时满足以下条件，才可称为“下一阶段 NanoJev 范式”而不是研究原型：

1. 至少一个明确任务包上，本地候选的质量/校准不劣于 FP32 baseline，且域外可靠弃权；
2. 同一 workload 下 artifact、内存、cold/warm 延迟和本地成本 proxy 有可复现改善；阈值、置信水平（建议 95% CI）和非劣/改善 margin 必须在 protocol 中预注册，baseline 明确为同一 N3/N4 paired receipt 的 `fp32_baseline`；
3. 本地 skill、主模型 shadow 网关和离线发布包使用同一 typed contract、权重/tokenizer hash 和回滚机制；
4. 零网络推理、日志无原文/凭据、fail-open、protected 零删除和 kill switch 均有 receipt；
5. 每次晋级都有 owner/独立 reviewer 记录，任何失败都能回滚；
6. 没有把测试通过、合成控制、外部 star、未经配对的 token 估算或一次 demo 写成能力证明。

## 9. 执行身份与记录规则

- 外部执行 Worker 只能走官方 DeepSeek API：`https://api.deepseek.com/v1`、`deepseek-flash`、`DeepSeek-V4.1-Flash`；禁止 CommandCode、SenseNova、过期别名或其他第三方回退。
- NanoJev 本地 skill 只做 bounded advisory，始终保留 `authorizes_execution=false`；低置信度必须由主模型和确定性门禁接管。
- 每个 development、testing、optimization、deployment 阶段先运行本地 lifecycle；保存 event ID、设备/精度、network calls 和反馈标签。
- 本文与 V3 roadmap 是条件式设计；V2 completion audit、N3-S readiness receipt 和 owner/reviewer 记录才是状态权威。

## 10. 权威入口

### W24 进度注记（2026-09-20）

T10/A5 已补齐 protocol/preflight/synthetic-control substrate：五臂 paired contract、A4 hash/family
binding、provider/tokenizer-only cost contract、protected-family fail rule 和 content-free receipts。
其状态仍是 `protocol_valid_not_authorized`，不能作为 V4 的质量、速度、体积、成本或部署证据；真实
paired measurement 依赖独立 review、下游 oracle 和 provider/tokenizer usage。主仓库回归更新为
893 OK / 2 skipped，安装版 skill 17 OK。该增量只提升 V4-S0/S1 的证据可执行性，不改变 V2 外部门禁、
N3-S blocked 状态或 active-pruning 默认关闭。

- 旧 roadmap 完成矩阵：[ROADMAP_V2_COMPLETION_AUDIT_V1.md](ROADMAP_V2_COMPLETION_AUDIT_V1.md)
- 当前交接：`CURRENT_PROGRESS_AND_HANDOFF.md`（内部文档，本地留存，不发布）
- V3 契约与阶段设计：[NANOJEV_V3_ROADMAP.md](NANOJEV_V3_ROADMAP.md)
- N3-S readiness：[NANOJEV_V3_N3_S_READINESS_CONTRACT_V1.md](NANOJEV_V3_N3_S_READINESS_CONTRACT_V1.md)
- 逐包审查日志：`EXECUTION_REVIEW_LOG.md`（内部文档，本地留存，不发布）

### W25 进度注记（2026-09-20）

官方 DeepSeek V4.1 Flash 分析 Worker `1789855446-75411a636af1` 对 V2/V3/V4 文档和当前
receipts 做了只读一致性复核（`success`、`files_changed=[]`）；修正后最终复核
`1789856306-d0f348dea40a` 再次确认一致性（`success`、`files_changed=[]`）。本轮修正了 handoff 的 `892/2`
陈旧回归计数、重复 B0 状态行、已撤回的 confidence 表述、T5/A5/T6 的陈旧状态、B0 与 T5
不同 scope 的 spread 口径，以及 N2 clean-receipt 尚不存在这一事实。当时口径为：N1 **676/2**
历史增量、N2 **715/2** 历史增量、W24 后主回归 **893/2**；B0 三场所诊断 spread **3,964.56**
（含 Aster appendix），T5 主表 full-window **3,517.11**、2026 YTD **6,454.37（>5%）**，
故 `headline_allowed=false`；当时 T6=`OWNER_PENDING`、T10=`READY_FOR_REVIEW`（仅 protocol/preflight）、
N3-S 仍因缺 N2 receipt/四项独立 review 而 blocked。此处只提升证据可读性，不产生质量、速度、
体积、成本或部署证据。

V4-S0 现进入“文档账本已收口、外部门禁仍开放”：下一安全切片是 owner/legal/independent-review
决策包、V4-M1/M2 指标预注册和只读 receipt/validator 复核；S1 之前不运行真实 N3 arm、量化、
蒸馏、active pruning 或部署。Worker doctor 仍为 healthy，生产 API 固定为
`https://api.deepseek.com/v1` + `deepseek-flash` → `DeepSeek-V4.1-Flash`，仅 `deepseek-official`
且无第三方回退；本轮本地 NanoJev development/testing events 分别为
`2f770b42-01a1-4161-966d-f365764d0e53` / `184eba81-0de6-45f8-8c4a-5f2d7e3a7fa5`，以及最终复核前的
development `380514c7-6e19-4f0c-b565-c4cb70815090`，均 MPS/FP32、零网络、低于 0.9 而 abstained；
development feedback=`fallback`、testing feedback=`fallback`，最终 development feedback=`fallback`。本地 lifecycle 仍是每个阶段的
advisory 前置步骤，不能替代 reviewer 或 owner；本轮主仓库 **893/2**、skill **17/17**，不构成
模型质量、速度、体积、成本或部署证据。

### W26 进度注记：V4-S0.4 metrics contract/preflight（2026-09-20）

新增 [`NANOJEV_V4_S0_METRICS_CONTRACT_V1.md`](NANOJEV_V4_S0_METRICS_CONTRACT_V1.md)、
`research/nanojev_v4_s0_metrics_contract_v1.json`、只读 validator 和 **29 项**专项测试。
contract 固定 5 个候选、`model_compute`/`paper_decision_e2e`/`local_serving` 三个 latency scope、
M1/M2 package/RSS/warm/cold target、paired quality/cost/reliability 门禁；local serving 强制
cold p99，目标字段绑定到 `local_serving`，preflight 带 canonical protocol SHA-256。

run2 收据 `results/nanojev_v4_s0_metrics_preflight_20260920_run2.json` 的状态为
`metrics_contract_valid_not_authorized`，`protocol_sha256=65bbf78a6459c0fe87354244256c03277abb9d865be29c3a9c42b9a012dbe20d`，
`measurement_authorized=false`、`training_authorized=false`、`deployment_authorized=false`、
`model_loaded=false`、`network_model_calls=0`。这只是结构/目标预注册，不是体积、速度、质量、成本
或部署证据；早期未带 hash 的 run1 仅保留历史。

官方 DeepSeek V4.1 analysis Worker `1789857061-b8bfe57eb1bf` 首轮审查发现并推动了上述缺口修复；
修正后的隔离 staging 再由官方 DeepSeek V4.1 analysis Worker `1789857734-79419c9726da` 只读复核，
终态 `success`、`files_changed=[]`，确认六项缺口全部闭合：local_serving cold p99 强制、M1/M2 latency
scope binding、run2 protocol hash、exclusive-create 重跑安全、freeze/approval 语义和
`pre_approval_only` 自批准阻断。Worker 静态核对 contract/validator/29 tests/run2 receipt 一致性；
它不能替代主模型执行测试或重算哈希，主模型已独立确认 canonical protocol SHA-256 仍为
`65bbf78a6459c0fe87354244256c03277abb9d865be29c3a9c42b9a012dbe20d`。残余风险（当前冻结批准键之外的未来
授权键扫描、cold/M2 scope 负向测试覆盖、输出 I/O 中断的部分文件边界）已登记，不改变 fail-closed。
两个 Worker job 均为 `https://api.deepseek.com/v1`、`deepseek-flash` → `DeepSeek-V4.1-Flash`、
`official_only=true`、`fallback_attempted=false`。V4-S0.4 因此为“contract/preflight 完成、官方只读复核通过、
owner/reviewer 冻结待定”，真实 N3/N4 measurement、量化、训练、serving、active pruning 和部署仍保持关闭。
本轮本地 NanoJev testing event `502663ab-8766-4924-bcb7-15a5847b90d7` 与文档收口后的 development event
`4ac5f220-c609-4406-bc61-f887843d5016` 均为 MPS/FP32、零网络、低于阈值而 `abstained`，feedback 分别为
`1a7b0818-d389-4360-b734-6127dad0c5bd`、`a22bbb3b-54b5-424e-8d1f-c7877171bcd4`（均 `fallback`）；
它们只提供 advisory receipt，不替代 owner/reviewer 或授权。
修正缺口后的 development event `cb6b1092-3c7f-49c3-9fe9-6d0ee2b7dd44` 的 feedback
`8ca3d2e7-5ab2-4466-8a0d-3245e5c2db60`=`fallback` 也已登记。
最终回归前 testing event `564e74da-782f-4cc6-8ffe-daa893a0a737` 亦为本地 MPS/FP32、零网络、低于阈值而
`abstained`，feedback `1b7df5e8-fdd4-4d2d-8b2e-44c263f202cc`=`fallback`。

### W27 进度注记：V1 corpus isolation plan（2026-09-20）

V4-S1 的安全前置又补齐一条只读隔离计划路径：`plan_engineering_corpus_v2_isolation_v1.py` 按
`pair_id` 血缘与 canonical model-visible input digest 生成 deterministic components，显式报告
`pair_id`/`source_group_id` split crossing、evaluation-derived provenance 与 cross-seed semantic-holdout
风险；输出 portable、content-free、exclusive-create receipt。它不写新 corpus、不改 split/label、不合并、
不训练。

run3 收据 `results/engineering_corpus_v2_isolation_plan_20260920_run3.json` 的 SHA-256 为
`904c48ab99ab333fcaa3e8dc6490929b51cd7909aed5cee5e5546258e148f884`，状态仍为
`blocked_isolation_plan_only`：246 records、28 components、6 split-conflicted components、27 canonical
cross-split groups、18 evaluation-derived records、cross-seed common inputs 129、affected records 174/174；
`training_authorized=false`、`merged_rows_written=0`。W27 专项 **10/10**，主仓库 **932/2**，skill **17/17**。
这证明隔离控制路径的可复核性，不证明数据清洁、准确率、体积、速度、成本或任务处理质量。

官方 DeepSeek V4.1 Flash analysis Worker `1789859941-6b1300836f21` 只读复核成功、无文件修改；身份固定为
`https://api.deepseek.com/v1`、`deepseek-flash` → `DeepSeek-V4.1-Flash`、`official_only=true`、
`fallback_attempted=false`。Worker 未在缺依赖的隔离 staging 执行测试，主仓库回归是实际验证来源。残余风险
（source-group 为 plan-level block、run3 未触发两类专用 violation、determinism 未有专门 byte-identical
单测）不解除 S1/T8g/T9/N2/N3-S 门禁。

V4 仍是条件式路线图：S1 未通过前不运行真实 S2；S2 无质量收据不运行 S3；S3 无 paired Pareto 收据不宣称
S4；真实 measurement、训练、量化、serving、active pruning 和部署保持禁止。W27 是只读隔离计划进展；
V4-S0.4 仍是最近完成的 contract/preflight 子项，owner/reviewer 冻结仍待外部记录。本轮 optimization lifecycle event
`c8eb834a-9470-426a-b024-38e2c63489ea` 在本地 MPS/FP32、零网络下低于 0.9 而 `abstained`，feedback
`3a92263d-207c-4352-a345-8a1b9093aa87`=`fallback`；它不构成优化选择或授权。
文档复核 job `1789860583-1bcca79905ea` 随后确认 W27/V4 文档口径一致；W27 receipt 未擅自扩展
`measurement_authorized`/`deployment_authorized` 字段，真实测量与部署仍由 V4-S0.4 及外部门禁控制。

### W28 进度注记：V4-S0.4 scope-binding negative coverage（2026-09-20）

新增三条只读 validator regression，覆盖 `v4_m1_research.cold_start_scope_id`、
`v4_m2_release.warm_single_decision_scope_id` 和 `v4_m2_release.cold_start_scope_id` 偏离
`local_serving` 的 fail-closed 行为。V4-S0.4 专项 **32/32**，主仓库 **935/2**，安装版 skill **17/17**；
clean contract 仍 `metrics_contract_valid_not_authorized`，没有模型加载、网络调用、真实 measurement、
量化、训练、serving 或部署。

官方 DeepSeek V4.1 Flash analysis Worker `1789861172-bc6de5143aaa` 只读复核成功、无文件修改；它确认
双 profile 的 warm/cold binding、cold p99 和授权扫描保持 fail-closed。官方身份为
`https://api.deepseek.com/v1`、`deepseek-flash` → `DeepSeek-V4.1-Flash`、`official_only=true`、
`fallback_attempted=false`；staging 未执行测试，主仓库结果是实际验证来源。

W28 只提升 S0.4 契约覆盖，不产生任何性能、质量、成本、体积或部署结果；owner/reviewer 冻结与 S1 数据隔离
门禁仍未关闭。

### W29 进度注记：plan receipt machine-readable boundary flags（2026-09-20）

W27 plan-only receipt 新增四个显式 false 字段：`measurement_authorized`、`measurement_performed`、
`deployment_authorized`、`deployment_performed`；旧 run3 原样保留，新 run4 exclusive-create，SHA-256 为
`8800a8192a51bb9accd7a825f4ae0a5040e2fefd64fb7efeba7df9ebf401ee6c`。run3/run4 的隔离计数、block reasons、
source hashes 和 components 保持一致；W29 focused 10/10，主仓库 935/2，skill 17/17。

官方 DeepSeek V4.1 Flash retry review `1789861830-a96a57205f73` 成功、无文件修改；首轮
`1789861770-86bd4828e5c2` 因 structured result 格式失败，同样无文件修改且无第三方回退。该增量只改善
receipt governance，不产生真实 measurement、训练、量化、serving 或部署证据。

### W30 文档一致性收口（2026-09-20）

官方 DeepSeek V4.1 Flash 文档复核 job `1789862099-a3fdc102aa0e` 返回 `success`、`files_changed=[]`，
确认 W1–W29 计数、run3/run4 hash 与边界语义一致，并指出三处陈旧措辞。W27 的 W29 前临时 run4 已明确
标为历史，V2 审计的当前回归已更新为 **935/2**，V2 的“now 787”已改为历史节点；V4 仍保持条件式设计，
不产生真实测量、训练、量化、serving、active pruning 或部署证据。官方链路仍固定为
`https://api.deepseek.com/v1`、`deepseek-flash` → `DeepSeek-V4.1-Flash`、`official_only=true`，无第三方回退。

修正后的最终只读复核 job `1789863133-df273189f1e7` 返回 `success`、`files_changed=[]`，独立重算 run3/run4
hash，确认四个 false boundary fields、共同字段和 W27/W28/W29 计数一致；没有改变 V4 的条件式状态。

文档收口使用本地 NanoJev development event `da37c160-883f-4619-a079-26d4a67d2967`（MPS/FP32、零网络、
abstain）；最近决策 event `cee7f210-776b-40bf-a8a3-5e59d92d983f` 的 `fallback` feedback receipt 为
`4c336585-b617-49c5-9929-a89b6bf35420`；收口 testing event `2e6f2c4d-eedb-4487-84d2-4a84395ce092`
的 feedback receipt 为 `37dd6893-bd8e-4c10-8536-7a4e532ed0b5`=`fallback`。NanoJev 只作 advisory，不替代
owner/reviewer 或任何阶段门禁。为设计 contract-first 优化顺序又记录 optimization event
`748a343b-54f2-4407-b256-8ae394fab089`，本地 MPS/FP32、零网络、`abstained`，feedback receipt
`5e57511f-c5ca-4d2f-87d0-62305de98481`=`fallback`；因此没有执行真实压缩、训练或部署。部署阶段
readiness event `888bb9a6-0c92-43bd-9c8a-50845a50d641` 同样在本地 MPS/FP32、零网络下 `abstained`，
feedback receipt `cc589821-ff0d-49ba-bdae-589c50fd69eb`=`fallback`；未启动服务或发布 artifact。

### W31 T15 ecosystem ledger refresh（2026-09-20）

只读读取 17 个公开 GitHub REST metadata，统一 snapshot 为 `2026-09-20T00:18:05Z` UTC，并把
`achimala/jevinci` 的 API redirect 记录为 canonical `achimala/jev-paint`。本轮没有读取或上传私有
项目数据，没有安装代码/权重，没有产生模型质量、速度、体积、成本或部署证据；star 与 README 自述
仍仅是公开 metadata/author claim。V4 的 S0–S7 条件式顺序和所有外部门禁不变。
