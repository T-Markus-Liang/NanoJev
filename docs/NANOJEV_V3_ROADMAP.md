# NanoJev V3 roadmap — local decision substrate

**状态：N1 benchmark、N2 contract/preflight、N3 协议/预检、synthetic-only control runner、N3-S readiness preflight、N4 footprint contract/preflight 与 synthetic-only footprint controls 已实现/验证，均待独立审核；真实 N2 pack 尚未通过，N3 真实 arm 与 N4 真实量化未运行，2026-09-20。**  
**目标：** 把 NanoJev 发展成一种可审计的本地决策基础设施：体积小、部署在本机、
不依赖远程推理、成本透明、反应快；在明确任务域内逐步提高准确率和任务处理质量，
在域外主动弃权，而不是把小模型包装成通用聊天模型。

这不是“把当前 checkpoint 直接升级成全能模型”的承诺。面向产品化的后续范式目标、指标阶梯和 90 天条件式执行设计另见
[V4 next-paradigm roadmap](NANOJEV_V4_NEXT_PARADIGM_ROADMAP_V1.md)。V2 的完整审计见
[ROADMAP_V2_COMPLETION_AUDIT_V1.md](ROADMAP_V2_COMPLETION_AUDIT_V1.md)。T6/T7/T8g/T9/T9d/T16
等旧门禁未关闭前，V3 只允许做契约、测量、合成/已授权数据和本地 shadow 工作，不得
跳过 owner、法律或独立审核。

## 1. 新范式定义

NanoJev V3 的产品单元不是自由文本回复，而是一个有边界的 typed decision：

```text
local input -> typed probabilities / rank / route / abstain
            -> content-free receipt -> deterministic caller action
```

核心特征：

1. **Local-first：** 默认 loopback、本地权重、本地 tokenizer、本地日志；网络模型调用数
   必须为 0，除非调用方明确选择且单独记录的外部模式。
2. **Small-by-contract：** 以权重体积、参数量、峰值内存和加载时间一起定义“小”，
   不用“能在 Mac 上启动”冒充可部署。
3. **Fast-by-measurement：** 分开 cold start、warm single decision、batch throughput、
   p95/p99 和 gateway overhead；不能把吞吐除以样本数当成单决策延迟。
4. **Quality-by-domain：** 每个任务包有冻结数据、source-group 隔离、校准/选择性风险、
   OOD/abstain 和 protected error；不以一个聚合准确率覆盖域外崩塌。
5. **Cost-by-receipt：** 报告每决策本地资源、provider token（若有）和重试/重建成本；
   字符数不等于 token，shadow estimate 不等于实际节省。
6. **Fail-open and reversible：** 不确定、解析失败、超预算、依赖不闭合、版本不匹配时
   保留原请求；任何 active reduction 都必须能够按 receipt 逐字节还原。
7. **Human/owner authority：** skill 是 advisory measurement，不是授权系统、交易系统、
   许可系统或生产发布按钮。

## 2. 当前基线（只列已测事实）

| 维度 | 当前证据 | V3 解释 |
|---|---|---|
| 模型 | Qwen3-0.6B `local_atomic_seed17`，任务拟合头 | 不是通用工程模型；schema readout 负面诊断不能直接替代 |
| 本地执行 | MPS/FP32、offline、`network_model_calls=0` | 需同时建立 CPU、MPS 和低精度基线 |
| 域内质量 | 迷宫 cohort 的 `>=0.9` band 观测准确率 97.8%，但 0.9 下仍有 62.5% abstain | 只作为冻结 baseline，不是泛化承诺 |
| 域外质量 | 工程调查显式 0.9 gate 下 13/13 abstain，最高置信度 0.736 | 这是安全行为和能力缺口的同时证据 |
| 工作流 | shadow、batch scoring、restore、fail-open 已实现 | active pruning 仍未授权；真实删除/节省为 0 |
| 权重体积 | 当前 checkpoint 目录约 2.2 GiB，最大 `best.safetensors` 约 2.39 GB（本机文件统计） | 这是 V3 压缩/量化 baseline，不代表未来目标已达成 |
| 回归 | 仓库 893 OK/2 skip；skill 17 OK | 测试通过不等于质量或生产 readiness；其中 25 项为 N1 harness、40 项为 W21 后 N2 validator、22 项为 N3 protocol validator、12 项为 N3 synthetic runner、53 项为 W21 修正后的 N3-S readiness validator、24 项为 N4 footprint validator、14 项为 N4 synthetic footprint runner、27 项为 W24 T10/A5 protocol/preflight |

## 3. 验收分层和建议目标

下面是**待预注册和实测的目标**，不是当前结果。任何目标调整都要新版本协议和审计记录。

| 层级 | 目标 | 必须同时满足 |
|---|---|---|
| L0 可复现 | 同一输入/模型/运行时生成相同 typed receipt | 权重、tokenizer、协议、设备和依赖哈希齐全 |
| L1 可本地运行 | 无网络加载和推理，loopback skill 可启动 | `network_model_calls=0`；offline smoke、kill switch、rollback 通过 |
| L2 小体积 | 建立 FP32/FP16/INT8/4-bit 候选阶梯 | 每个候选记录磁盘、参数、峰值内存、加载时间和质量差异；不能只报压缩率 |
| L3 快响应 | 标准 workload 的 warm p50/p95、cold、batch 分开达标 | 初始建议目标：warm p95 ≤250 ms；最终阈值须以本地 baseline 和任务 SLA 预注册 |
| L4 可校准 | 每个已批准任务包有 reliability/ECE、NLL/Brier、coverage-selective risk | calibration、test、OOD 和 source groups 严格隔离 |
| L5 任务质量 | 任务包准确率/成本/abstain 改善，且无 protected error 回归 | 95% bootstrap/Wilson 区间、最小样本和最坏族门禁先写入协议 |
| L6 工作流价值 | 主模型下游成功不下降，净 provider token/cost 有真实 paired receipt | 至少 3 个主模型族；同一请求 unfiltered/filtered provider usage 成对存在 |
| L7 可部署 | 本地包可安装、升级、回滚、审计 | 默认 shadow；active 需要全部 Track A gate + 人工批准 |

## 4. V3 工作包

### N0 — V2 closure and evidence ledger

**目的：** 先把 V2 的完成、负面结果和外部门禁固定下来。  
**交付：** 本文档、V2 completion audit、最新 handoff/review log、owner decision checklist。  
**门禁：** 不得把 T6/T7/T8g/T9d/T16 标成完成；所有数字带来源和时间。  
**状态：** 本轮设计已准备，等待独立审核/owner 处理旧门禁。

### N1 — Contract-first benchmark harness

**目的：** 建立跨设备、跨精度、跨任务包的单一 benchmark runner。  
**记录：** model/weights/tokenizer/chat-template SHA、设备、精度、线程、温度、batch、
  cold/warm、p50/p95/p99、peak memory、energy proxy（若可测）、网络调用计数和 receipt hash。  
**门禁：** 任何模型/量化/架构变更先过 parity、determinism、protected-segment、
malformed/fail-open 和 no-network tests。  
**不做：** 不根据一个设备或一次 warm run 选择赢家。

**本轮状态（2026-09-20）：`IMPLEMENTATION/VERIFICATION COMPLETE — READY_FOR_REVIEW`。**

- 新增 [N1 benchmark contract](NANOJEV_V3_BENCHMARK_CONTRACT_V1.md)、
  `scripts/benchmark_nanojev_v3.py` 和 `scripts/test_benchmark_nanojev_v3.py`；harness
  延迟导入现有本地 predictor，固定记录 checkpoint/tokenizer/chat-template/依赖哈希、设备、
  精度、线程、cold/warm p50/p95/p99、峰值 RSS、零网络调用和 content-free receipt。
- N1 单测 **25/25**；N1 时点主仓库回归 **676 OK / 2 skipped**；安装版本地 NanoJev skill
  **17/17 OK**；`git diff --check` 与编译检查通过。N2/N3 后的最新主仓库回归为
  **737 OK / 2 skipped**（N1 的 676 保留为历史快照）。
- 真实离线 smoke 收据：
  [`results/nanojev_v3_n1_smoke_20260920.json`](../results/nanojev_v3_n1_smoke_20260920.json)，
  checkpoint `local_atomic_seed17` 在 MPS/FP32 下 `network_model_calls=0`，两次输出摘要稳定。
  本次 workload 的 cold **10,529.1 ms**（含加载）、warm **3,254.2 ms**、峰值 RSS
  **5,274,910,720 B**；这些只是一次收据，不是 L3 目标达标或模型质量结论。
- 第二个独立进程收据
  [`results/nanojev_v3_n1_smoke_20260920_run2.json`](../results/nanojev_v3_n1_smoke_20260920_run2.json)
  的 `output_digest`、`receipt_sha256`、`canonical_sha256` 与首轮逐项一致；第二轮 cold
  **10,525.0 ms**、warm **3,227.2 ms**、峰值 RSS **5,274,877,952 B**。这只支持当前
  workload 的跨进程确定性，不外推为性能目标或质量改进。
- Worker 使用官方 DeepSeek API `https://api.deepseek.com/v1`、`deepseek-flash` →
  `DeepSeek-V4.1-Flash`；因未返回合格结构化结果，Worker 状态记为 `partial`，但实际变更
  仅限上述三个允许文件，已逐文件复核后应用。没有 provider 降级或重复写入。

**N1 待审边界：** 当前 workload 每个进程仍只有一个 44-state warm 样本，不能据此选择性能赢家；
CPU/低精度阶梯、更多 batch/并发、质量配对和部署包仍属于 N4–N7，且 active pruning 仍关闭。

### N2 — Data and domain quality packs

**目的：** 从“一个游戏域”升级为少量、明确边界的任务包：routing、verification、
context safety、tool-history triage、金融状态分类（仅在 R1 冻结后）。  
**要求：** 每包独立 source groups、train/dev/calibration/test/OOD、human-reviewed labels、
  hard negatives、protected cases、数据许可和 provenance receipt。  
**门禁：** 复现 T8g 的内容去重/连通分量审计；任何跨 split 同输入、事实来源冲突或 endpoint
  身份不清都阻塞训练。  
**目标：** 先提高已批准任务的 coverage-selective quality，不以降低 abstain 为单一目标。

**当前状态（2026-09-20）：`IMPLEMENTATION/VERIFICATION COMPLETE — READY_FOR_REVIEW`。**

- 新增 [N2 domain-pack contract](NANOJEV_V3_DOMAIN_PACK_CONTRACT_V1.md)、
  `scripts/validate_nanojev_v3_domain_pack_v1.py` 和 `scripts/test_validate_nanojev_v3_domain_pack_v1.py`。
  validator 只读校验 manifest、五个 split、source-group/lineage、canonical visible-input、
  provenance、heldout 和 protected cases；hash 发生变化、schema 不完整或输出覆盖时 fail-closed。
- Worker 初始交付 37 项合成测试；主模型复核后补齐 required target/question/provenance、protected
  reason 和 manifest 路径边界，W21 又固定 portable report identity/source_paths，最终 N2 专项
  **40/40**，主仓库最新 **893 OK / 2 skipped**。
- 合成 clean pack CLI 退出 0，但报告明确 `preflight_passed_not_training_authorized`、
  `training_authorized=false`；现有冻结 V1 corpus 直接运行退出 2（schema/heldout/split/
  protected 缺口），没有被转换、合并、改 split/label 或训练。
- T8g 的 27 个跨 split canonical-input groups / 63 条记录、18 个 evaluation-derived provenance
  hits 和跨 seed 129 个相同输入仍保持为阻塞证据；N2 不替代 owner/reviewer 对新 corpus 的决定。

### N3 — Architecture and readout ladder

**目的：** 比较任务头、schema-conditioned readout、轻量 encoder/router 和共享 prefill，
  但把每个读法作为独立架构假设。  
**门禁：** 所有 Choice 排列、option-count、label permutation、Boolean/Score、OOD 和
  constant baseline 都要通过；任何“高置信位置伪影”立即拒绝。  
**输出：** 不是一个全局赢家，而是按任务包记录 accuracy/ECE/coverage/latency/memory 的
Pareto front。

**当前状态（2026-09-20）：`PROTOCOL/VERIFICATION COMPLETE — READY_FOR_REVIEW`；未运行任何 arm。**

- 新增 [N3 readout protocol](NANOJEV_V3_N3_READOUT_PROTOCOL_V1.md)、冻结协议
  [`research/nanojev_v3_n3_readout_protocol_v1.json`](../research/nanojev_v3_n3_readout_protocol_v1.json)、
  只读 validator `scripts/validate_nanojev_v3_n3_readout_protocol_v1.py` 和 22 项合成测试。
- 协议固定三条 paired arm（trained head、schema readout、encoder/router）、五类冻结 split
  角色、option/label permutation、Boolean/Score、OOD、protected-case、constant baseline 和
  完整 paired receipt 字段；validator 对缺失/重复/越界/授权式字段 fail-closed。
- 真实协议 preflight 收据
  [`results/nanojev_v3_n3_protocol_preflight_20260920.json`](../results/nanojev_v3_n3_protocol_preflight_20260920.json)
  为 `protocol_valid_not_authorized`；`training_authorized=false`、`deployment_authorized=false`、
  `production_pruning_authorized=false`、`model_loaded=false`、`network_model_calls=0`。
- 新增 synthetic-only paired-control runner
  [`scripts/run_nanojev_v3_n3_synthetic.py`](../scripts/run_nanojev_v3_n3_synthetic.py) 及其
  [`12 项测试`](../scripts/test_run_nanojev_v3_n3_synthetic.py)，并生成
  [`synthetic smoke receipt`](../results/nanojev_v3_n3_synthetic_smoke_20260920.json)。收据覆盖
  3 arms、8 个内存 fixture cases、123 条 content-free receipts（option 96、label 15、OOD 18、
  protected 30、baseline 12）；状态为 `synthetic_controls_passed_not_model_evidence`，
  `network_model_calls=0`、`model_loaded=false`、training/deployment/production-pruning
  authorization 全为 `false`，protected confident errors 为 0。该 runner 只证明配对控制和
  fail-closed 收据路径，不证明任何真实模型质量、速度、体积或部署能力。

- 新增 N3-S readiness contract
  [`NANOJEV_V3_N3_S_READINESS_CONTRACT_V1.md`](NANOJEV_V3_N3_S_READINESS_CONTRACT_V1.md)、
  冻结声明 [`research/nanojev_v3_n3_s_readiness_v1.json`](../research/nanojev_v3_n3_s_readiness_v1.json)、
  只读 validator `scripts/validate_nanojev_v3_n3_s_readiness_v1.py` 和 W21 后的 53 项专项测试。当前
  预检收据 [`results/nanojev_v3_n3_s_readiness_preflight_20260920_run3.json`](../results/nanojev_v3_n3_s_readiness_preflight_20260920_run3.json)
  为 `n3_s_readiness_blocked`：N2 domain-pack receipt 缺失，四项独立审核均为 `pending`；
  N3 protocol/synthetic receipts 虽 hash 匹配，也不会被当作真实测量授权。所有 measurement/
  training/deployment/production-pruning flags 保持 false，network/model activity 为零。
- W20 修正 N2 producer/consumer schema 不兼容，用真实 producer 生成临时合成 pack 收据代替
  手造结果字段，并补齐严格类型、固定必检字段、异常读取及 JSON 歧义回归。旧 19 项单测没有
  发现这个缺陷，旧收据保留不覆盖；新增测试只提升接口/门禁可信度，不是模型能力提升。
  `review_declarations_only=true` 明示这里只检查声明，reviewer 身份和权限仍需 owner 核对。
- W21 让 N2 receipt 的 `pack` 使用 `pack_id@pack_version`，并新增 `source_paths`；N3-S 现在
  要求 source hashes 与 manifest 加五个 split 的相对路径精确绑定。readiness report 使用相对
  `source` 和 `workspace_root="."`，避免 pinned bytes 随 checkout 绝对路径变化。W21 的新测试
  覆盖跨目录报告等价与 source-path/hash 漂移；这些是证据完整性改进，不是模型能力提升。
- 官方 Worker job `1789843972-126092f86bc5` 使用 `https://api.deepseek.com/v1`、
  `deepseek-flash` → `DeepSeek-V4.1-Flash`，但未返回合格五字段结构化结果，状态保留为
  `partial`；主模型补齐测试并复核实际文件，没有 provider 回退、训练或模型推理。

**N3 待审边界：** 当前是协议/结构验证、synthetic-only 控制烟测和 readiness preflight，不是架构质量结果；真实 N3 pack、label/OOD 组成、
语义正确性、校准、延迟/内存 Pareto、训练、量化和部署仍未开始。readiness blocked 是预期的
fail-closed 结果；即使未来变为 `passed_not_authorized`，也不授权训练或发布。

### N4 — Distillation, quantization and footprint

**目的：** 在不破坏 typed probability 和 abstain 语义的前提下，逐级测试 FP16、INT8、
  4-bit、结构化蒸馏、低秩/adapter 和 tokenizer/context 优化。  
**门禁：** 每个候选与 FP32 baseline 配对；quality delta、protected error、ECE、OOD、
  load/cold/warm、peak memory 和 artifact hash 全记录。  
**停止条件：** 任何压缩候选出现 confident-wrong protected decision、概率不归一、
  restore/fail-open 回归或不可复现，就停止晋级，不以体积换安全。

**当前状态（2026-09-20）：`CONTRACT/PREFLIGHT + SYNTHETIC CONTROLS COMPLETE — READY_FOR_REVIEW`；没有真实量化。**

- 新增 [N4 footprint contract](NANOJEV_V3_N4_FOOTPRINT_CONTRACT_V1.md)、冻结协议
  [`research/nanojev_v3_n4_footprint_contract_v1.json`](../research/nanojev_v3_n4_footprint_contract_v1.json)、
  只读 validator `scripts/validate_nanojev_v3_n4_footprint_contract_v1.py` 和 24 项专项测试。
- 候选阶梯固定为 `fp32_baseline`、`fp16_cast`、`int8_weight_only`、`int4_weight_only`、
  `distilled_student`；所有候选必须和 FP32 配对，收据要求 artifact/tokenizer/protocol hash、
  bytes、参数量、峰值内存、加载/冷/暖延迟、质量/校准/OOD/protected 和归一化指标。
- 真实 preflight 收据
  [`results/nanojev_v3_n4_footprint_preflight_20260920.json`](../results/nanojev_v3_n4_footprint_preflight_20260920.json)
  为 `footprint_contract_valid_not_authorized`；5 候选、23 个 required metrics、28 个 receipt
  fields，`model_loaded=false`、`network_model_calls=0`、quantization/training/deployment performed
  与全部 authorization flags 均为 `false`。这只证明契约结构，不证明体积、速度、成本或质量。

- 新增 synthetic-only footprint-control runner
  [`scripts/run_nanojev_v3_n4_synthetic.py`](../scripts/run_nanojev_v3_n4_synthetic.py) 与
  [`14 项测试`](../scripts/test_run_nanojev_v3_n4_synthetic.py)，并生成
  [`synthetic smoke receipt`](../results/nanojev_v3_n4_synthetic_smoke_20260920.json)。收据覆盖
  5 个固定候选、5 个 content-free receipts、receipt/report hash、字段/顺序、隐私和
  fail-closed activity/authorization controls；状态为
  `synthetic_footprint_controls_passed_not_measurement_evidence`。所有真实 artifact/体积/延迟/
  质量/校准/OOD/成本字段均为 `null`，因此该 runner 不是模型或资源测量证据。

- W18 复现把主仓库回归更新为 **787 OK / 2 skipped**，安装版 NanoJev skill 仍为 **17/17 OK**；
  `py_compile` 与 `git diff --check` 通过。773→787 只表示新增测试覆盖，不是性能、质量、体积、
  成本或部署提升。

### N5 — Serving and latency engineering

**目的：** 让小模型“快”可被复现测量。  
**候选：** persistent load、prefix/prompt cache（只缓存公开可复用前缀）、bounded batch、
  CPU/MPS kernel、request coalescing、compiled graph；所有候选都要保持请求隔离。  
**门禁：** p50/p95/p99、cold start、queueing、timeouts、cancellation、memory pressure、
  concurrent requests、错误回退和 no-network 都必须在同一 workload manifest 下比较。  
**不做：** 不把并行吞吐写成 single-decision latency，不以取消真实 provider 请求为前提。

### N6 — Workflow integration and shadow value

**目的：** 证明 NanoJev 作为主模型前置层确实有净价值。  
**顺序：** shadow → paired replay → controlled active experiment → human review。  
**门禁：** 至少 3 个主模型族、受保护段零删除、下游成功率不降超过预注册 margin、真实
  provider prompt tokens/cost paired、网关评分/重试/重建/工具重执行成本计入。  
**默认：** active 继续关闭；无真实 paired receipt 就只能说“shadow estimate”。

### N7 — Local packaging and release

**目的：** 形成可在本机离线安装、升级、回滚的最小发布包。  
**发布材料：** artifact manifest、license/SBOM、weights/tokenizer hashes、Python/runtime
  matrix、health/doctor、migration、rollback、privacy and retention policy。  
**门禁：** 新机离线安装、断网推理、损坏权重拒绝、版本不匹配拒绝、kill switch、日志不含
  原文/凭据、active 默认关闭；部署 lifecycle 只能提供 advisory readiness，必须保留
  `authorizes_execution:false`，不授予发布权限。

### N8 — Continuous improvement loop

**目的：** 让准确率和任务质量逐步提升，而不是靠一次性“换更大模型”。  
**闭环：** 失败 receipt → 去标识化 error taxonomy → 新 hard-negative/contrastive pair →
  frozen calibration → candidate train/quantize → paired evaluation → promotion review。  
**门禁：** 任何新样本不得回流到 test/OOD；每个 candidate 保留可回滚 checkpoint；
  confidence 不能单独决定 promotion；所有训练/优化/部署阶段记录本地 NanoJev lifecycle event。

## 5. 指标、成本和证据协议

每个 V3 实验必须同时报告：

- **质量：** accuracy、NLL、Brier、ECE/reliability、coverage-selective risk、OOD、
  protected-error count、task-family worst case；
- **速度：** cold/warm p50/p95/p99、batch throughput、queue/wait、timeout 和取消；
- **体积/资源：** params、weight bytes、package bytes、peak RSS/unified memory、load time；
- **成本：** provider prompt/completion tokens（只能使用 provider usage）、本地运行时间、
  可复现的 energy/cost proxy、重试/重建/工具重执行；
- **可靠性：** determinism、hashes、no-network、fail-open、restore identity、日志隐私；
- **证据等级：** author claim / source audit / local reproduction / paired comparison，
  以及 snapshot time、comparison scope 和 artifact path。

任何一项缺失时，结论只能是“部分测量”，不能叫作“更快”“更便宜”“更准确”或“已部署”。

## 6. 90 天建议顺序（不替代 owner 门禁）

| 时间窗 | 主动作 | 退出条件 |
|---|---|---|
| 0–2 周 | N0/N1：V2 audit、benchmark runner、设备/精度/体积 baseline | N1 收据可重建；V2 外部门禁清单被 owner/reviewer 看到；N1 已实现/验证，仍待独立审核 |
| 2–5 周 | N2：新任务包和 T8g provenance 修复；只做数据审计和标注 | N2 contract/preflight 已完成；split/endpoint/source-group 仍需审核通过 |
| 4–7 周 | N3/N4：架构读法与量化/蒸馏 paired ablation | N3 synthetic harness、N4 footprint contract/preflight 与 N4 synthetic controls 已完成，但尚无真实 arm/量化结果；N4 只能在 N2/N3 审核后启动，并须产出无 protected regression 的 Pareto receipt |
| 6–9 周 | N5：本地 serving 和并发/长上下文 benchmark | 达到预注册 latency/memory target，失败路径仍 fail-open |
| 8–12 周 | N6/N7：shadow value、离线包、回滚和人工 review | 三主模型族 paired evidence；否则保持 shadow |
| 持续 | N8：失败驱动数据闭环和 T15 ledger | 每次候选都有版本、事件、收据和可回滚路径 |

## 7. 必须长期保持的红线

1. V2 R1 未冻结前不做正式 Track B 训练；Aster 未获书面许可前不纳入 primary。
2. 不将本地 NanoJev 输出当成授权、代码审查结论、交易信号或 provider token 账单。
3. 不为追求 coverage 降低 abstain threshold；先修数据、域适配和校准。
4. 不启用生产 active pruning，除非 Track A 全部门禁、三主模型 paired quality/cost、
   protected stress suite 和人工批准全部通过。
5. 不把 star、作者 benchmark、一次本地 demo 或测试通过写成能力证明。
6. 每个 development/testing/optimization/deployment 阶段都先调用本地 NanoJev lifecycle，
   保存 event ID 和反馈；Worker 只允许官方 DeepSeek V4.1 Flash，禁止第三方回退。

## 8. V3 成功定义

V3 成功不是“模型更自信”，而是：在至少一个明确任务包上，经过冻结、可复现、成对的
质量/校准/延迟/体积/成本/隐私测试，本地 NanoJev 以更小 artifact、更低本地成本和更快
响应提供不劣于 baseline 的决策质量，并在域外可靠弃权；同时可以被本地 skill、主模型
shadow 网关和离线部署包一致调用。没有这些 receipts，V3 只能停留在研究原型阶段。

## 9. 下一阶段执行设计（N3-S → N8）

这是在当前 V2 外部门禁和 N3 synthetic 收据之上的**条件式执行路线**，所有“目标”均为待
预注册/实测值，不是当前能力声明。每一阶段都必须先调用本地 NanoJev lifecycle，保存
event ID，再由确定性门禁和 owner/reviewer 决定是否晋级。

| 阶段 | 目的 | 必须交付 | 晋级门禁 |
|---|---|---|---|
| N3-S | 从结构烟测进入可审计真实测量 | 经审核的 N2 domain pack；三 arm paired receipts；完整 permutation、label、OOD、protected、baseline 控制 | source-group/lineage、标签、许可和 holdout 审核通过；任何 protected confident error 或位置伪影即阻断 |
| N4-S | 体积与成本阶梯 | FP32 基线、FP16/INT8/4-bit/蒸馏候选的 artifact/内存/加载/质量收据 | 同一 workload 下质量、ECE、OOD、protected error 不劣；哈希可复现；不得以压缩率替代质量 |
| N5-S | 快速本地 serving | CPU/MPS、cold/warm、并发、p50/p95/p99、超时/取消/回退收据 | 预注册 warm p95（建议 ≤250 ms）和内存目标；超时与错误仍 fail-open；不把吞吐冒充单决策延迟 |
| N6-S | 证明工作流净价值 | 至少 3 个主模型族的 unfiltered/filtered paired receipts，包含 scorer、重建、重试和工具重执行成本 | 下游成功率不降超 margin、protected 段零删除、真实 provider token/cost 可核对；否则保持 shadow |
| N7-S | 离线可部署发布包 | SBOM/license、权重/Tokenizer 哈希、health/doctor、断网安装、回滚、kill switch | 新机离线安装与损坏/版本不匹配拒绝通过；active 默认关闭；deployment lifecycle 仍 `authorizes_execution=false` |
| N8-S | 质量持续提升 | failure taxonomy → hard-negative/contrastive pair → calibration → paired replay 的版本化闭环 | test/OOD 永不回流；每候选可回滚；不得用置信度单独 promotion；每次改动保留 receipt 和 reviewer 决定 |

N3-S 当前停在 readiness preflight：`results/nanojev_v3_n3_s_readiness_preflight_20260920_run3.json`
明确报告 `n3_s_readiness_blocked`，因为 N2 domain pack 尚无通过收据，且四项独立审核仍为
`pending`。在这些证据出现前，不运行真实 paired arm，也不把 N3 synthetic receipt 当作质量、
校准、延迟或部署证据。

当前 N4-S 已完成契约、结构预检和 synthetic-only 控制收据；它仍没有开始 artifact 生成、真实量化、
蒸馏或性能比较。`synthetic_footprint_controls_passed_not_measurement_evidence` 不提供任何
体积、延迟、质量、校准、OOD、成本或部署证据。

**阶段顺序约束：** N3-S 未通过前不启动 N4；N4 没有 Pareto 收据不启动 N5；N5/N6/N7
只能在 shadow 或人工批准范围内推进。任何外部许可、R1/R2/R3/R4、T8g/T9/T9d/T16 未关闭
时，相关阶段保持 `BLOCKED` 或 `READY_FOR_REVIEW`，不以路线图文字代替批准。
