# NanoJev V4-S0 metrics contract V1

**状态：`frozen_research_contract`；这里只冻结 schema/candidate ladder，数值目标仍待 owner/独立 reviewer 冻结，不是当前成绩或发布批准。**  
**日期：2026-09-20（Asia/Shanghai）**

V4-S0.4 把“更小、更快、质量不退化、成本可核对”变成机器可验证的指标契约。实现和协议分别是：

- `research/nanojev_v4_s0_metrics_contract_v1.json`
- `scripts/validate_nanojev_v4_s0_metrics_contract_v1.py`
- `scripts/test_validate_nanojev_v4_s0_metrics_contract_v1.py`

本阶段只做预注册和 fail-closed 结构校验，不加载模型、权重、tokenizer 或数据，不运行真实量化、
训练、serving、provider 调用或部署。预检状态 `metrics_contract_valid_not_authorized` 不表示
任何候选已经更小、更快、更准确、更便宜或可以发布。

## 1. 目标档位

| 指标 | V4-M1 研究目标 | V4-M2 发布目标 | 共同约束 |
|---|---:|---:|---|
| package bytes | ≤ 1.5 GB | ≤ 800 MB | 权重、tokenizer、运行时、SBOM 全计入；同一 hash manifest |
| peak memory / RSS | ≤ 2.0 GB | ≤ 1.25 GB | 同一设备、同一 workload、包含服务进程 |
| warm single-decision p95 | ≤ 250 ms | ≤ 100 ms | p50/p95/p99 都必须收据化；不能用吞吐替代单决策延迟 |
| cold start p95 | ≤ 5 s | ≤ 2 s | 与 warm 分开，含加载；p99 必须报告但本版本不预设数值门槛 |
| quality vs FP32 | accuracy 非劣 margin 暂定 2 pp；NLL/Brier/ECE 95% paired CI 不劣 | 同左，且 task-family worst case 不劣 | protected error=0；OOD abstain 与排列稳定性不退化 |
| local cost proxy | wall time、设备、加载、重试、工具重执行、energy proxy | 同硬件同 workload 下 95% paired CI/margin 改善 | 不写未测电费、云价或 provider 节省 |
| reliability | deterministic、zero-network、可回滚 | 断网安装、损坏拒绝、kill switch 全通过 | active pruning 默认关闭，scope guard 不可关闭 |

以上是候选目标，不是 NanoJev 当前表现。最终数值、硬件、workload、CI 和 margin 必须由 owner 与
独立 reviewer 在真实测量前冻结；若硬件证明目标不现实，应保留失败收据并修改协议，不能放宽保护门禁。

## 2. 必须分开的 latency scope

每份真实 paired receipt 必须同时声明并分别测量；V4-M1/M2 的 single-decision/cold 目标明确绑定到
`local_serving` scope，不能误套到 V2 B7 的 model-compute 或 paper-decision scope：

1. `model_compute`：准备好 tensor 后的模型计算，不能混入特征提取；
2. `paper_decision_e2e`：特征、校验和本地 paper-decision 全路径；
3. `local_serving`：常驻服务的 cold p50/p95/p99、queue、warm p50/p95/p99、timeout、取消和 fail-open。

V4 目标不覆盖 V2 B7 的既有 scope；两者都必须在同一硬件和同一 workload 下给出 p50/p95/p99。

## 3. 配对、数据和质量门禁

- 候选阶梯固定为 `fp32_baseline`、`fp16_cast`、`int8_weight_only`、`int4_weight_only`、
  `distilled_student`；每个候选都必须 `paired_with=fp32_baseline`。
- workload manifest、hardware profile、runtime、tokenizer、输入顺序和 frozen split 必须一致，
  并写入 hash；缺一项不得称为配对。
- `train`/`dev`/`calibration` 的用途先固定；`test`/`ood` 只能评估，不能 fit、select、calibrate、
  tune、train 或 promote。
- accuracy、NLL、Brier、ECE、coverage、selective risk 必须按 task family 报告；平均值掩盖
  protected family 回归时，整体判定为失败。
- `protected_error_count=0`、概率归一误差 ≤ `1e-9`、OOD abstain 不退化、语义排列不稳定为零；
  任一失败停止候选晋级。

## 4. 收据与边界

validator 只读协议 JSON，报告包含 canonical protocol SHA-256、schema、hash、目标和边界状态，不含原始输入、凭据或权重。
所有执行与授权字段在本阶段固定为 `false`，`network_model_calls=0`，`measurement_authorized=false`。

真实 measurement 只有在 N2/N3-S、owner、法律和独立 reviewer 门禁关闭后，且再次运行本地 NanoJev
testing/optimization lifecycle，才可以启动。V4-S0.4 不能解冻 V1 corpus、替代 T6/T7/T8g/T9d/T10/T16，
也不能授权 N4 量化、N5 serving、N6 workflow、N7 packaging 或生产 active pruning。

本版本的 `approval_mode=pre_approval_only` 是有意的：validator 只验证“尚未批准”的 preregistration，
不接受把 `owner_approved` 或 `independent_reviewer_approved` 改为 true 来伪造晋级。真正批准后必须
产生新的、带 reviewer/owner 身份和签名引用的协议/收据版本。

## 5. 复现

```bash
.venv/bin/python scripts/validate_nanojev_v4_s0_metrics_contract_v1.py \
  --protocol research/nanojev_v4_s0_metrics_contract_v1.json \
  --output results/nanojev_v4_s0_metrics_preflight_20260920_run2.json
.venv/bin/python -m unittest discover -s scripts \
  -p 'test_validate_nanojev_v4_s0_metrics_contract_v1.py'
```

首次创建 output 时应为 `metrics_contract_valid_not_authorized`；output 使用 exclusive-create，重复运行同一路径
会安全返回 exit 2，复核时应省略 `--output` 或改用新的临时路径。预检只证明 contract 完整，不证明任何模型指标。

当前 run2 preflight 的 canonical protocol SHA-256 为
`65bbf78a6459c0fe87354244256c03277abb9d865be29c3a9c42b9a012dbe20d`；run2 preflight 文件 SHA-256 为
`bf13472ba76000f0e304d8e9669e30dbd8e5ef2c24fc65be3adaa7c6dea4211d`。早期未带 protocol hash 的
`results/nanojev_v4_s0_metrics_preflight_20260920.json` 仅保留为历史尝试，不是当前权威 preflight。

## 6. W26 官方只读复核

官方 DeepSeek V4.1 analysis Worker `1789857734-79419c9726da` 对隔离 staging 的 contract、validator、测试
和 run2 receipt 返回 `success`、`files_changed=[]`，确认上一轮发现的 cold p99、scope binding、protocol
hash、exclusive-create、freeze/approval 与 `pre_approval_only` 六项缺口均已闭合。该 Worker 只做静态复核，
不能替代本地测试或哈希重算；主模型已独立确认 canonical hash、contract 文件 SHA-256
`b65d92e31d5cc4a39f1d1512d4d28d97e0f4a057180aa59ac27f3115dec45c1a` 和 run2 文件 SHA-256
`bf13472ba76000f0e304d8e9669e30dbd8e5ef2c24fc65be3adaa7c6dea4211d`。残余风险（未来新增批准键的扫描覆盖、
cold/M2 scope 负向测试覆盖、输出 I/O 中断的部分文件边界）不改变 validator 的 fail-closed 语义。

Worker 身份固定为 `https://api.deepseek.com/v1`、`deepseek-flash` → `DeepSeek-V4.1-Flash`、
`official_only=true`、`fallback_attempted=false`。这份复核只提升契约可审计性，不解冻真实 measurement、
训练、量化、serving、active pruning 或部署。

## 7. W28 scope-binding 负向覆盖（2026-09-20）

新增三条 validator regression，分别覆盖 M1 cold、M2 warm、M2 cold target scope 偏离
`local_serving` 的 fail-closed 行为。V4-S0.4 专项现为 **32/32**；clean contract 仍为
`metrics_contract_valid_not_authorized`，所有 authorization flags 为 `false`，`network_model_calls=0`。

官方 DeepSeek V4.1 Flash analysis Worker `1789861172-bc6de5143aaa` 只读复核成功、无文件修改；确认
新增测试、双 profile scope binding、cold p99 和递归 authorization scan 一致。Worker 未在隔离 staging
执行测试；主仓库回归为 **935 tests OK（2 skipped）**，安装版 skill **17/17**，静态检查通过。
这只是负向覆盖提升，不是延迟、质量、体积、成本或部署结果，也不授权真实 measurement。
