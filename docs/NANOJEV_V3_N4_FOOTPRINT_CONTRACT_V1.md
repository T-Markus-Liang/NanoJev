# NanoJev V3 N4 Footprint Contract V1

**状态：contract/preflight design；不加载模型、不量化、不训练、不部署，2026-09-20。**

N4 把“体积小、成本低”变成可复现的候选阶梯，而不是压缩率宣传。该文件只定义候选、配对
字段、资源/质量指标和 fail-closed 边界；实现为
`scripts/validate_nanojev_v3_n4_footprint_contract_v1.py`，协议为
`research/nanojev_v3_n4_footprint_contract_v1.json`。

## 1. 非目标与安全边界

1. clean preflight 只表示契约结构完整，状态 `footprint_contract_valid_not_authorized` 不代表
   量化、蒸馏、训练、发布或 promotion 获准。
2. validator 只读一个 JSON 文件，标准库实现，不读取 checkpoint、dataset、权重或 tokenizer，
   不访问网络；可选报告只能 exclusive-create。
3. N2/N3 source-group、label、OOD、protected-case 与独立 reviewer 门禁仍是前置条件；本契约
   不能替代它们，也不能解冻 V1 corpus。
4. 任何候选必须与同一 FP32 baseline 配对，报告 artifact/package bytes、参数量、峰值内存、加载
   时间、cold/warm latency、质量/校准/OOD/protected 指标和哈希；缺一项不得称为更小、更快或更便宜。

## 2. 候选阶梯

| candidate | 表示 | 用途 | 当前状态 |
|---|---|---|---|
| `fp32_baseline` | FP32 | 现有参考 | 仅声明，未由 N4 预检加载 |
| `fp16_cast` | FP16 cast | 低精度候选 | 待真实 paired measurement |
| `int8_weight_only` | INT8 weight-only | 体积/内存候选 | 待真实 paired measurement |
| `int4_weight_only` | 4-bit weight-only | 极小体积候选 | 待真实 paired measurement |
| `distilled_student` | 结构化蒸馏 student | 质量/速度候选 | 需要独立训练协议，当前不启动 |

候选可在后续协议版本中增加，但必须新增 schema/version 和完整配对收据；不得把未列入阶梯的
候选混入比较，也不得用一个候选的成功替代其他候选的验证。

## 3. 配对和 split 规则

- 所有候选使用相同 frozen workload manifest、tokenizer、输入顺序和 N3 controls。
- `train` 仅用于蒸馏拟合，`dev` 仅用于候选配置，`calibration` 仅用于校准；`test`/`ood` 只
  评估，不得 fit/select/calibrate/tune/train/promote。
- 每个候选都必须声明 `paired_with=fp32_baseline`、parent candidate、artifact/tokenizer/protocol
  hash 和完整 receipt 字段。
- 允许的质量门禁是预注册的 delta/CI，而不是看到 test/OOD 结果后改阈值；`protected_error_count`
  必须为 0，概率向量必须归一，OOD abstain 不能被压缩候选削弱。

## 4. 必需收据字段

每个候选必须输出以下字段（字段列表本身也固定在 JSON 协议中）：

`candidate_id`、`representation`、`parent_candidate_id`、`paired_with`、`artifact_sha256`、
`tokenizer_sha256`、`protocol_sha256`、`parameter_count`、`weight_bytes`、`package_bytes`、
`peak_memory_bytes`、`load_time_ms`、`cold_latency_ms`、`warm_p50_ms`、`warm_p95_ms`、
`warm_p99_ms`、`quality_metrics`、`calibration_metrics`、`ood_metrics`、`protected_error_count`、
`probability_normalization_error`、`deterministic`、`network_model_calls`、`model_loaded`、
`quantization_performed`、`training_performed`、`deployment_authorized`、`receipt_sha256`。

N4 preflight 只校验字段和边界，不生成这些真实测量值；报告中的 performed/authorization 标志
始终为 `false`。

## 5. 晋级与停止条件

只有 N2/N3 独立审核通过后，才可在新的本地 NanoJev testing/optimization lifecycle 下运行真实
paired measurement。任何候选出现 confident-wrong protected decision、概率不归一、OOD 过度作答、
哈希不一致、不可重放、内存/延迟收据缺失或 fail-open 回归，立即停止该候选，不以压缩率交换安全。

N4 clean preflight 之后的真实阶段仍必须经过 reviewer；N5 serving、N6 workflow、N7 packaging
不能因为 N4 契约通过而提前启动。

