# NanoJev V3 N3-S readiness contract V1

**状态：只读 readiness contract，2026-09-20。**

N3-S 是从 synthetic controls 进入真实 paired readout measurement 的前置门。这个文件和
`scripts/validate_nanojev_v3_n3_s_readiness_v1.py` 只核对收据、哈希和独立审核声明；它们不加载
模型、权重、tokenizer、dataset 或 artifact，不访问网络，也不授予真实推理、训练、量化、部署或
promotion 权限。

## 当前结论

当前预检收据
[`results/nanojev_v3_n3_s_readiness_preflight_20260920_run3.json`](../results/nanojev_v3_n3_s_readiness_preflight_20260920_run3.json)
为 `n3_s_readiness_blocked`，原因是：

1. 审核所需的 N2 domain-pack preflight 收据尚不存在；
2. N2/N3 数据、标签、holdout、protected cases 和许可/来源的独立审核均为 `pending`；
3. N3 protocol 和 N3 synthetic receipt 虽然存在且 hash 匹配，但它们只证明协议/控制路径，
   不能替代真实 domain pack 或 reviewer。

这是一条预期的 fail-closed 结果，不是 N3-S 失败，也不是开始真实测量的授权。

## 必须固定的证据

契约固定三个输入收据：

| Evidence | 当前期望 | 作用 |
|---|---|---|
| N2 domain-pack preflight | `preflight_passed_not_training_authorized` | 证明 source-group/lineage、canonical input、provenance、holdout、protected cases 结构通过；不授权训练 |
| N3 protocol preflight | `protocol_valid_not_authorized` | 证明三条 paired arm、排列、OOD、protected、baseline 控制结构有效；不运行 arm |
| N3 synthetic controls | `synthetic_controls_passed_not_model_evidence` | 证明 receipt/hash/fail-closed 控制路径；不是模型质量证据 |

每个收据必须有固定的小写十六进制 SHA-256。缺失文件、读取失败、hash 不匹配、状态不符、
字段类型不符或非零网络/模型加载/授权字段，均退出 2 并保持 `measurement_authorized=false`。

### W20 接口修正（2026-09-20）

W19 的 N2 声明错误地要求 `valid`、`network_model_calls` 和 `model_loaded`，而实际 N2
producer 不输出这三个字段。原单测手造了同样的字段，因而漏掉不兼容；**W19 的 19 项通过
不能证明真实 producer/consumer 兼容**。W20 改为检查真实 N2 输出：

- `schema_version=nanojev-v3-domain-pack-preflight-v1` 与固定通过状态；
- `input_files_changed=false`，`manifest_errors`、`violations`、`block_reasons` 均为空列表；
- `training_authorized=false`、`training_performed=false`、`merged_rows_written=0`；
- manifest 加五个 split 的六项 source hashes 均有效，且 `source_paths` 恰好绑定这六个相对
  文件名、before/after 一致。

N3 两类收据也检查各自固定 schema/report version。上述必检值在 validator 内固定；JSON
声明不能删除、改类型或改状态以降低要求。比较为递归严格类型比较，`false`、`0`、`0.0`
不互相替代。N2 测试通过真实 `validate_pack` 生成临时**合成**收据，不创建真实 N2 通过证据，
也不改 N2 producer 的 schema。

专项回归现为 **53 tests OK**，包括真实 N2 producer、source-path/hash 绑定、N2 CLI 完整报告/摘要区分、
异常读取、软链接越界/循环、schema/type/hash 漂移、重复 JSON key 与非有限数拒绝。W19 原收据与 W20 run2 均保留为历史快照；
W21 run3 仍为 `n3_s_readiness_blocked`，未因修复变成 approved 或 ready。

SHA-256 固定的是**保存的完整报告字节**，不是 N2 CLI 使用 `--output` 时打印的简短摘要。
W21 起 N2 报告用 `pack_id@pack_version` 和 `source_paths`，不再把 checkout 绝对路径写进
报告；相同 pack bytes 在不同目录生成相同报告身份。仍应携带原报告和其完整 hash，不能把
schema/hash 当作 producer 或 reviewer 身份认证。readiness 报告本身的 `source` 使用相对标记、
`workspace_root="."`，避免下游再引入路径依赖。

## 必须显式批准的审核项

- N2 domain pack 独立审核；
- N3 protocol 与 controls 独立审核；
- labels、holdout、OOD 和 protected cases 审核；
- 数据许可与 provenance 审核。

只有四项声明都明确为 `approved`，且三个收据全部存在并通过字段检查，validator 才会报告
`n3_s_readiness_passed_not_authorized`。即使如此，报告仍保留：

```text
measurement_authorized = false
real_n3_measurement_authorized = false
authorizes_execution = false
review_declarations_only = true
```

这里检查的是审核**声明**，没有验证签名、reviewer 独立身份或其审批权限。`ready=true` 不代表
独立审核已被机器认证；owner 仍须核对实际审核记录，主模型不能把字段改为 `approved` 代签。
V1 的 `contract_valid`/`valid` 表示本次完整 gate 校验无 violations（含缺失证据与 pending
review），不是“JSON 结构有效”；pending 时它们与 `ready` 同为 false。

真实 N3-S 测量还需要单独的人类/owner 权限和新的本地 NanoJev `testing` lifecycle event；
validator 不能替代这些权限。

## 复现

```bash
.venv/bin/python scripts/validate_nanojev_v3_n3_s_readiness_v1.py \
  --protocol research/nanojev_v3_n3_s_readiness_v1.json \
  --output results/nanojev_v3_n3_s_readiness_preflight_20260920_run3.json
# 当前预期：exit 2，status=n3_s_readiness_blocked
# 若输出文件已经存在，省略 --output 可重查；exclusive-create 不覆盖历史收据。
```

测试只在临时目录构造正向/负向 JSON receipts，不读取真实 dataset 或 checkpoint：

```bash
.venv/bin/python -m unittest discover -s scripts \
  -p 'test_validate_nanojev_v3_n3_s_readiness_v1.py'
```

该 contract 不改变 V2 T1–T16 状态，不解冻 T8g/T9/T9d，不授权真实 N3 arm、训练、量化、active
pruning 或部署。
