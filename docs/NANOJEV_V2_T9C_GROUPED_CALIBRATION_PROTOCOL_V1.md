# T9c — grouped calibration protocol V1

**状态：协议/预检草案，待独立审核；未运行分组拟合，未改变服务温度。**

T8d 只测试了一个全局 temperature，并得到负面结果：单调重标定不能修复语义排序或位置伪影。
T9c 的问题更窄：在冻结的 `question_type × option_count` 分组内，是否存在可复现的后验温度，
能在不改变 argmax 的前提下改善 NLL/Brier/ECE 或 coverage-selective risk？它不是降低 0.9 门限的
许可，也不是将工程域外弃权变成授权。

## 冻结设计

协议文件：[`research/nanojev_v2_t9c_grouped_calibration_protocol_v1.json`](../research/nanojev_v2_t9c_grouped_calibration_protocol_v1.json)。

JSON 中的 `frozen_research_protocol` 只表示协议字节和输入角色在测量前冻结，**不表示 owner 或
独立 reviewer 已批准**；批准状态仍由本文件的 `待独立审核` 和后续审查日志决定。

- 分组键只允许 `question_type` 与 `option_count`，并从冻结输入行重新计算；不能信任报告中的分组字段。
- 只允许 `calibration` split 拟合；`test` 与 `ood` 只能评估，不能 fit/select/tune/calibrate。
- 每组至少 32 个 calibration questions、8 个 state；不足组记录 `unestimated`，保持 baseline T=1.0，
  不得用相邻组或全局温度填补。
- 主目标 NLL，辅以 Brier、固定 10-bin ECE、coverage-selective risk；搜索范围和 deterministic refine
  参数固定，baseline T=1.0 必须同一行成对比较。
- 不改 `serve_decisions.py`、checkpoint、tokenizer 或默认 temperature；候选只存在于离线 receipt。
- uncertainty 按 `state_id` 聚类 bootstrap，置信水平与 non-inferiority margin 必须在 protocol 中预注册。
- 每组和 aggregate 都报告 protected error、OOD coverage、abstention 与缺组情况；任何 protected confident
  error 都阻断候选。

## 预检范围

`scripts/validate_nanojev_v2_t9c_grouped_calibration_protocol_v1.py` 是标准库、只读、fail-closed
validator。它只检查协议结构、split 角色、分组键、目标/搜索参数、控制项和授权边界；不加载模型、
checkpoint 或数据，不执行 inference，不产生 calibration 结果。

```bash
python3 scripts/validate_nanojev_v2_t9c_grouped_calibration_protocol_v1.py \
  --protocol research/nanojev_v2_t9c_grouped_calibration_protocol_v1.json \
  --output results/nanojev_v2_t9c_grouped_calibration_preflight_20260920.json
```

退出码 0 只表示 `protocol_valid_not_authorized`；退出码 2 表示协议缺失、结构非法或报告路径已存在。

## 未来真实测量的晋级条件

必须先取得独立 protocol review，再使用真实冻结数据生成完整 receipt；之后仍需单独 review 才能讨论
任何服务配置变更。若所有组都缺样本、held-out CI 跨过 margin、OOD/protected 退化，或 group temperature
在不同非测试 split 间不稳定，结论应为负面/保持 T=1.0，而不是降低门限。

当前 T9c 的**真实 grouped calibration measurement**仍是 `NOT_STARTED`；协议/预检子项已
`READY_FOR_REVIEW`，协议草案不等于测量任务完成。
