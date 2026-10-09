# NanoJev V3 N1 Benchmark Contract V1

**状态：契约（N1 交付物），2026-09-20。**
**范围：** 只定义本地 benchmark harness 的记录字段、确定性与 fail-closed 行为。
**实现：** `scripts/benchmark_nanojev_v3.py`；测试：`scripts/test_benchmark_nanojev_v3.py`。

本契约对应 roadmap 的 N1（Contract-first benchmark harness）。它不实现模型训练、
量化、serving 或生产 pruning，只建立跨设备、跨精度、跨任务包可复现的单一测量入口。

## 0. 三条必须保持的声明

1. **目标是目标，不是当前结果。** 本文档、roadmap 和 harness 中的任何阈值（例如
   warm p95 ≤250 ms）都是待预注册和实测的 *target*。harness 只输出一次本地运行的
   测量值，不宣称模型已经达标。
2. **本地 token 数不是 provider 计费。** harness 记录的是本地 tokenizer 的输入规模与
   本地资源；只有 provider 自己的 usage 报告才能用于计费。字符数、本地 token 数、
   shadow estimate 都不等于真实账单或真实节省。
3. **本 harness 不启用生产 pruning。** report 中固定写入
   `contract.enables_production_pruning=false`，receipt 中固定写入
   `enables_production_pruning=false`。active pruning 仍需 Track A 全部门禁与人工批准。

## 1. CLI 契约

```
python3 scripts/benchmark_nanojev_v3.py \
    --checkpoint DIR --input FILE_OR_DIR \
    [--device auto|cpu|mps|cuda[:index]] \
    [--precision auto|fp32|bf16] \
    [--threads N] \
    [--batch-states N] [--batch-questions N] \
    [--temperature FLOAT] [--max-length N] \
    [--repeats N] [--output report.json]
```

- `--checkpoint`、`--input` 必填；其余有默认值。
- `--threads` 会在导入 torch 前设置 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`，并在
  torch 就绪后读取实际线程数（`threads_effective`）。
- `--repeats` 为 workload 遍数；第一次 predict 记为 cold，其余全部记为 warm。默认 2，
  保证至少有一个 warm 样本。
- `DecisionPredictor` 只在真正执行 run 时惰性导入；`--help`、导入模块和单元测试都不会
  加载 torch 或权重。

## 2. 必须记录的字段

### 2.1 文件身份（identity）

- 输入：单文件或目录下 `*.jsonl`，逐个记录 `path / bytes / sha256`，并给出整体
  `input.sha256`。输入为空或缺失时 fail closed。
- checkpoint 必填文件：
  `config.json`、`best.safetensors`、`backbone_config/config.json`、
  `tokenizer/tokenizer.json`、`tokenizer/tokenizer_config.json`。
- checkpoint 可选文件（存在即记录）：`tokenizer/chat_template.jinja`、
  `tokenizer/special_tokens_map.json`、`tokenizer/merges.txt`、`tokenizer/vocab.json`。
- **tokenizer chat_template**：优先取 `tokenizer/chat_template.jinja` 的文件内容；
  否则取 `tokenizer_config.json` 的 `chat_template` 字段；记录 `source / bytes / sha256`。
- benchmark 脚本自身、Python 版本与解释器路径、依赖版本（torch/transformers/
  safetensors/numpy）与硬件描述符。

### 2.2 运行时配置

`device`、`precision`、`threads_requested`、`threads_effective`、`temperature`、
`batch_states`、`batch_questions`、`max_length`、`repeats`。

### 2.3 延迟与资源

- `latency.load_ms`：构造 predictor（含本地加载）耗时。
- `latency.cold`：`load_ms + 第一次请求`，并给出 `p50/p95/p99` 与 `samples`。
- `latency.warm`：第一次之后的请求，给出 `p50/p95/p99`、`min/max/mean` 与 `samples`。
- `latency.total_elapsed_ms`、`resources.peak_rss_bytes`、`resources.energy_proxy`
  （当前不可测时为 `null`）。
- cold 与 warm 必须分开报告；不得把吞吐除以样本数当作单决策延迟。

### 2.4 网络调用

`execution` 中每个 predict 结果的 `network_model_calls` 必须恰为 `0`。字段缺失、
类型错误或非零都会 fail closed（抛 `ContractError`），不会写出部分 report。

## 3. receipt 与确定性 canonicalization

### 3.1 隐私保护 receipt

`receipt` 是 content-free 的：只包含各种 sha256、运行时配置和 contract 声明，**不包含**
原始 state/question 文本、样本 id 或输入文件路径。`receipt_sha256` 由 receipt 的
canonical JSON 计算，因此相同输入与配置得到相同 receipt。

### 3.2 确定性 report canonicalization

`canonical_report()` / `canonical_report_sha256()` 会在计算前**递归剔除显式 time-varying
字段**（`generated_at`、`latency`、`resources`、`peak_rss_bytes`、`load_ms`、
`total_elapsed_ms`、`elapsed_ms` 等）以及派生的自引用哈希
（`canonical_sha256`、`receipt_sha256`）。因此同一输入、模型、运行时在测量噪声范围内
产生相同的 `canonical_sha256`，可直接用于 parity/determinism 对比。任何新的 time-varying
字段都必须显式加入 `TIME_VARYING_KEYS`。

## 4. Fail-closed 清单

出现以下任一情况即 `ContractError`，进程非零退出，不输出 report：

1. checkpoint 目录不存在或缺必填文件；
2. 输入不存在、为空、为非法 JSON 行，或缺少 `id/state/questions`；
3. `batch-states < 1`、`batch-questions < 0`、`repeats < 1`；
4. `temperature` 非有限正数；
5. `threads` 非正整数；
6. 任一 predict 的 `network_model_calls` 缺失、类型错误或 ≠0（V3 默认 network calls=0）。

## 5. 测试

`scripts/test_benchmark_nanojev_v3.py` 使用临时文本 fixture 和内存 fake predictor，
不加载权重、不访问网络、不导入 torch，覆盖：

- canonicalization 对 time-varying 噪声不敏感、对实质变化敏感；
- checkpoint/input identity、chat_template 提取、缺失文件 fail closed；
- `network_model_calls` 与参数的 fail-closed 校验；
- receipt 的确定性、content-free 与输入变化敏感性；
- 惰性导入行为；
- 用 fake engine 跑通完整 `run()` 并核对 cold/warm、peak RSS、receipt 与 canonical hash 稳定。

运行方式：

```
python3 -m unittest test_benchmark_nanojev_v3 -v   # 在 scripts/ 目录下
```

## 6. 不做的事

- 不修改既有 v2 脚本；
- 不下载模型或数据，不调用任何真实 provider；
- 不启用生产 pruning，不把一次 warm run 当成设备/模型赢家；
- 不把测试通过或 benchmark 数字当成能力证明或部署许可。
