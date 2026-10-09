# PROVENCE_REAL_EVAL_V1 — 已发布 Provence 权重在 real-context eval 上的外部基线

> 日期：2026-09-29。范围：**仅推理，无训练**；使用公开发布的 checkpoint，未微调。
> 脚本：`scripts/eval_provence_real_context_v1.py`；逐记录分数：
> `results/provence_real_eval_v1.jsonl`（590 行：record_id, provence_score,
> mapped_noul + 诊断列）。

## 1. 模型选择

HF 上存在两个可用 Provence 系 checkpoint（`naver/provence-reranker-debertav3-v1`
与 `hotchpotch/open-provence-reranker-*` 家族）。按"ModernBERT 优先"指令选用：

- **`hotchpotch/open-provence-reranker-v1-gte-modernbert-base`**（快照
  `911517e2`）：MIT 许可、149M、`Alibaba-NLP/gte-reranker-modernbert-base`
  底座（ModernBERT 系），官方 `process()` 推理路径，快照内含
  `modeling_open_provence_standalone.py`。训练数据为 MS MARCO / GooAQ / NQ
  的 QA 句级 keep/drop 银标（Qwen3-4B teacher），**英语 only**。
- 未选 `naver/provence-reranker-debertav3-v1`：CC-BY-NC-ND 权重 + DeBERTa
  512-token 上限；OpenProvence 与其 MLDR 评测打平且许可/底座更贴合本仓库方向。

## 2. 映射（Provence → noul 契约）

| Provence | 本 eval |
|---|---|
| question | `user_messages_in_order[-1]`（超过 1200 字符截断，影响 82/590 行） |
| context | **候选段文本**（`candidate_pointer` 指向的 conversation 项） |
| sentence keep prob | `provence_score = max(per-sentence P(keep))`；段内任一句相关即判相关 |
| P(irrelevant) | `mapped_noul = 1 - provence_score` |

附加诊断列：`provence_ctx_score`（变体 B：把 `[上一段, 候选, 下一段]` 作为
预切"伪句"列表喂入，候选得分 = 其 chunk 的 max keep prob；邻居各截 1500 字符）、
`provence_rerank`（模型自带 sigmoid rerank 分）、`provence_mean_prob`、
`provence_n_sent`、`provence_kept_frac_t01`（官方 T=0.1 下保留句比例）。

## 3. 运行环境（如实记录）

- `external/valen/.venv`：transformers **5.4.0** + torch **2.6.0**；模型卡要求
  的 nltk **不在** venv 中——以 `pip install --target /tmp/provence_pydeps nltk`
  （nltk 3.10.3）+ `punkt_tab` 数据挂 PYTHONPATH 运行，**venv 与仓库
  requirements 均未改动**。
- 兼容性 shim（remote code 按 transformers 4.57 编写，5.4.0 上三处已移除的 API
  需补回，均已在脚本内注明并验证）：`build_inputs_with_special_tokens`
  （`[CLS] q [SEP] c [SEP]`，与 `tokenizer(q,c)` 成对编码结果核对一致）、
  `create_token_type_ids_from_sequences`（全 0，ModernBERT 无 segment embedding）、
  `all_tied_weights_keys` 惰性计算（远程类未调 `post_init()`）。
- **MPS 内存问题**：变体 B 在 MPS 上跑到 ~半量时 `MPS backend out of memory
  (MPS allocated: 180.9 GB)`——remote `process()` 的分配在 MPS 池内不释放。
  变体 B 改走 **CPU**（batch 24 / infer-batch 8，190s 跑完剩余 278 行）；
  变体 A MPS 全程 84s。增量落盘 + `record_id` 续跑已实现。
- 权重加载干净：safetensors 140 张量全量加载，无 missing/unexpected key。

## 4. 覆盖

590/590 记录全部打分成功（main 525 + drop_supp 65；56 个真 drop）。
`language_bucket`：multi 497 / en 93——**英语 checkpoint 跑中英混合转录属域外
使用**，这是解释结果时的主要保留项。

## 5. 核心发现：已发布模型在官方工作点上"什么都不删"

`provence_kept_frac_t01` 对全部 590 行 = **1.000**：在 Provence 官方推荐
T=0.1 下没有任何候选段被整句剪除（keep 概率全域压缩在 ~[0.10, 0.55]）。
即零样本迁移到 agent 转录 + 任务式长 query 时，该剪枝器退化为"全保留"——
所以下面的阈值扫描本质是借用其排序信号，而非其设计语义。

| 信号 | AUROC |
|---|---|
| `1 - provence_ctx_score`（±1 邻段变体） | **0.765** |
| `1 - provence_score`（候选单段） | 0.593 |
| `provence_rerank`（排序头） | 0.460（≈随机/略反） |

### 阈值扫描（`mapped_noul_ctx ≥ t → drop`，全 590 行）

| t | acc | TP | FP | FN | recall | prec |
|---|---|---|---|---|---|---|
| 0.50 | 0.0949 | 56 | 534 | 0 | 1.000 | 0.095 |
| 0.70 | 0.2356 | 54 | 449 | 2 | 0.964 | 0.107 |
| 0.75 | 0.6424 | 44 | 199 | 12 | 0.786 | 0.181 |
| **0.80** | 0.8814 | 16 | 30 | 40 | 0.286 | 0.348 |
| **0.82**（best-acc） | 0.9068 | 5 | 4 | 51 | 0.089 | 0.556 |
| **0.84**（FP=0 最好点） | 0.9068 | 1 | 0 | 55 | 0.018 | 1.000 |
| ≥0.85 | 0.9051 | 0 | 0 | 56 | 0.000 | — |

多数类基线（全 keep）acc = 0.9051。`mapped_noul`（单段）在所有阈值下
**不超过多数类基线**（best-acc 0.9051 @ recall 0）；`1 - rerank` 最好点
0.9068 @ recall 3.6%。

分文件看（ctx 信号）：main 525 行 @t=0.82 acc 0.9905 / FP=3；@t=0.84 FP=0；
drop_supp 65 行 @t=0.80 recall 16/54=29.6% / FP=1——drop_supp 的 drop 段
ctx 分数 p50=0.224 与 keep 段 p50=0.263 大面积重叠。

## 6. 与内部基线对比（同 590 行口径）

| 模型 | acc | recall(drop) | FP |
|---|---|---|---|
| lora（内部） | **0.9453** | **62.5%** | 11 |
| winnow | 0.9151 | — | 7 |
| kev | 0.8927 | — | 34 |
| lora ∧ winnow | — | 17.9% | **0** |
| **Provence ctx（best-acc）** | 0.9068 | 8.9% | 4 |
| **Provence ctx（FP=0）** | 0.9068 | 1.8% | **0** |
| Provence 单段 / rerank | ≤0.9068 | ≤3.6% | ≤1 |

结论：已发布 Provence 权重零样本用作段级 irrelevant 判定，**弱于全部三个
内部模型**；其 FP=0 工作点 recall 1.8% 远低于 lora∧winnow 的 17.9%。作为
"外部候选基线"它证明了两点：(a) 任务式长 query + agent 段粒度 + zh 混合语料
的域迁移损失是真实且大的（设计上 T=0.1 的剪枝语义整域失效，kept_frac=1.0）；
(b) ±1 邻段上下文对 Provence 帮助显著（AUROC 0.593→0.765），提示单段孤立
打分本身丢弃了判定所需的局部上下文——与 PROVENCE_REPRO_PREP §5(b) 的粒度
风险一致。

## 7. 错误模式（≤10 词片段）

- FP @ctx0.80（30 个，全部 lang=multi）：多为中文任务指令/汇报段，如
  "你只需改动全局的codex的配置文档就可以"、"已按你的要求处理：- 撤销工作区新增的"——
  英语 checkpoint 对中文段系统性低分。
- TP @ctx0.80（16 个）：多为无关闲聊/跨主题段，如 "Hi! What are you working on
  today?"、"你可以联网搜索资料吗"、无关 plist/Chunk 工具输出。
- FN：被漏掉的 drop 多为**当前任务相关但应被替换/过期**的长 tool_result 与
  assistant 汇报段（如 replay_solution 相机链路分析）——Provence 判"与 query
  主题相关"而不是判"该段是否值得保留在上下文里"，任务语义本就有别。

## 8. 复现实命令

```bash
python3 -m pip install --target /tmp/provence_pydeps nltk   # venv 之外
NLTK_ALLOW_PROXIED_URLOPEN=1 python3 -c "import nltk; nltk.download('punkt_tab')"
PYTHONPATH=/tmp/provence_pydeps external/valen/.venv/bin/python \
  scripts/eval_provence_real_context_v1.py --variant a \
  --data data/real_context_eval_v1/eval.jsonl \
      data/real_context_eval_v1/drop_supp/eval.jsonl \
  --output results/provence_real_eval_v1.jsonl
# 变体 B 建议 --device cpu（MPS 上 remote code 存在内存泄漏）
PYTHONPATH=/tmp/provence_pydeps external/valen/.venv/bin/python \
  scripts/eval_provence_real_context_v1.py --variant b --device cpu \
  --batch 24 --infer-batch 8 --data <同上> --output <同上>
```

## 9. 保留意见

- 域外三重错位：QA 短 query → 任务式长 request；passage → agent 消息段
  （含 tool_result JSON/日志）；en-only → 84% multi（中英混合）。
- `provence_score` 是"段内最相关句"的聚合，对"整段是否 irrelevant"是间接
  代理；阈值的绝对数值不可与 noul 校准语义直接对读。
- query 截断 1200 字符影响 82 行（13.9%）；候选段 >512 token 时由 remote code 内部
  切块，长段尾部信息在 T=0.1 语义下仍参与句级判定但跨 block 无主从关系。
- 分数仅供基线对照；未做任何 threshold 拟合到生产策略（与
  CONTEXT_FILTER_THRESHOLD_POLICY 一致：只报 grid，不选生产阈值）。
