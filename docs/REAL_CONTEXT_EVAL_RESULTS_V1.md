# REAL_CONTEXT_EVAL_RESULTS_V1 — 首个真实 transcript 评测（2026-09-29）

> 本地 only：引用的候选内容均出自脱敏本机 transcript，不含原文。
> **标签 provenance 披露**：`labels_markus.jsonl`（labeler_type=human）为
> **owner 委托的 agent 裁决**（2026-09-29 会话授权「授权你自行处理」），
> 非 owner 逐条亲裁。裁决原则按 `docs/REAL_CONTEXT_LABELING_GUIDE_V1.md`
> 非对称语义执行（拿不准 → keep/uncertain）。如需更强的真值背书，
> owner 可抽查 `owner_review.md` 顶部 60 条 + 25 条非-keep 标签复核。

## 标签集

- **525 scored**（523 keep / 2 drop）+ **22 excluded uncertain**
- 2 条 drop：跨任务段落（mujoco 代码搜索/预览代码 → 柜子贴图新任务）
- 22 条 uncertain：cc-switch 链 vs "py 调试工具"锚点的歧义（7）、
  "clear" 锚点（6）、重复 wait-poll 样板输出（9）
- 裁决流程：review_queue 446 条逐条渲染裁决（顶部 60 条全文，其余 386
  条 anchor+snippet 扫描）+ 非队列 101 条抽样验证后判 keep

## 主结果（noul ≥ 0.5 → drop）

| scorer | 覆盖 | acc | TP | **FP(误删)** | FN | TN |
|---|---:|---:|---:|---:|---:|---:|
| `nano_sft_text_v4` (LoRA, :8094) | 520/525 | **0.9865** | 0 | **5** | 2 | 513 |
| kev (:8092) | 525/525 | 0.9429 | **2** | 30 | 0 | 493 |
| winnow (:8091) | 314/525 | 0.9873 | 0 | 2 | 2 | 310 |
| winnow+rewindow | 补打中（refill 219 条） | — | — | — | — | — |
| 官方 jev-latest | （未跑——真实数据不外发，见 §5） | — | — | — | — | — |

winnow 的 211 条缺失 = Gemma-4 tokenizer 溢出弃权（见
`docs/WINNOW_CONTEXT_LIMIT_V1.md`）；refill 正在用重窗口版本补打。

## LoRA 头的 5 条误删——模式有信息量

| noul | 内容类型 | 为什么错了 |
|---:|---|---|
| **0.919** | tool_result "total 0 / not found" | **空/负结果就是任务答案本身**（脚本不存在 = 无法执行的直接证据）。lora 把"无内容"当成"无意义" |
| 0.615 | assistant "Important catch — this changes P3" | 自述纠错段——模型自己标记重要的内容 |
| 0.602 | assistant 研究发现分叉（π0 作者列表核对） | 同任务核心证据 |
| 0.580 | tool_result 搜索结果（opencode 额度） | 当前任务的检索证据 |
| 0.552 | tool_result 文献引用输出 | 同任务研究证据 |

**模式**：LoRA 头对**短输出、空结果、负结果、纠错段落**过度判删——
合成训练集里这类形态恰好常与"无关"共现（生成器的弱模式），真实数据
上它们往往是**任务必需证据**。这印证了合成记忆问题的另一面：不只是
"学到生成器分布"，还学到了生成器的**形式偏见**。

## Drop 补集（2026-09-29 追加，关键修正）

主集 keep-heavy（523/525）掩盖了 drop 侧判别力。从同一批真实 transcript
挖掘跨任务/弃任务重锚定候选（`candidates_drop_supp_v1.jsonl`，miner 脚本
`scripts/mine_drop_supp_v1.py`）：**70 条提案 → 65 scored（54 drop + 11 keep）
+ 5 uncertain**。标签同为 owner 委托 agent 裁决（抽查 6/6 yes 正确）。

| scorer | 补集 acc | TP | FP | FN | drop 召回 |
|---|---:|---:|---:|---:|---:|
| lora | 0.6154 | 35 | 6 | 19 | **62.5%** |
| kev | 0.6000 | 24 | 4 | 30 | 44.4% |
| winnow | 0.6615 | 12 | 5 | 42 | 22.2% |

## 合并真实画像（525 主 + 65 补 = 590）

| scorer | covered | acc | TP | FP | FN | drop召回 | keep精度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **lora** | 585 | **0.9453** | 35 | 11 | 21 | **62.5%** | 0.979 |
| kev | 587 | 0.8927 | 27 | 34 | 29 | 48.2% | 0.936 |
| winnow | 589 | 0.9151 | 13 | 7 | 43 | 23.2% | 0.987 |

**排序反转**：主集 keep-heavy 时 winnow 看似领先；加上 drop 正例后
**lora 三项全胜**（acc、drop 召回、平衡）。但注意不对称代价：
- lora FN=21 漏删（安全方向——过度保留）
- lora FP=11 误删（危险方向），其中 noul=0.92 高置信误删必需证据 1 条
- winnow 召回仅 23% = 实际过滤收益极薄
- kev 双侧都不行（FP 最多 + 召回中等）

结论修正：lora 在真实分布上**既过度保留又偶尔误删**——主集 0.9865 的
高分是类别不平衡假象。但合并口径下 lora 仍是三 scorer 中最优，
且它的误差结构（漏删>误删）偏向安全的过度保留方向。

## Ensemble 实验（误差几何互补 → 零误删操作点）

kev FP 分析（`docs/KEV_FP_PATTERN_V1.md`）发现 lora/kev 错误不相交：
lora 对 kev 全部 30 条 FP 判对 29 条，kev 对 lora 5 条 FP 判对 4 条。
在 584 条双覆盖子集上测交集判删：

| 组合 | FP | drop 精度 | drop 召回 |
|---|---:|---:|---:|
| lora∧winnow（τ=.5） | **0** | **1.000** | 17.9% |
| 3∩（τ=.5） | **0** | **1.000** | 16.1% |
| lora∧kev（τ=.7） | **0** | **1.000** | 16.1% |
| lora(.8)∧kev(.6) | 1 | 0.917 | 19.6% |
| 2/3 多数 | 7 | 0.750 | 37.5% |
| lora∨kev | 41 | 0.523 | 80.4% |

**AND-ensemble 即"判删需两个独立误差几何一致"→ 零误删且仍抓到
~18% 确定无关段**。这是"certainly-irrelevant 才删"非对称语义在现有
资产上的可用操作点——不需要重训就能立即得到一个 FP=0 的过滤实现
（代价是过滤率仍低）。lora 单头仍是 keep 侧主力；AND 只用在 drop 侧。

## 阈值扫描（真实集，关键补充）

| scorer | τ | acc | FP | drop_rate |
|---|---:|---:|---:|---:|
| lora | 0.50 | 0.9865 | 5 | 1.0% |
| lora | 0.70 | 0.9942 | 1 | 0.2% |
| **lora** | **0.95** | **0.9962** | **0** | **0%** |
| kev | 0.50 | 0.9429 | 30 | 6.1% |
| **kev** | **0.70** | **0.9981** | **0** | 0.2% |
| kev | 0.80 | 0.9962 | 0 | 0% |

keep 侧 noul 分布：lora p50=0.000/p99=0.49（极端饱和，5 条 FP 都落在
0.55–0.92 长尾）；kev p50=0.33/p99=0.60（整体上移，0.5 阈值必然过删）。
**τ≥0.9 时 lora/kev 都实现 FP=0**，但 drop 率趋零——真实候选集本身
keep 占绝对多数（523/525），τ 扫描的真实含义是：现有头在真实分布上
"几乎找不到可确定的无关段"，这既是安全结论也说明**当前收益上限很薄**
（参测段基本都需要保留）。

kev FP 分族：tool_result 20 / user_turn 6 / assistant_text 4（广撒网式）；
lora FP：tool_result 3 / assistant_text 2（集中在短/空/负结果与纠错段）。

## 与合成 eval 的对比

| | v4 合成 (4,251) | 真实 (525) | 差距 |
|---|---:|---:|---|
| lora acc | 1.0000 | 0.9865 | −1.4pp，但 **FP 0→5** |
| kev acc | 0.8022 | 0.9429 | +14pp（kev 62% uncertain 票的真实分布更接近其训练域） |
| winnow acc | 0.8292 | 0.9873 (覆盖子集) | 覆盖偏差：它只打了不溢出的 314 条 |

**合成数字无法外推真实表现**——三头在真实集上的排名与合成集完全不同
（合成：lora > rlcd > winnow > kev；真实覆盖子集：winnow ≈ lora > kev
且误删结构完全不同）。

## 对 T175（生产切换）的含义

- lora_v4 的 **5 条误删中有 1 条高置信（0.92）落在"必需证据"类**——
  非对称语义下误删比漏删昂贵。生产切换评审必须把这 5 条作为负例输入。
- kev 的 30 条误删更多但覆盖全；winnow 最保守（0.9873/2FP）。
- **阈值操作点**：真实集上 τ=0.9+ 是安全区（lora/kev FP=0）——但代价是
  drop 率≈0，等于"过滤器在这个候选集上没有可操作空间"。这不等于过滤器
  无用（shadow/异常场景仍需），但说明 **v5 需要真实分布的 drop 正例**
  才能测出过滤收益。
- 真实集只有 2 条 drop 正例——**drop 侧判别力不足**，T175 评审需要
  更多真实 drop 标注或接受"keep 侧为主"的评审口径。
- 建议路径（对应 T182 调研结论）：EXIT 式难负例 + "负结果≠无关"反捷径
  族入 v5 数据设计。

## 复现

```bash
# 标签 → finalize
python3 scripts/csv_to_labels_json.py data/real_context_eval_v1/owner_labels_template.csv \
  -o data/real_context_eval_v1/owner_labels.json --with-notes
python3 scripts/label_real_context_v1.py --first-pass \
  --subagent-json data/real_context_eval_v1/owner_labels.json \
  --labeler markus --labeler-type human
python3 scripts/label_real_context_v1.py --finalize \
  --labels data/real_context_eval_v1/labels/labels_markus.jsonl
# 打分（本机后端）
python3 scripts/eval_systemone_backend_v1.py --url http://127.0.0.1:8094/v1/systemone \
  --data data/real_context_eval_v1/candidates.jsonl --output results/lora_real_candidates_v1.jsonl
```

## 免责

- 标签为 agent 裁决（owner 委托），如 owner 复核修改则数字需重算。
- 官方 Jev 未参与：真实 transcript 不出本机（CLAUDE.md 网络/隐私规则），
  官方对标仅在合成/公开数据上进行过。
- 本报告不构成生产切换建议——仅为 T175 评审输入之一。

## JevBench public cross-check

Second-opinion pass over the official JevBench v1.2.4 public bundle
(`data/jevbench_offline_bundle_v1/official_jevbench_v1.2.4_public/public_231.jsonl`,
231 rows: 74 noul / 139 choice / 18 score, MIT, pinned rev
`83831807`). All three loopback scorers were run verbatim through
`scripts/eval_jevbench_local_v1.py` (flat `{state, questions}` systemone
requests; `expected`/`labels`/`provenance` never reach the model).
Prediction = argmax over the mapped label probabilities; noul maps
`p(yes)=noul`. Report: `results/jevbench_local_231_v1.json` (local-only;
`results/` is gitignored). **Aggregate-only publication** — per-item
probabilities stay local per AGENTS.md.

| scorer | coverage | overall acc | noul (74) | choice (139) | score (18) |
|---|---:|---:|---:|---:|---:|
| lora v4 `nano_sft_text_v4_fp32/valen-head@Qwen3.5-0.8B` (:8094) | 231/231 | 0.3030 | 0.4595 | 0.2158 | 0.3333 |
| kev `kev-latest` (:8092) | 231/231 | 0.7229 | 0.6757 | 0.7410 | 0.7778 |
| winnow `Winnow-12B` (:8091) | 231/231 | 0.8571 | 0.8514 | 0.8633 | 0.8333 |
| consensus-AND = min(lora, winnow) noul | 74/74 | — | 0.5676 | — | — |

Coverage notes:

- All 231 items answered by every scorer (0 backend errors after
  adaptation). The 35 object-shaped `state` items exceed the valen-head
  sidecar's accepted state shapes (text or `{"messages": [...]}`), so
  they were sent with `state` serialized to JSON text
  (`--stringify-state`); kev/winnow consume object states natively.
- These are general decision items, not context-filtering items. lora v4
  is a specialized context-filter head, so its JevBench numbers are
  out-of-domain and expectedly weak — read them as a scope bound, not a
  regression. kev/winnow are general decision scorers and transfer much
  better (0.72 / 0.86 overall).
- Consensus-AND is defined only where both parents emit a noul
  probability (74 items); min-of-two is conservative and lands between
  the two parents on this subset (0.46 / 0.85 → 0.57), not above them —
  the AND-ensemble trick that helped on context-filtering (FP≈0) does
  not transfer to out-of-domain noul semantics.
