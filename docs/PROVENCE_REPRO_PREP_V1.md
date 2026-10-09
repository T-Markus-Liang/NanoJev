# PROVENCE_REPRO_PREP_V1 — Provence 配方复刻前置调研（侦察 + 本地 smoke，无训练）

> 对应 `docs/ECOSYSTEM_RESEARCH_V1.md` §2 top-1："ModernBERT 底座上的 Provence 配方"。
> 范围声明：本文档只做配方提取、许可证核对、**本地推理 smoke（不训练）**、移植映射与工作量
> 估算。未做：任何训练、GPU 服务器、对 `external/valen/.venv` 的安装改动、对现有仓库脚本/
> 数据的改动。smoke 脚本写在 `/tmp/provence_smoke.py`（仓库外）。
> 日期：2026-09-29。生命周期 advisory: event `ea58a381-82b1-448a-a5e4-c10757cab106`
> （decider 对"下一步"弃权 out-of-scope，按规则独立决策）。

## 1. Provence 精确配方（来源核实：论文 HTML + bergen 代码 + HF 卡）

来源：arXiv:2501.16214v1（ICLR'25）；`github.com/naver/bergen/scripts/provence/`
（`gen_silver_labeling_provence.py` / `modeling_provence.py` / `train_provence.py` /
`readme.md`）；`hf.co/naver/provence-reranker-debertav3-v1`（模型卡 + 仓库内
`modeling_provence.py` 推理代码）。

### 1.1 银标生成（三步管线）

1. **建 datastore + 检索**：MS MARCO doc 集合（370k 训练 query；文档切成 N∈1..10
   随机句数、偏向长段的 passage，前置页标题 → 34M passages）。SPLADE-v3 检索 top-50
   → `naver/trecdl22-crossencoder-debertav3` 重排 → 每 query 取 **top-5** passage。
   NQ（87k query）同流程，最终模型用两者混合。
2. **LLM 银标**：`meta-llama/Meta-Llama-3-8B-Instruct`（readme 注明实际用 3.1 版；
   vLLM，temperature=0 贪心，max_new_tokens=256，max_model_len=2048，fp16 + fp8 KV）。
   passage 用 `nltk.sent_tokenize` 切句，渲染为 `[i] sent_i`。默认 **answer oracle**
   prompt（§消融中优于 relevant/straightforward）：

   > "Answer the Question, using ONLY information provided in the Context. If no
   > useful information is provided, you MUST output 'No answer'. If some parts of
   > the Context are used to answer, you MUST cite ALL the corresponding sentences.
   > Use the symbols [ ] ... e.g [0] for a fact from sentence 0."

   解析规则（`gen_silver_labeling_provence.py`）：响应含 "No answer" → 空选择；
   否则 regex `\[([\d, ]+)\]` 抽引用下标；解析失败或无引用且无 "No answer" →
   **丢弃该样本**（论文：~90% 情况 LLM 给出引用，过滤的是"答了但忘引用"的噪声）。
   产出：每 (query, passage) 一个 JSON：{query, context(句列表), selected_sents,
   response}。公开下载：readme 给出 MS MARCO / NQ 两份已标注数据 + 对应 .trec
   run 的 Google Drive 链接（几个 GB，可跳过 step 1–2 直接训练）。
3. **训练**：见 §1.2。

### 1.2 模型与损失（`modeling_provence.py`）

- 底座 DeBERTa-v3-large（0.43B，512 ctx）。两类初始化：
  standalone 从 `microsoft/deberta-v3-large`；**unified 从已训练 cross-encoder**
  `naver/trecdl22-crossencoder-debertav3`（排序头保持，剪枝头随机初始化——注意
  排序层必须叫 `classifier` 以对齐 checkpoint 命名）。
- 双头：`classifier = Linear(pooler_out, 2)` 取 logit[0] 为 **ranking score**
  （pooled [CLS]/BOS）；`token_classifier = Linear(hidden, 2)` **逐 token 二分类
  keep/drop**。
- 输入打包：`[CLS] query [SEP] context_sent_tokens... [SEP]`。句级标签广播到该句
  全部 token；query/special/pad 位标签=0。CE 作用于全序列。
- 损失：token CE + λ·MSE(ranking_score, teacher_score)，**λ=0.05**（MS MARCO dev
  上"不显著损伤重排"的最小值）；teacher score = 初始 reranker 对同 pair 的分数
  （pointwise 蒸馏当正则）。
- 超参：lr 3e-6，batch 48，**1 epoch**；全量 MS MARCO 一轮在 A100 级 GPU 数天，
  readme 说明 1/10 数据也有好结果。

### 1.3 推理与阈值语义（HF 仓库 `modeling_provence.py`）

- `process(question, context, threshold=0.1, always_select_title=False, ...)`；
  query 先做 SQuAD 式 normalize（小写/去标点/空白修复——与 MS MARCO 训练分布一致）。
- 逐 token softmax → P(keep)，`token_pred = p > T`；**sentence_rounding**：句内
  keep-token 占比 > T 才整句保留（占比阈值复用同一个 T，注释里 threshold=0.5 是
  论文叙述、代码里用传入 T；`always_select_title` 在"至少选中一句"时强制保留首句
  标题）。
- 输出 `{pruned_context, reranking_score, compression_rate}`。
- **阈值语义**：T=0.1 = 保守剪枝（高召回、近乎零性能损失，官方推荐默认）；
  T=0.5 = 更高压缩；同一 T 在 7 个数据集上迁移良好，压缩率随数据自动在 50–80%
  浮动。**这正是我们要的"certainly irrelevant 才 drop"语义**：低阈值=只删确信
  无关句。
- 超长 context 按 max_len(512) 切成多个 block，各带同一 query 前缀，逐 block 处理
  后按句坐标拼接。

### 1.4 评测协议

BERGEN 库；7 数据集跨域（NQ/HotpotQA/TyDi/PopQA 用 Wikipedia datastore；
BioASQ/Pubmed、SyllabusQA、RGB 各自域）；SPLADE-v3 + DeBERTa-v3 top-5 → 剪枝 →
Llama-2-7b-chat 生成；**LLM-as-judge（SOLAR-10.7B）+ match/recall** 对短答案；
报告压缩率 vs 质量 Pareto。附加分析：needle-in-haystack 位置鲁棒性（首尾位置弱，
因训练数据分布）、选中句数 vs 银标 oracle 一致性、2/6/10 句与 100 词粒度鲁棒性、
rerank 保持度（MRR/nDCG/BEIR 13 集均值 55.4→55.9）。

消融结论（对我们的设计直接有用）：token 级标签 + 句级 rounding ≥ 句级单点标签；
answer oracle > relevant oracle > straightforward oracle；数据量大有帮助
（370k > 87k 等量）；standalone ≈ unified 质量（unified 只为省一次前向）。

## 2. ModernBERT 事实核查（`answerdotai/ModernBERT-large` 卡）

- ModernBERT-large：28 层 **395M** 参数；base：22 层 **149.6M**（smoke 实测）。
- 原生 **8192 token** 上下文（RoPE + local-global alternating attention +
  unpadding/FlashAttention），预训练 2T token 含代码。
- **License: Apache-2.0**（large/base 同）——解决 Provence 权重 CC-BY-NC-ND
  不可分发问题。
- transformers ≥4.48 支持；本仓库 `external/valen/.venv` 为 **transformers 5.4.0
  + torch 2.6.0（MPS 可用）**。可用头：`ModernBertForTokenClassification` /
  `ModernBertForSequenceClassification`（已验证 import）。注意 ModernBERT
  **无 token_type_ids**。
- GLUE 88.4（base）/large 仅次 DeBERTa-v3-large；BEIR ColBERT 51.3 同尺寸最强。

## 3. 本地 smoke 结果（推理，无训练）

脚本 `/tmp/provence_smoke.py`（不进仓库）。用 valen venv 直接跑（5.4.0 足够新，
**无需新建 venv**）；HF 直连可达（huggingface.co 200/2.1s；hf-mirror.com 不通，
未使用）。模型 `answerdotai/ModernBERT-base`（149.6M）。

| 项 | 结果 |
|---|---|
| 下载+加载 | OK（直接 HF；分类头 MISSING 属预期——随机初始化） |
| Provence 式打包 `[CLS] q [SEP] segs [SEP]` | OK，87 tok 3 段 agent 风格样本前向正常 |
| 逐 token keep 概率 → 句均值聚合 | OK，P(keep)∈[0.54,0.65]（随机头，仅验证管线） |
| 长上下文 | 60 段 1159 tok 一次前向正常（DeBERTa/Provence 需切 3 block） |
| CPU 稳态 | 322.7 ms / fwd @1159 tok ≈ **3.6k tok/s** |
| **MPS 稳态** | **50.6 ms / fwd @1159 tok ≈ 22.9k tok/s** |
| 冷启动首调 | CPU 756 ms / MPS 467 ms @87 tok（含编译/加载路径） |

**Smoke 判定：PASS。** Mac 本机 MPS fp32 跑 Provence 式 seq-label 前向完全可行；
~1.2k token 上下文单次约 50ms，上百段的 transcript 大概率仍在单 block（8192）内。
ModernBERT-large（395M）预计 MPS 约 3–4× 慢、仍 <1s/次，可选。

## 4. 移植映射（Provence → NanoJev 契约）

| Provence 概念 | NanoJev 等价物 | 备注/差异 |
|---|---|---|
| question | `state` 中的**当前 user request**（`user_messages_in_order[-1]`，可拼 verbatim `noul` instructions） | Provence 是短 QA query；我们是任务式请求，更长更脏，normalize 需改成不去标点或保留原文 |
| context passage | `state.conversation` 全量 segment | Provence 单 passage ≤512；我们 transcript 可数千 token → 8192 ctx 覆盖 |
| sentence（nltk 切分） | **candidate segment**（pointer 指定的 message/tool_result 片段） | 粒度更粗且带结构；多行 tool output 可能需要子切分或保持整段一个候选 |
| per-token CE + sentence rounding | 逐 token/逐 segment keep 概率 → 段均值 | 我们的 segment 边界由 pointer 给定，不必 nltk；token 级方案照旧适用 |
| keep prob + 阈值 T | `questions.irrelevant` 的 `true` 概率 = **1 − P(keep)** | 语义天然对齐：低 T = "只删确信无关" = "certainly irrelevant" |
| 银标（Llama-3-8B answer+cite） | 三条路：(a) valen v3/v4 生成器标签（纯函数、已有~17k 记录）；(b) real_eval owner 人工标签（保守、~数百条）；(c) **本地 teacher**（Qwen3-8B/14B via mlx-lm，answer+cite prompt 原样移植） | (a) 现成但每记录单 candidate，需 adapter 把同 group 记录合并成 (request, 段列表, mask)；(b) 量小只能 eval；(c) 最接近原配方——注意 **provider/Jev 输出永远不得进训练标签**（红线），只能用本地开源 teacher |
| reranking 头 + λ·MSE 蒸馏 | 可选：无现成 reranker teacher；若做，用本地 cross-encoder（如 bge-reranker-v2-m3，MIT）当 teacher | 不需要可砍——我们只训 standalone compression 头，λ 项归零 |
| retrieval top-5 多样性 | transcript 内自然噪声 + 生成器 hard-negative families | 无需检索步；v4 families（tool_pairing/topic_shift 等）直接提供难度结构 |
| MS MARCO 370k×5 = ~1.85M pair | 我们量级 ~5–20k (request, transcript) pair | 数量差两个数量级；靠 base-size + 1 epoch + 冻结底层可缓解 |
| T=0.1/0.5 双工作点 | 现有 abstention grid {0.5..0.99} + CONTEXT_FILTER_THRESHOLD_POLICY | 报告照旧按 grid 重读，不选生产阈值 |
| BERGEN LLM-eval | real_context_eval_v1 指标族（acc/NLL/Brier/ECE/per-family/CW@0.9）+ 压缩率 | 端到端"删后任务质量"评测需要额外 harness（见 §5 缺口 d） |

## 5. 缺口与风险

- (a) **域标签**：Provence 银标来自 MS MARCO/NQ + Llama-3-8B；我们的 agent 域
  有 v3/v4 生成器标签（结构性强、覆盖真实转录弱）与 real_eval 人工标签（量少）。
  最忠实复刻需要本地 teacher 对真实 transcript 银标——**必须用本地开源模型**
  （mlx-lm Qwen3-8B/14B 或 Llama-3.1-8B），provider/Jev 输出禁用。
- (b) **粒度**：Provence 句级 vs 我们 segment 级。契约本身每请求只问一个
  candidate；Provence 式是**一次前向给全部段打分**——更高效，但打分粒度从
  "单 candidate 概率"变成"段序列掩码"，与 `/v1/systemone` 出参形状的桥接要新建
  （段 keep prob → 每 candidate 一题批量构造或直接改 scorer 出多段概率）。
- (c) **打包差异**：valen `state` 是含 `candidate_pointer` 的序列化 JSON，
  不是 query+context 两段；Provence 输入要求 query 短而 context 是段序列。
  需写 state→(request, segment list, span 坐标) 的 adapter（纯解析，~百行）。
- (d) **端到端 eval**：Provence 量"删后 QA 质量"，我们只有段级 keep/drop 真值 +
  abstention 指标；下游主模型任务质量评测是独立工作（CONTEXT_FILTER_VALUE 系）。
- (e) **尾部位置弱点**（论文 needle 实验）：句 0/末句召回差——我们的 candidate
  常在尾部（最近工具输出），需在训练数据构造时保证位置分布均匀。
- (f) 权重不迁移：一切从头训；英语 only 配方，zh 数据需自证。

## 6. 可复用 vs 必须新建

**可直接复用/改写：**
- bergen `scripts/provence/*`：银标 prompt + 解析规则、`token_classifier` 头结构、
  `sentence_rounding`、训练循环骨架、TREC/验证脚本模式。
- 已公开银标数据（MS MARCO+NQ Drive 包）可作域外预训练/ sanity 对照。
- `transformers` 的 `ModernBertForTokenClassification` + valen venv（5.4.0/torch
  2.6/MPS）就地可用；llama.cpp 对 `ModernBertForSequenceClassification` 已支持
  （GGUF 部署路径成立，待验证 token-classification 导出）。
- NanoJev 侧：`data/valen_nano_v3/v4` 标签契约与一致性校验器、real_eval 协议与
  指标族、winnow_calibration ECE、候选 pointer/segment 结构。

**必须新建：**
1. `state` → (query=末轮请求, segments+span 坐标, per-段 label) adapter（合并同
   group 的 per-candidate 记录为一条 Provence 样本）。
2. ModernBERT 版 `modeling`：骨架 + `token_classifier(Linear(hidden,2))`（可选
   ranking 头砍掉）——`modeling_provence.py` 直译，~150 行。
3. 训练脚本：CE over context token 位、lr 3e-6 / bs 视 MPS 内存（base 可 16–32，
   large 8–16，必要梯度累积）、1 epoch、MPS fp32（对齐仓库 fp32 参考约定）。
4. 本地 teacher 银标脚本：answer+cite prompt 原样 + transcript 段编号 `[i]` +
   同套 regex/丢弃规则（vLLM 不可用→mlx-lm 或 transformers generate）。
5. 评测桥：段掩码 → noul 概率映射 + 接入现有 eval.jsonl/abstention 报告。

## 7. 工作量估算（全复刻 = 数据标注 + 训练 + eval harness）

| 阶段 | 内容 | 估时（人日） |
|---|---|---|
| E0 | adapter + modeling 移植 + 单元 smoke（本机 MPS） | 0.5–1 |
| E1 | 合成数据版最小复刻：v3/v4 标签 → Provence 样本（~15k pair），ModernBERT-base 训练 1 epoch（MPS 数小时），eval.jsonl 指标 | 1–2 |
| E2 | 本地 teacher 银标管线 + 真实 transcript 标注（含 redaction/授权流程）：teacher 推理 ~5–10k transcript，M 系列约 1–2 天机时 | 2–4（含协议审批） |
| E3 | eval harness + 与 valen 头/winnow shadow 对照、位置/粒度鲁棒性测试 | 1–2 |
| E4 | ModernBERT-large 升级 + 阈值扫描 + 文档化 | 0.5–1 |
| **合计** | 到"可对照的复刻结论" | **5–9 人日**；E0+E1 的 MVP 仅 **1.5–3 人日** |

机时估算：ModernBERT-base MPS ~23k tok/s 前向；训练按 ~3× 前向成本、~15k 样本
× ~1k token × 1 epoch ≈ 15M token → MPS 上 **小时级**；large ~3–4×。无需 GPU
服务器即可完成 MVP；E2 若扩规模再上 A100 也只是加速器时，不是阻塞项。

## 8. Go / No-Go 建议

**GO —— 建议立项，按 E0→E1 先做 MVP（合成标签最小复刻），再决定是否投 E2 真实
teacher 银标。** 理由：

- 配方完整公开且每一步可核实（脚本、prompt、超参、数据下载）；
- ModernBERT-large（Apache-2.0、8192 ctx）恰好补齐官方权重两个硬伤（NC-ND 许可、
  512 上限），且本机 MPS 推理已实测可行；
- 任务同构度高：低阈值 keep-概率语义 = 我们 "certainly irrelevant 才删" 的保守
  语义；逐段概率天然输出我们需要的 per-segment drop prob；
- MVP 成本低（1.5–3 人日），不依赖 GPU 服务器、不碰生产路径。

**主要保留意见**：(i) 真实域标签仍是最大不确定项——合成标签 MVP 只能证明管线
成立，不能证明真实 transcript 上的质量，E1 结论须按 "synthetic-only" 口径写；
(ii) 单 candidate 契约 → 全段掩码的接口改造意味着新 scorer 形态，需要与现有
`/v1/systemone` 兼容层另行设计；(iii) Provence 论文已知首/尾句弱点要在我们的
数据构造时主动对冲。

## 9. 引用与凭据

- arXiv:2501.16214v1 HTML（§3 方法、§4.1 超参、§4.4 消融、App A/B 数据与模型表、
  Table 6 银标 prompt）。
- `github.com/naver/bergen/scripts/provence/`：`gen_silver_labeling_provence.py`
  （teacher=vLLM Llama-3-8B-Instruct、三种 prompt、regex 解析与丢弃规则）、
  `modeling_provence.py`（双头结构）、`train_provence.py`（打包/标签广播/loss）、
  `readme.md`（三步管线 + Drive 数据包）。
- `hf.co/naver/provence-reranker-debertav3-v1`：卡 + 仓库内推理代码
  （`process`/`sentence_rounding`/block 切分）；License CC-BY-NC-ND（权重不可分发）。
- `hf.co/answerdotai/ModernBERT-large`：395M/8192/Apache-2.0；base 149.6M。
- 本机 smoke：`/tmp/provence_smoke.py`，valen venv transformers 5.4.0 + torch
  2.6.0，MPS fp32，输出见 §3 表。
