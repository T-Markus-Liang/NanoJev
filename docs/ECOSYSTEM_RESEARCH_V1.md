# ECOSYSTEM_RESEARCH_V1 — 开源上下文过滤生态调研（2026-09-29）

> 目的：为 v5 数据集设计 + 候选复刻方案搜集开源参照（"站在别人肩膀上"）。
> 生命周期 advisory: event `0a6d7452-2673-4536-88d4-93b60bb93754`。
> 调研方式：web_search/webfetch 广泛检索（LLMLingua 系、RECOMP、Selective-Context、
> Provence、EXIT、reranker 系、可借用 eval 套件、真实分布语料）。
> 声明：第三方许可证与数字为调研时点快照，复刻前以源仓库 LICENSE 为准；
> 本文档不授权任何部署或生产变更。

## 0. 生态发现（重要背景）

我们的对标家族本身是活跃开源生态——**多数可直接复用训练代码与冻结评测**：

- **Jev** = TypeSafe AI 托管 System One（闭源无权重）。契约：共享 `state` +
  并行 typed questions（`noul`/`choice`/`score`），~70–500ms，RLCD 校准。
  https://www.jevtypesafeai.com/jev/system-one · https://docs.rs/typesafe-jev
- **kev**（Jared Palmer + Devin）= Jev 架构的 Apache-2.0 开源复刻：LoRA r16 +
  pointer head，冻结 Qwen 底座（0.5B/0.8B/4B/9B/27B），**Apple MPS fp32 原生
  设计**，`/v1/systemone` 兼容，`--init_from` 迁移配方，冻结 eval manifests +
  checksums，公开 "trained vs new sources" 泛化表。
  https://github.com/jaredpalmer/kev · https://huggingface.co/jaredpalmer/kev-4b
- **Winnow-12B** = `EldanRing/Winnow-12B`（GGUF BF16/Q8 已发布）；
  JevBench 公开 231 项子集 ~85.3–85.7%（托管 Jev 1.13 为 85.7%）。
- **JEMM-27B** ≈ 很可能 `autotrust/JEV-27B`（六组均值 84.07 vs Jev 83.85）；
  未确证，待核对我们的 checkpoint 实际来源。
- **winnow（筛子）** = Ghaleb Dweikat 的 Claude Code hook：工具输出切 ~25 行块，
  每块一次 `noul` 判断，含错误检测覆盖 + 不确定即保留带（drop 阈值 0.1，
  公开 ECE 0.182）。https://madewithjev.com/builds/winnow ——**本质上就是
  NanoJev 要驱动的那个产品形态**。
- 公开评测：**JevBench**（231 项公开子集 + v1.4.2 榜单，`fstandhartinger/jevbench`）、
  kev 冻结套件（`decision-v7`/`transfer-v4`/`transfer-v9`）、OpenJev text、
  Nimble、VitaminC、MASSIVE-en。
- 架构参照：Archer Hume "Jev's Architecture Unmasked"——prefill-only 读出、
  隔离 question 分支、pointer-head 选项打分。

**kev 自己的数字量化出我们的记忆化问题**：kev-0.8B 在**新源**上 0.648/0.697，
在**训练源**上 0.827/0.838——同尺寸模型 train→transfer 差距 ~15–18pp，
与我们"lora_v3 在 v3 满分 / v4 只有 0.446"是同一现象。

## 1. 候选表

| 候选 | 链接/大小 | 许可证 | Mac/MPS？ | 段级 keep/drop 概率？ | 可复用训练数据？ | 公开数字 |
|---|---|---|---|---|---|---|
| **LLMLingua-1** | arxiv.org/abs/2310.05736；github.com/microsoft/LLMLingua；Llama-2-7B PPL | MIT(code) | 是（GGUF/MPS） | token 级自信息，非 query-aware | 否 | GSM8K/BBH 20× 压缩近乎无损 |
| **LongLLMLingua** | arxiv.org/abs/2310.06839；同 repo | MIT | 是 | query-aware token 分 + 文档重排；无校准概率 | 否 | NQ +17–21% @4× 压缩 |
| **LLMLingua-2** | arxiv.org/abs/2403.12968；mBERT-multi ~178M / XLM-R-L ~560M | code MIT；**权重 CC-BY-NC-SA** | 是 | 是——per-token `p_preserve`，可聚合到段 | **是**——`MeetingBank-LLMCompressed`（5,169 对）+ 标注工具 | LongBench 2–5×，比 v1 快 3–6× |
| **Selective-Context** | arxiv.org/abs/2310.06201；github.com/liyucheng09/Selective_Context | MIT(待核) | 是 | token/句级自信息，**非 query-aware** | 附带 eval 数据 | −50% ctx，BERTscore −0.023 |
| **RECOMP** | arxiv.org/abs/2310.04408；github.com/carriex/recomp | repo 公开（许可待核） | extractive 版可跑 | 否——输出摘要/空串 | 配方公开，数据可重建 | NQ/TriviaQA 摘要 ~6% 长度近无损 |
| **FILCO** | arxiv.org/abs/2311.08377；github.com/zorazrw/filco | repo 公开 | Flan-T5-XL/Llama-2-7B 偏重 | 否——生成式句过滤 | **是**——STRINC/lexical/CXMI 银标 | −44–64% 长度，6 任务优于 passage 过滤 |
| **Provence** | arxiv.org/abs/2501.16214；hf.co/naver/provence-reranker-debertav3-v1；code in github.com/naver/bergen | **CC-BY-NC-ND**（权重，受限） | 0.43B DeBERTa-v3 可跑；**512-token 上限** | **是**——句级二分类概率 + 可调阈值（0.1 推荐） | **是**——标注脚本 + MS MARCO 银标（Llama-3-8B 标注）经 bergen 放出 | 多域 ~0 性能损失 |
| **EXIT** | arxiv.org/abs/2412.12559；github.com/ThisIsHwang/EXIT；hf.co/doubleyyh/exit-gemma-7b | Gemma ToU | Gemma-2B/7B-it LoRA 可跑 | **是**——句级 P(yes) vs τ=0.5，文档条件 | **是**——HotpotQA supporting-facts 正例 + 同文档难负例 + 跨文档随机负例 | NQ/TriviaQA/HotpotQA/2Wiki 超过前代压缩器 |
| **AdaComp** | arxiv.org/abs/2409.01579 | — | 小预测模型 | 否——预测压缩率 k | 三元组配方 | 近等精度更低成本 |
| **ACon** | arxiv.org/abs/2510.00615 | — | LLM compressor | 否——抽象式压缩 agent 观测/历史 | 指南优化管线 | AppWorld/OfficeBench −26–54% 峰值 token |
| **DTOC** | arxiv.org/abs/2609.26121 | — | 框架（外部记忆+占位符） | 否 | 否 | DeepSWE +solve，−10–13% 输入 |
| **winnow sieve** | github GhalebDweikat/winnow | 开源 | 是 | 取决于 judge（每块 `noul`）；带 Jev + Haiku-4.5 适配器 | 手工标注 replay bins 引用但未明确发布 | ECE 0.182；公开校准带 |
| **Qwen3-Reranker-0.6B** | hf.co/Qwen/Qwen3-Reranker-0.6B | Apache-2.0 | 是（社区 GGUF；另有 4B/8B） | 文档级 yes/no logits→0–1 | Apache-2.0 模型 | MTEB-R 65.80，0.6B 最强 |
| **bge-reranker-v2-m3** | hf.co/BAAI/bge-reranker-v2-m3 | MIT | 是 | 是——sigmoid pair score | 否 | BEIR 56.51 |
| **jina-reranker-v3** | hf.co/jinaai/jina-reranker-v3 | **CC-BY-NC** | 0.6B Qwen3 底，listwise | 是 | 否 | BEIR 61.94 |
| **gte-multilingual-reranker-base** | Alibaba mGTE | Apache-2.0 | 是 | 是 | 否 | MTEB-R ~59.5 |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | hf.co/cross-encoder/... | Apache-2.0 | 是——22M 参数 ~1800 docs/s | 是 | MS MARCO 真实 Bing 查询 | TREC-DL19 NDCG@10 74.3 |
| **kev (0.5B–27B)** | github.com/jaredpalmer/kev | **Apache-2.0** | **是——MPS fp32 原生** | **是——校准 noul 概率，就是我们这个任务** | **是——训练代码、eval manifests、`--init_from` 迁移** | kev-27B 新源 0.848 vs Jev 0.857 |

## 2. 最值得本机复刻的 Top-3

**1. ModernBERT 底座上的 Provence 配方。** 与 NanoJev 工作最接近的已发表
方案：查询条件下的句级二分类序列标注 + reranking 头统一 + 阈值可调
（保守剪枝用 0.1）。完整训练管线（retrieval+rerank → LLM 银标 → seq-label
训练）在 `naver/bergen/scripts/provence` 发布。权重 CC-BY-NC-ND 不能分发，
但配方可整体迁移到 **ModernBERT-large（Apache-2.0, 396M, 8192 ctx）**——
同时解决 Provence 的 512-token 上限和许可证问题。llama.cpp 已支持
`ModernBertForSequenceClassification`（GGUF 可行）；MLX 训练器存在。

**2. EXIT 的句级决策配方。** 小 decoder（Gemma-2B-it LoRA）输出每句归一化
P(yes)，条件 = 查询 + 全文；标签 = HotpotQA supporting-facts 正例 + 同文档
非支撑句难负例 + 跨文档随机负例——**真实人工标签，不是生成器伪影**。直接
可译：Qwen3-0.6B/0.8B LoRA，把维基句子换成 agent transcript 段。

**3. LLMLingua-2 的数据蒸馏 + 标注管线。** `MeetingBank-LLMCompressed` 数据、
token 级标注工具、"压缩 = 二分类 token 标注"范式全公开。复刻**管线**（而非
NC 权重）到自己域：教师压缩 transcript 对 → 对齐 → 段级标签 → encoder 头。
注意：LLMLingua-2 是任务无关的（无查询条件）——需自己加请求条件化，这正是
Provence 的创新点。

**荣誉提及——kev 训练仓库**：NanoJev 属 Jev 族，`jaredpalmer/kev` 就是现成
Apache-2.0 脚手架（block-causal 分支掩码、pointer head、`--init_from`、
冻结 eval manifests）。即使保留自有架构也值得研读。

## 3. 可借用 eval / 数据集

**直接 keep/drop 真值：**
- **HotpotQA / 2WikiMultihopQA / MuSiQue supporting-facts**——句级"问题需要"
  人工标签，免费二分类真值。
- **Provence 银标**——Llama-3-8B 标注的 MS MARCO 句标签（bergen 管线）。
- **FILCO 过滤上下文**——STRINC/lexical/CXMI 句级银标，6 任务。
- **MeetingBank-LLMCompressed**——GPT-4 token 级 keep/discard（可聚合到段）。
- **RGB**（github.com/chen700564/RGB）——doc 级 positive/noise/negative 切分，
  专为噪声鲁棒设计。
- **MS MARCO**——真实 Bing 查询→段落相关性。

**可改造的长上下文/会话 eval：**
- **LongMemEval**——历史聊天项中哪条对当前问题必需；与"会话段相关性"最接近。
- **HELMET**（MIT, princeton-nlp）——RAG + passage-rerank 子集带真值。
- **LongBench / LongBench v2 / ZeroSCROLLS / LooGLE**——上下文+问题；部分子集
  可推导 gold-supporting spans。
- **RULER**（Apache-2.0, NVIDIA）——合成 needle 任务；可作"needle=keep,
  distractor=drop"但同样有合成记忆风险。
- **LOFT**（Apache-2.0 code / CC-BY data）——35 数据集含检索任务，1M+ ctx。
- **JevBench 公开子集（231 项）** + kev 冻结套件——与既有对标数字直接可比。

**v5 真实分布语料（见 §4）：**
- **SALT-NLP/SWE-chat**——真实开发者 Claude Code/Codex/Gemini CLI 会话全程：
  transcript、tool call、tool result。工具输出段的黄金来源。
- **nebius/SWE-rebench-openhands-trajectories**（67k）、
  **nvidia/SWE-Zero-openhands-trajectories**（318k，宽松许可 repo）、
  **SWE-Gym/OpenHands-Sampled-Trajectories**——真实工具输出；带 `resolved`
  标记 + gold patch → 可推导哪些文件 Read/Grep **真的被用上**。
- **TOUCAN-1.5M**（Apache-2.0）——真实 MCP 工具执行，495 个 server。
- **WildChat-1M**（ODC-BY）——100 万真实用户会话含话题切换；
  **LMSYS-Chat-1M**（需申请许可）。
- **Salesforce/xlam-function-calling-60k**（CC-BY-4.0，gated）——工具调用
  结构多样性。

**去重工具：** github.com/ChenghaoMou/text-dedup（Apache-2.0：MinHash+LSH、
SimHash、后缀数组、RETSim）+ SemDeDup 嵌入聚类去重。

## 4. 给 v5 设计的建议

"合成生成器会被记忆"已被 kev 公开表实证（0.8B：训练源 0.83 vs 新源
0.65–0.70）。照此设计：

1. **按源切分而非按条切分。** 泛化失败的基本单元是*生成器/域*：整语料留出
   （如 train = SWE-chat + HotpotQA，eval = OpenHands 轨迹 + WildChat），
   对齐 kev 的 "trained vs new sources" 报告口径。
2. **正例锚定结果推导标签，而非教师意见。** 最佳免费信号：SWE 轨迹带
   `resolved` + `reference_patch`——内容命中 gold patch 的文件 Read/Grep
   （或被修复的失败测试输出）是"agent 实际用到"的硬 keep 标签，**真实分布
   真值，任何生成器造不出来**。
3. **真实分布负例比聪明正例更重要：** 同任务被取代的陈旧工具输出（文件改后
   的旧 Read）、上个任务残留块、主题相似但未用的搜索命中（EXIT 的同文档
   难负例模式）、重复内容（SemDeDup 聚类）、"看起来相关但事后无用"段。
4. **保持非对称规范。** Winnow 的设计是正确目标：类别 = *"certainly
   irrelevant"*（drop 阈值 0.1、uncertain→keep、含错内容必保留）。按非对称
   代价训/评——保留一个无关 token 成本几分钱，删了必需段毁整个任务。报告
   ECE + recall-at-low-p（像 winnow 的公开校准带）。
5. **多粒度 + 反捷径特征。** 混句/块/轮/工具结果粒度；剥离生成器指纹（句式
   模板、长度先验）；带标签噪声和中带 uncertain 样本，让 p 校准而非双峰——
   winnow 发现 0.1–0.2 带实际有 39% 是需要的。
6. **train↔eval 激进去重**（MinHash + 嵌入去重）——近重复记忆是分数的
   隐形注水器。
7. **教师标签做 ensemble：** Provence 式 Llama-3/8B 银标 + FILCO STRINC/CXMI
   启发式作弱标签特征，再与结果推导标签调和（§3），不迷信单一标注源。

## 5. 不做的事（死胡同及原因）

- **Soft-prompt 压缩器**（ICAE、Gist Tokens、AutoCompressor、500xCompressor、
  Activation Beacon、CEPE）：输出绑定特定 decoder 的隐向量，无 keep/drop
  概率，训练重。做错的问题。
- **纯自信息方法**（Selective-Context、LLMLingua-1 原味）：保留"意外"token
  而非"任务相关"token——构造上 query-blind，信息量大的错误日志一律 keep。
- **LLMLingua-2 权重直接用**：CC-BY-NC-SA（非商用）+ 任务无关——测冗余不
  测相关。只有配方价值。
- **抽象式压缩**（RECOMP-abstractive、CompAct、ACon 式摘要）：改写而非标注；
  无段级概率；增加生成延迟与幻觉风险。
- **纯语义去重**（SemDeDup/text-dedup）：正交——压缩跨条目冗余，与"对当前
  请求是否相关"无关。可作特征/管线环节，不作 scorer。
- **通用 reranker 当 NanoJev**：发 pair 相关性分，但训的是检索排序（"主题
  相关"）非"任务必需"；分数未按非对称 drop 语义校准。可当弱标签教师/eval
  基线，不当最终 scorer。
- **托管决策 API**（Jev、Anthropic context-editing `clear_tool_uses_20250919`）：
  Jev 正是 NanoJev 要替代的（无权重、仅托管）；Anthropic 的工具结果清理是
  私有服务端启发式——设计参考，非可复现基线。
- **Provence 权重分发**：CC-BY-NC-ND。可作 baseline 评测、可抄配方；不可
  微调/分发。

## 6. 邻近方向参照：Naive-N0.5-Flash（2026-09-27 发布，非候选）

> 非 keep/drop 候选；记录它是因为其核心技术验证了我们路线，且定义了
> "模型内部上下文选择"这条平行路线。

**事实**（HF `NaiveAI/Naive-N0.5-Flash` + GitHub `NaiveAI-Labs/Naive-N0.5-Flash`，
均 2026-09-27 创建，MIT）：

- **309B MoE / 15.5B 激活**，48 层，256 routed experts × top-8，hidden 4096，
  vocab 152576（MiMo 词表）。基座 = 小米开源 `MiMo-V2.5-Base`，非从头训。
- **零全注意力层**：39 SWA（窗口 128 token）+ 9 DSA（DeepSeek Sparse
  Attention，indexer 16 头从全历史选 top-2048 token），8×六层模块 5:1 布局，
  GQA4，原生 1M context。
- 训练：续训 3.25T tokens（50B indexer warmup + 3T sparse-attention CPT +
  200B decay）+ SFT。配套 NaiveRT 推理（mega-kernel + PDL + 投机解码，
  `-FP8-Draft` 草稿模型已在 HF）；API 定价宣称 $0.10/$0.40/$0.01 per M tok。
- "Built with AI" 叙事：AI 自主做架构探索与 infra 优化，人做关键决策——
  与 Valen/递归 R&D 同一故事线。
- 第三方同 harness 对比（Claude Code）：DeepSWE 67.8 vs DeepSeek-V4.1-Flash
  69.8；Terminal-Bench 2.1 86.7 vs 88.0。官方对比图里竞品分数抄自对方
  model card 的最优 scaffold，差距被放大——**自己复述时要用同 harness 数字**。
- 需 `trust_remote_code`（`modeling_naive_n05_flash.py`）；49 shard BF16
  ≈600GB，本地不可跑，仅 API/FP8 路径可用。

**与 NanoJev 的关系——同构技术验证，非竞品：**

- DSA indexer = **模型内部的可学习上下文选择器**：小打分器扫全历史、
  选 top-k 进 backbone。概念上与 NanoJev context gate 同一件事，区别在位置：
  内部选 token 只省 attention 计算（KV cache 仍全量保留、仍全扫历史）；
  外部过滤省真实输入 token → 省成本 + 延迟 + 隐私，且对黑盒 API 模型有效。
- FlashMemory（karminski3 转发）同路数：索引器当双编码器单独训练、不加载
  基座——与我们"独立小模型做上下文打分"的训练方法学同构。**前沿厂商把
  "学到的上下文选择"当正经方向在做，说明 NanoJev 的技术路线成立。**
- 战略含义：若 1M context + 稀疏注意力普及，外部过滤的卖点从"模型装不下"
  转为"**没必要付这个钱**"+ 风险/隐私过滤 + 小模型前置路由。pitch 与
  benchmark 设计需按此口径。
- 潜在用途：main-model/worker 候选（对标 AGENTS.md 指定的 DeepSeek-V4.1-
  Flash），不是 decision primitive；API 价格出来后值得做同 harness 对比。

**待核实：** NaiveAI 与 karminski3 的关联为推测（时间与评测风格吻合，无公开
归属证据）；官网 naive.ai 自报 benchmark 细节、NaiveRT 是否开源均待确认。

## 待核实 / 免责

- "JEMM-27B" 身份为推断（大概率为 `autotrust/JEV-27B`）；核对实际 checkpoint。
- RECOMP、EXIT repo code、Selective-Context 数据、MeetingBank 衍生品的许可证
  未完全核实——再分发前查各 repo LICENSE。
- Qwen3-Reranker GGUF 经社区渠道存在，但 llama.cpp 的 rerank 模式不如
  embedding 标准——上马前验证；MLX/transformers 路径安全。
- Provence 银标经 bergen 管线分发（脚本+中间产物），非单一干净 HF 数据集——
  预留提取工作量。
- jina-reranker-v3 为 CC-BY-NC——仅可 eval。
