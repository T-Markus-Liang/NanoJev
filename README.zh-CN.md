# NanoJev — A nano replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**简体中文** | [English](README.md)

**一个 0.6B 并行决策模型：输入状态与问题，直接得到完整概率分布，无需生成答案 token。**

[模型](https://huggingface.co/C-Tianyu/NanoJev) · [数据集](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)

## NanoJev 实测记录

公开仓发布 NanoJev 指标，以及固定、可披露协议下的聚合对比结果。第三方模型原始输出
和私有逐条收据保留在本地，不进入公开源码树。

### 找到出口：50×50 迷宫

模型判断四个局部方向是否可通行；代码记住碰撞、探索未知边，并沿已经走通过的
路径重新定位。NanoJev 在记录运行中以 **244 次行动、36 次碰撞**到达目标。

### 持续成长：12×12 贪吃蛇

共同规划器先排除立即碰撞的动作，再寻找通向当前食物的静态路径。模型在剩余候选
之间选择；仅剩一个候选时由代码直接执行。在记录运行中（**种子 61005**、贪心控制），
NanoJev 在 **256 步内吃到 27 个食物**，并在评测上限时保持存活。

## 本地决策 benchmark

仓库包含一个冻结的 **1,720 行 / 893 个有标签样本** 离线 bundle，覆盖 official JevBench public、SemIf authored/perturbation/shape、WANLI 和 Every retrieval。所有运行仅用于评测；公开内容只包含聚合指标，不发布 provider 原始响应或逐条收据。

| Rank | 系统 | Acc | Δ vs Jev | Speed× | BalAcc | NLL ↓ | Brier ↓ | Cov@0.9 | CW@0.9 | 扰动翻转 | Every R@1 | Wall s | p50/p95 s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Official Jev direct | **0.8858** | — | 1.00× | **0.8443** | **0.9648** | **0.1775** | 0.7180 | **29** | **0.0000** | **1.0000** | 642.7* | 0.292 / 0.729 |
| 2 | **Winnow-12B Q8** | **0.8824** | **-0.0034** | **1.30×** | **0.8374** | **0.9295** | 0.2069 | **0.8744** | 59 | 0.0278 | **1.0000** | **494.6** | **0.152 / 0.831** |
| 3 | Kev-9B | 0.8186 | -0.0672 | 0.92× | 0.7548 | 0.9985 | 0.2599 | 0.5058 | 35 | 0.0093 | 1.0000 | 700.4 | 0.174 / 1.444 |
| 4 | Kev-4B | 0.8018 | -0.0840 | 1.07× | 0.7048 | 1.1326 | 0.3055 | 0.4942 | 47 | 0.0185 | 1.0000 | 601.5 | 0.128 / 1.399 |
| 5 | Decider-2B | 0.7917 | -0.0941 | 0.61× | 0.6875 | 1.1044 | 0.3142 | 0.5558 | 43 | 0.0370 | 1.0000 | 1051.7 | 0.282 / 1.163 |
| 6 | Reflex stable | 0.7895 | -0.0963 | **2.61×** | 0.7803 | 1.0509 | 0.2913 | 0.2343 | **10** | 0.0741 | 1.0000 | **246.4** | **0.089 / 0.404** |
| 7 | this-that-model-1.0 | 0.7850 | -0.1008 | 0.62× | 0.6366 | 1.4306 | 0.3658 | 0.7436 | 101 | 0.0278 | 0.9474 | 1037.9 | 0.229 / 1.195 |
| 8 | SemIf Qwen3.5-4B MLX4 | 0.7671 | -0.1187 | 1.57× | 0.7423 | 1.1401 | 0.3365 | 0.5384 | 39 | 0.1389 | 1.0000 | 409.6 | 0.099 / 0.450 |
| 13 | NanoJev MiniCPM seed20 | 0.6529 | -0.2329 | 0.26× | 0.6694 | 0.7705 | 0.4603 | 0.2721 | 21 | 0.0833 | 0.9474 | 2519.0 | 0.355 / 3.031 |

`*` Official Jev direct 使用远端响应延迟，和本机 wall time 不完全可比；`Cov@0.9` 是置信度阈值 0.9 下的覆盖率，`CW@0.9` 是高置信错误数。完整表见 [BENCHMARKS.md](BENCHMARKS.md)；bundle 在 [`data/jevbench_offline_bundle_v1`](data/jevbench_offline_bundle_v1)（manifest SHA-256 `1fc3234acf806016adfd9906fade547ca597f87acf749aecac1ba706f6dd8ee5`）。

## 域内训练决策头（Valen 训练栈）

第二代实验：在开源 [Valen](https://github.com/Valen-Team) SFT+RLCD 栈上训练的
轻量决策头（Qwen3.5 底座，head-only 阶段冻结 backbone）。以下数字全部在
单卡 A100-80G / Apple Silicon MPS 上用同一套评测代码测得；官方 Jev 走
TypeSafe 直连 API（`jev-latest`，实测版本 `jev-1.13.0`）。仅发布聚合结果——
逐题原始输出保留在本地，不进公开仓库。

### 上下文过滤任务 —— `valen_nano_v1` held-out eval（678 道 noul 题）

自有 CC0 数据（`context_relevance_v1` + `oracle`，世界变体跨 train/eval 分裂）。
目标用途：候选上下文相关性过滤。

| 评分器 | Acc | Brier ↓ | NLL ↓ | 备注 |
|---|---:|---:|---:|---|
| **nano_rlcd_v2**（0.8B，SFT→RLCD） | **0.9012** | **0.1349** | **0.2058** | 当前 `:8093` sidecar 生产 backend |
| nano_rlcd_2b（2B，SFT→RLCD） | 0.8953 | 0.1521 | 0.2529 | RLCD 比其 SFT 初始化 +3.4pp |
| nano_sft_v2（0.8B，12 ep） | 0.8894 | 0.1570 | 0.2380 | |
| nano_rlcd_v1（0.8B，RLCD pilot） | 0.8746 | — | — | ECE 0.0377 |
| nano_sft_2b（2B，12 ep） | 0.8614 | 0.2032 | 0.3250 | 2B 在此任务不及 0.8B |
| Winnow-12B Q8（生产） | 0.8599 | — | — | ECE 0.1191 |
| 官方 Jev（`jev-1.13.0`） | 0.7341 | — | — | ECE 0.1038；662/678 作答，16 题 API ~5s 超时 |
| nano_sft_v1（0.8B，4 ep） | 0.6445 | — | — | 欠拟合参照 |
| Valen-Preview-0923 头 | 0.5100 | — | — | 通用域头，不迁移 |
| 未训练头 | 0.5220 | — | — | |

**在自有任务上，用约 1 万条域内数据训出的 0.8B 头以 +16.7pp 击败官方 Jev**，
延迟也更优（本地 ~0.9s vs 远端 ~1.5s，免费、离线）。RLCD 在已收敛 SFT 之上
仍有实测增益（+1.2pp acc，Brier/NLL 更优）；在欠拟合的 2B 初始化上增益 +3.4pp。

*标签注意：v1 数字含至多 ~1.3pp 标签噪声水分——干净标签 `valen_nano_v2` 复测与 `valen_nano_v3` 重建见 `BENCHMARKS.md` / `docs/VALEN_NANO_V2_REEVAL_V1.md`。*

### 上下文过滤 —— `valen_nano_v4`（4,251 道留出题）

任务定义：给定一条用户请求、对话历史和一个候选上下文片段，判断该片段
是无关（可安全丢弃）还是必需（必须保留）。两个错误方向：误删会丢失证
据（高风险）；漏删会浪费上下文（低风险）。

本评测取代 v3——4,251 题 = v3 的 2,950 条干净行 + 1,301 条新增困难
家族样本。**请勿与上方 678 题 v1 表的数字直接比较**（数据 sha256 前
缀见 `data/valen_nano_v4/manifest.json`）。

| 评分器 | 这是什么 | 准确率 | 误删 | 漏删 | 延迟中位数 |
|---|---|---:|---:|---:|---:|
| nano_sft_text_v4 | NanoJev 0.8B + LoRA 微调主干 | 1.0000 | 0 | 0 | 70ms |
| nano_rlcd_v4 | NanoJev 0.8B，偏好优化 | 0.8946 | 245 | 203 | 47ms |
| Winnow-12B Q8 | 本地 12B 通用模型 | 0.8292 | 15 | 711 | 542ms |
| JEMM-27B | 外部无决策头 27B 基线 | 0.8165 | 148 | 632 | 192ms |
| Kev | 本地编码器后端 | 0.8022 | 375 | 466 | 1116ms |
| nano_sft_v4 | NanoJev 0.8B 仅决策头 | 0.7944 | 765 | 109 | 45ms |
| 官方 Jev 1.13.0 | TypeSafe 云端 API | 0.6765 | 0 | 1375 | 687ms |

**说明：**

1. 准确率 = 答对比例。误删 = 把必需片段错判为无关——丢失证据；漏删 =
   把无关片段错误保留——浪费上下文。4,251 题 = v3 的 2,950 行 + 1,301
   条新增困难家族样本（工具结果依赖、跨指针证据、长上下文稀释、话题
   切换、对抗改写）。延迟为每请求中位数：NanoJev 各头为 A100 fp32
   前向；Winnow/Kev 为本地 Mac；JEMM 为 bf16 分片于 2×L40；官方 Jev
   为 API 往返。数据 sha256 前缀见 `data/valen_nano_v4/manifest.json`。
2. 满分 1.0000 意味着这套合成套件再次饱和——包括新增困难家族。真实
   转录评测仍待进行；应视为可学习基准的天花板，而非生产证明。
3. 官方 Jev 的失败仍是结构性的：0 误删对 1,375 漏删——它几乎从不判
   内容为可丢弃。JEMM 呈同方向的"保留偏置"（632 漏删对 148 误删），
   但更弱。
4. 旧 v2 头与无头读出对照仅在 v3 子集上测得（见下方归档 v3 表），未
   带入 v4。
5. JEMM-27B 跨硬件复现：A100 上 0.8165 对 L40×2 上 0.8167
   （25/4,251 决策翻转）——同权重跨设备一致性确认。

#### 归档：`valen_nano_v3` eval（2,950 道留出题）

已被上方 v4 表取代；保留以存档未带入 v4 的旧 v2 头与读出对照行。

任务定义：给定一条用户请求、对话历史和一个候选上下文片段，判断该片段
是无关（可安全丢弃）还是必需（必须保留）。两个错误方向：误删会丢失证
据（高风险）；漏删会浪费上下文（低风险）。

本评测集更新、更难——**请勿与上方 678 题 v1 表的数字直接比较**
（`data_sha256 c95286ff00512b28…`）。

| 评分器 | 这是什么 | 准确率 | 误删 | 漏删 | 延迟中位数 |
|---|---|---:|---:|---:|---:|
| nano_sft_text_v3 | 自研 0.8B 模型，LoRA 微调主干 | 1.0000 | 0 | 0 | 228ms |
| nano_rlcd_v3 | 自研 0.8B 模型，偏好微调 | 0.9847 | 34 | 11 | 213ms |
| nano_sft_v3 | 自研 0.8B 模型，仅训练决策头 | 0.9173 | 5 | 239 | 219ms |
| Kev | 本地通用编码器后端 | 0.8600 | 177 | 236 | 553ms |
| Winnow-12B Q8 | 本地 12B 通用大模型 | 0.8166 | 7 | 534 | 425ms |
| nano_sft_v2 | 上一代头，基于 v1 数据训练 | 0.7268 | 235 | 571 | 300ms |
| nano_rlcd_v2 | 上一代头，当前生产环境 | 0.7210 | 307 | 516 | 266ms |
| 官方 Jev 1.13.0 | TypeSafe 云端 API | 0.6871 | 0 | 923 | 625ms |
| LM-head 读出，LoRA 主干 | 标签 token 读出对照组 | 0.5620 | — | — | — |
| LM-head 读出，裸主干 | 完全未训练的空白对照 | 0.5530 | — | — | — |

**说明：**

1. 准确率 = 答对比例。误删 = 把必需片段错判为无关——丢失证据；漏删 =
   把无关片段错误保留——浪费上下文。延迟为每请求中位数；本地行均在
   Apple Silicon fp32 上测得，官方 Jev 经其云端 API 测得。
2. 官方 Jev 的失败是系统性的而非噪声：0 误删对 923 漏删——它几乎从不
   回答"无关"，属于系统性过度保留。
3. 最下面两行是对照组：同一主干去掉训练头后塌缩为永远回答"保留"——
   在 0.8B 规模上，真正承载技能的是训练头。
4. 满分 1.0000 意味着这套合成基准已被完全学透（饱和），并不证明生产
   环境表现；真实流量评测仍待进行。

### 通用域对照 —— JevBench 公开 231 题

同一个头跑官方 JevBench 公开集（139 choice / 74 noul / 18 score）。专项头
**不迁移出域**——与上方 v1 表互为镜像：

| 评分器 | Acc |
|---|---:|
| 官方 Jev 直连* | ~0.886 |
| Winnow-12B Q8* | 0.8824 |
| OmniJev-4B | 0.688 |
| OmniJev-2B / NanoJev MiniCPM seed20 | 0.654 |
| OmniJev-0.8B | 0.524 |
| **nano_rlcd_v2** | **0.307** |

`*` 在包含这些题目的冻结离线 bundle 上测得。专项头分题型：noul 0.473、
choice 0.216、score 0.333。

### 金融 regime 任务 —— `valen_fin_v1` eval（3,000 条 / 9,000 问）

加密永续日频决策：`regime_gate`（从 state 读取 `btc_ret20`）、
`fwd5_bucket`（5 日前向收益分桶，随机 0.25）、`xs_outperform_5d`
（跑赢截面中位，随机 0.50）。

| 评分器 | Overall | fwd5 | regime | xs5d |
|---|---:|---:|---:|---:|
| fin_sft_v1（0.8B） | **0.6202** | 0.329 | 0.9997 | 0.532 |
| fin_rlcd_v1（0.8B） | 0.6086 | 0.300 | 1.0000 | 0.526 |
| OmniJev-4B | 0.435 | 0.174 | 0.648 | 0.484 |

`regime_gate` 主要是"读 state"而非市场预测；真正的前瞻预测题（`fwd5`、
`xs5d`）信号很弱——与项目纸面账本的结论一致：信号在 regime/排序结构里。

**结论**：架构是通用的，先验是域特异的。小的域内头在自己的任务上胜出
（0.90 vs 0.73），在域外落败——这正是生产服务保留 Winnow 作通用默认、
只把上下文过滤问题路由给专项头的原因。

## 核心能力

| 能力 | 已实现 |
|---|---|
| 多状态并行 | 同一批处理多个独立环境状态 |
| 多问题并行 | 每个状态同时回答多个问题 |
| 动态候选 | Choice 每题支持 2–255 个候选，共享决策头 |
| 多种决策类型 | Choice 候选分布、Boolean 概率、Score 的 2–10 级分布及期望 |
| 直接输出概率 | 一次前向得到完整候选分布，无输出 token 解码 |
| 轻量底座 | Qwen3-0.6B，支持单卡训练与部署 |
| 持久推理服务 | 模型加载一次，复用权重处理后续请求 |

实际运行已验证：**6 个状态 · 18 个问题 · 44 条候选路径 · 1 次 backbone 前向**。[并行调用记录](research/parallel_example_v3.json)

## 更大的游戏与校准决策

- **完整环境：** 8×8、16×16、32×32、50×50 迷宫，四类拓扑，每图多个位置，并支持配置更大尺寸。
- **局部判断与代码规划：** 统一 5×5 局部观察、四方向并行安全判断、移动记忆及模型引导探索。
- **贪吃蛇规则：** 可复现食物生成、身体增长、碰撞与尾部移动、动态动作候选和安全问题。
- **概率学习：** 观测事件数据、CE/Brier 训练、成对适当奖励学习、精确梯度检查及已完成的 Qwen3-0.6B 训练。
- **完整评测：** 按地图划分数据、固定游戏集合、真实模型执行与独立轨迹重放核验。

局部安全模型的测试题准确率为 **77.84%**，50×50 OOD 题为 **76.56%**。概率学习先导中，成对适当奖励组的分布误差为 **测试 0.11844 / OOD 0.06202**；该误差是模型分布与模拟器事件概率之间的差值平方和。

[RLCD 实现与结果](docs/RLCD_EXPERIMENT.md) · [输入契约](docs/TYPESAFE_CONTRACT.md) · [V2 路线图](docs/NANOJEV_V2_ROADMAP.md)

## 此前完整 40 图 NanoJev 导航评测

使用 T=1 概率采样时，NanoJev 完成 **19/20（95%）** 的 4×4 测试地图和
**18/20（90%）** 的 6×6 OOD 地图。这些是固定 20-test/20-OOD cohort 上的
NanoJev 自有指标。

## 实现流程

每个决策由**状态、问题和候选集合**定义。模型编码候选路径，再由共享决策头输出每题的完整分布。Choice 使用共享标量头与集合注意力；Boolean 使用单路径 sigmoid；Score 返回等级分布和概率加权期望。

1. **构建问题：** 生成状态、问题、候选描述和目标分布。
2. **组织数据：** 相关地图、规则及其变体保留在同一分区。
3. **训练模型：** 初始化 Qwen3-0.6B，预热决策头，再使用完整问题的分布损失训练。
4. **执行评测：** 测量概率质量，运行游戏控制器并记录真实动作。
5. **服务与可视化：** 复用持久模型接口，在浏览器重放完整轨迹。

[完整训练与运行手册（English）](research/pipeline_runbook.md)

## 下载演示使用的模型

| 用途 | [模型仓库](https://huggingface.co/C-Tianyu/NanoJev/tree/main/variants)中的检查点 |
|---|---|
| **50×50 迷宫演示** | `variants/local_atomic_seed17` |
| **Snake 演示** | `variants/games_gold_seed17` |
| 整图问题评测 | `variants/games_api_seed17` |
| 校准决策实验 | `variants/events_ce_seed17`、`variants/events_brier_seed17`、`variants/events_paired_seed17` |

```python
from pathlib import Path
from huggingface_hub import snapshot_download

variant = "local_atomic_seed17"  # Snake 使用 "games_gold_seed17"。
snapshot = snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    allow_patterns=[f"variants/{variant}/*"],
)
checkpoint_dir = Path(snapshot) / "variants" / variant
```

[游戏数据包](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data/tree/main/games_v4)包含匹配的训练分区和固定评测输入。

## 下载并运行模型

模型和数据集均可公开下载。在 NVIDIA CUDA 或 Apple Silicon 环境中安装 [Python 依赖](requirements-toy.txt)：

```bash
python -m pip install -r requirements-toy.txt
```

下载基础 checkpoint 和数据集。根目录权重对应此前的导航版本，也是后续训练的初始化模型：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev", local_dir="checkpoints/NanoJev",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
snapshot_download(
    repo_id="C-Tianyu/NanoJev-Data", repo_type="dataset", local_dir="data/NanoJev",
)
```

启动持久服务：

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev \
  --web-root web --port 8765
```

打开 **http://127.0.0.1:8765**，或向 **`POST /api/evaluate`** 发送批量请求。模型只加载一次，后续请求复用权重。

Apple Silicon 使用 MPS FP32 运行 checkpoint 推理，参见 [Apple Silicon 推理说明](docs/APPLE_SILICON.md)。

[完整手册](research/pipeline_runbook.md)包含数据生成、训练、评测、checkpoint 创建，以及从下载模型和数据继续运行的命令。

## 路线图

当前计划维护在 [NANOJEV_V2_ROADMAP.md](docs/NANOJEV_V2_ROADMAP.md)。W45 之后，关键路径改为架构收敛与独立外部验收，不再自建另一套全模型排行榜。

- [x] **建立本地基线：** 独立 heldout、head/full/LoRA 三 seed 训练，以及分组校准诊断。
- [ ] **收敛 readout：** 比较 direct logits、当前 LoRA head 与 LoRA+pointer；候选顺序和 shared-prefix parity 必须通过。
- [ ] **扩大独立证据：** 构建 corpus v4 和更大的工程 heldout，不使用 benchmark/provider 输出。
- [ ] **外部验收：** 实现固定版本的 [JevBench](https://github.com/fstandhartinger/jevbench) adapter，完成一次冻结公开评测，再申请维护者 heldout。

架构参考：[SemIf](https://github.com/TheoLeeCJ/SemIf)。本地应用与 runtime 参考：[laya-mlx](https://github.com/mizorewww/laya-mlx)。外部 benchmark 与排行榜：[Benchmark Heaven 的 JevBench](https://benchmarkheaven.com/jev-models)。
