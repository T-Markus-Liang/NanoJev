# NanoJev 当前进度与 AI 执行交接计划

基于 2026-09-19 本地实验产物与工作区核对。**这是下一位执行 AI 的首读入口。**

**最新执行前置要求（2026-09-19）已验收**：Worker 只走 DeepSeek 官方 API 的 V4.1 Flash
（当前正式 API ID 为 `deepseek-flash`，禁止第三方回退）；本地 NanoJev skill 已安装并增加
开发/测试/优化/部署四阶段调用入口。真实调用、测试及限制见
[本地工具链验收](LOCAL_TOOLING_ACCEPTANCE_V1.md)，后续遵循项目 `AGENTS.md`。
T1/B0 尚未关闭；当前 paper driver 的在途修改不能当成已验收交付。

协作方式：执行 AI 负责范围明确的实现、测试与证据整理；原审阅 AI 在关键设计/数据/实验门禁处审核与调整。用户转交审核材料后再启动审核，不假设后台有人持续监听或自动批准。本文不授权主动上下文裁剪、真实交易、付费资源采购、私密数据外传或全局 provider 配置变更。

## 1. 当前状态：不要重新从零开始

| 工作项 | 当前事实 | 仍缺什么 |
|---|---|---|
| V1 决策服务及游戏研究 | Qwen3-0.6B 决策头、HTTP 服务、Codex 本地技能及既有游戏/RLCD-inspired 研究产物 | 不等于通用上下文安全或金融能力 |
| Phase 0 测量 | 冻结基线、哈希收据、paired/group CI、816 题 workflow 与 3264 题挑战 | 人工审阅的新任务族；三份匹配 baseline/candidate 训练种子；完整发布门禁 |
| Track A shadow | 三 wire format 的字节保真适配，3000 guardrail case；旧 checkpoint 的 24 个真实 HTTP 请求全部保留 | 更广 provenance；端到端任务/成本评测 |
| **G1 主模型工作流网关** | **已建成并通过五层验证**：库 42 项测试、CLI 入口回归、真实 scorer 端到端、进程级 active 缩减、`actual` 记账路径（412−300=112 已验证）。默认 shadow、12 类 fail-open、kill switch、可插拔 scorer、收据无原始内容。审阅方发现并修复了 CLI 完全无法启动的高严重度缺陷 | **RG 门禁**：默认 shadow、active 关闭；启用生产主动裁剪仍需 Track A 全部门禁 + ≥3 主模型族配对证据 |
| **G2 可逆过滤** | **已修复并通过独立验证**：65 项还原测试；审阅方验证模块级与 HTTP 闭环的**逐字节还原**、篡改被拒（`removed_segment_hash_mismatch`）。审阅方发现的契约违反（末位 list 内容消息被清空时发送不可还原的缩减请求）已修复：网关现在发送前做**真实往返自检**，不可还原即 fail-open | 已知残余限制：模块层对"中间位清空 + 后一条为 list 内容"仍不可还原，**由网关守卫拦截为 fail-open**（不违反契约，但该形状无法缩减）。根本修法是让清单记录每条消息原始 part 数 |
| **W1 真实数据模拟盘** | **四场所全部接通**（Binance 880 文件 / Bybit 30 / Hyperliquid 15 / **Aster 35**；后两者经本地出口代理——直连被本环境 DNS 污染/拒绝）。真实数据 PIT 队列 **6,554 条**经**未修改**校验器 **exit 0**；三场所模拟盘收据齐全；新增 15 项管线测试 | ⚠️ **同一策略在三场所产出 +21,174 / −77,919 / +194,026（极差 271,945 = 本金的 2.7 倍），而三场所 mark 价只差 1–2 bps** → **策略对输入混沌，结果不构成证据**。必须先修三个放大器（见 B0）。队列是 **PILOT 9 特征**（非 R1 闭集 12） |
| **数据许可审计** | Binance（已读全文：CC BY-NC-SA、§4.2 禁实盘）与 **Aster**（已读全文：§6.1(b) 下载需事先书面同意、§6.2(e) 自动化需明确许可，**无研究豁免**） | **Aster 是唯一已读且未解决的禁止条款**；Bybit（JS SPA，浏览器无代理）与 Hyperliquid（文档无条款页）**仍为未读/unverified**。详见 [许可审计](VENUE_DATA_LICENSING_V1.md) |
| Track B PIT | 时间戳/标签隔离/chronological split 校验器；**真实数据队列已建成并通过** | 完整状态/动作契约；金融基线；R1 冻结 |
| **P1 金融数据契约** | **已交付并停在 R1**：14 个数据源分级调研；永续-only 四场所；协议含 event/label/split/holdout/种子，校验器投影**机器验证通过**。**真实数据已下载 925 文件**（全部公开归档/API，未购买、未使用密钥） | **R1 必须裁定**：PILOT 9 特征 vs 闭集 12（强平特征无场所提供，永久拿不到）；71 项参数；计价货币范围 |
| **P2 执行模拟器** | **永续层 + 模拟盘回测已交付**（80 项测试）：守恒残差在容差内、两次运行字节相同、强平/保证金充足性/资金费已建模、现货购买力缺口已补齐；71 项参数待 R1 | **R2 门禁**：策略语义冻结；四场所真实 tick/lot/杠杆/资金费表未来源化；isolated 损失上限未建模；**成本参数与场所成交量耦合使跨场所不可比** |
| Relevance V1 | 三种子训练/结果汇总完成，本轮新增结论文档；**不晋级** | 独立代码/证据审核，版本管理收口；不是继续跑种子找赢家 |
| A4 工具历史 shadow | **已实现**：11 个 fixture 用例（8 scored + 3 bypass）、确定性构建器、13 项测试、设计文档；复用 shadow core，字节保真 | RA 门禁项：三操作 shadow receipt、结果省略 vs 含证据 judge state 的配对比较、三主模型族下游成本/正确性、bounded-result 编码 |
| E1 社区本地实测 | **laya 已在本机 MPS 跑通**：自动选 `mps:0`，4 问题进程内 warm p50 = 40.8 ms；本机 NanoJev 服务同形 1 状态/3 问题 warm p50 ≈ 74.9 ms。reflex 页面 WebGPU 可用（零安装） | B8 共享打分及视觉金融迁移未验证；laya 编码器闸门架构问题与 reflex direct-logits 消融均待 RE 审核 |

**战略优先级：金融数据与执行模拟基础优先；Track A 是并行基础设施，不得以 UI、插件打包或更多 relevance 调参挤占主线。**

## 用户当前重点方向（2026-09-19，最高优先级，覆盖上述默认排序）

本项目当前只有两条重点推进线，其余工作都归入服务这两条线：

**方向 1 — 所有主模型前置的 NanoJev 决策/压缩层。** 以**可逆过滤**与**严格回退**降低无效输入 token。
- 严格回退（fail-open）已实现并经 12 条失败路径测试；可逆过滤是新增的明确要求：被移除的片段必须能**逐字节还原**，且网关**绝不持久化**被移除的原文（原文始终由调用方持有）。
- **接入 ≠ 被授权主动裁剪**：网关默认 shadow、active 关闭。启用生产主动裁剪仍需 Track A 全部门禁 + ≥3 主模型族配对下游质量与净 token 成本证据。

**方向 2 — 金融二级交易 RLCD 模型训练及模拟盘交易回测。** 市场范围已由用户收窄为：
- **只做加密二级合约（永续合约）交易，不做现货**；场所为 **Binance、Bybit、Aster、Hyperliquid**。
- 交付物包含 **RLCD 模型训练**与**模拟盘（paper）交易回测**；实盘仍然禁止。
- 该范围变更的影响：P1 原草案是**现货、long-only、毛价格变动标签**，须改为**双向、含杠杆/保证金/资金费/强平**的永续语义；P2 模拟器**目前明确拒绝做空**且无保证金/资金费/强平模型，这是永续回测的**硬阻塞**，已在修订工作包中处理。
- **许可证冲突必须正视**：原推荐的 Binance Vision 归档为 CC BY-NC-SA 4.0（非商用）且其 §4.2 禁止实盘自营交易执行。研究/模拟盘用途或可接受，但该许可**永远无法支撑实盘执行阶段**；修订工作包被要求逐场所核查许可与自动化/执行条款并显式标注。

**用户明确指令（2026-09-19）：本地决策模型必须接入主模型工作流，以提速并降低 token 消耗。** 具体交付物是 G1 网关；其 CLI 入口缺陷已由审阅方发现并修复（详见 [执行审核日志](EXECUTION_REVIEW_LOG.md) 的 G1 条目）。

**用户明确指令（2026-09-19）：先推进真实数据仿真模拟交易（比 RLCD 训练容易）。** 已完成端到端：真实数据下载 → PIT 队列 → 未修改校验器 → 模拟盘收据。详见 [真实数据模拟盘 V1](PAPER_TRADE_REAL_DATA_V1.md)。

**用户明确指令（2026-09-19）：授权 Bybit / Aster / Hyperliquid 作为数据源并立即打通。** 四场所已全部接通。**两个阻塞本环境的出口问题（非场所侧封锁）由本地代理解决**——`api.bybit.com` 被 DNS 污染、`fapi.asterdex.com` 直连被拒。审阅方首轮只测直连就判定 Aster 不可达，**这是疏漏**。

**用户明确指令（2026-09-19）：更新文档进度与后续规划，并把重要信息提交同步到 git。**

### 本轮最重要的结论：**测量本身是坏的，修好之后结论反转**

真实数据上同一参考策略最初产出**四个互相矛盾**的结果，极差 **271,945（本金 2.7 倍）**：
Binance 修复前 −49,920.90、修复后 +21,173.77、Bybit −77,919.27、Aster +194,025.96。

**此前"策略对输入混沌"的结论已撤回。** 极差主要来自两个**工具缺陷**，不是策略行为：

1. **收据谎报自己的样本窗口**：`--first-day/--last-day` 被写进收据却**从未用于过滤数据**，每次运行实际使用整个归档（2023-01-01 起）而声称 2024-01-01 起——三个场所**从未在同一区间上比较过**。靠一个不可能通过的复现检查发现："一半"样本的结果与"全部"完全相同。
2. **参与率上限与场所报告成交量耦合**（`fill ≤ 10%×bar volume`），造成 Aster 的部分成交。

修复后同区间同策略：

| 场所 | 决策 | 净 PnL | 分歧 |
|---|---:|---:|---:|
| Binance | 61 | −34,080.29 | 0 |
| Bybit | 61 | −37,597.40 | 0 |
| Aster | 61 | −38,044.85 | 0 |

**极差从 271,945 降到 3,964（本金约 4%），三场所同号同量级。** 容量耦合的单独效应是 **820**，不是 279,000。详见 [B0 窗口缺陷](B0_WINDOW_DEFECT_V1.md)。

仍然成立的诚实表述：参考策略在三个场所**都亏钱**（机械均线交叉 + 声明式临时成本，不构成任何策略结论）；**约 4% 本金的离散度仍然太大**，单场所数字仍不得引用；**测量完整性在这里不是形式主义，它就是发现本身**。

另经核实：仓位规模**路径无关**（仅用 initial_cash 与决策日收盘价）；在 `fill_probability=1.0`、`reject_probability=0.0` 下 replay **对种子不敏感**，"多种子敏感性报告"在此配置下是**退化轴**，有意义的轴是场所与时段。

### 本地 NanoJev 能力：**域内可用，域外崩塌**（独立诊断结论）

独立诊断（subagent，含精确复现基线收据）判定置信度崩塌是**域外现象**：

| | 域内（迷宫 test/ood） | 域外（13 道工程题） |
|---|---|---|
| 中位置信度 | **0.779** | 0.505 |
| 最大置信度 | **0.998** | 0.736 |
| ≥0.9 的题 | **37.5%，其中 97.8% 正确** | 0%（13/13 弃权，0.9 门限**观测**运行）|
| 准确率与置信度 | **正相关（点二列 r=+0.397）** | 3/6 |

**因此：域内置信度是有信息量的，域外崩塌是游戏训练模型的正确行为。** 但残余问题存在：域内在 0.9 下仍弃权 62.5%，且 41.7% 的域内题置信度 ≤0.736。

**三项表述经对抗性核验后被撤回**：① 错答占据置信度前三 —— **不成立**（真实排名 1/3/5）；② 置信度与正确性反相关 —— n=6 下不成立（置换检验 p=0.80，两位独立核验者连符号都相反）；③ 历史弃权率 **43.5% 是错的**，我调查之前是 **7/23 = 30.4%**（43.5% 把本次调查自己的调用算了进去）。另：3/6 是**随机下的众数**（P=0.3125），不构成发现；检验 0.8 vs 0.5 需约 18 题。见 [就绪度实测](NANOJEV_SKILL_READINESS_V1.md)、[域外诊断](SKILL_ABSTENTION_DIAGNOSIS_V1.md)、[对抗核验](SKILL_READINESS_VERIFICATION_V1.md)。

### 信号灯（2026-09-19）

| 面 | 状态 | 含义 |
|---|---|---|
| 数据链路（Track B） | 🟢 Green | 四场所接通、真实 PIT 队列过未修改校验器、manifest 含来源与哈希 |
| 执行经济性（Track B） | 🟠 **转好** | B0 大部分达成：窗口缺陷与容量耦合已修，极差 271,945 → **3,964（本金 4%）**、三场所同号。仍是**非结果**（成本为临时猜测、无 edge 声明），T5 归因未完 |
| 模型闸门能力（Track A） | 🟡 Amber | 网关/可逆过滤/A4 已验证；真实 checkpoint 提议 0 删除、`MAX_SCORED=32` bypass、Catalog Choice 受干扰 100%→54.17%。**skill：域外 13/13 弃权、域内 ≥0.9 者 97.8% 正确** → 崩塌是域外现象，已加范围守卫；**温度重标定无法修复**（负面结论） |
| 生态吸收 | 🟡 Amber | 第二轮生态扫描已分类入账（见下）；未核实项不得当事实引用 |

### 生态第二轮扫描的入账方式（2026-09-19）

用户提供的 Jev 生态更新已按证据等级分类并写入 roadmap 的「Ecosystem update」节，**只作为任务与观察，不作为能力或证据**：

- **已核实存在**（GitHub API 快照，2026-09-19）：fast-jev-compaction 3,729⭐（其 README 表明 token 为字符估算、无 tokenizer；所述 156k→62k 数字**不在 README 中**，未核实）、json-render 16,681⭐、fx 3,064⭐、ai-python 184⭐、cline/plugins 23⭐、kev 287⭐（MacBook 可训练，直接相关）、**bespokelabsai/nimble 147⭐（开放配方+Apache-2.0 权重，对比式数据策展——直接对应 Track A 零删除瓶颈，已有完整 pinned 评审）**、jevinci 19⭐、postgres-Jev 0⭐、jev_stock 6⭐、上游 TianyuCodings/NanoJev **626⭐**（仓库创建于 2026-09-17）。
- **未找到/未核实**：Atomic、Jevinik、Monad 实盘 bot、DuckDB 扩展、成本案例（$0.09/$2.17/$0.19）、Decider-2B、System-One 4B、Jev 兼容公开 API、「HF 模型 Jev 化」库。
- **吸收的路线变更**：新增 **A5 过滤 vs 重建对照实验臂**；Phase 0 新增**成本原生报告**与**外部声明证据账本**两条门禁；「判断即代码原语」三面（provider resolver / 数据库行谓词 / 评估算子）登记为 **C-track 候选**（不承诺）；社区实盘 bot 仅作安全需求来源；Mac 可训练小模型采用**五级验收阶梯**（能加载≠能训练）。

实验详情见 [CONTEXT_RELEVANCE_V1.md](CONTEXT_RELEVANCE_V1.md)：seed 17 迷宫准确率超回归预算；seed 18 在 0.99 下有误删；seed 19 不能因已看过留出结果而被挑为赢家。三个种子的迷宫 NLL/Brier 点估计均恶化。生产参考仍为 `checkpoints/local_atomic_seed17/variants/local_atomic_seed17`，实际 token 节省为零。

## 2. 阅读顺序与事实来源

1. 本文：任务边界、下一工作包、暂停审核点。
2. [V2 roadmap](NANOJEV_V2_ROADMAP.md)：战略与硬性验收规则。本文补充执行步骤，不降低 roadmap 门槛。
3. [Relevance V1](CONTEXT_RELEVANCE_V1.md) + [冻结协议](../research/context_relevance_v1_protocol.json) + [原始报告](../results/context_relevance_oracle_v1_report.json)。
4. [Financial PIT](FINANCIAL_PIT_V1.md)、`scripts/financial_pit_v1.py` 及对应测试。
5. [Shadow V1](CONTEXT_SHADOW_V1.md)、`scripts/context_gate_v1.py`、`scripts/context_gate_local.py` 及对应测试。
6. [Workflow baseline](WORKFLOW_V2_BASELINE.md)、[社区参考](JEV_COMMUNITY_REFERENCES.md)、[输入契约](TYPESAFE_CONTRACT.md)。

代码在 `scripts/`，评测收据在 `results/`，协议/历史研究在 `research/`，本地数据与 run 在 `data/`、`runs/`，已下载公共数据在 `dataset/`，权重在 `checkpoints/`。`web/` 主要是既有演示；当前不要优先重做前端。

没有发现仓库内 `AGENTS.md`。入口状态以 `git status` + 实际文件为准；README/旧 milestone 文档中的测试数与“尚未实现”是历史快照，不应覆盖后续收据。遇到冲突时记录差异，不默默选取有利说法。

## 3. 接手前的工作区保护

本次核对 HEAD：`be0303c Add shadow context gating and financial PIT validation`。存在前序未提交修改：

- 已跟踪修改：`docs/NANOJEV_V2_ROADMAP.md`、`scripts/context_gate_v1.py`、`scripts/train_pipeline_decisions.py`。
- 未跟踪：社区参考文档、relevance 冻结协议、数据构建/报告脚本、`test_build_context_relevance_v1.py`、`test_report_context_relevance_v1.py`、`test_training_runtime.py`，以及初始化/三种子迷宫/汇总的 6 个 relevance 结果 JSON。
- 本轮文档更新另新增本文、relevance 结论页，并同步 roadmap、shadow、workflow 和社区参考链接/状态；**未提交 Git**。

必须先检查 `git status --short`、`git diff` 和本地数据完整性。禁止 `git reset --hard`、`git clean`、覆盖原报告、删除无效前身审计文件、批量 `git add .` 或误纳入权重/私密日志。提交/推送遵循用户另行授权，不把本交接视为授权。

验证基线：主 unittest 165 项，OK（2 skip，163 执行通过）；本地 skill 3 项通过。本轮重跑了这两个套件，未重训、未重跑 checkpoint 推理、未重新生成大报告，未做独立源数据审计。

## 4. 工作包与顺序

下列未存在的交付路径均是**建议的新文件名，不代表已经实现**。执行者可以为合理的项目结构调整命名，但须保持可追溯链接。

### 4.0 工作包状态总览（2026-09-19 更新）

**执行顺序已升级为任务板**：见 [roadmap 的 Current development slice](NANOJEV_V2_ROADMAP.md)（T1–T16，每个任务带目标/改动文件/验证命令/门禁/依赖/工作量）。下表是状态快照，任务板是唯一权威顺序。

| 包 | 状态 | 证据/入口 |
|---|---|---|
| P0 证据收口 | ✅ 完成 | 冻结哈希 MATCH；独立重建报告语义相同 |
| G1 网关 | ✅ 完成并五层验证 | [网关文档](MAIN_MODEL_GATEWAY_V1.md)；CLI 缺陷已修复 |
| G2 可逆过滤 | ✅ 完成并独立验证 | [可逆过滤文档](REVERSIBLE_FILTERING_V1.md)；65 项测试 |
| A4 工具历史 | ✅ 完成，含跨包验证 | [A4 文档](TOOL_HISTORY_SHADOW_V1.md)；11/11 经网关语义一致 |
| P1/P1b 金融契约 | ✅ 交付，**停在 R1** | [数据计划](FINANCIAL_DATA_PLAN_V1.md) |
| P2/P2b 模拟器+回测 | ✅ 交付，**停在 R2** | [模拟器文档](FINANCIAL_SIMULATOR_V1.md)；80 项测试 |
| W1 真实数据模拟盘 | ✅ 交付 | [真实数据模拟盘](PAPER_TRADE_REAL_DATA_V1.md) |
| 许可审计 | ✅ 交付（2 家已读、2 家未读） | [许可审计](VENUE_DATA_LICENSING_V1.md) |
| E1 社区本地实测 | ✅ 部分（laya 已测；reflex 计时未做） | [参考项目评审](JEV_COMMUNITY_REFERENCES.md) |
| E2 生态核实账本 | ✅ 第二轮扫描已入账（核实 10 项、未核实 9 项） | roadmap「Ecosystem update」节；待持续更新 |
| **B0 测量完整性** | ⚠️ **未达成，阻塞所有金融结论** | 本轮新增，见下 |
| **B0 测量完整性** | 🟡 **大部分达成**（2026-09-19）：T1 冻结协议 / T2 分歧审计 / T3 sizing 锁定 / T4 容量解耦全部交付；**并发现并修复了使此前所有数字失效的窗口缺陷** | 残余：T5 PnL 归因与分时段报告。修复后同区间同策略 **Binance −34,080.29 / Bybit −37,597.40 / Aster −38,044.85**，极差 **3,964（本金 4%）**，三场所同号——此前的"策略混沌"结论**已撤回**。见 [B0 窗口缺陷](B0_WINDOW_DEFECT_V1.md) |
| **本地模型域外崩塌诊断** | ✅ 完成（独立 subagent，复现基线收据 ≤1e-8） | 判定置信度崩塌是**域外**现象：域内 240 题中位置信度 0.779、≥0.9 者 **97.8% 正确**、点二列 **r=+0.397**；域外 13 题最高仅 0.736、**13/13 弃权**。见 [域外诊断](SKILL_ABSTENTION_DIAGNOSIS_V1.md) |
| **置信度崩塌归因** | ✅ 完成（2026-09-19） | **拆开了"内容域"与"表层形式"**：域内迷宫题加长无关文本／改写为工程措辞，中位置信度仅变动 ≤0.021；域外工程题缩短成一句话、简化措辞后置信度仍为 0.275／0.444（未恢复）。**结论：障碍是内容域，不是长度或措辞——没有提示工程捷径，必须补域内训练数据。** 见 [归因](CONFIDENCE_ATTRIBUTION_V1.md) |
| **温度重标定修复尝试** | ✅ 完成（**负面**，含 bootstrap） | 拟合仅用**已存在的冻结校准集**（`calibration.jsonl`，192 题，与 test/OOD/train 的 state 与 source-group 重叠均为 0）得 **0.854**；调查题最大置信度 0.736→0.769，**仍低于 0.9 门限 0.131，弃权仍 13/13**；要在 0.9 答出**第一题**需 T≤0.467，而那题正是错在安全关键方向的 `safe_to_drop`——**温度单调，无法重排序**。域内无实质收益（state 聚类 bootstrap：准确率差恰为 [0,0]，NLL/Brier/ECE 区间全部含 0，fixed-bin ECE 的"改善"是分箱假象）；且 T 在不同非测试划分上**符号翻转**（校准集 Brier 最优 ≈0.9994、dev 最优 >1），说明 0.854 是样本噪声。**建议服务温度保持 1.0、不放宽门限。** |
| **对抗性核验** | ✅ 完成，**撤回我 3 项表述** | 错答占前三（假，实为 1/3/5）、置信度反相关（n=6 不成立，p=0.80）、历史弃权率 43.5%（实为 **30.4%**）。另：3/6 是随机众数（P=0.3125）。见 [对抗核验](SKILL_READINESS_VERIFICATION_V1.md) |
| **skill 范围守卫** | ✅ 完成（17 项测试） | `nanojev-scope-guard-v1`：5 类短语命中即标 `out_of_scope`、清空 value 但**保留概率**、附 `operating_envelope`、无关闭开关；安装副本未改（属独立受审步骤） |
| **A5 过滤 vs 重建对照** | ⬜ 新增待启动 | roadmap A5；复用 A4 fixture |
| **本地 skill 就绪度实测** | ✅ 已测（2026-09-19） | **技术可用**（按需启动/模型输出比特一致/离线/0.27–0.33s）；**域外不可用作决策**（13/13 弃权、最高置信答案错在安全关键方向），**域内可作参考**（≥0.9 者 97.8% 正确）。已加 `nanojev-scope-guard-v1` 守卫（17 项测试）。**对抗核验撤回了我 3 项表述**：错答占前三、置信度反相关、历史弃权率 43.5%（实为 30.4%） |
| **A2 分批评分（MAX_SCORED=32）** | ⬜ 待启动，**先于** laya/reflex 模型对比 | roadmap Track A slice 第 8 条 |
| P3 基线 / P4 RLCD | ⬜ 未启动 | 依赖 B0 与 R1 |

### B0 — 测量完整性（**新增，最高优先，阻塞一切金融结论**）

**背景**：同一策略在三个场所产出 +21,174 / −77,919 / +194,026（极差 271,945 = 本金 2.7 倍），而三场所 mark 价只差 1–2 bps。在此之前，任何金融数字（包括未来的 RLCD 结果）都无法与路径噪声区分。

**执行**（按序五个子项；**代码事实已于 2026-09-19 核实**，纠正了此前两处错误表述）：
0. **B0-0 冻结可复现基线**：`research/paper_trade_b0_protocol.json`（已建）固定场所、品种、窗口、策略与策略参数、费率、容量模式、种子与 run id；驱动加 `--protocol` 加载并记录 `protocol_sha256` 与输入 manifest 哈希，不匹配即拒绝运行。
1. **B0-A 订单/成交分歧显式化**：`long`/`short` 是**目标仓位动作**，模拟器按**账本实际持仓**算 delta，所以仓位**水平**会自愈；不自愈的是每笔成交的**尺寸与时机**（容量夹取 `10%×bar volume`、拒单、TTL 过期），它改变手续费、资金费与 PnL 路径。实现：回放后用 `result["orders"]` 的 `requested_quantity`/`filled_quantity`/`status_history` 做分歧审计，`--on-divergence {error,report}`（默认 error：写收据、标记 `divergence_error`、非零退出）。
2. **B0-B 路径无关仓位（核实+锁定）**：驱动**已经**用 `initial_cash` 与决策日收盘价算量（`0.25×initial_cash×leverage/close`），此前文档"按当前权益复利"的说法**有误，已更正**。本项 = 加 `--sizing fixed_notional` 并用测试锁定该性质；按权益复利的变体需要交错 decide/replay，推迟到 T12。
3. **B0-C 容量与成交量解耦**：`CapacityPolicy.max_participation_fraction=0.1` × 场所报告成交量是 Aster 27 笔部分成交的**确认来源**。实现 `--capacity {fixed,venue_volume,off}`，默认 `fixed`（所有场所同一声明参考成交量），跨场所对比表必须用 `fixed` 运行。
4. **B0-D 集成回放 + PnL 归因 + 强制敏感性报告**：新脚本 `scripts/paper_trade_report_v1.py` 产出归因（信号/价格、点差、手续费、资金费、容量未成交残余，与净 PnL 对账到 1e-6）与敏感性矩阵（种子 × 场所 × 日期子集，含极差与 `headline_allowed` 布尔）。存在未解释残差或离散度超阈值时**不得输出头条数字**。

**验收**：跨至少三场所、同区间同策略的结果离散度小到可解释，**或** harness 明确声明无法给出并拒绝输出头条数字。1–2 bps 的价格差不得再产生无法解释的巨大残差。

### P0 — 证据收口与接手核验（先做，范围小）

**输入**：冻结协议、三 run、原始报告、当前 diff、本文。

**执行**：
1. 检查所有必要本地产物是否存在；核对配置、数据/权重/预测哈希和每种子 dev 选择路径。任何缺失都列明，不能以占位结果补齐。
2. 阅读新数据构建、报告与运行时修改，运行两个 unittest 套件；若重建汇总报告，必须使用全新输出路径并保留原收据。重建只汇总已有预测，不追加测试集调参。
3. 在 `docs/EXECUTION_REVIEW_LOG.md` 建首条记录：范围、输入版本、检查结果、未验证项、风险；引用而非复制大报告。

**验收**：三种子及坏结果全部保留；不晋级；能区分“已跑单测”“报告重建”“独立过程审核”；冻结产物未改变。

**审核点 R0**：提交当前 diff、哈希核对结论、失败/跳过列表给审阅 AI。等待审核时可继续 P1 的只读数据源调查，不必闲置；不要启动新 relevance 训练。

### G1 — 主模型工作流网关（用户明确要求的必做项）

**背景**：项目已有字节保真的 shadow 评分核心（`scripts/context_gate_v1.py`）和一个只写 receipt 的本地 CLI（`scripts/context_gate_local.py`，其自身文档明确说明"不是转发代理"）。**真正把本地决策模型接进主模型工作流的组件此前完全不存在**——没有网关，本地模型不可能参与任何主模型请求。

**执行**：
1. 建立网关：监听回环地址，暴露 OpenAI 兼容 `POST /v1/chat/completions` 与 Anthropic 兼容 `POST /v1/messages` 透传端点；用可插拔 scorer 调用既有 shadow core。
2. **默认 shadow**：向上游转发**原始字节**，返回上游响应原样；只记录假设性删除计划。**active 必须显式开启**且必须 fail-open：闸门异常、超时、分数缺失/畸形/非有限，或删除集含任何受保护片段时，一律转发原始请求。
3. 复用既有 sidecar/保护逻辑，不重新实现。系统/开发者指令、当前用户意图、任务所需工具 schema、引用证据、依赖上下文、pin 片段永不删。
4. **token 口径**：只有真正发送了缩减请求时，才用上游响应回报的 prompt token 数声明**实际**节省；shadow 模式只能给出**估计**并标注 tokenizer 身份。
5. 每次请求写 JSONL receipt（mode、wire format、片段哈希、保留/删除指针、原因码、分数、闸门延迟、token 数），**绝不写原始提示词或凭据**。提供全局 kill switch。

**建议产出**：`scripts/main_model_gateway_v1.py`、`scripts/scorer_adapters_v1.py`、`scripts/test_main_model_gateway_v1.py`、`docs/MAIN_MODEL_GATEWAY_V1.md`。

**验收**：shadow 字节等价；active 缩减可测；所有 fail-open 分支有测试；receipt 无原始文本；kill switch 生效；测试用**回环假上游**，绝不调用真实 provider。scorer 需可切换（NanoJev 本地服务 / 进程内 scorer / 可配置的 laya 型编码器服务）。

**审核点 RG（必须停）**：**默认保持 shadow 与 active 关闭**。启用生产主动裁剪需要：Track A 全部验收门禁 + 配对的下游任务质量与净 token 成本证据（≥3 个主模型族）+ 受保护片段压力集零删除。在拿到这些证据前，任何"已降低 token 消耗"的表述都只能标注为估计值，不得作为结论。

### P1 — 金融数据与首个实验契约（最高优先，下一主工作包）

**输入**：PIT V1、roadmap B1–B6。首轮市场/品种、时间范围、频率、预测 horizon、预算尚未冻结，不能当成已批准条件。

**执行**：
1. 调查可用数据源，列明来源 URL/版本、训练与再分发权利、可访问性/费用、实际字段、时间戳语义、修订/复权/退市覆盖、缺失与时钟规则。市场选择给出建议及替代项，不擅自购买数据。
2. 设计最小结构化二元事件预测任务，分别定义事件概率与策略动作。冻结 label predicate、观测/成交价格约定、可用时刻、horizon、feature fit cutoff、universe 构建与 regime 定义（不得用未来观察划分决策时状态）。
3. 提出 chronological train/dev/calibration/test、purge/embargo、instrument/regime holdout；记录样本依赖处理、选择规则及允许查看的结果。
4. 提交数据许可/PIT 真伪证据与风险，不把事后填入 `available_ns` 视为真实性证明。真实来源拿不到时，标记 blocker；可继续模拟器合成单测，但不得声称真实金融验证。

**建议产出**：`docs/FINANCIAL_DATA_PLAN_V1.md`、`research/financial_source_manifest_v1.json`、`research/financial_experiment_protocol_v1.json`（审核前明确标记 draft）。重要架构选择独立记录决策依据与备选方案。

**验收**：权利/来源/可用时点可审计；市场、任务、分割与指标无悬空定义；拟议规则在看到 test/OOD 结果前冻结；缺失信息显式列明。

**审核点 R1（必须停）**：市场/数据权利、事件定义、分割与 holdout 方案交审阅 AI/用户确认。在通过前不开展正式金融训练或为收益挑选参数，不购买受限数据。

### P2 — 执行模拟器与模型外风控

**依赖**：R1 认可的状态、时间、价格和成本口径；可提前编写不依赖真实源的纯合成单测。

**执行**：模拟事件顺序、订单提交/生效延迟、成交、部分成交/拒单、手续费、价差、滑点、容量/冲击和适用的借券约束。实现 no-trade、仓位/敞口/损失/换手/频率限制、过期数据/时钟异常检查和 kill switch，全部在模型外。

数据不足以支撑某项微观结构时，不得虚构精确成交；使用显式保守模型、适用范围与敏感性分析，或阻断该策略。若扩展 PIT V1 schema，需要版本迁移与旧契约回归测试。

**建议产出**：`docs/FINANCIAL_SIMULATOR_V1.md`、`scripts/financial_simulator_v1.py`、`scripts/financial_risk_v1.py`、相应测试与 `results/financial_simulator_v1_stress.json`。

**验收**：确定性重放；未来价格不能提前影响决策；现金/仓位/PnL/费用账目一致；每笔成交有可追溯来源和顺序；手算小案例与独立参照对照；异常必定阻断或 no-trade；合成压力测试不冒充真实绩效。

**审核点 R2（必须停）**：事件时间线、撮合假设、成本/风险测试与泄漏控制通过后，才进入正式策略效用比较。

### P3 — 确定性与监督金融基线，先于 RLCD

**依赖**：R1、R2 通过，真实可审计数据可用，预算与模型/种子方案冻结。

先建立 no-trade/passive（适用时）/random/simple-rule 参照和简单概率模型；再做匹配数据、初始化、种子、预算的 CE 与 exact-Brier 比较。选模型只看 dev，校准只看 calibration；最终测试不得驱动参数选择。协议预先声明种子数与 seed 列表（建议至少三种子），不得看结果后增删种子。

**建议产出**：`docs/FINANCIAL_BASELINES_V1.md`、独立版本的训练/评测协议、`results/financial_baselines_v1.json` 及逐 fold/seed/instrument/regime 收据。

**验收**：NLL/Brier/ECE、selective risk/coverage 与 action utility 分开；gross/net、费用、换手、回撤、尾部损失、敞口、CI 和延迟范围分开报告；坏时期不能隐藏。若只证明事件预测，不可称策略盈利。

**审核点 R3（必须停）**：审核监督基线、经济性与数据稳定性；即使收益不好也先报告，不得直接用 RL“补救”。

### P4 — RLCD-like 与效率实验（暂不启动）

仅在 R3 通过后单独预注册 estimator 的偏差/方差、样本效率、校准/效用对照。共享打分须对独立前向做数值等价、候选置换、padding/长度和请求隔离检查；报告 feature/model/end-to-end 延迟。视觉金融状态只作 structured baseline 之后的可选消融；不先改成视觉模型。

**审核点 R4**：目标函数/奖励和 parity 方案变更、候选晋级、paper/shadow 阶段推进均需审核。实盘仍不在本计划授权内。

### A4 — 可并行的工具历史 shadow 工作包

可在 P1–P2 期间交给另一个执行者，限定文件所有权，避免双方同时编辑 `context_gate_v1.py`/训练运行时。

- 新建独立 source-group fixture，不复用 relevance test/OOD 作为训练数据；覆盖 parallel/error/orphan/duplicate/pending 工具、可变文件、不可重跑结果、用户修正与受保护信息。
- 先冻结 pairing/provenance 契约。操作只在 shadow receipt 中表达：整对保留、保留调用并明确标记有界结果、成对移除；发往 provider 的原始 bytes 不变。
- 比较结果缺失 judge state 与包含完整证据的 state；不能确立相关性就保留，禁止靠自动重跑工具补证据。
- 先通过保护/回退/隐私/依赖闭包测试，再设计三种主模型的配对下游任务与真实 token/缓存/重试/闸门净成本测量。

**建议产出**：`docs/TOOL_HISTORY_SHADOW_V1.md`、独立 fixture manifest、测试和 shadow 收据。

**审核点 RA**：新 eligibility/截断规则、外部模型请求（内容、权限、费用）、主动裁剪前必须审核。未获数据外传许可只能用自创或有明确授权的样例。维持既有 0.99 协议，不复制 upstream 0.5。

### A5 — 过滤 vs 重建对照实验臂（2026-09-19 新增）

**背景**：社区最大项目（fast-jev-compaction，已核实 3,729⭐）采用与我们 A2/A4 相同的过滤语义；最响亮的公开反对意见认为压缩本质是重建。两种立场目前都无证据。这个工作包把争论变成冻结的配对实验，而不是立场之争。

**执行**：
1. 在 A4 冻结 fixture 上跑五条臂：原文保留（对照）、确定性安全去重、相关性过滤（A2/A4 语义）、抽象摘要、检索重建。
2. 测量：依赖对保留率、证据忠实性、受保护段删除数（必须为 0）、下游任务成功率、还原成本、**端到端每请求成本**（含评分、重建、重试、工具重执行；只用 provider 回报或 tokenizer 计量的 token，**禁止字符估算**）。
3. 按任务族报告配对置信区间；总体赢但受保护族输 = 失败。
4. fail-open 在所有臂中都是"转发原始字节"；摘要兜底是另一种策略，不得静默替换。

**边界**：本实验单独不产生采用决定；采用仍需 Track A 全部门禁。字节级可还原 ≠ 缩减后推理等价。

### E2 — 生态核实账本（2026-09-19 新增，持续维护）

**执行**：把 roadmap「Ecosystem update」节的未核实项逐条补一手来源（Atomic、Jevinik、Monad bot、DuckDB 扩展、成本案例、Decider-2B、System-One 4B、Jev 兼容 API、HF 适配库）；引用任何星数前先重新快照；若 fast-jev-compaction 出现官方基准收据，核对其 156k→62k 声称的口径（注意其 README 自述 token 为字符估算）。

**边界**：纯只读核实；不为核实而安装插件、上传私有数据、用账户或下单；每条记录 evidence_level（作者声称 / 源码审计 / 本机复现 / 配对对照）+ URL + 快照时间 + 口径。

### E1 — 探索性：laya / reflex 的 Mac 本地只读评估（可并行，低优先）

两个新参考项目（2026-09-19 纳入 [社区参考](JEV_COMMUNITY_REFERENCES.md)）面向本机 Mac 笔记本场景：

- **laya**（Apache-2.0，421M ModernBERT 编码器，`pip install laya`）：比当前 0.6B decoder 更轻，是 Track A 闸门"编码器架构"的候选假设。允许：隔离环境中安装 pinned revision（`4937897c`），跑其自带 quickstart，测 MPS/CPU 延迟与内存，并在冻结 workflow 样本上做明确标注的 zero-shot 探测。
- **reflex**（MIT，Qwen3.5 + direct logits，无训练头）：其 WebGPU 演示（Safari 18+，Qwen3.5-0.8B ONNX 约 650MB）可在本机浏览器零安装验证；其 packed-mask 分支隔离 + 数值等价测试是 prefix sharing 的直接参照。

**边界**：只读评估收据，标注为 upstream-weights 测量；不改冻结 cohort、生产 checkpoint 或服务代码；不引用其 T4/GB10/第三方 Jev 数字作对比结论；reflex direct-logits 上限 26 候选，仅作消融臂，不得替代 255 候选契约。

**审核点 RE**：编码器闸门架构问题（laya deliverable 4）与 direct-logits 消融实验（reflex deliverable 2）在执行前必须交审阅 AI 审核；纯安装/演示/测速不需要。

## 5. 什么时候必须找审阅 AI

除 R0–R4/RA 外，以下任一情况应暂停相关实施并给出决策材料：

- 想变更数据 split、标签、目标函数、阈值、评估/晋级标准或生产 checkpoint。
- 发现未来信息泄漏、标签矛盾、跨 split 同源、哈希不一致、无法解释的收益/延迟跃升。
- 准备删除受保护上下文、上传用户转录、引入有副作用工具重放或连接 broker。
- 因测试失败希望删测试、放宽门槛或只报告某个 seed。
- 重要架构/依赖选择、额外付费资源或训练预算超出既定范围。

普通的局部实现、补测试、修 bug 可在当前已审核范围内继续。审核等待时可以做独立的只读调查/合成单测，但不得跨越被冻结的决策门。

## 6. 每个工作包的审核提交格式

在 `docs/EXECUTION_REVIEW_LOG.md` 追加，不覆盖历史：

```text
工作包 / 状态：NOT_STARTED | IN_PROGRESS | READY_FOR_REVIEW | BLOCKED | ACCEPTED
输入版本：commit + dirty diff，协议/数据/源码哈希
本次范围：改动路径、设计决定、明确未做事项
复现：命令、环境、退出码、日志/收据路径、耗时/费用
结果：完整种子/分组指标、失败、跳过、回归、区间
证据限制：合成/真实、回放/新推理、自动检查/独立审核分别说明
待审问题：推荐方案、替代方案、风险与回滚路径
下一步：通过后做什么；不通过如何收敛
审核结论：留空待审核，不自行填写 ACCEPTED
```

负面结果可以让实验完成，不能让部署 gate 通过。缺少真实数据时把对应工作包标 BLOCKED，列出具体缺口及可推进的独立事项，不把整项目标记完成。

## 7. 可直接转交执行 AI 的提示词

> 你接手 NanoJev。先读 docs/CURRENT_PROGRESS_AND_HANDOFF.md，再读 roadmap、CONTEXT_RELEVANCE_V1.md 和 FINANCIAL_PIT_V1.md。保护现有未提交变更与冻结产物，不重训 relevance、不调 0.99、不挑 seed 晋级。先完成 P0 核验与审核记录，并推进 P1 金融来源/权利/PIT/事件和 holdout 方案；R1 前不做正式金融训练。若无合法可用真实数据，明确 blocker，可做独立模拟器合成测试但不宣称金融效果。按工作包交付代码/测试/收据/失败列表和待审决策，达到 R0/R1 等门禁后整理材料给用户转交原审阅 AI。不要自动提交/推送、上传私密数据、改 provider、主动剪上下文或实盘交易。
