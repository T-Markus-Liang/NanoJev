# Track B 信号/数据假设短名单（T62）

状态：调研完成，**未训练**。用途：把排名最高的假设送入 review gate（T63），通过后才允许在
6,554 条 PIT cohort 上跑 T11/T14 类实验。全部假设必须与现有 PIT 合约兼容（1 日决策 bar，
11 特征 allowlist 范围内或明确标注需要新数据）。

## 现有数据（不需要新数据源）

`mark_price, index_price, mark_index_basis_bps, last_funding_rate, funding_interval_hours,
open_interest_level, open_interest_log_change_1d, quote_volume, trade_count, taker_buy_ratio,
realized_vol_24bar`（4 场所：Binance/Bybit/Aster/Hyperliquid）

T11/T14 阴性结果说明：**这 11 个特征的线性/常规监督用法没有信号**。下面的假设区别在于
**条件化形式**——文献表明这些变量只在尾部/特定状态下含信息，全样本线性会把它平均掉。

## 排名

| # | 假设 | 外部证据 | 数据可行性 | 测量计划 |
|---|---|---|---|---|
| **H1** | **极端负 funding → 正向 24h 收益**（非线性、不对称、只在尾部） | AIJMR 2026：Binance 实测，极端负 funding 后 24h 均值 +0.506%，t/Wilcoxon/方向检验+FDR 全部显著；极端正 funding 无对称效应 | ✅ `last_funding_rate` 已有；需 180 日滚动分布做分位 | cohort 内重放：funding 分位<10% 的子样本 vs 基线收益；控制 pre-settlement 4h 收益；HAC 回归 |
| **H2** | **funding z-score 拥挤度反向（fade）** | crypto-edge-search OSS：funding 作拥挤/情绪代理，z-score 极值 fade；CTF 研究：funding 对单资产预测弱、**横截面更有效** | ✅ 同 H1；横截面版需要多资产 cohort（当前主要是 BTC/ETH? 需查） | funding z>2 → 反向持仓；z<-2 → 正向；与 carry 剥离（只测方向 overlay） |
| **H3** | **mark-index basis 极值收敛** | 机制先验：basis 是 perp 与 index 的偏离，理论上回归 | ✅ `mark_index_basis_bps` 已有 | basis |z|>2 子样本的次日 basis 变化；basis 极值与次日收益的联合分布 |
| **H4** | **OI 变化 × 价格背离（OI 升+价格平 → squeeze 前置）** | 经典微结构假设，无强文献但机制合理 | ✅ `open_interest_log_change_1d` + mark_price 已有 | OI 分位 >90% 且 |价格变化|<中位 的子样本次日收益 vs 基线 |
| **H5** | **taker_buy_ratio 极值（订单流失衡代理）** | 文献中 OFI 效应集中在盘中/分钟级；日级弱 | ⚠️ 特征有但频率可能不足 | taker ratio 分位尾部子样本次日收益；预期弱，作为对照臂而非主假设 |
| **H6** | **跨场所 lead-lag**（Binance 领先→小场所跟随） | 微结构文献支持，但**效应在盘中**，日 bar 基本无滞后空间 | ⚠️ 4 场所数据有但 1d 粒度不够 | 仅记录为"需要更高频数据"的候选，不在本批测量 |

## 明确否决（文献已证伪或数据不可得）

- **delta-neutral funding carry**：Mykola-Quant 预注册证伪——BTC/ETH/SOL 全变体 OOS 净收益为负，成本下限问题。不进 shortlist。
- **清算瀑布（liquidation cascades）**：R1 已记录无历史清算数据源；MarketTensor 同样注明"feature hooks with explicit source errors until a reproducible historical source is added"。**数据不可得，阻塞。**
- **链上/交易所净流量**：免费历史源覆盖不足，阻塞。

## 推荐进入 review gate 的顺序

**H1 → H2 → H3**。H1 的外部证据最强（专门设计过非线性尾部检验且结果显著），且恰好解释
T11 阴性的可能原因——线性模型把尾部效应平均掉了。H2 与 H1 共享数据但方向框架不同
（拥挤度 fade vs 极端反弹），可同时进 gate 做对照。H3 是机制先验，成本最低的独立检验。

## 边界

- 所有测量计划都是 **PIT cohort 重放**，不产生新交易行为；
- 任何假设通过测量后仍需独立 review 才进入 estimator/RLCD 实验；
- 本调研不构成任何信号有效性或盈利性声明——文献结果不等于在我们数据/协议下复现。

## 第二轮调研更新（2026-09-24，T64）

**Meta-prior（最重要的一条信息源）**：github.com/kimlage/crypto-edge-search —— 开源证伪实验室对 ~167 个 crypto 假设做反过拟合筛选，最终 **0 个干净幸存**。与我们相关的判决：funding fade **方向就是反的**（"extreme funding persists"，placebo 动量方向反而赢）；XS funding-rank carry / funding momentum / 跨场所 funding 离散度 / DOW 季节 / vol-targeting / 残余动量（30 资产太薄）/ perp-spot cash-and-carry 全部 KILL。

**实测更新**：H1 证伪、H3-sharp 证伪（证实 H3 是均值回归同义反复）、H4 仅波动效应、H5 噪声（`financial_signal_hypotheses_v1/v2`）。

### 修正后的排名

| # | 假设 | 证据 | 状态 |
|---|---|---|---|
| **H6** | **funding 同向跟随**（level/分位高 → P(up) 高；与 H2 fade 方向相反） | SSRN Crypto Carry（funding 正向预测收益）；edge-search 实测 fade 方向反转 | 下一轮首选 |
| **H7** | **异常成交量 → 次日负向收益**（disagreement/discount） | Garfinkel & Sokobin：**Binance 日频实测 −0.498%/day，α 显著**——和我们数据源+粒度完全匹配 | 下一轮第二 |
| **H8** | **basis 动量**（Δbasis 5 日 → 次日方向） | Chi et al：日频因子 t=7.10（dated futures，perp 需验证） | 下一轮第三 |
| **H9** | **vol 条件层**（`realized_vol_24bar` 分位做交互而非独立信号） | JFQA：高 RV → 低后续收益；Sentinal 案例 regime gate 提升 Sharpe | 作为 H6-H8 的分层条件 |
| H10 | 滞后收益 × 成交量交互反转 | 文献：低活跃币有日频反转、大盘币是动量 | 可选 |

### 纪律提醒（沿用证伪实验室的教训）

- 5 资产做横截面排序只有 ~4 自由度——只做**池化 per-asset 时序臂**，不碰 cross-sectional rank。
- funding 方向假设的外部证据自相矛盾（carry 正 vs fade 杀 vs momentum 杀）——在我们自己的 cohort 上按同协议测，别信任何单边文献。
- 每条新臂成本极低（同一统计机器），但预注册少而精：一次最多 3-4 臂进 FDR。
