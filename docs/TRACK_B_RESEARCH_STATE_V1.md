# Track B 研究状态汇总（结构化版，截至 2026-09-26）

本文是 Track B 的唯一权威"我们到底知道什么"。原逐实验判定表（T11–T136）已收敛为
机制/袖子/坟场/基建/开放问题五层结构；实验→收据索引在文末附录。全部测量为 PIT-safe
trailing 窗口重放，protocol+owner 授权收据在 `research/` 与 `results/financial_signal_*`。

**数据基座**：`data/perp_pit_v1`（6,554 rec，5 资产日频 Binance）、`perp_pit_v2`
（3,266 rec，含 OI）、`perp_pit_xs_v1`（13,380 rec / 10 资产 / 2023-01→2026-08）、
`perp_pit_xs_v2`（~46k rec / top-30-qvol）、`perp_pit_mega_v1`（287,514 rec / 277 资产×5 年，
含 11 退市）、`perp_pit_mega_4h_v1`（1.28M rec / 283 标的）、Bybit/Hyperliquid 本地镜像。

## 1. 确认机制（durable findings）

- **总门 = BTC ret20（市场级开关，7+ 独立确认）**：spec 内 f2b2 +228.6 vs −52.3bps
  p=1.3e-5（T79）；Hyperliquid 场外复现 ρ=+0.061、不门控=powered null（T99）；
  XS dfh 上门 +111 / 下 −14bps（T106）；Bybit（T107）；277 资产 mega gate-on +192.6 /
  gate-off −63.7（T112）；参数化对决 ret20 完胜 atrp（t=+4.32 vs −0.83，atrp 是精化
  非替代；唯一例外是 funding carry 走 vol 门）（T114）；退出唯一可靠信号=门关闭后
  fwd5 −195.7bps t=−7.07 5/5 年（T118）；书级镜像对称（T134）。
  收据：`financial_signal_master_gate_v1.json`、`financial_signal_gate_shootout_v1.json`。
- **XS dfh 动量 = 三尺度复现的第二信号**：日频 5 资产 ρ=+0.031（T101）→ 10 资产 XS
  +54bps t=3.34 4/4 fold（T104）→ 277 资产 mega +68.9bps t=3.31 FDR 过 4/5 fold（T112）；
  Bybit XS +54.5 ≈ Binance +54.06（T107）；benchmark v4 `dfh20_three_scales` 全 PASS。
  资产内 ρ≈0 → 纯横截面效应；月尺度高点才有（dfh10/1d null）；含 beta 成分。
- **镜像结构：动量住门内、反转住门外**：日频反转 ρ=−0.035 真实但净 −3.9bps/日
  不可交易，且集中在 gate-OFF（T132）；书级完美对称 mom +16.3/−11.1 × rev +9.8/−7.5
  （T134）——反转边被换手吃掉，只有动量边可货币化。
- **上市年龄条件（交互而非年龄本身）**：dfh×age 8/8 门——d90-365 新币 +110.8bps、
  gt365 老牌币**反转** −85.2bps（最清晰统计量）；纯年龄排序 null（T131）。
  用法=book overlay，不进模型（T135）。
- **funding carry 是条件信号不是常数**：日频 crowded-long carry follow 只在 BTC 上行
  门内（spec v1）；小时级只在正 funding regime（ρ +0.048/−0.007，T94）；4h mega 翻号
  不泛化（T124）；XS funding carry 的 +76bps 是 2021 事件+3 币集中运气（T115）。
  尺度不对称定论：小时级用 funding-regime 门，日频用 BTC 趋势门。
- **结算锯齿污染窗（诊断必修）**：结算为中心 ±2h 弱 dip、中周期正、两 funding 符号
  同 dip（非支付机械）、2h 最强 −6.27bps、高 rv 限定（T93）；剔除该窗后 funding_pct
  PASS→FAIL（污染确认，T96）；mega 4h 复现 −5bps（T124）。文献未见发表→潜在小型
  新发现；扣费不可交易。
- **mom_4h 反转 2025 真死**：70× 复现证伪"负 regime 排他"（T96 是小样本伪影），
  但 2025 衰减至 0 且 incumbent 同死（非组成伪影）（T124/T126）。对照：日频 XS dfh
  2025 未衰减（标准化后最佳年）；funding 类信号因费率钉 0 饿死。
- **入场/退出状态测量（滚仓时机）**：门内近高/extended 入场 +189-195bps 优于
  pullback +1.7bps（日频反转了兄弟项目 4h 结论）；pullback+低 funding=一致性最优；
  avoid breaking/底分位（T117）。退出定型=hold until gate-off + E4 快速回撤灾难止损；
  E2 狂热 +511bps 是延续信号非退出（T118）。

## 2. 可货币化袖子（记账口径，非可交易声明）

| 袖子 | 形态 | 净表现 | 状态 |
|---|---|---|---|
| spec v1 `crowded_long_carry_follow_v1` | funding≥80pct ∧ basis≥66pct ∧ BTC ret20>0 → 5-10d 多头，门关闭/时间停退出 | 净 +151.8bps/笔（基准 60.8）；条件策略 +294.8bps/笔但稀疏（110 笔） | **冻结评审候选**（`research/financial_signal_spec_v1.json`，T81）；ledger v1 前瞻中；残差 fold1 时期效应/右尾 |
| `xs_dfh_carry_v1` | XS dfh top2/bot2 + ret20 门 | 净 +13.5bps/日，Sharpe 0.99，turnover 27%；全 horizon/深度/lookback 过、placebo 归零（T106） | **spec_candidate_pending_review**（`research/financial_signal_spec_xs_dfh_v1.json`，T109——冻结前抓回 SMA20 漂移）；ledger v2 回填 +10.8kbps |
| ridge_min3@top30 | 3 特征 {dfh20, btc_ret20, 交互}，LS top2/bot2 | frozen-fit 净 +7.28bps/日 Sharpe 0.666 四年全正（T127）；expanding_monthly 重训 +24.2bps/日（T136）；top30 域匹配修复 2026 反传：OOS book +83 vs +5bps（T130） | 生产模型；ledger v4/v4_xs2_ridge_min3_top30；14 特征膨胀反噬、long-only 边际 +1.75 |
| age overlay | dfh×age 反传感知书叠加（book-only） | +8.20bps/日 Sharpe 0.778 vs 普通 0.245（T135）；v5 回填 +15.2kbps Sharpe 0.653（T148） | 上线；**年轻池已空**——固定宇宙新币均老过 365d，短期只有空头腿直至宇宙扩新 |
| M-gate-on 书 | 门内动量单边（镜像书动量腿） | 净 +141.8%，Sharpe 0.744（T134）；联合门控系统 +160% Sharpe 1.43 mdd 17.9% active 53% vs always-long +137%/0.69/69.2%（T111） | 描述层最优书；反转腿不叠加（MR 组合 0.66<0.744） |
| campaign v3 | 复合入场战役模拟（majors-only） | **0.432x（−56.8%）如实报亏** | 上线但诚实反映"该宇宙不适用"——保留作对照而非声称 |

统一 caveats：① 记账口径非成交——无市场冲击模型、funding 计提为近似；
② xs_v2/mega 宇宙按 qvol 选有幸存者偏差味道（mega 含 11 退市、survivorship control
过，但 live 宇宙是幸存者集）；③ **2026 是 hard window**——联合系统该年 Sharpe 0.72
最弱（T111）、v4 曾 OOS 反传（T128，域匹配已修但窗薄）；④ 全部 paper/前瞻 ledger，
无真实交易授权；⑤ ρ≈0.04 属弱但真实成分——定位是状态特征/条件层而非独立策略。

## 3. 证伪/关闭（坟场，一行一尸）

- OI 臂：v2 cohort null（T72）→ 52 标的复测全 |t|<0.7、FDR 幸存=日组成伪复制（+456→−3.9bps/日）——**此粒度彻底关闭**（T116）
- XS funding carry：+76bps = 2021 事件（ex-2021 −0.24bps）+ 前 3 币 65.6% PnL 集中——噪声非 spec；stealth-rally 亚细胞留观察（T115）
- BTC→alt lead-lag：滞后残差≈0 微负=beta 同步衰减+共同均值回归（T95）；真实发现是大 BTC bar 时 alt vol +50%——诊断用
- Amihud 流动性翻转：方向反了（T101b）；Zaremba 翻转再证伪（T132）
- 战役/金字塔加仓机制：roll 机制 4h 减损（rolled 0.63x < flat 1.48x < hold 2.99x）、原则化重建 0/9 格胜 hold（T121/T123）
- 手调止损/参数：36 格中位 0.653x 跑输 hold、默认格 rank 2/36=过拟合、ex-2021 0.219x——**34.4x 判 parameter_lucky**（T122）；可辩护残差仅"回调带+真止损"定性结构
- SMA20 门变体：实测门是 ret20>0，spec 冻结前抓回漂移（T109）；T114 再确认 ret20 完胜
- funding-regime 排他性：负 regime 反转排他=小样本伪影（70× 下普适，T124）；T105 排他性仅 confirm 半区成立
- 特征膨胀 ridge：14 特征破坏 fold 一致性（2/4），min3 唯一净正；仅交互+xs_rank_mom 系数跨期稳（T127）
- 年龄进模型：pooled ρ 升但 OOS 净 Sharpe 降（0.594 vs 0.802）——只做 book overlay（T135）
- recent-only/窄域训练：recent 窗 −0.107 反噬、xs10-domain 减半反传；**只有 top30-qvol 域匹配有效**（已应用，T130）
- 冻结模型 vs 重训：frozen 最差（+4.8bps/日）、rolling/recency 单调变差——定案 expanding_monthly 全史扩窗（T136）
- 早期坟场（保留）：funding fade 方向、H3 同义反复、H4 波动非方向、funding 动量/streak/极值非单调、跨场所 funding 价差（Binance↔Bybit 近同步，T80）、DOW 季节性、25bps/1d 二元标签（压掉连续结构——本研究最重要方法学发现）、全局线性形态（T78：pooled 真但跨期不稳——后被 T125 以交互项+门控结构复活）

## 4. Live infrastructure

- **9 个前瞻 ledger**（`results/forward_ledger_*.jsonl`，append-only 幂等）：
  v1=spec-v1 5 资产；v2/v2_xs2=XS dfh（xs_v1/xs_v2 宇宙）；v3/v3_xs2=战役（报亏对照）；
  v4/v4_xs2=ridge_all；v4_xs2_ridge_min3_top30=域匹配生产臂；v5=age overlay。
- **健康探针** `check_forward_ledgers_all_v1.py` → `results/forward_ledgers_all_v1_state.json`
  （9 账本，最近检查 7 个新鲜、v4_xs2 孤儿标注；已知伪影：尾行 cumulative 漂移=
  append-only+迟到 bar 重定价）。
- **launchd 链** `ai.nanojev.forward-ledger.plist`：每日 07:10 refresh→ledgers→snapshot→
  health log，已 installed+kickstart exit 0。运维教训：**launchd 下 TCC 授权按解释器
  二进制算**——必须用 `.venv/bin/python` 非 `/usr/bin/python3`（T97）。
- **refresh 管线** `refresh_binance_cohort_v1.py`（5 资产）/ `refresh_binance_xs_v1.py`
  （xs_v1）/ `refresh_binance_xs2_v1.py`（xs_v2 top30）；已修复尾重写在瞬时失败时丢 bar 的
  脆弱点（T129）。
- **benchmark v4 记分卡** `financial_signal_benchmark_v4.py` →
  `results/financial_signal_benchmark_v4.json`：7 块 19 PASS / 6 FAIL，7 个独立 BH-FDR
  家族、确定性×2 字节一致；dfh 三尺度 cross_scale 全 PASS；ridge_min3 fold-rho 2/4
  如实记 FAIL；mega_4h 反转+结算窗复现入榜。

## 5. 开放问题（下一判定点）

- **2026 OOS verdict pending**：域匹配后 v4_xs2_ridge_min3_top30 OOS spearman
  −0.027→+0.144、净值 1.126→1.494（T129/T130），但 2026 本身是 hard window（T111
  该年最弱 S0.72）；前瞻 ledger 继续积累，未到判定点。
- **young-pool revival**：固定宇宙新币全部老过 365d——age overlay 多头腿需宇宙扩新币
  才复活（T148）。
- **Valen regime 头 vs ridge**：`data/valen_fin_v1/` 20k train/3k eval 已建（T143）、
  T145 head-only 训练中断续跑中——自训决策头能否在 gate/regime 判断上胜过 ridge_min3。
- **paper-trade→spec-v2 晋级门**：xs_dfh_carry_v1 停在 spec_candidate_pending_review——
  需定义前瞻 ledger 证据阈值（多少 OOS 日、什么 net/drawdown 下限）才晋级 spec v2。
- **残留时期效应**：spec v1 fold1 测试窗所有门下仍为负——门控消不掉的时期项未解。

## 6. 发表边界（per AGENTS.md）

- **可发表**：pinned protocol + 公开输入下由本项目实测的聚合 benchmark 对比
  （NanoJev/官方 Jev 分数），或有出处+快照日期+对比范围的公开数字引用。
- **本地限定**：全部 ledger/receipt/探索笔记、官方 Jev raw per-item 输出与
  side-by-side 收据（publish aggregates, not raw responses）；Jev 输出永不入训练/
  校准数据。
- **本文所有信号净收益=记账口径研究结论**，非可交易策略声明、非盈利承诺；
  RLCD/金融拟合需新 protocol+owner 授权；无真实交易授权。

## 附录：实验→收据索引（原判定表压缩）

T63-65 假设重放 `financial_signal_hypotheses_v{1,2,3}`；T66 标签重设计 `labels_v4`；
T67 净成本 `net_backtest_v1`；T68-71 网格/精化/harness
`robustness_v1`/`conditional_v1`；T72 OI `oi_v1`；T78 首个拟合 `fit_v1`；T79 门
`regime_v1`；T80 价差 `crossvenue_v1`；T82 benchmark v1/v2 `benchmark_v{1,2}`；T86-88 小时级 `intraday_v1`；
T89 5 月扩展同前；T93 结算 `settlement_v1`；T94 funding regime `funding_regime_v1`；
T95 lead-lag `leadlag_v1`；T96 反转 `reversal_v1`；T99 HL `hyperliquid_v1`；T101 dfh `dfh_v1`；T102 XS pilot `xs_pilot_v1`；T103 合成 spec v2
`composite_v1`；T104 全史 XS `xs_v1`；T105 反转 OOS `reversal_oos_v1`；T106 稳健
`xs_dfh_robust_v1`；T107 Bybit `bybit_dfh_v1`；T108 benchmark v3 `benchmark_v3`；
T109 spec `research/financial_signal_spec_xs_dfh_v1.json`；T110 ledger v2
`forward_ledger_v2*`；T111 总门 `master_gate_v1`；T112 mega `xs_mega_v1`；T114 门对决
`gate_shootout_v1`；T115 carry 解剖 `xs_carry_diag_v1`；T116 OI mega `oi_mega_v1`；
T117 入场 `entry_state_v1`；T118 退出 `exit_state_v1`；T119 战役 `campaign_sim_v1`；
T120 v3 ledger `forward_ledger_v3*`；T121 4h roll `campaign_4h_v1`；T122 网格
`campaign_grid_v1`；T123 原则化 `campaign_v2_principled_v1`；T124 mega 4h
`mega_4h_v1`；T125 ML 探针 `ml_probe_v1`；T126 衰减解剖 `decay_v1`；T127 ridge v2
`ridge_v2_v1`；T128 v4 ledger `forward_ledger_v4*`；T129 宇宙 30 `xs_v2` 管线；
T130 域匹配 `domain_match_v1`；T131 年龄 `newlist_v1`；T132 反转 mega
`reversal_mega_v1`；T133 benchmark v4 `benchmark_v4`；T134 镜像书 `mirror_book_v1`；
T135 年龄进书 `age_model_v1`；T136 重训 `refit_eval_v1`；T143 决策数据集
`data/valen_fin_v1/`；T148 v5 ledger `forward_ledger_v5*`；T149 全探针
`forward_ledgers_all_v1_state.json`。
