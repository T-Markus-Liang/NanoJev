# 金融信号标准 Benchmark V1

统一评测合约下的 Track B 信号记分卡。一条命令重放全部臂 + 对照。

```bash
python3 scripts/financial_signal_benchmark_v1.py
# -> results/financial_signal_benchmark_v1.json (deterministic, byte-identical x2)
```

## 评测合约（冻结）

| 项 | 值 |
|---|---|
| Cohort | `data/perp_pit_v1/records.jsonl`（6,554 recs，receipt 内钉 sha256） |
| 主标签 | `log(mark[t+5]/mark[t])` 连续 5 日收益 |
| 参考标签 | `up>25bps/1d` 二元（列存） |
| Folds | `research/financial_r1_pit_validator_core_v2.json` 冻结 test 窗 |
| PIT | trailing-180 mid-rank 分位，per-asset |
| 成本 | 5bps taker×2 + funding 计提（rate×24/interval 每持有日） |
| 持仓 | 5 bar，per-asset 不重叠 |
| 判定 | **PASS** = FDR(α=0.05) 过 + \|rho\| > placebo 带；ECON_ONLY = 净胜基准但无统计显著；否则 FAIL |

## 控制（benchmark 纪律）

- **Placebo**：标签在 asset 内确定性打乱（3 seeds）——真信号 rho 必须超出 placebo 带
- **确定性**：`run()` 两次 payload 字节一致（实测 ✅）
- **基线**：always-long 同节奏净统计（mean +22.4bps，med −7.0）

## 记分卡（results/financial_signal_benchmark_v1.json @ sha 673528c8…）

| cell | ρ / p | arm n | arm 毛 bps | 净 bps | folds+ | placebo ρ | 判定 |
|---|---|---|---|---|---|---|---|
| **funding_pct** | +0.042 / 0.0015 | 562 | 204.3 | 168.0 | 3/3 | ≤0.012 | **PASS** |
| **basis_pct** | +0.037 / 0.0057 | 562 | 241.2 | 207.4 | 2/3 | ≤0.019 | **PASS** |
| rv_pct | −0.020 / 0.13 | 562 | 12.4 | −5.7 | 0/3 | ≤0.019 | FAIL |
| abnvol_pct | +0.030 / 0.026 | 562 | 101.9 | 81.4 | 2/3 | ≤0.020 | **PASS** |
| taker_pct | +0.009 / 0.52 | 562 | −3.9 | −18.3 | 1/3 | ≤0.040 | FAIL |
| fmom_pct | +0.000 / 0.98 | 562 | 79.9 | 56.0 | 2/3 | ≤0.016 | ECON_ONLY（尾部假象） |
| **fxb（funding×basis）** | +0.043 / 0.0011 | 562 | 232.0 | 194.2 | 2/3 | ≤0.014 | **PASS** |
| **f2b2（指示）** | in-vs-out t=5.6 / ~0 | 862 | 185.9 | 152.0 | 2/3 | — | **PASS** |
| **f2b2_btc（+BTC趋势门）** | t=6.4 / ~0 | 731 | 228.6 | 193.1 | 2/3 | — | **PASS** |
| **tri（+vol门）** | t=3.97 / 7e-5 | 355 | 225.0 | 187.5 | 2/2 | — | **PASS** |

## 读法

- **4 个连续特征真实**：funding / basis / fxb / abnvol（弱）。rv 和 taker 纯噪声。
- **指示单元最强**：f2b2_btc（双高+BTC上行）臂内净 +193bps，对比基准 +22——9 倍。
- **placebo 带 ≈ ±0.02**：任何 rho<0.02 的"发现"不可信；funding/fxb 超出 3-4×。
- **folds+ 列是稳定性指示不是门**：3/3 最好，2/3 是常态（crypto 时期效应真实存在）。
- fmom 的 ECON_ONLY 正是 benchmark 价值：它净胜基准纯属右尾噪音——ρ=0 且无 placebo 区分度。

## 边界

记账口径（无冲击模型）；日频；5 资产；ρ~0.04 级弱信号；PASS ≠ 可交易。新臂接入合约：
在 `CONTINUOUS`/`INDICATORS` 加特征名 + `build_dataset` 算特征 + 进同一 FDR 家族。

## v2：新增 intraday 跨尺度层（T91）

```bash
python3 scripts/financial_signal_benchmark_v2.py
# -> results/financial_signal_benchmark_v2.json (deterministic, byte-identical x2)
```

- **daily 块原样重放** v1 合约（导入 v1 代码路径，payload 与 v1 receipt 字节等价），**不改 v1**。
- **intraday 块**：`data/perp_pit_intraday_v1/records.jsonl`（17,960 recs，5 资产 × ~3,592h，2026-04-02..08-31，sha 钉在 receipt），标签 `label.forward_return_bps`（4h 前向 mark bps），trailing-90-record PIT；7 cells：funding/basis/rv/taker/abnvol_pct + mom_1h/mom_4h；同构指标（spearman + Welch quintile + top-decile arm 净统计 + placebo + PASS/ECON_ONLY/FAIL）。n_eval=17,490（warmup 470 丢弃）。
- **FDR 家族按尺度分开**：daily 10 cells 与 intraday 7 cells 各自做 BH-FDR（α=0.05），不跨尺度合并。
- **intraday 判定（5 月扩展集）**：funding_pct **PASS**（ρ=+0.026/p=6.5e-4，placebo 带 ±0.016）、taker_pct **PASS**（ρ=−0.030/p=8.9e-5）、mom_1h **PASS**（ρ=−0.035/p=3.6e-6）、mom_4h **PASS**（ρ=−0.047/p=4.5e-10）、abnvol_pct ECON_ONLY、basis_pct FAIL、rv_pct FAIL（ρ=−0.018 落在 placebo 带 ±0.034 内）。注：mom/taker 的 ρ 为负（反向关系）；顶十分位臂净均值 taker −6.6 / mom_1h −8.3 / mom_4h −9.5bps vs 基准 −8.1——mom 两臂低于基准，"PASS" 是统计检出而非多头经济价值。
- **cross_scale 块**：对比两尺度 funding_pct 的 ρ/p/方向并标复制状态。扩展集结果：daily ρ=+0.042/p=0.0015 PASS，hourly ρ=+0.026/p=6.5e-4 PASS（placebo 带 ±0.016）→ **replicated_both_scales**，同向。相对 2 月窗口（hourly ρ=+0.051/p=2e-5）funding ρ 减半——与 T89 结论一致：level 效应在正 funding regime 更强；扩展后仍过 FDR+placebo 双门但边际明显变薄。
