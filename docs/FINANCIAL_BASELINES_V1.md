# T11 金融基线实现与数据预检

日期：2026-09-20。状态：**协议/预检完成；真实 fit 尚未开始**。

## W39 更新：T11 专用协议已分离出 R1/T12 冲突

上一版 `results/financial_baselines_preflight_20260920_v1.json` 是保留的诊断收据：它把 R1
日历压力窗口当作训练/开发/校准排除，因此第一折 dev=0。它没有修改数据，也不代表 R1 失败。

当前 T11 专用协议为 [`research/financial_baselines_protocol_v1.json`](../research/financial_baselines_protocol_v1.json)，
冻结收据为 [`financial_baselines_protocol_freeze_20260920_v1.json`](../results/financial_baselines_protocol_freeze_20260920_v1.json)。
它保留 R1 三折与 instrument holdout（primary SOL/ETH、A BNB/XRP、B BTC），只允许 primary
进入 train/dev/calibration；三折 primary 计数为 746/24/24、928/24/24、1108/24/24。
R1 的五个 calendar stress windows 在 T11 仅作预注册的 test reporting strata，不驱动选择，
PIT regime masks 明确延后到 T12。这样没有删除或重写旧窗口，也没有用 test 解决 dev 冲突。

协议 validator 以代码锚定 SHA-256、根目录路径约束、上游 receipt 全量 hash、feature-only
投影、固定预算/seed、CE/Brier/ECE 定义和 false authorization flags fail-closed。最新预检为
`results/financial_baselines_protocol_preflight_20260920_v5.json`，状态
`protocol_valid_not_fit_authorized`。这仍不是独立 reviewer 的批准，也不是模型质量证据。

## 已实现

- `scripts/financial_baselines_v1.py`：验证 T6 v5 所绑定的 protocol/core/report/dataset/validator
  全部文件哈希；逐行检查 R1 schema/event；使用未修改 PIT validator 重建全部折分并严格比较。
- 预检按 instrument 分组，禁止 holdout instruments 进入 train/dev/calibration；
  压力窗口与标签区间有交集的行也从这三个阶段排除。每项排除保留 ID，不修改原数据。
- `scripts/financial_baseline_estimators_v1.py`：NumPy-only 训练集标准化、L2 logistic CE、
  exact binary Brier、Bernoulli gradient-boosted regression stumps、NLL/Brier/accuracy/
  10-bin probability-ECE、paired whole-group bootstrap CI。仅在合成单测中拟合过。
  Stumps 是浅层 GBM 基线，不冒称完整 GBM 库或最优树模型。

CE/Brier 使用相同全零初始化、seed 固定排列与更新步数、16 样本 minibatch（样本不足时使用
全体）。每步跨尾循环取样，保证匹配的样本预算。默认 steps=200、lr=0.05、l2=0.001；
GBM 默认 30 rounds/lr=0.1。**这些是实现默认值，不是已获审阅冻结的金融训练协议**。
二元 Brier 为 `(p-y)^2`，与项目其他多类别 summed Brier 的量纲不可直接混用。
Bootstrap 返回 candidate-minus-baseline 的 95% percentile CI；要求至少两个 group，
但具体时间块/source-group 定义还须冻结，不能因工具支持分组就声称已解决金融时间相关性。

## 旧版真实预检发现（保留为失败诊断）

收据：`results/financial_baselines_preflight_20260920_v1.json`，退出 2，`blocked_before_fit`。

| 折（从 0 编号） | 排除 holdout/日历压力后的 train | dev | calibration |
|---|---:|---:|---:|
| 0 | 564 | 0 | 22 |
| 1 | 670 | 24 | 24 |
| 2 | 850 | 24 | 24 |

冻结 R1 第一折 dev 为 2025-03-27 至 2025-04-28，整个落入
2025-02-15 至 2025-04-30 压力窗口。因协议要求从开发/选择中排除压力窗口，dev 必然为空。
这不是缺少用户授权，也不是 T6 哈希校验失败；T6 只证明原始 PIT 折分，未证明完整 holdout
方案可用于训练。不能跳过该折、用 test 选参或删除压力窗口以制造可运行结果。

旧版预检中的 instrument 序列化提案为 UTF-8 `venue:linear:asset_id`，按 SHA-256 排序取 2/2/1：
primary=SOL/ETH；A=BNB/XRP；B=BTC。原协议未定义连接分隔符，而 asset_id 本身不带 venue/type，
故本轮显式标为**待冻结提案**，不原地改 asset_id。压力窗口结束日期暂按包含当日解释，
转为半开区间；标签恰好结束在窗口起点也保守排除。这两项细节都需在新版中固定。

四类 expanding point-in-time regime 标记也未交付。尤其 `quote_volume` 不等于 30-bar median，
`last_funding_rate` 不等于 trailing-day settled funding；不能用现有字段静默替代协议定义。
预检的 train/dev/calibration 数量尚未应用这些 regime masks，因此不是最终可训练数量。

## 后续实施顺序

1. 由独立 reviewer 审阅 T11 专用协议/validator；在审阅与项目级 fit 授权前，runner 必须保持 fail-closed。
2. 补齐规则/base-rate/random/no-trade/long/flat/short 参照、train-only preprocessing 集成、
   dev-only 选择、calibration-only 校准、step-zero baseline 与三 seeds 全量 runner。
3. 从真实源历史生成四类 point-in-time regime masks（T12），声明 expanding quantile 的初始化、相等值、
   截止时刻和 label-overlap purge 规则，绑定源哈希；不能以重新命名普通 test 子组冒充 heldout。
4. 冻结 source-group/时间块 bootstrap 方案及各 fold/seed/instrument/regime 报告；
   按 review 条件运行真实基线，然后按区间报告优于确定性或未优于，不能预承诺正面结果。

用户已给予实验执行授权；这里保留的是未满足的技术条件，不反复要求相同授权。
本模块没有真实拟合 CLI；不得把其合成单测或预检当作模型质量、交易收益或 T11 完成证据。

## 验证

```sh
.venv/bin/python -m unittest discover -s scripts -p 'test_financial_baseline*.py' -v
.venv/bin/python -m unittest discover -s scripts -p 'test_financial*.py'
python3 scripts/financial_baselines_v1.py \
  --receipt results/financial_pit_r1_bound_receipt_20260920_v5.json \
  --output /tmp/new-unique-t11-preflight.json
```

30 项 T11 tests / 114 项金融 tests 通过。最后一个命令预期退出 2，输出采用 exclusive-create，
不覆盖已有收据。系统 python3 无 NumPy 时仅跳过 estimator suite；项目 `.venv` 已实跑全部测试。
