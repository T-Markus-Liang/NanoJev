# 执行审核日志

按 [进度与 AI 交接计划](CURRENT_PROGRESS_AND_HANDOFF.md) 第 6 节格式追加记录。**历史条目只追加，不覆盖。** `审核结论` 留空表示待审阅 AI 填写；执行者不得自行填写 ACCEPTED。

---

## P0 — 证据收口与接手核验

**状态**：READY_FOR_REVIEW
**记录时间**：2026-09-19（本地）

### 输入版本

- HEAD：`be0303c4c8360902f3fc670800e9a5ef866cd79a`（`Add shadow context gating and financial PIT validation`）
- 已跟踪修改（工作区）：`docs/NANOJEV_V2_ROADMAP.md`、`docs/CONTEXT_SHADOW_V1.md`、`docs/WORKFLOW_V2_BASELINE.md`、`scripts/context_gate_v1.py`、`scripts/train_pipeline_decisions.py`
- 未跟踪新增（收口对象）：`docs/CONTEXT_RELEVANCE_V1.md`、`docs/CURRENT_PROGRESS_AND_HANDOFF.md`、`docs/JEV_COMMUNITY_REFERENCES.md`、`research/context_relevance_v1_protocol.json`、6 个 `results/context_relevance_oracle_v1_*.json`、`scripts/build_context_relevance_v1.py`、`scripts/report_context_relevance_v1.py`、`scripts/test_build_context_relevance_v1.py`、`scripts/test_report_context_relevance_v1.py`、`scripts/test_training_runtime.py`
- 冻结协议哈希：`b5d03717d91ad13e…`（`research/context_relevance_v1_protocol.json`）
- 数据清单哈希：`b08cc0058d19cdc3…`（`data/context_relevance_oracle_v1_seed20260919/manifest.json`）
- 环境：Python 3.14.6，torch 2.14.0，MPS 可用

### 本次范围

只做核验与文档化，未重训、未跑 checkpoint 推理、未改冻结产物：

1. 用冻结脚本独立重建 relevance 汇总报告到**新路径**（不覆盖原报告）。
2. 比对新报告与冻结报告的语义内容、部署结论与 provenance 哈希。
3. 运行主单元测试套件与本地 skill 测试套件，记录基线通过数。
4. 建立本审核日志。

### 复现

```bash
.venv/bin/python scripts/report_context_relevance_v1.py \
  --output results/context_relevance_oracle_v1_report_recheck_001.json
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
.venv/bin/python -m unittest discover -s integrations/codex-skill/nanojev-local-decider/scripts -p 'test_*.py'
```

退出码均为 0。

### 结果

- 独立重建报告：`results/context_relevance_oracle_v1_report_recheck_001.json`
- 与冻结报告比对（剔除仅反映"报告时刻"的 `source_files` 后）：**语义完全相等**。
  - 三种子 `best_step` 均为 68；`splits`、`maze`、`checkpoint` 逐项相等。
  - `deployment` 仍为 `not promoted; production checkpoint unchanged`。
  - 冻结与重建的 `source_files` 哈希集合完全一致；协议哈希、数据清单哈希一致。
- 测试基线（本轮文档化之前的工作区状态）：主套件 **165 项，OK（2 skip = 163 执行通过）**；本地 skill 套件 **3 项 OK**。
- **测试计数对账（A4+P2 并入后）**：主套件 **213 项 OK（2 skip）**，且 165（基线）+ 13（A4）+ 35（P2）= 213，**无未解释增量**。合并后全量回归通过，说明新增工作包未破坏既有测试。
- 本地 V1.0 MPS 参考（用于后续 E1 对照，取自既有 `results/nanojev_v2_baseline_seed17.json`）：device `mps`，precision `fp32`，load 7,109.9 ms，warm p50 571.3 ms，warm p95 590.1 ms，49.15 questions/s，peak RSS 5,276,991,488 B；test acc 77.84%，OOD acc 76.56%。

### 未提交运行时改动审查

对两项已跟踪但未提交的脚本改动做了逐行审查：

1. `scripts/context_gate_v1.py`：把原本内联在 `shadow_request` 里的评分载荷构造抽成共享函数 `scoring_payload(segments, candidates)`，训练与推理因此使用**同一个渲染器**。这是必要的正确性改进（训练/推理提示词漂移会导致闸门失效），且不改变 `shadow_request` 的对外行为。`users` 仍在 `shadow_request` 内用于"无用户消息则 bypass"的守卫（第 287 行），无死代码。
2. `scripts/train_pipeline_decisions.py`：改为设备无关运行时。`autocast` 使用实际设备类型而非硬编码 `"cuda"`；`model.to(device)` 取代 `model.cuda()`；CUDA 随机种子仅在 CUDA 下设置；新增 `--device {cuda,mps,cpu,auto}`；配置记录 `runtime_device`；MPS 下 `max_gpu_allocated_gb` 记为 `null`。
   - **关键协议实现**：`best` 现在初始化为未改动的初始化 dev target CE、`best_step = 0`，并在训练前把初始权重写入 `best.safetensors`。这正好实现冻结协议里的"包含初始化 step 0；平局保留更早 checkpoint"，避免未训练候选"默认获胜"。
   - 新增守卫：初始 dev target CE 非有限时直接报错；`reward_generator` 仅在 `paired_brier_pg` 时创建。
   - **必须记录的约束**：新增的 `training_runtime()` 明确拒绝在 MPS 上运行 `paired_brier_pg`（"MPS sampled policy-gradient training is not validated; use CPU or CUDA"）。即 **RLCD 式采样式策略梯度估计器无法在本机 Mac 的 MPS 上运行**，Track B 的 RLCD-like 实验只能走 CPU（慢）或 CUDA。这是诚实的限制声明，但直接影响本机金融工作包的可执行范围，已列入待审问题。
   - `resolve_runtime` 为 HEAD 已有函数（非本次引入），其错误信息为中文，与仓库其余英文风格不一致 —— 既有小瑕疵，非本轮回归，建议列入清理候选而非阻塞。
3. 新增 `scripts/test_training_runtime.py`（3 项）覆盖新函数的 CPU/MPS/标志组合，含 MPS + `paired_brier_pg` 必须报错、CUDA-only 标志在非 CUDA 报错。

### 金融 PIT 基座复核

P1/P2 都建立在 PIT 校验器之上，因此在推进金融工作包前先核验其可复现性：

```bash
.venv/bin/python scripts/benchmark_financial_pit_v1.py --output results/financial_pit_v1_stress_recheck_001.json
```

- 结果：`status=passed`，12,000 条合成记录，0.61 秒完成。
- 与冻结收据 `results/financial_pit_v1_stress.json` 比对：除 `elapsed_ms`（计时，本就应变化）外**全部字段一致**，包括 `dataset_sha256=5bcd04f0105258e8…`、three-fold 计数（train 2979 / dev 1179 / calibration 1179 / test 1200）、`future_data_mutations_rejected=1000`、`label_isolation_mutations=1000`、`embargo_ns`、`asof_ns`。
- 结论：合成时间戳/划分完整性检查是确定性的、可独立重建的。这**仍然只证明构造样例上的完整性**，不证明真实数据真实性、预测准确率、成本或收益 —— 与 `FINANCIAL_PIT_V1.md` 的声明一致。

### 证据限制

- 报告重建只重新汇总既有预测并复核数据/配置/收据哈希；它**不是**对训练过程无泄漏的独立审计，也不是新的 checkpoint 推理复现。
- 单元测试通过证明代码路径自洽，不证明 relevance 数据集 oracle 正确或划分无争议。
- 报告内 `provenance_note` 明确：源码哈希是报告时捕获，不是训练时的追溯签名。
- 原收据含本机绝对路径；跨机器迁移需显式记录路径映射，本次未做迁移。

### 待审问题

1. 是否接受 P0 的核验深度（重建一致性 + 哈希复核 + 套件通过）作为 relevance 实验的收口证据？还是要求额外的独立数据/oracle 审计？
2. relevance 三种子产物是否现在随文档一起提交（Git 提交/推送仍需用户单独授权）？
3. 冻结无效前身产物（`data/context_relevance_v1*`、`runs/context_relevance_v1_seed17`）继续保持不删除、仅审计的现状，是否确认？
4. 是否接受把上述两项未提交脚本改动（`context_gate_v1.py` 共享渲染器、`train_pipeline_decisions.py` 设备无关运行时与初始化 checkpoint）作为 relevance 实验的配套运行时一并收口？
5. MPS 不支持 `paired_brier_pg` 这一限制，是否接受"本机 Mac 上金融 RLCD-like 实验只走 CPU 或外部 CUDA"作为既定约束（还是要求先做 MPS 采样 PG 的验证任务）？

### 下一步

- 通过后：P0 状态置 ACCEPTED，转入 P1/P2 审核门（见后续条目）。
- 不通过：把缺口列回 P0，不进入金融工作包的正式训练阶段。

### 审核结论

（留空待审阅 AI 填写）

---

## A4 — 工具历史 compaction shadow fixture 与工具链

**状态**：READY_FOR_REVIEW
**记录时间**：2026-09-19

### 输入版本

- 基线 HEAD `be0303c`；未修改任何既有文件（`git status` 仅显示新增）。
- 冻结协议阈值保持 0.99，未调参。

### 本次范围

新建（4 个文件）：

- `research/tool_history_fixture_manifest_v1.json`（55,214 B，`status: frozen_before_shadow_materialization`，11 用例）
- `scripts/build_tool_history_fixtures_v1.py`（种子确定性构建器；拒绝非空输出目录；`--self-test` 不写盘）
- `scripts/test_tool_history_fixtures_v1.py`（13 项测试）
- `docs/TOOL_HISTORY_SHADOW_V1.md`（设计规格 + 明确未做清单）

### 复现

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_tool_history*.py'
.venv/bin/python scripts/build_tool_history_fixtures_v1.py --self-test
```

执行者报告：13 项测试 OK（13/13）；`--help` 退出码 0；`--self-test` → `status ok`，cases 11 / source_groups 11 / scored 8 / bypass 3 / `wrote_output false`；两次干净重建字节相同；`--seed 12345` 改变字节但仍通过校验；对非空目录二次写入被拒绝。**审阅方独立复跑确认：13 项 OK，self-test `status ok`。**

### 结果

- 用例覆盖：success、error、parallel_calls、duplicate_ids、orphan_result、pending_call、non_repeatable_result、mutable_file_read、stale_output、user_correction、secrets_credentials。
- 字节保真：每个用例断言 `shadow_request(...)[0] is raw`、`request_sha256 == forwarded_sha256`、删除片段/token 为 0、阈值 0.99。
- 复用方式：只读 import `context_gate_v1`（`shadow_request`、`parse_segments`、`Bypass`、`FORMATS`、`serialized`），未 fork。
- 无网络、无 provider 代理、无主动删除、无生产路径改动。

### 已知歧义（执行者记录，待审阅方裁定）

1. `openai_chat` 的并行调用会合并为单个 `/messages/N/tool_calls` 片段，无法建立严格 1:1 call→result 映射；`parallel_calls` 因此用 `anthropic_messages` 表达，配对校验改为"不重叠 + 全覆盖"而非基数相等。
2. orphan / duplicate id / pending 场景下 core 抛 `Bypass("unresolved_tool_link")` 且枚举零片段；manifest 声明**意图**配对并强制 `pairing_resolved: false`，不猜测归属。
3. `pending_call` 的 `result_segments` 合法为空，校验器要求 `pairing_resolved: false`，避免"空结果集看起来完整"。
4. `depends_on` 方向：仅当**受保护**片段声明依赖某可删片段时才保护它；`user_correction` 中依赖闭包被真实触发。
5. `keep_call_bounded_result` 仅在 `mutable_file_read` 允许；error/credential/non-repeatable/stale/correction 禁止，因为被省略的部分**就是**证据。`remove_pair_atomically` 仅限可重复、非易变、无标志的读取对。
6. `ambiguity_probe`（修复 call-id 拼写 / 补上缺失结果）只用于机器校验声明的结构契约，**从不物化**。

### 证据限制

- 全部为合成自撰 fixture；无真实用户转录、无真实凭据、无真实工具输出。
- 未做 shadow receipt 实跑、未做下游质量对比、未做阈值拟合。
- `keep_call_bounded_result` 是**已声明但未实现**的契约：core 没有"显式标记有界"的字段。
- 单一 seed 下的确定性已验；跨平台字节一致性未验。

### 待审问题

1. 上述 6 条歧义裁定是否接受？特别是 `parallel_calls` 改用 anthropic 格式、以及 orphan/pending 强制 `pairing_resolved: false` 的处理。
2. `openai_chat` 无法表达 1:1 配对这一事实，是否需要在 gateway（G1）侧单独设计 call/result 归属，还是接受该限制？
3. RA 门禁剩余项（三操作实跑 receipt、结果省略 vs 含证据 judge state 配对比较、三主模型族下游成本/正确性）是否按现顺序推进？

### 下一步

通过后进入 RA：先补 bounded-result 编码契约，再用假上游/本地服务做三操作 shadow receipt，最后才是跨模型族的下游质量与净成本比较。

### 审核结论

（留空待审阅 AI 填写）

---

## E1 — 社区参考的 Mac 本地只读实测

**状态**：READY_FOR_REVIEW
**记录时间**：2026-09-19

### 本次范围

只读实测，未改冻结产物、未安装进 `.venv`、未接入生产：

1. reflex 演示页零安装可用性检查。
2. laya 隔离安装（`pip --no-deps --target /tmp/laya_pkg`）+ MPS 实测。
3. 本机 NanoJev 服务的同形延迟对照（临时端口 8799，测完即停）。

### 复现与结果

**reflex（浏览器，零安装）**

- `https://kshetrajna12.github.io/reflex/` 加载成功，标题 `reflex · decisions, with probabilities, in your browser`，页面说明与上游 README 一致；`navigator.gpu` 存在（WebGPU 可用）。
- **未完成的验证**：未在本机实际加载其 0.8B ONNX 模型（约 650MB）并计时。浏览器实例当时被子代理占用，故模型加载/推理计时**留待补测**。因此本项只证明"页面可用、WebGPU 具备"，不证明其浏览器推理速度或与 Python 路径的一致性。

**laya（隔离安装，MPS）**

- 隔离安装 `laya==0.3.2` 到 `/tmp/laya_pkg`（`--no-deps`，复用 `.venv` 已有的 torch 2.14 / transformers 5.17），**未污染项目环境**。
- `laya.load("convaiinnovations/laya")` 成功，**自动选择 `mps:0`**；加载 241.8 s（含 28 个文件下载，非纯加载时间）。
- 4 个问题（1 choice / 1 score / 2 noul）单状态，进程内 warm 延迟 5 次：[48.5, 47.3, 40.8, 38.4, 40.4] ms → **p50 = 40.8 ms，min = 38.4 ms**。
- 输出合理：department = billing（0.9628）、urgency = 1.44、churn_risk = 0.8248、is_phishing = 0.112。

**本机 NanoJev 服务对照（同形请求）**

- `scripts/serve_decisions.py`，checkpoint `local_atomic_seed17`，MPS fp32。
- 请求 `local_test_input.json`：1 状态 / 3 问题（1 choice 3 选项 + 1 boolean + 1 score 4 级）。
- HTTP warm 延迟 5 次：首次 818.3 ms，随后 [77.6, 73.9, 73.9, 75.2] ms → **warm p50 ≈ 74.9 ms**。

| 系统 | 设备 | 问题数 | 测量方式 | warm p50 |
|---|---|---:|---|---:|
| laya 0.3.2（421M 编码器） | mps:0 | 4 | 进程内 | **40.8 ms** |
| NanoJev V1.0（0.6B 解码器 + 决策头） | mps | 3 | HTTP（含 tokenize/往返） | 74.9 ms |

### 结果解读（严格限定）

- 在本机 MPS 上，laya 用**更多问题**（4 vs 3）达到了约 **1.8×** 更低的 warm 延迟。这与"编码器比解码器更适合做本机低延迟闸门"的假设一致，也是 E1 的核心发现。
- **但这不是受控对比**：问题数不同、测量方式不同（进程内 vs HTTP 含往返）、候选数不同、模型任务不同。不得据此声称 NanoJev 慢于 laya，也不得据此声称闸门能降低 token。
- 与 README 记录的"warm MPS ≈65 ms / 3 问题 8 候选"量级一致，说明本机环境正常。

### 证据限制

- 全部为 upstream-weights 测量，不构成 NanoJev 能力声明。
- laya 加载时间含下载，未单独测量纯加载。
- 未做内存峰值测量（对照的 NanoJev 峰值为既有收据 5,276,991,488 B）。
- 未跑 laya 的 `laya-multilingual` 或 `laya-typed-decisions`，未做其自带 benchmark。
- 未做本地 zero-shot 探测（laya deliverable 2 未执行）。

### Scorer 选型分析（供 G1 接线与 RE 审核使用）

用户指令要求"提速 + 降低 token 消耗"，而 G1 网关的 scorer 是可插拔的，因此第一个要定的是**用哪个本地模型当闸门**。基于本机实测与既有收据：

| 候选 | 本机 MPS 实测 | 已知删除行为 | 校准状态 | 上下文 | 许可 |
|---|---|---|---|---|---|
| NanoJev V1.0（0.6B 解码器+决策头） | warm p50 ≈ 74.9 ms / 3 问题（HTTP） | 0.99 下**全部弃权**，实际节省 0 | ECE 0.0851/0.0734（迷宫） | 512 | 本项目 |
| relevance 三种子（同 backbone 微调） | 未单独测延迟 | seed 17 低删除率/0 误删、seed 18 **有误删**、seed 19 删除率 22–28%/0 误删；**均未晋级** | 相关性二分类 ECE 0.019–0.059 | 512 | 本项目 |
| laya 0.3.2（421M 编码器） | **warm p50 = 40.8 ms / 4 问题（进程内）** | **未评级**：其能力在自己的 typed-decisions 上，未在我们的相关性任务上验证 | 出厂过自信（自述 ECE 0.466→0.081 需重拟合温度） | 512/1024 | Apache-2.0 |

**建议（待 RE 审核，非既定结论）**：

1. **G1 首个接线 scorer 用 NanoJev 本地服务**（已实现、可复现、`/api/evaluate` 现成）。这能立刻让网关端到端跑通，但**预计 shadow 报告的假设性节省为 0**——这本身是正确的基线事实，不是失败。
2. **不要**为了得到非零节省而把 seed 19 直接接成生产闸门——那是用已看过的留出结果挑赢家，roadmap 明令禁止。
3. laya 的约 1.8× 延迟优势只支持"编码器路线值得立项"，**不支持**把它直接当闸门：它没有在我们的相关性任务上验证过删除安全性，且需在 calibration split 上重拟合温度。其 Apache-2.0 许可对本地使用友好。
4. 真正的非零节省只能来自**一个通过验证的闸门模型**。合法路径有两条：(a) 用**新的**预注册协议与新鲜确认集重训 relevance 类闸门；(b) 把 laya 编码器在我们的相关性数据上微调后按同样门禁验证。

**证据限制**：上表延迟数字跨越不同测量方式（进程内 vs HTTP）、问题数与任务，**不是受控对比**；"删除行为"一列来自合成最新记录依赖课程，不等于真实长上下文压缩表现。任何"能省 X% token"的表述，在拿到配对下游质量与上游回报 token 数之前都只能是估计值。

### 待审问题

1. 是否要求补做 reflex 浏览器模型加载/推理计时（需独占浏览器一段较长时间）？
2. laya 编码器闸门架构问题（laya deliverable 4）是否授权立项？如授权，应给出与 G1 网关的接口边界。
3. 是否要执行 laya 在冻结 workflow 样本上的 zero-shot 探测（deliverable 2）？注意其自述 base zero-shot 接近随机，预期结果可能是"无区分度"，该结果本身有信息量但也可能无助于决策。

### 下一步

- 通过后：E1 收口为只读收据；架构立项需单独 RE 审核。
- 不通过：保留测量，按审阅意见补测或终止该探索线。

### 审核结论

（留空待审阅 AI 填写）

---

## P2 — 执行模拟器与模型外风控（合成基础）

**状态**：READY_FOR_REVIEW
**记录时间**：2026-09-19

### 输入版本

- 基线 HEAD `be0303c`；未修改任何既有文件。
- 全部数据为合成构造；未下载行情、未连接 broker、未声称收益。

### 本次范围

新建（5 个文件）：

- `scripts/financial_simulator_v1.py`（65,760 B）
- `scripts/financial_risk_v1.py`（28,322 B）
- `scripts/test_financial_simulator_v1.py`（31,143 B，34 项测试）
- `results/financial_simulator_v1_stress.json`（压力收据）
- `docs/FINANCIAL_SIMULATOR_V1.md`（17,588 B）

### 复现

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_financial_simulator*.py'
.venv/bin/python -m py_compile scripts/financial_simulator_v1.py scripts/financial_risk_v1.py
```

**审阅方独立复跑：35 项测试 OK（Ran 35, OK）；语法检查通过。**（注：审阅方首次复跑时为 34 项，执行者随后补齐至 35 项，以 35 为准。）

### 结果（来自 `results/financial_simulator_v1_stress.json`）

- **守恒**：`ok = true`，10 项残差全部在 `tolerance = 0.001`（scale 1e6）内；最大残差为 `cash_vs_cash_entries = 2.33e-10`（浮点噪声）；`violations = []`。
- **确定性**：两次运行 `byte_identical = true`，`ledger_sha256 = 0b197edaa333af53…` 一致。
- **无未来信息**：`decision_leaks_attempted = 200` → `decision_leaks_blocked = 200`；另有 `post_execution_price_leaks_blocked = 1`、`strict_execution_leaks_blocked = 1`；`replay_guard` 决策检查 38 次、执行检查 35 次，`violation_count = 0`。
- **成本单调性**：5 档 `fee_bps` 下 `violations = []`（提高费用不会提高净 PnL）。
- **订单生命周期**：`order_statuses_observed` 覆盖 accepted/filled/rejected/partially_filled/expired；四个场景各自体现 no_fill / rejection / partial / baseline 行为（如 `partial`：33 单全部部分成交、583 笔 fill、33 单到期）。
- **风控**：position / gross_exposure / net_loss / turnover / order_rate / stale_data / clock_drift / kill_switch 各自验证 block 与（position、turnover）reduce，并带 reason code；`replay_all_blocked` 下 0 成交、5 次风控拦截。
- **待 R1 冻结参数**：收据显式列出 `provisional_pending_r1` 共 **38 项**（fees/spread/slippage/capacity/latency/fills/marks/risk_limits 等），策略版本标记 `v1-provisional-pending-r1`。

### 审阅发现（我方）

1. **决策计数残差未暴露（系统性，非个例）**：`decisions` 恒为 60，但 `abstain + no_trade + risk_blocked + orders` 无法闭合，残差**随场景变化**（baseline 2、no_fill 22、partial 7、rejection 22）。

   **审阅方的归因是错的，已由执行者在 P2b 中纠正并修正**：我当时追查代码看到 `outcome` 默认 `no_order`，便判定残差是"未生成订单"的决策。执行者实现该计数字段后发现 `no_order` 在四个场景中**均为 0**，真正的残差是 **`rejected_decisions`**——**因库存不足被拒绝的卖单**。计数现在精确闭合，且闭合性在压力测试中被断言：`abstain + no_trade + risk_blocked + rejected + no_order + ordered == decisions`。

   **教训（对审阅方法）**：我依据"默认值即真相"做了推断，未验证默认值在该路径上是否会被覆盖。**这是本次审阅中我唯一的实质性判断错误**，如实保留以警示后续审阅者不要以代码默认值代替实际计数。
2. ~~**文件权限不一致**：新文件为 `600`，仓库既有文件为 `644`。~~ **审阅方已于本次复查中统一为 `644`**（内容未改动），该项已关闭。
3. `forbid_post_decision_prices_in_executions = False` 是**有意**的保守策略（允许执行延迟内价格变动），但该项也属于待 R1 裁定的语义，需在 R1 明确其对"无未来信息"的实际边界。

### 执行者自报的未建模缺口（审阅方确认其已写入规格，非隐瞒）

- **无现金/购买力/保证金检查**：订单可能超出可用现金，该路径**未建模也未测试**。对 P3 策略效用比较是实质性缺口，R1 必须裁定。
- **成交模型是种子抽样 × 参与率上限**，不是队列/订单簿重建；换种子即换成交。不得当作真实成交证据。
- 延迟为固定叠加、无抖动/排队；B7 的三个延迟层级未在此测量。
- 场所级库存与借券约束未建模（做空被拒绝而非建模）。
- 报价的 point-in-time 真实性无法由合成测试证明，属 P1 来源/权利审计范围。
- `marks.fallback` 仅在构造性 as-of 路径可达，用直接策略测试覆盖而非真实端到端回放。
- 严格泄漏开关编码的是**解释选择**而非可证不变量，故默认关闭。

### 证据限制

- 全部为合成报价与合成决策；无真实品种、交易所、行情或 broker。`scope` 字段已如此声明。
- 通过测试只证明**实现自洽**，不证明撮合模型真实、成本假设正确或策略有效。
- 38 项策略参数与 4 项契约裁定均为初始猜测/未定，必须在 R1 冻结后才可用于任何正式实验；在此之前不得用于选参或比较策略收益。
- 未与 PIT 校验器的真实分折协议联调（仅消费其概念契约）。

### 待审问题

1. 是否接受合成基础作为 P2 第一切片通过，并在 R1 冻结上述 38 项参数？
2. `no_order_decisions` 计数是否要求立即补齐（我方建议：补齐，成本极低）？
3. 无现金/购买力检查这一缺口，是在 R1 一并裁定（新增账户层）还是先声明"资金无限"边界？
4. 4 项契约裁定（持仓按资产聚合/成交模型形状/参考价取中间价/借券与做空）如何在 R1 定稿？
5. `forbid_post_decision_prices_in_executions` 的边界如何在 R1 表述，才能既允许真实执行延迟又保持"无未来信息"门禁？

### 下一步

- 通过后：等 R1 冻结策略语义 → 与 PIT 分折协议联调 → 确定性/CE/exact-Brier 基线（P3）。
- 不通过：按审阅意见修改策略契约与计数，不改动合成测试的完整性。

### 审核结论

（留空待审阅 AI 填写）

---

## P1 — 金融数据/权利/PIT/实验契约草案（停在 R1）

**状态**：READY_FOR_REVIEW —— **按计划停在 R1 审核门，未越过**
**记录时间**：2026-09-19

### 输入版本

- 基线 HEAD `be0303c`；未修改任何既有文件；**未购买数据、未下载任何数据集、未连接 broker**。
- 冻结参照：`docs/FINANCIAL_PIT_V1.md`、`scripts/financial_pit_v1.py`。

### 本次范围

新建（3 个文件，全部标记 draft）：

- `docs/FINANCIAL_DATA_PLAN_V1.md`（55,738 B）
- `research/financial_source_manifest_v1.json`（57,110 B，14 个数据源）
- `research/financial_experiment_protocol_v1.json`（37,569 B，`draft_pending_R1: true`）

### 复现与独立验证

```bash
.venv/bin/python -c "import json,pathlib; [json.loads(pathlib.Path(p).read_text()) for p in ('research/financial_source_manifest_v1.json','research/financial_experiment_protocol_v1.json')]"
```

**审阅方独立验证（机检）**：

- 两个 JSON 均可解析。
- **校验器兼容性声明经验证为真**：`pit_validator_core` 的顶层键集**恰好**等于 `{folds, embargo_ns, asof_ns}`；真实校验器签名为 `walk_forward(rows, folds, embargo_ns, asof_ns)`，**接受该 core 的全部三个键**；3 个 fold 各自 train/dev/calibration/test 窗口有序，测试窗**相邻且不重叠**（fold2 test 起点 == fold1 test 终点）。
- 该文件**故意不扁平化**：携带 event/label/feature/universe/holdout/seed 声明，超出校验器顶层契约；协议内已给出把 `pit_validator_core` 投影为独立文件的明确命令，而非假装直接可读。

### 结果

**14 个数据源已调研**，含 URL、许可、成本、访问方式、字段、时间戳语义、修订策略、退市覆盖、风险与决策：

| 决策类别 | 数据源 |
|---|---|
| `recommended_first_market` | **binance_vision_public_archive**（CC BY，许可已验证） |
| `ranked_alternative` | databento、norgate_data（许可已验证） |
| `plumbing_fixture_only` | kaggle_huge_stock_market_dataset（CC0） |
| `supplementary_context_only` | fred_alfred |
| `deferred` | coinbase_exchange_api、cme_group_eod |
| `blocked` | crsp_us_stock（机构许可，无法核实） |
| `rejected` | kaiko（商业）、alpha_vantage（个人非商用）、stooq、yahoo_finance_yfinance（许可未核实） |

推荐理由与替代项已在计划文档中给出；`cannot_verify_without_payment`、`open_questions`、`scope_limitations` 三节显式列出无法在不付费情况下核实的项。协议含 `event_definition`、`label_predicate`、feature allowlist、universe 构建、instrument/regime holdout、≥3 个声明种子与 `expected_exclusions`。

### 证据限制

- 全部为**桌面调研**：未购买、未下载、未做真实性验证。**许可条款必须由用户/人工独立确认**后才能使用任何数据源。
- 计划自身声明：设置 `available_ns` **不证明** point-in-time 真实性；来源许可、修订/复权处理、退市完整性、事件定义正确性均需单独的原始来源审计。
- 协议是 draft，**尚未**在任何真实数据上运行；校验器兼容性是结构性的，不代表数据可用。
- 推荐首个市场落在加密资产（Binance Vision）主要因其公开、免采购、许可清晰；这不构成"加密更适合策略"或任何收益性主张。

### 待审问题（R1 必须裁定）

1. **是否批准 Binance Vision 公开归档作为首个市场**？若否，指定替代或要求先解决某来源的许可。
2. **事件定义与 label predicate** 是否按草案冻结（含 horizon、成交/观测价格约定、可用时刻）？
3. **split 几何与 embargo** 是否按草案冻结（3 fold、expanding train、单侧 embargo 3600 s、asof 2027-04-01、持仓期与 regime holdout 方案）？
4. **≥3 个训练种子**的具体取值是否批准？
5. 真实数据不可得时是否接受"标记 BLOCKED 并只推进合成测试"，而非降级为无数据声称？

### 下一步

- **通过 R1 前：不开展任何正式金融训练、不购买受限数据、不为收益挑参。**
- 通过后：按 R2 推进（执行模拟器策略语义冻结 → PIT 联调 → P3 确定性/CE/exact-Brier 基线）。

### 审核结论

（留空待审阅 AI / 用户填写）

---

## A4×G1 — 跨工作包集成验证（审阅方）

A4 的 fixture 是直接针对 `context_gate_v1.shadow_request` 构建的，G1 是 HTTP 网关；**没有任何验证确认过 A4 声明的语义能在网关路径上成立**。审阅方补齐了这一空白：把全部 11 个 fixture 通过**真实网关进程**（`--mode active` + 恒删存根 scorer，即对每个候选都返回 `p_irrelevant = 0.999`，最激进情形）发送，再与每个 fixture 自带的 `expected_gate` 声明比对。

```bash
.venv/bin/python scripts/build_tool_history_fixtures_v1.py --output-dir /tmp/th_fx --seed 20260919
.venv/bin/python /tmp/a4_through_g1.py
```

### 结果

| 检查 | 结果 |
|---|---|
| 用例数 | 11 |
| gate `status` 与声明一致 | **11/11** |
| gate `reason` 与声明一致 | **11/11** |
| `all_retain` 保留语义一致 | **11/11** |
| 收据含 `leak_tokens` 的用例 | **0** |
| 收据全部内容无关 | ✅ |

### 逐用例行为（最激进 scorer 下）

| 用例 | status | reason | 是否缩减 |
|---|---|---|---|
| success / error / parallel_calls / non_repeatable_result / mutable_file_read / stale_output / user_correction / secrets_credentials | `scored` | `shadow_only` | 是 |
| duplicate_ids / orphan_result / pending_call | **`bypass`** | **`unresolved_tool_link`** | **否（零删除）** |

**三个工具链接歧义用例（重复 ID、孤儿结果、挂起调用）在网关路径上正确整体 bypass，不做任何归属猜测**——这正是 A4 设计的核心安全属性，现已确认它在 HTTP 网关层同样成立。

### 对"删了什么"的逐条核查（审阅方）

在恒删 scorer 下 8 个用例真的发生了删除，因此审阅方逐条检查了被删对象：

- 删除**仅限助手草稿或对工具值的复述**（如 `'...note the answer does not need.'`、`'The ... setting was v1 at read time.'`、`'The recorded ... quote was 100.'`）。
- **权威证据始终保留**：`mutable_file_read` 的工具结果 `{"zqi_setting":"v1"}` 与 `stale_output` 的工具结果 `zqj-SYN=100`、用户修正均未被删。
- 与 A4 的设计说明一致：`stale_output` 的说明本就写明"对过期值的冗余助手复述仍属可删候选"；`mutable_file_read` 禁止的 `remove_pair_atomically` 未被使用。
- 交叉验证：fixture 的 `leak_tokens`（合成实体标记）在**转发字节中保留**、在**收据中不出现**——保留与隐私两个方向同时成立。

**结论：未发现安全违规。** 最激进 scorer 加上 active 模式这一最坏组合下，工具结果、系统指令与用户消息仍全部保留，且歧义链接整体回退。

### 证据限制

- 存根 scorer 是**人为构造的最激进情形**，不代表任何真实模型的评分分布；真实 checkpoint 提议删除 0 个。
- 该验证证明**结构保护与回退在网关层有效**，不证明真实语义相关性判断正确。
- 11 个 fixture 为合成自撰；不构成真实工具历史的覆盖。
- 全程 loopback，无真实 provider、无网络。

---

## G1 — 主模型工作流网关（用户指令）

**状态**：READY_FOR_REVIEW（代码与测试已完成并验证；设计文档 `docs/MAIN_MODEL_GATEWAY_V1.md` 仍在收尾）
**记录时间**：2026-09-19

### 背景与范围

用户明确指令：本地决策模型必须接入主模型工作流以提速、降低 token 消耗。项目此前**没有**任何网关/代理组件，本地模型无法参与主模型请求，因此这是真实的功能空白，而非已有能力的参数调整。

已确定的实现约束（写入派发规格）：

- 回环监听；OpenAI `POST /v1/chat/completions` 与 Anthropic `POST /v1/messages` 透传。
- **默认 shadow**：转发原始字节、上游响应原样返回；active 显式开启且必须 fail-open。
- 复用 `context_gate_v1` 的解析与保护逻辑，不重新实现。
- token 口径：只有真正发送缩减请求时才用上游回报的 prompt token 数声明实际节省；shadow 只给估计值并标注 tokenizer。
- receipt 不含原始提示词/凭据；提供全局 kill switch。
- scorer 可插拔：NanoJev 本地服务 / 进程内 scorer / 可配置的 laya 型服务。
- 测试必须使用**回环假上游**，绝不调用真实 provider，不新增第三方依赖。

### 代码级审查（审阅方，测试与文档尚在收尾）

对 `scripts/main_model_gateway_v1.py`（27,790 B）与 `scripts/scorer_adapters_v1.py`（10,444 B）逐段审查，两个文件均通过 `py_compile`。**已核实的不变量**：

1. **默认 shadow**：`MODE_SHADOW` 为默认，`--mode` 默认 `"shadow"`；`GatewayConfig.__post_init__` 拒绝未知模式。
2. **fail-open 链完整**（`handle()`）：`unsupported_method` / `unsupported_wire_format` / `kill_switch` / `gate_error`（`shadow_request` 任意异常）/ `plan_error`（受保护片段进入删除集）/ `reduction_error`（缩减失败或重校验失败）——**任一情况均置 `forwarded = raw` 并记录 `forward_reason`**。上游不可达时返回 502 JSON 占位，不泄漏上游细节。
3. **受保护片段结构上不可能被删**：`removal_plan()` 只接受 `role == "assistant"` **且** `reason == APPLIED_DROP_REASON` 的 `suggestion == "drop"`；其余一律返回空计划 + `protected_segment_in_removal_set`。
4. **耦合点已验证**：网关常量 `APPLIED_DROP_REASON = "high_irrelevance_score"` 与 core 在 `context_gate_v1.py:330` 对可删片段发出的原因码**一致**，故 active 模式**确实能产生缩减**，而非永远 fail-open。若 core 未来改码，网关会安全地退化为 fail-open（失效方向安全）。
5. **缩减仍然守门**：`build_reduced_request()` 要求非空计划、指针格式与索引边界合法、结果非空，并强制 `reduced_request_would_lose_user_intent`（**不得丢失任何 user 消息**）；随后在 `handle():367` 用 `parse_segments(candidate, wire_format)` 做 core 重校验（非重新实现）。
6. **token 口径诚实**：shadow 且未缩减 → `claim="estimate"`（`basis="shadow_estimate_only"`）；active 未缩减 → `claim="none"`；仅在拿到**上游回报的配对基线**时 → `claim="actual"`（`provider_paired_baseline`）。骨架标注 `scope="content_free_accounting"` 与 `local_estimate_not_provider_billing`。
7. **收据无原始内容**：骨架字段仅含 schema、event_id、mode、哈希、原因码、指针、计数与 policy 阈值；嵌套的 `gate_receipt` 来自本就无内容的 shadow core。日志写入失败被吞掉，不影响转发。
8. `log_message` 被覆写为不记录请求行/头。

**次要观察（非缺陷）**：`build_reduced_request` 的 docstring 称"用 core parser 重新校验"，该调用实际位于调用方 `handle()`；对整体流程的表述属实，仅位置描述不够精确。

**待验**：`scripts/test_main_model_gateway_v1.py` 与 `docs/MAIN_MODEL_GATEWAY_V1.md` 尚未落盘。回环假上游测试是本节结论能否成立的关键——**在测试通过前，以上仅为静态代码审查，不是运行时证据**。

### 运行时验证（测试已落盘，审阅方独立复跑）

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_main_model_gateway*.py'
```

**结果：30 项测试 OK（约 14–16 s）。** `docs/MAIN_MODEL_GATEWAY_V1.md`（15,008 B）已落盘。

**审阅过程记录（重要）**：审阅方首次复跑时**捕获到 6 项失败**（`shadow`/`active`/`fail-open` 各组），失败原因多为断言 `gate_receipt.status == "scored"` 却得到 `"bypass"`。随后复跑转为全绿。**审阅方已核查该修复是修 fixture 而非放宽断言**：`test_shadow_forwards_original_bytes_byte_identically_and_only_estimates` 至今仍严格断言 `status == "scored"`、`proposed_pointers == ["/messages/1/content"]`（精确值）、`claim == "estimate"`、`basis == "shadow_estimate_only"`、`tokens > 0`；修复方式是为 fixture 补上 `eligible_sidecar()`，使闸门真正进入评分路径。**结论：修复正当，非测试弱化。**

**fail-open 覆盖（执行者自报 12 类，审阅方抽查确认测试强度充足）**：scorer 异常 / 超时 / 缺失、分数畸形/部分/非有限/重复/非单位和、不确定分数、唯一候选为受保护或 pinned、gate 计划含受保护删除（防御性重检）、sidecar 头不可读、缩减构建或重校验失败、不支持的 wire format 与非 POST、kill switch、gate 抛错——**全部转发原始字节**。另验上游 503 原样透传、上游不可达 502。

**审阅方抽查的三个高强度测试**：
- `test_protected_drop_in_gate_plan_is_refused_defensively`：**注入伪造的 gate receipt**（含 `protected_structure` 与 `required_dependency` 删除），断言 `forward_reason == "protected_segment_in_removal_set"`、原始字节转发、未应用。
- `test_removal_plan_helper_rejects_every_non_eligible_drop`：对 system 角色、`caller_protected`、`required_dependency`、user 角色四种情况做子测试，全部必须返回空删除集。
- `test_scorer_timeout_fails_open`：慢 scorer + `score_timeout=0.05`，断言 `reason == "scorer_error"`、`scorer_failure_kind == "timeout"`、转发未变、未应用。

**无外部依赖验证**：`import main_model_gateway_v1, scorer_adapters_v1` 后 `'laya' in sys.modules` 为 **False**（laya 从未被导入/安装/启用）。测试文件中的 `https://` 仅出现在**配置校验**用例（拒绝 `ftp://`、拒绝 URL 内嵌凭据 `https://user:pass@…`、断言默认 shadow），实际测试服务器全部绑定 `127.0.0.1` 回环。

**测试计数对账**：165（基线）+ 13（A4）+ 35（P2）+ 31（G1，含审阅方新增的 CLI 回归测试）= **244**，全量套件 244 项 OK（2 skip）。

### 审阅方发现并修复的缺陷（高严重度）

**`main()` 无法启动网关**——CLI 入口完全不可用。

- 现象：`.venv/bin/python scripts/main_model_gateway_v1.py --upstream … --receipt-log …` 立即崩溃：
  `AttributeError: 'GatewayConfig' object has no attribute 'config'`（`main_model_gateway_v1.py:562`）。
- 根因：`make_server(gateway)` 读取 `gateway.config.listen_host`，因此期望 `ContextGateGateway` 实例；而 `main()` 传入的是裸 `GatewayConfig`（第 598 行）。测试之所以全绿，是因为它们走 `gateway = ContextGateGateway(config)` → `make_server(gateway)` 这条**库路径**，从不执行 `main()`。这是典型的"测试通过、产品不可用"。
- 影响：用户明确要求的"接入主模型工作流"交付物**无法从命令行启动**，因此静态审查与 30 项库测试都不足以宣告可用。**这也是对本条目先前"已核实默认 shadow/`--mode` 默认值"结论的更正——我此前从源码读取 CLI 而未曾实际运行它。**
- 修复：`main()` 改为 `server = make_server(ContextGateGateway(config))`（一行）。修复后 CLI 正常打印 banner（`mode: shadow`、`scorer: none`）并持续服务，无 stderr。
- 回归覆盖：审阅方在 `test_main_model_gateway_v1.py` 新增 `CliEntryPointTest`，以子进程方式真实启动 CLI、读取 banner、断言默认 shadow 且进程存活。
- **反向验证**：审阅方把缺陷重新注入后运行该测试 → `FAILED (failures=1)` 并复现原始 `AttributeError`；随后用带 `trap` 的备份恢复原文件并确认 CLI 再次正常。
- 影响面（需 R1/RG 知悉）：先前一切"网关已通过验证"的说法仅在**库层面**成立；CLI 可用性是本次修复后才成立的。CLI 现已实际跑通。

### 真实 scorer 端到端验证（审阅方，CLI 修复后）

单元测试只使用**假 scorer**，真实 `/api/evaluate` 路径从未被端到端跑过。CLI 修复后，审阅方用**真实的 NanoJev 服务（`local_atomic_seed17`，MPS）** + 回环假主模型完成了第一次真实集成验证：

```bash
# 真实决策服务 + 网关(shadow, --scorer http --scorer-url http://127.0.0.1:8799) + 假上游
.venv/bin/python /tmp/g1_e2e_shadow.py
```

请求：OpenAI chat 格式，system（受保护）+ assistant（sidecar 标记 eligible）+ user（当前意图）；上游返回 `usage.prompt_tokens = 412`。

| 检查 | 结果 |
|---|---|
| 网关启动 | ✅ banner `mode: shadow`、`scorer: http` |
| shadow 转发原始字节 | ✅ `forwarded_bytes_identical: true`；`request_sha256 == forwarded_sha256` |
| 上游响应原样透传 | ✅ `upstream_reply_passed_through: true` |
| Bearer 凭据不透明转发 | ✅ `auth_forwarded_opaquely: true` |
| 内部头不泄漏上游 | ✅ `internal_headers_leaked_upstream: []` |
| 收据无原始文本 | ✅ `receipt_leaks_raw_text: false`（探测 `warehouse moved`、`invoice #4411`、`sk-fake` 均未出现） |
| token 口径 | ✅ `claim="estimate"`、`basis="shadow_estimate_only"`、`tokens=0` |
| 上游 token 计数 | 412（来自上游 usage 字段） |
| 闸门耗时 | 223.8 ms（真实 MPS 推理往返） |
| 总往返 | 227.6 ms |

**真实闸门提议删除：0 个**（`gate_status: bypass`、`gate_reason: uncertain_score`）。

**这是本次交付最重要的诚实结论**：网关在真实模型下端到端工作正常，但**当前 checkpoint 不提议任何删除，因此实际 token 节省为 0**。这与此前 shadow 基准记录的结论一致，不是网关缺陷。任何"已降低 token 消耗"的表述在此证据下都不成立。

**尚未做**：`active` 模式的真实端到端缩减（需先有一个通过验证的闸门模型）；真实 provider 调用；配对下游质量。

### 进程级 active 模式验证（审阅方）

CLI 缺陷正藏在 `main()`，因此"active 缩减路径从未作为真实进程跑过"是此前的盲区。审阅方用 `--mode active` + 存根回环 scorer（对唯一的 eligible 候选给出 `p_irrelevant = 0.999`，即**确定性提议删除**）+ 回环假主模型完成验证：

```bash
.venv/bin/python /tmp/g1_active_e2e.py
```

| 检查 | 结果 |
|---|---|
| 网关启动 | ✅ banner `mode: active`、`scorer: http` |
| **发生了真实缩减** | ✅ `active_mode_reduced: true` |
| `applied` / 指针 | ✅ `true`；`["/messages/1/content"]` |
| 转发角色 | `["system", "user"]`——被删 assistant 消息整条移除 |
| 归档片段已移除 | ✅ `archived_segment_removed: true` |
| **用户意图保留** | ✅ `user_intent_preserved: true`（`invoice #4411` 仍在） |
| **系统指令保留** | ✅ `system_preserved: true` |
| `forward_reason` | ✅ `active_reduced` |
| **token 口径（关键）** | ✅ `claim="estimate"`、`basis="provider_usage_observed_no_paired_baseline"`——**即使真的发送了缩减请求，未提供配对基线时仍拒绝声称 `actual`** |
| 估计节省 | 10（空白分词估计值） |
| 上游回报 token | 300（取自上游 `usage` 字段） |
| 收据无原始文本 | ✅ `receipt_leaks_raw_text: false` |
| 上游响应原样透传 | ✅ |

**结论**：两种模式均已通过真实进程验证——shadow 转发原始字节并如实报告 0 节省；active 转发缩减字节、保留受保护片段与用户意图、并如实区分 estimate 与 actual。

### token「actual 节省」记账路径验证（审阅方）

`actual` 是唯一能合法支撑"降低 token 消耗"这一说法的口径，而它此前从未作为真实进程跑过。审阅方用 active 模式 + 恒删存根 scorer，对同一请求跑两次（仅差一个配对基线头）：

```bash
.venv/bin/python /tmp/g1_actual_savings.py
```

| 情形 | claim | basis | tokens |
|---|---|---|---|
| **无**配对基线（但**确实发送了缩减请求**） | `estimate` | `provider_usage_observed_no_paired_baseline` | 6（空白分词估计） |
| **有**配对基线（`x-nanojev-baseline-provider-prompt-tokens: 412`） | **`actual`** | `provider_paired_baseline` | **112** |

- 算术已核对：`412（调用方提供的上游基线）− 300（上游回报的缩减请求 prompt token）= 112`。
- 两种情形 `reduced: true` 且 HTTP 200——**关键点：即使真实发送了缩减请求，缺少配对基线时仍拒绝声称 `actual`**。
- 五项断言（A 保持 estimate、B 变为 actual、B 使用 paired 依据、B 算术正确、两者均发生缩减）**全部通过**。

**意义**：这条路径是"降低 token 消耗"最终能被证实（或证伪）的**唯一合法测量机制**，现已在真实进程上验证可用。它同时确立了测量前提——需要一个**上游回报的配对基线**，而这要求对同一请求在未过滤与已过滤两种条件下各调用主模型一次。

### 证据限制

- 验证范围：库层面 31 项测试（含审阅方新增的 CLI 回归测试）+ 真实 scorer 端到端 + 进程级 active 缩减 + actual 记账路径 + A4×G1 跨包集成（11/11）。**仍未做**：真实 provider 调用、配对下游质量、生产启用。
- **接入 ≠ 授权主动裁剪**：默认 shadow、active 关闭。启用生产主动裁剪仍须通过 Track A 全部验收门禁 + ≥3 主模型族的配对下游质量与净 token 成本证据 + 受保护片段压力集零删除。
- 端到端验证中的"主模型"是回环假上游，不是真实 provider；其 `usage.prompt_tokens` 是构造值，但**取自上游响应字段**这一读取路径是真实的。
- 真实闸门延迟 223.8 ms 是单次 MPS 往返，不是尾延迟分布，也不构成 Track A 的 p95 结论。

### 待审问题

1. 网关的上游配置方式（环境变量 / 配置文件）与凭据处理策略，是否接受当前实现（**不读取、不记录、仅透传调用方自带头**，并剥离 `x-nanojev-*` 内部头）？
2. 真实端到端验证的目标主模型族选哪个？需要用户提供可用凭据，或指定一个本地模型作为真实上游。当前仅用回环假上游验证过转发与记账路径。
3. CLI 缺陷这一发现是否接受为"审阅方发现并修复"，还是要求改为由原执行者修复并重新提交？（现状：审阅方已修复、已加回归测试、已做反向验证。）

### 审核结论

（留空待审阅 AI 填写）

---

## G2 — 可逆过滤还原路径（用户明确要求）

**状态**：**已修复（审阅方）** —— 审阅方发现的可逆性契约违反已修复并加了回归测试；剩余已知几何限制被网关守卫拦截为 fail-open。
**记录时间**：2026-09-19

### 背景与范围

用户明确要求方向 1 使用**可逆过滤**与**严格回退**。严格回退（fail-open，12 条路径）此前已实现并测试；**可逆性此前只是隐含的**——收据只记录哈希，没有任何机制证明或执行原文还原。

### 产物

- `scripts/context_restore_v1.py`（内容无关还原清单 + 逐字节还原）
- `scripts/test_context_restore_v1.py`
- `scripts/main_model_gateway_v1.py`（增量：缩减生效时经 `x-nanojev-restore-manifest` 响应头返回清单）
- `scripts/test_main_model_gateway_v1.py`（增量：头只在真实缩减时出现）

### 复现与独立验证（审阅方，不复用作者的测试）

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_context_restore*.py'      # 63 OK
.venv/bin/python -m unittest discover -s scripts -p 'test_main_model_gateway*.py'   # 41 OK
```

**1. 模块级往返（审阅方自写检查）**

| 检查 | 结果 |
|---|---|
| 确实发生缩减 | ✅ `reduced != raw` |
| 清单不含原文 | ✅ content-free |
| **还原与原请求逐字节相同** | ✅ `restored == raw` |
| 角色完整复原 | ✅ `['system','assistant','user']` |
| **篡改被拒绝** | ✅ `RestoreError: removed_segment_hash_mismatch` |

**2. HTTP 层完整闭环（审阅方自写检查，真实网关进程）**

```bash
.venv/bin/python /tmp/g2_reversible_loop.py
```

原请求 →（网关 active 缩减）→ 缩减字节 →（上游）→ 响应头携带还原清单 → 调用方用清单 + 自持内容还原 → 与原请求比对。

| 检查 | 结果 |
|---|---|
| 响应含 `x-nanojev-restore-manifest` | ✅ |
| 响应头内容无关（无原文） | ✅ |
| 缩减确实改变字节 | ✅ |
| 缩减后角色 | `['system','user']` |
| **还原后与原请求逐字节相同** | ✅ `restored_equals_original: true` |
| 还原后角色 | `['system','assistant','user']` |

### 设计要点（审阅方认可）

- **所有权划分**：原文始终由调用方持有；网关只持哈希与指针，清单**绝不持久化原文**。
- **诚实的字节精确性限定**：还原结果是**规范序列化**，仅当原请求本身为规范形式时才逐字节相同；用 `canonical_json` 标志区分，而非默认假设。
- **诚实的安全提示**：清单中明确写出"哈希不是加密，短片段可被字典攻击"——没有把摘要说成保护。
- **还原验证严谨**：除逐段哈希校验外，还把记录的删除操作**重新施加于重建结果**并断言其逐字节复现调用方的缩减字节——这才能证明片段属于该请求对、无重复插入、空消息处理一致；最后用 core parser 复核。

### 高严重度缺陷（审阅方发现，可达；作者自测未覆盖）

**网关会发送"无法还原"的缩减请求，同时发出一个不可用的还原清单——可逆性契约被静默违反。**

作者自测 63 项还原测试 + 41 项网关测试全部通过，但审阅方的独立边界扫描发现一个未被覆盖的几何形状。

**最小复现**：`anthropic_messages` 格式，**list 内容**（content blocks）的助手消息位于消息列表**末位**，其唯一 text 部分被删除 → 消息被清空并整条移除。

| 形状 | 结果 |
|---|---|
| 整条删除 string 内容的助手消息（首位/末位） | OK |
| 部分删除 list 内容助手消息（**首位**） | OK |
| 部分删除 list 内容助手消息（中间位） | OK |
| **部分删除 list 内容助手消息（末位）** | ❌ `RestoreError: reduced_bytes_do_not_match_manifest` |

**经真实网关进程确认的契约违反**（`/tmp/g2_lastmsg_gateway.py`）：

| 检查 | 结果 |
|---|---|
| 网关发送缩减请求 | ✅ `true` |
| 网关报告 | `active_reduced` |
| 发出还原清单头 | ✅ `true` |
| **调用方能否还原原文** | ❌ **`false`** |
| 还原错误 | `RestoreError: reduced_bytes_do_not_match_manifest` |
| **契约违反** | ⚠️ **`CONTRACT_VIOLATED: true`** |

**根因（两层）**：

1. `context_restore_v1.restore_request` 对"list 内容消息因删除最后一个部分而被清空、且该消息位于列表末位"这一几何形状重建失败——重新施加记录的删除操作无法逐字节复现缩减字节。
2. **更关键**：网关的"可逆或发送"守卫只检查 `build_restore_manifest(...)` **能否构建**，而不检查 `restore_request(...)` **能否真正往返**。因此清单构建成功、缩减被发送，而调用方拿到一个不可用的清单。作者的"applicable but not reversible → fail open"规则**实现不完整**。

**影响**：调用方始终持有原请求，故**无数据丢失**；但可逆性**机制**被静默破坏，调用方只有在尝试还原时才会发现，而那时缩减请求已被发送。这正是"可逆过滤"作为用户明确要求所不允许的失效模式。

**可达性**：很高。Anthropic 格式以 content blocks 为原生形态，且"会话末尾的助手消息只含一个 text 块"是极常见的形状。

**建议修复**：网关必须在发送前用 `restore_request` 做真实往返自检（或在 `build_restore_manifest` 阶段做等价的几何校验），任何不可还原的计划一律 fail-open；同时修复末位清空消息的重建逻辑。修复后必须补上该几何形状的回归测试。

### 修复实施（审阅方）

**修复 1 — 几何逻辑**：`context_restore_v1._dropped_message_indexes` 原先在 `position >= len(reduced_items)` 时直接报错，但 `position == len(reduced_items)` 恰恰是"末位消息被清空后塌出列表"的特征。改为：`== ` 判定为整条移除并 `continue`；`>` 仍然报错（真不一致）。

**修复 2 — 网关守卫（防御纵深）**：`main_model_gateway_v1.handle()` 在发送缩减前，用 `removed_segments_from_raw()` 从**调用方自己的原始字节**恢复被删值，执行**真实** `restore_request` 往返并与原请求的**规范形式**比较；任何不一致 → `reduction_error` → fail-open。恢复出的片段仅用于进程内自检，**从不持久化或记录**。

**审阅方在实施修复 2 时自己引入并修正了一个 bug**：首版直接与 `raw` 比较，而 `restore_request` 返回**规范序列化**，导致非规范（如 pretty-printed）请求被误判为不可还原——7 项既有测试失败。已改为与原请求的 `canonical_bytes` 比较。**这是我自己的失误，如实记录。**

**修复后验证（真实网关进程）**：

| 几何形状 | 网关行为 |
|---|---|
| 末位 list 消息被清空（已修复） | ✅ `sent_reduced=True`、`active_reduced`、清单可还原 |
| 中间位 list 消息被清空且后一条也是 list（模块层仍不可还原） | ✅ `sent_reduced=False`、`reduction_error`、**无清单头** |

第二行即"可逆或发送"契约的正确执行：**不可还原的计划绝不发送**。

**回归测试**：`test_context_restore_v1.LastMessageEmptiedRegressionTest`（2 项，覆盖 5 种位置几何）+ `test_main_model_gateway_v1.test_irreversible_plan_fails_open_instead_of_being_sent`（1 项）。

**剩余限制（诚实记录）**：模块层对"中间位清空 + 后一条为 list 内容"的组合仍无法还原。当前由网关守卫拦截为 fail-open，**因此不会违反契约，但该形状的请求无法获得缩减**。根本修法是让清单记录每条消息的**原始 part 数**，从而无需启发式推断——建议作为后续改进。

**审阅方法说明**：该缺陷是在审阅方**不使用作者测试**、自行扫描消息位置 × 内容形态组合时发现的；作者 104 项测试全绿仍未能捕获，说明覆盖存在系统性盲区（只测了首位/中间位）。

### 审核方已确认的缺口（低-中严重度；边界经实证修正）

**还原清单头没有显式长度上限保护，但实际尺寸被 core 的评分预算封顶。**

审阅方先做了分析（清单约 220 字节/片段 → 推测 50 段会超 8KB），随后**用真实网关进程做了实证，结果修正了分析结论**：

| 删除片段数 | 清单头字节 | 客户端是否收到 | 是否超 8KB |
|---:|---:|---|---|
| 1 | 666 | 是 | 否 |
| 10 | 2,604 | 是 | 否 |
| **32（core 上限）** | **7,378** | **是** | **否（余量约 10%）** |
| 60 | — | **未发生缩减** | — |

`60` 段的情形**根本没有缩减**：core 返回 `bypass / scoring_budget_exceeded`，`forward_reason = active_no_reduction`，零删除 → 不产生清单。`context_gate_v1.py` 的预算常量为 `MAX_BYTES = 128_000`、`MAX_SEGMENTS = 128`、**`MAX_SCORED = 32`**。

**修正后的结论**：清单的最大尺寸被 `MAX_SCORED = 32` 限制为约 **7.4 KB**，在常见的 8 KB 单头上限内，**但余量仅约 10%**。

- 采用 8 KB 上限的服务器（nginx/Apache 默认）：**当前可通过**。
- 采用 **4 KB** 上限的服务器或中间代理：约 **18 个删除**时即会失败——这在 `MAX_SCORED` 之内，**属于可达情形**。
- 任何未来上调 `MAX_SCORED` 的改动都会**静默破坏可逆性机制**，因为没有任何尺寸守卫会报警。

**建议（低成本）**：仍应加入显式尺寸上限，超出时返回内容无关的"清单过大"标记（调用方始终持有原请求，仍可依自持指针还原），或对大清单改用响应体附加字段。**不阻塞 G2 通过**，但应在启用 active 前解决。

**审阅方法说明**：本条最初由分析得出，随后被实证推翻其可达性判断；记录保留修正过程，以免后人重复同一误判。

### 审阅方发现的战略级限制（对方向 1 直接影响）

同一次实证暴露了一个比头长度更重要的约束：

```python
MAX_BYTES    = 128_000   # 请求字节预算
MAX_SEGMENTS = 128       # 片段总数上限
MAX_SCORED   = 32        # 可评分的候选数上限
```

**当一个请求的可删候选超过 32 个时，core 直接返回 `bypass / scoring_budget_exceeded`，整个请求零删除。**

这对"降低无效输入 token"这一目标构成直接矛盾：**最需要缩减的恰恰是长上下文**（长编码代理历史、多文档检索），而这类请求的候选数极可能超过 32。实测中 60 个可删片段的请求就完全 bypass，节省为 0。

`MAX_SEGMENTS = 128` 与 `MAX_SCORED = 32` 之间的差距意味着：一个片段总数在 33–128 之间的请求，只有在**受保护片段占多数**时才能被评分；候选一多就整体放弃。

**影响**：这是当前闸门设计的**结构性限制**，不是缺陷——bypass 是安全的（fail-open），但它意味着"长上下文节省"这一最主要的用例目前**无法被服务**。任何声称"已降低长上下文 token"的说法在此限制下都不成立。

**建议**：分块或分批评分（把候选切成 ≤32 的批次并合并删除计划），或在设计上接受长请求 bypass 并在产品层面明确该边界。**需在 Track A 门禁中被显式裁定**，因为它决定了方向 1 的可服务范围。

### 证据限制

- 作者自测（59 项）与审阅方独立检查（模块级 + HTTP 闭环）均已通过，但**均在同一台机器、同一实现假设下**完成；尚无第三方独立实现或跨实现互操作验证。
- 全程 loopback 存根 scorer；**未**证明在真实模型评分分布下的行为。
- 可逆性**不解决**"是否该删"的问题——它只保证删得可还原。删除安全性仍取决于闸门模型与保护契约。
- **未**证明生产可用：active 仍默认关闭；仍需 Track A 全部门禁与配对下游质量证据。

### 待审问题

1. **（阻塞）** 上述可逆性契约违反必须修复：是否要求由原执行者修复并补齐该几何形状的回归测试，还是由审阅方直接修复？修复前 G2 **不得**标记为通过。
2. 修复方式是否接受"发送前用 `restore_request` 做真实往返自检，不可还原即 fail-open"这一最保守方案（相比仅修 `_rebuild_original` 的几何逻辑，前者能同时挡住未来同类缺陷）？
3. 还原清单经**响应头**返回是否可接受？替代方案是经响应体附带或调用方自行保留指针——头方案不改动上游响应体，但受头长度限制（见上节实测：8 KB 上限下余量约 10%，4 KB 上限下约 18 个删除即失败），需确认边界处理。
4. 是否接受由审阅方完成模块级与 HTTP 闭环的独立验证，还是要求另行指定第三方复核？

### 审核结论

（留空待审阅 AI 填写）

---

## P2b — 永续合约层与模拟盘回测（用户范围变更后）

**状态**：READY_FOR_REVIEW
**记录时间**：2026-09-19

### 背景

用户把 Track B 范围收窄为**只做加密二级永续合约**（Binance/Bybit/Aster/Hyperliquid）。原 P2 模拟器**明确拒绝做空**、无保证金/资金费/强平模型，构成硬阻塞。

### 产物

- `scripts/financial_simulator_v1.py`（扩展：`ContractSpec`、`PerpPolicy`、`AccountPolicy`、双向签名动作、净额、保证金充足性、资金费、强平、扩展守恒账本）
- `scripts/financial_backtest_v1.py`（**新增**，44,249 B：离线模拟盘回测 + CLI）
- `scripts/financial_risk_v1.py`（扩展：6 项永续守卫 + 8 项限额）
- `scripts/test_financial_simulator_v1.py`（**35 → 80 项**）
- `docs/FINANCIAL_SIMULATOR_V1.md`（重写）
- `results/financial_simulator_v1_stress.json`（重新生成，跨进程字节相同）

### 复现

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_financial_simulator*.py'
```

**审阅方独立复跑：80 项 OK。** 全量套件 **365 项 OK**（2 skip）。

### 关键实现（执行者报告，审阅方抽查）

- **强平**：在每个可用合成价格观测点**与**每个资金费边界求值；时间线单向遍历一次，因此"先破位后回升"仍会被捕获，且求值后的价格无法影响状态。`isolated` 只平该腿；`cross` 平所有永续腿；现货腿不参与永续保证金（有测试）。每笔强平写独立记录 + 交易现金条目 + 单独费用条目，守恒器独立断言其归零。解析解已验证：10× 多头在 100 入场、触发价 90.452。
- **保证金充足性被强制执行而非假设**：每个增险订单的开仓部分必须按声明杠杆适配 `available_margin`，否则减量并在无可用额度时拒单。**P2 审阅指出的现货购买力缺口也已补齐**（`insufficient_cash_for_spot_purchase`），两条路径都不会让现金变负。
- **资金费**：按声明区间与锚点、从**声明的来源字段**取率；`amount = -q × multiplier × mark × rate`；每笔独立记账；单调性已验证（费率 0/1e-4/5e-4/1e-3 → 成本 0/0.6/3.0/6.0，零违反）。无可用标记价时跳过支付，绝不臆造。
- **回测**：合成种子路径 + 合成 regime 计划，因果 MA 参考策略或调用方决策流；完整权益/保证金曲线、回撤、换手、成本、保证金占用，以及**归因和可证等于账本净 PnL** 的逐品种/逐 regime 分解；示例运行：39 笔成交、6 次资金费、238 个曲线点、最大回撤 0.99%、两次运行字节相同。

### 对 P2 审阅意见的纠正

见 P2 条目第 1 点：**审阅方关于残差成因的判断是错的**（不是 `no_order` 而是 `rejected_decisions`），执行者已修正并断言计数闭合。

### 待 R1 冻结参数：**71 项**（原 38）—— 48 项模拟器 + 23 项风控，另加 19 项回测构造值

### 证据限制（执行者自报，审阅方确认已写入规格）

- **无真实场所参数保真**：Binance/Bybit/Aster/Hyperliquid 的 tick/lot/杠杆档位/资金费表/强平费/价格带**均未来源化或校验**；四个场所仅作为**声明的标签**，未联系、未复刻规则。
- 未建模：跨场所保证金划转、反向/夸托/组合保证金、队列位置、订单簿重建、延迟抖动、部分强平、保险基金/ADL/坏账社会化。
- **isolated 损失未验证被限制在分配保证金内**（有意未建模，仅报告已实现损失）。
- 参考 MA 策略**无预测价值声明**；无真实 PnL 分布。
- 合成观测点之间的区间内权益波动未表示。
- 全程合成数据；未下载行情、未连接 broker、无盈利声明。

### 待审问题

1. 是否接受永续层与回测作为 P2 第二切片通过？71 项参数是否进入 R1 冻结清单？
2. 四场所的真实 tick/lot/杠杆/资金费表是否要求来源化（仍不购买、仅查公开文档）？若不要求，是否接受"场所仅为标签"的明确边界？
3. `isolated` 损失上限不建模是否可接受，还是必须在启用 paper 阶段前补齐？

### 审核结论

（留空待审阅 AI 填写）

---

## W1 — 真实数据模拟盘（用户指令：先推进实盘数据仿真模拟交易）

**状态**：READY_FOR_REVIEW
**记录时间**：2026-09-19

### 背景

用户指令：先推进实盘数据仿真模拟交易（先于 RLCD 训练）。同时用户明确授权接入 Bybit / Aster / Hyperliquid 数据源。

**范围界定（重要）**：模拟盘回测属于**离线研究**，落在 Binance 许可 §4.1 允许范围内；§4.2 禁止的是**实盘执行**。本轮**不下单、不接 broker、不调用交易 API、不读账户**。

### 产物

| 脚本 | 作用 |
|---|---|
| `scripts/fetch_binance_vision_v1.py` | 有界公开归档下载 + 来源/哈希清单 |
| `scripts/fetch_venue_perp_v1.py` | Bybit / Hyperliquid 公开行情 API 抓取（Aster 记录为不可达） |
| `scripts/build_perp_pit_v1.py` | 真实数据 → `nanojev-financial-pit-v1` 记录契约 |
| `scripts/paper_trade_perp_v1.py` | 真实 bar → P2b 模拟器 → 模拟盘收据 |

数据落 `data/`（已被 git 忽略，未污染版本库）；收据落 `results/paper_trade_perp_v1.json`；文档 `docs/PAPER_TRADE_REAL_DATA_V1.md`。

### 执行结果

- **Binance**：880 文件 / 1,185,364 B / 5 品种 / 2023-01..2026-08（20 个 "missing" 全为未发布的 2026-09 月度归档）
- **Bybit**：30 文件 / 1,165,457 B / 0 错误
- **Hyperliquid**：已接通（`metaAndAssetCtxs` / `candleSnapshot` / `fundingHistory`）
- **PIT 队列**：6,554 条记录，2,951 正例（**45.03%**）
- **PIT 审计**：**未修改**的 `financial_pit_v1.py` + 冻结协议核心 → **exit 0**，三折四阶段全非空（test 905/905/465，dev/cal 各 120）

### 模拟盘结果 —— **本轮最重要的产出是"这不是一个结果"**

同一策略、同一区间、同品种，因数据集与场所不同产出**四个互相矛盾**的数字：

| 运行 | 数据 | 成交 | 净 PnL | 最大回撤 |
|---|---|---:|---:|---:|
| A | Binance（修 3 天 index 数据之前） | 82 | **−49,920.90** | — |
| B | Binance（修复后） | 83 | **+21,173.77** | 78.86% |
| C | Bybit（同区间同品种） | 81（2 拒） | **−77,919.27** | 94.22% |
| D | Aster（同区间同品种） | 83（27 部分成交） | **+194,025.96** | 64.79% |

**三场所净 PnL 极差 = 271,945，是初始资金 100,000 的 2.7 倍。** A 与 B 只差三个 BTCUSDT 日期（2023-02-13、04-07、04-08）和**一笔决策**——该决策在 A 中因库存不足被拒、在 B 中成交，**这一笔翻转了符号**。

### 发散**不是**数据质量问题（已交叉验证）

在下结论前先直接核对了各场所的 mark 价：

| 对比 | 共同天数 | 平均绝对偏差 |
|---|---:|---:|
| Aster vs Bybit mark 收盘 | 1,358 | **0.0199%**（≈2 bps） |
| Aster vs Binance mark 收盘 | 1,337 | **0.0122%**（≈1.2 bps） |
| 最差单日（Aster vs Bybit，2024-10-13） | — | 0.199%（≈20 bps） |

**场所之间只差约 1–2 个基点**，而同一策略在这三条价格序列上返回 +194,026 / +21,174 / −77,919。**策略对其输入是混沌的**——路径上 1 个基点的差异，经仓位规模与被拒/部分成交路径放大成最终 PnL 的 2.7 倍摆动。

因此 ~2 bps 的真实跨场所基差**不是** 271,945 极差的原因，只是**触发器**。**本页任何数字都不得被引用为收益、亏损、优势或结果。**

### 待修机制

策略的 `long`/`short` 是**目标仓位动作**，模拟器按账本实际持仓算 delta，因此仓位**水平**自愈；但**被拒或部分成交**会静默改变成交的**尺寸与时机**（容量夹取、拒单、TTL 过期），从而改变手续费、资金费与 PnL 路径。三个场所分别产生 **0 / 2 / 27** 次此类事件。任何推断之前，必须先让 harness 把"实际成交 ≠ 预期成交"显式化（B0-A）。

~~第二个放大器：数量每次按当前价重算（`0.25 × equity × leverage / price`），因此仓位规模随路径**复利**而非固定。~~ **2026-09-19 更正**：经代码核实，sizing 用的是 `initial_cash` 而非当前权益（`0.25 × initial_cash × leverage / close`），数量仅随 1/价格变化、**不随路径复利**。此条作废，B0-B 改为"核实+锁定"该性质。

### 运行 B（Binance）完整账本

| 项 | 值 |
|---|---:|
| quotes / decisions / fills | 4,008 / 83 / 83 |
| 资金费支付 | 10,251 |
| 强平 | 0 |
| 初始 → 最终权益 | 100,000.00 → 121,173.77 |
| 净 PnL | +21,173.77 |
| 手续费 | 4,722.50 |
| **资金费成本** | **27,167.86** |
| 成交名义额 | 9,445,009.73 |
| 最大回撤 | 117,970.85（峰值的 78.86%） |
| **守恒校验** | ✅ ok |

资金费是手续费的 **5.8 倍**。**确定性与归因已验证**：两次运行收据**逐字节相同**；逐 regime 归因之和与曲线权益变化残差 **−2.18e-11**。

### 逐 regime 分解（运行 B，因果阈值）

阈值是**扩张分位数**（只用决策时点及之前的观测、最近秩、无插值、48 bar 最小历史）。优先级：basis blowout → funding extreme → 高波动 → 低流动性 → normal。schedule 以单一参考品种（BTCUSDT）为键。

| Regime | 观测 | 权益变化 | 资金费 | 手续费 | 成交 | 最大回撤 |
|---|---:|---:|---:|---:|---:|---:|
| basis_blowout | 476 | +67,939.98 | 11,096.48 | 196.66 | 4 | 113,514.16 |
| vol_high | 832 | −58,539.96 | 3,488.59 | 706.43 | 11 | 117,106.13 |
| normal | 2,727 | +33,180.07 | 7,015.57 | 2,779.51 | 47 | 117,970.85 |
| liquidity_low | 955 | +1,920.64 | 1,769.86 | 967.31 | 19 | 114,514.21 |
| funding_extreme | 120 | −32,822.14 | 3,797.35 | 0.00 | 0 | 87,118.72 |
| unlabeled | 2 | +3,979.57 | 0.00 | 0.00 | 0 | 55,102.12 |

承担主要亏损的两个 regime（`vol_high`、`funding_extreme`）正是只看成本的回测会忽略的条件——但它们分别只有 11 笔和 **0 笔**成交，**buckets 太薄，不支持任何推断**。

### 修一个 Bybit 分页 bug（审阅过程中发现）

首次 Bybit 抓取每个序列恰好返回 **1,000 天**。Bybit v5 kline 返回区间内**最近**的 `limit` 条，所以按 `start` 向前分页会在一页处静默截断历史。已改为**向后回溯**（把 `end` 移到收到的最早行之前）。重抓后从 1,165,457 → **3,734,650 字节（3.2 倍）**，Bybit 运行的决策数从 61 → **83**（与 Binance 触发数一致）。**修复前的跨场所对比是建立在不等区间上的。**

### 目标第③项：本驱动现在产出

- 权益曲线（`perp_curve`，5,113 点）、**回撤指标**（峰值/谷值/恢复/最长回撤）
- **逐 regime 分解**（含归因自洽校验）
- **逐场所分解**（Binance vs Bybit 同策略同区间对比）
- 保证金比率与强平距离的最小/最大值

### 场所连接的真实状态 —— **四场所全部接通（Aster 经代理打通）**

| 场所 | 状态 | 说明 |
|---|---|---|
| **Binance** | ✅ 已通 | 公开归档，无需密钥 |
| **Bybit** | ✅ 已通 | 官方主机 `api.bybit.com` 在本环境**被 DNS 污染**（解析到 179.60.193.16 无关地址、永不连接）；经本地出口代理后**官方主机可用** |
| **Hyperliquid** | ✅ 已通 | 直连可用 |
| **Aster** | ✅ **已通（经代理）** | `fapi.asterdex.com` 在本环境**直连被拒**；经代理返回真实数据（klines / mark / index / funding / `fundingInfo` / `premiumIndex` / `exchangeInfo`，602 个品种） |

**关键发现：两个阻塞都是本环境的出口问题，不是场所侧的封锁。**

| 主机 | 直连 | 经 `http://127.0.0.1:7890` |
|---|---|---|
| `api.bybit.com` | HTTP 000（DNS 污染） | **HTTP 200** |
| `fapi.asterdex.com` | HTTP 000（连接被拒） | **HTTP 200** |
| `api.hyperliquid.xyz` | HTTP 200 | HTTP 200 |

抓取器现默认走本地出口代理（`NANOJEV_PROXY`，默认 `http://127.0.0.1:7890`，可用 `--proxy` 覆盖，`--proxy ""` 强制直连），代理地址记入每份 manifest 的 `egress_proxy` 字段，保证抓取可复现。**这是审阅方此前的疏漏：第一轮只测了直连就判定 Aster 不可达，未检查是否存在可用代理。**

**代理解决的是"可达性"，不解决"许可"**：三个场所的条款页从本环境仍不可达，权限仍为 **unverified**；用户的授权是**项目决策，不是法律认定**。

### Aster 抓取过程中修的两个自身缺陷

1. **默认符号表用错**：`aster` 落到了 Hyperliquid 的默认值（`BTC` 而非 `BTCUSDT`），导致 25 个 400 错误。
2. **端点名猜错**：我猜的 `fundingRateConfig` 返回 404；正确端点是 **`fundingInfo`**（提供 `fundingIntervalHours`，当前 BTCUSDT 与 SOLUSDT 均为 8，但 P1b 曾观察到 4h，说明**区间可变**）。Aster 的 funding 行**本身不含区间字段**。

修复后：**35 文件 / 4,032,704 字节 / 0 错误**。

### 场所连接结果汇总

| 场所 | 文件 | 字节 | 每序列日线 |
|---|---:|---:|---:|
| Binance | 880 | 1,185,364 | 1,337–1,358 |
| Bybit | 30 | 3,734,650 | 1,358 |
| Hyperliquid | 15 | 14,131,314 | 1,358 |
| **Aster** | **35** | **4,032,704** | **1,358** |

### 仍需裁定的门禁问题

1. **PILOT 队列不是 R1 队列**：R1 白名单是**闭集 12 特征**（"不可增删、不可条件省略"），但其中三个无法从本归档构建——`open_interest_level` 与 `open_interest_log_change_1d` 需要 Binance daily metrics 归档（**可补下**），**`liquidation_intensity_1d` 没有任何场所发布历史数据**（补不下）。本队列因此声明了文档化的 **9 特征子集**并标记为 PILOT。**需 R1 裁定**：补下 metrics 并只放弃强平特征，还是修改白名单。
   - 补充：Aster 接通后，Bybit / Aster 的 OI 端点可用于交叉核对（**Aster 完全没有 OI 端点**，Hyperliquid 有但为链上口径）。三场所 OI 单位与单双边约定不同，**不可直接汇总**。
2. ~~Aster 的替代路径~~ —— **已解决**：经本地出口代理接通，35 文件 / 4,032,704 字节 / 0 错误。

### 证据限制（必须随任何引用一起声明）

- **不是结果**：同一策略在三天数据差异下、以及两个场所之间**翻转符号**（−49,921 / +21,174 / −77,919）。本页任何数字都是管线产物。
- **不是收益**：执行成本为待 R1 的**声明性猜测**；归档无订单簿，bid/ask 取 mark 收盘，全部点差成本集中在**一个**声明参数里。
- **不具预测性**：MA 参考策略只是执行路径校验器；它亏损不是关于任何模型的证据，盈利也同样不是。
- **不是 as-of vintage**：归档是当前快照，Binance 官方文档说明文件曾被就地替换。
- **不具实盘就绪性**：无成交真实性、队列位置、容量、保险基金、ADL、部分强平、场所参数保真。
- **两处约定是近似**：决策于 bar d 收盘、成交于 bar d 的 mark 价（最新可得观测，属"收盘决策"惯例、略偏乐观）；每个资金费边界使用**最近已结算**费率而非该边界结算的费率。
- **regime buckets 是描述性的、不是实验**：schedule 以单一参考品种为键，多个 bucket 成交数不足 20（其中 `funding_extreme` 为 **0**）。
- **拒单路径是已知的 harness 缺陷**：必须在任何结果可能有意义之前修复。
- 新增脚本已有单元测试：`scripts/test_perp_pipeline_v1.py`（**15 项**，全量套件 **380 项 OK**）。覆盖：扩张分位数的**因果性**（追加未来不改变历史标签）、regime 优先级与最小历史门槛、窗口合并无重叠、定义哈希可复现、Bybit **向后分页有界**（不会死循环）、真实归档的 PIT 契约合规（逐条过**未修改**校验器）、**标签谓词与声明公式逐条一致**（该测试队列 48 正/43 负，非平凡）、特征可用时间不晚于决策。
- 仍未测试：真实 venue 参数保真、点时可得的真实 feed、跨场所保证金、逆合约/组合保证金、队列位置/订单簿重建、延迟抖动、部分强平、保险基金/ADL/坏账、isolated 损失上限、参考策略的预测价值。

### 待审问题

1. 是否接受 W1 作为"真实数据打通 + 模拟盘跑通"的第一切片通过？
2. PILOT 9 特征子集是否接受，还是要求补下 Binance daily metrics（OI）后再跑？（强平特征无论如何都拿不到）
3. Aster 走哪条路：换网络出口 / 提供本地抓取脚本 / 本轮放弃？
4. 是否要求把拒单路径修好（显式状态分歧或只用减仓语义）后重跑，作为下一轮的前置？
5. 是否为新增的 4 个脚本补单元测试（建议补：PIT 转换的边界与缺失值规则、标签谓词、资金费对齐、Bybit 向后分页）？

### 审核结论

（留空待审阅 AI 填写）

---

## W2 — B0 测量完整性闭环 + 本地模型能力调查（2026-09-19）

本工作包回答两个问题：金融模拟盘的测量能否被信任，以及本地 NanoJev 能否作为 skill 无缝使用。**两者的答案都以"发现缺陷"为主，而非"产出数字"。**

### 一、B0：发现并修复使此前所有数字失效的缺陷

**缺陷**：`--first-day`/`--last-day` 只出现在两处——从协议赋值、写进收据元数据——**从未用于过滤数据**。因此每次运行实际使用整个归档（2023-01-01…2026-08-31），而收据声称 2024-01-01…2026-08-31。**三个场所从未在同一区间上比较过。**

**发现方式**：做 T5 分时段敏感性时，"一半"样本的结果与"全部"**完全相同**——这在数学上不可能，除非窗口从未生效。**不是靠读代码发现的，是靠一个不可能通过的复现检查发现的。**

| 检查 | 修复前 | 修复后 |
|---|---:|---:|
| 加载 quotes（协议窗口） | 4,008 | **2,919** |
| 决策数 | 83 | **61** |
| Binance 净 PnL | +21,173.77 | **−34,080.29** |

**修复**：`day_window_ms()` 解析为半开区间 `[start_ms, end_ms)`（含末日）并应用到 bar open time；反向窗口报错；4 项回归测试（区间语义、选择谓词、反向、过滤器存在性）。

### 二、修复后的三场所结果

协议冻结、窗口生效、容量解耦、**分歧全部为 0**：

| 场所 | 决策 | 净 PnL | 最大回撤 |
|---|---:|---:|---:|
| Binance | 61 | −34,080.29 | — |
| Bybit | 61 | −37,597.40 | — |
| Aster | 61 | −38,044.85 | — |

**极差 3,964（本金约 4%），三场所同号同量级**；此前为 **271,945（本金 2.7 倍）**且 Aster 为 +194,026 异常值。容量耦合的单独效应现测为 **820**，不是 279,000。

**结论撤回**：roadmap 曾写"策略对输入混沌"。**该结论撤回**——极差主要来自两个工具缺陷，不是策略。仍成立：参考策略在三场所**都亏钱**（机械均线交叉 + 声明式临时成本，不构成策略结论）；约 4% 本金的离散度**仍然太大**，单场所数字不得引用。

**另一项实测**：`fill_probability=1.0`、`reject_probability=0.0` 下 replay **对种子完全无影响**，故"多种子敏感性报告"在此配置下是**退化轴**；有意义的轴是**场所与时段**。

### 三、B0-A/B/C/D 交付

| 子项 | 交付 | 验证 |
|---|---|---|
| B0-0 | `research/paper_trade_b0_protocol.json` + `scripts/paper_trade_protocol_v1.py` | 摘要不匹配 exit 2；未知 schema / 缺字段 / 缺失文件均拒绝 |
| B0-A | `find_divergences()` + `--on-divergence {error,report}` | 耦合模式下 Aster **11 笔分歧 → exit 2**；解耦后 0 |
| B0-B | `--sizing fixed_notional` + 锁定测试 | 数量仅由 initial_cash 与决策日收盘价决定 |
| B0-C | `--capacity {fixed,venue_volume,off}` | 解耦后三场所分歧归零 |
| B0-D | ⏳ 未完成 | 归因分解与分时段报告待做 |

**测试**：全量 **396 项 OK**（2 skip）。

### 四、本地 NanoJev 能力：域内可用、域外崩塌

分派 5 个并行 subagent 调查"频繁弃权"。结论：

**独立诊断（复现基线收据 ≤1e-8）**——崩塌是**域外**现象：

| | 域内（迷宫 240 题） | 域外（13 工程题） |
|---|---:|---:|
| 中位置信度 | 0.779 | 0.505 |
| 最大置信度 | 0.998 | 0.736 |
| ≥0.9 的题 | **37.5%，其中 97.8% 正确** | 0%（**13/13 弃权**）|
| 置信度-准确率 | **正相关（r=+0.397）** | 3/6 |

**"13/13 弃权"已从重算升级为观测事实**（`abstain_below:0.9` 实跑，收据 `research/skill_abstention_survey_gated_run_v1.json`，事件 `4641c61b`）。**注意**：`decide` 路径**无默认阈值**，0.9 只存在于 `lifecycle`——早期表述把两者混为一谈。

**温度重标定（负面结论）**：仅在校准集拟合得 0.854；准确率处处不变（温度不改 argmax 与排序），ECE 在 test 改善 −0.031 却在 ood 恶化 +0.018、校准集/dev 亦恶化 → **不泛化**；调查题中位置信度 0.505→0.515，**仍全低于 0.9**。**"重标定即可修好弃权"这条路关闭。**

**skill 加固**：`nanojev-scope-guard-v1`（5 类短语、命中即 `out_of_scope`、清空 value 但保留概率、附 `operating_envelope`、无关闭开关）；17 项测试 OK；安装副本未改（属独立受审步骤）。

### 五、对抗性核验撤回了我三项表述

| 我此前的表述 | 核验结果 |
|---|---|
| 三个错答占据置信度前三 | ❌ **假**，真实排名 1/3/5 |
| 置信度与正确性反相关 | ❌ n=6 不成立（置换 p=0.80；两位核验者符号相反）|
| 历史弃权率 43.5% | ❌ 实为 **7/23 = 30.4%**（43.5% 含本次调查自身调用）|

另：3/6 是**随机下的众数**（P=0.3125），不构成发现；`check_secrets` 是通用实践而非项目记录事实，故 3/6 对标签选择敏感；检验 0.8 vs 0.5 需约 18 题。**撤回以"已撤回"方式记录，未静默删除。**

### 六、证据限制

- B0-D 未完成，归因分解与分时段报告缺失；PILOT 队列仍是 9 特征而非 R1 闭集 12。
- 域外诊断的差异同时来自语言、题型、选项措辞与状态长度，**只能定位、不能归因**；仅一个 checkpoint、一台设备；域内 0.9 下 62.5% 弃权**不能**外推为闸门删除率。
- 旧收据保留其错误窗口元数据，作为缺陷证据；正确产物为 `results/paper_trade_b0_baseline_*_v1.json`。
- Aster 数据许可冲突仍未解决（T7），其数字应置于附录。

### 审核结论

（留空待审阅 AI 填写）
