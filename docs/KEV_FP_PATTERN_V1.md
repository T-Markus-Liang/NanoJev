# KEV_FP_PATTERN_V1 — kev 真实候选集 30 条误删的模式分析

> 本地 only：候选/锚点均出自脱敏本机 transcript，本文只含压缩描述与极短引文
> （每条 ≤ ~10 词），不含原文段落。
> 标签 provenance 继承 `docs/REAL_CONTEXT_EVAL_RESULTS_V1.md`：525 scored
> （523 keep / 2 drop）为 owner 委托的 agent 裁决（非对称语义，拿不准→keep）；
> 本文 30 条 FP 全部 label_confidence=high。

## 0. 口径

- join：`eval.jsonl`（525，record_id→label）× `candidates.jsonl`（547 行序）
  × `results/kev_real_candidates_v1.jsonl` / `lora_real_candidates_v1.jsonl`（行 i↔行 i）。
- FP = label keep（irrelevant.true ≤ 0.5）且 kev noul ≥ 0.5 → **30 条**，与主报告一致
  （tool_result 20 / user_turn 6 / assistant_text 4）。
- kev 525/525 全覆盖；lora 5 条 500 弃权（均在表外，不影响对照）。

## 1. 30 条 FP 明细

`row`=candidates.jsonl 行号；`band`=transcript 段数档（非候选长度）；锚点为窗口
末条 user 消息的压缩描述。

| # | row | record_id | kev | lora | kind | band | 候选（截断） | 锚点摘要 |
|--:|----:|---|---:|---:|---|---|---|---|
| 1 | 28 | `12aa8112a2ab:seg0001` | .54 | ~0 | user_turn | s | 「先把链路全部测试验证通过再录制视频」 | 回报进度/本地失败路径测试 |
| 2 | 29 | `12aa8112a2ab:seg0002` | .53 | ~0 | user_turn | s | 「只修改 joypad_server.py…connect() 等 socket…」(详细修复指令) | 同上 |
| 3 | 104 | `4086d4526f55:seg0015` | .57 | .04 | tool_result | s | `Script completed…` + HF URL 探测(200/redirect/length) | 下载打包模型部署文件 |
| 4 | 177 | `59bc83dccef3:seg0029` | .57 | ~0 | tool_result | m | `Chunk ID…exit 0` + Unitree C++ 源码 dump | G1 Gazebo 跌倒只读分析 |
| 5 | 178 | `59bc83dccef3:seg0030` | .59 | ~0 | tool_result | m | 同上 + locomotion.h dump | 同上 |
| 6 | 179 | `59bc83dccef3:seg0031` | .64 | .005 | tool_result | m | 同上 + launch.xml dump | 同上 |
| 7 | 209 | `61587fdcfb74:seg0000` | .50 | ~0 | user_turn | m | 「研究1」（3 字会话开场） | 罗剑岚老师…研究调研 |
| 8 | 210 | `61587fdcfb74:seg0001` | .50 | ~0 | user_turn | m | 「请你阅读 世界模型研究 目录以及日志」 | 同上 |
| 9 | 211 | `61587fdcfb74:seg0002` | .50 | ~0 | assistant_text | m | 「我先按恢复审计方式只读查看…不改文件」 | 同上 |
| 10 | 215 | `61587fdcfb74:seg0045` | .61 | .001 | tool_result | m | `Chunk ID…exit 0` + git status 列表 | 同上 |
| 11 | 257 | `75220e6af930:seg0058` | .50 | ~0 | tool_result | m | 行号 python 源码 dump（Atlas decision brief） | AtlasOS trace 任务 |
| 12 | 294 | `8444684933cc:seg0017` | .56 | ~0 | tool_result | s | `Script completed` + 3×HF repo `200` 列表 | AGX Orin 资源下载验证 |
| 13 | 295 | `8444684933cc:seg0019` | .51 | ~0 | tool_result | s | `Script completed` + repo/file `size-not-in-api` 表 | 同上 |
| 14 | 296 | `8444684933cc:seg0021` | .52 | ~0 | tool_result | s | `Script completed` + size/`no-lfs-oid` 表 | 同上 |
| 15 | 340 | `946287c815cb:seg0000` | .52 | ~0 | assistant_text | s | 部署完成报告（Gazebo 已验证、操作说明、修复列表） | 新任务：只建会话日志 |
| 16 | 341 | `946287c815cb:seg0001` | .56 | ~0 | user_turn | s | 「请你进行多次测试验证，确保gazebo…90%…」 | 同上 |
| 17 | 342 | `946287c815cb:seg0003` | .56 | ~0 | assistant_text | s | 「我会仅创建并同步新的 active 会话记录…」 | 同上 |
| 18 | 343 | `946287c815cb:seg0005` | .52 | ~0 | tool_result | s | `Script completed` + codex_app 工具 schema dump | 同上 |
| 19 | 376 | `987eca2e79b9:seg0046` | .57 | .005 | tool_result | m | `(Bash completed with no output)` | py 脚本调试工具求推荐 |
| 20 | 377 | `987eca2e79b9:seg0048` | .58 | ~0 | tool_result | m | 行号 markdown dump（autoware rviz plugin README） | 同上 |
| 21 | 493 | `c4ec1cad6847:seg0023` | .58 | ~0 | tool_result | m | `Script completed` + 校验输出（rg not found / diff check passed） | ZIP 清理收尾指令 |
| 22 | 525 | `e9769c90884f:seg0087` | .50 | ~0 | tool_result | m | `Script completed` + 嵌套 session 输出（README patch 日志） | grass-fix 收尾指令 |
| 23 | 526 | `e9769c90884f:seg0089` | .53 | ~0 | tool_result | m | `Script completed` + `"output":""`（空轮询） | 同上 |
| 24 | 538 | `fc3e19d63e48:seg0001` | .51 | .07 | assistant_text | m | 「Let me look into OpenCode's free tier policy」 | 找 mimo free opencode plugin |
| 25 | 541 | `fc3e19d63e48:seg0027` | .60 | .38 | tool_result | m | 「Unable to verify if domain …safe to fetch」(fetch 拦截) | 同上 |
| 26 | 542 | `fc3e19d63e48:seg0030` | .50 | **.58** | tool_result | m | `Web search results for query…` + REMINDER 尾巴 | 同上 |
| 27 | 543 | `fc3e19d63e48:seg0031` | .53 | .49 | tool_result | m | 同上（pricing query） | 同上 |
| 28 | 544 | `fc3e19d63e48:seg0056` | .64 | .07 | tool_result | m | 「The user doesn't want to proceed…rejected」 | 同上 |
| 29 | 545 | `fc3e19d63e48:seg0057` | .69 | .01 | tool_result | m | 同上（第二次拒绝） | 同上 |
| 30 | 546 | `fc3e19d63e48:seg0058` | .68 | .04 | user_turn | m | `[Request interrupted by user for tool use]` | 同上 |

## 2. 失败聚类

| 簇 | 条数 | rows | 表面特征 | 为何实为 keep |
|---|---:|---|---|---|
| **A. 执行信封/模板化工具输出** | 13 | 104,177,178,179,215,257,294,295,296,343,377,493,525 | `Script completed`/`Chunk ID…exit 0` 信封、行号文件 dump、URL/git-status 列表 | 信封是 harness 常量；payload 是任务证据（代码=诊断对象、URL 探测=下载可行性答案本身、校验输出=验收证据） |
| **B. 空/负结果输出** | 2 | 376,526 | "no output"、`"output":""` | 与 lora 最大 FP 同族：空结果就是答案（命令无输出=无异常证据） |
| **C. Web fetch/search harness 壳** | 3 | 541,542,543 | `Web search results for query:` + REMINDER 样板、domain-verify 拦截文案 | 查询词本身就是当前任务检索证据；拦截是任务阻塞事实 |
| **D. 拒绝/中断控制面记录** | 3 | 544,545,546 | tool-use rejected、`[Request interrupted by user]` | 轨迹边界信号——解释"为什么停下/转向"；kev 当噪声 |
| **E. 短用户引导 turn** | 5 | 28,29,209,210,341 | 3 字开场、单行指令、测试要求 | 任务转向/约束来源；低词量 ≠ 低相关 |
| **F. assistant 微应答/范围声明** | 3 | 211,342,538 | "我先只读查看…"、"仅创建会话记录"、"Let me look into…" | 计划/范围承诺是轨迹语义的一部分 |
| **G. assistant 完成/状态报告** | 1 | 340 | 757 字部署验证报告 | 同会话前任务的实质总结 |

上位划分：A–D = **21/30 全是"过程性/harness 表面形式"**（信封、壳、空结果、
控制面）；E–G = **9/30 是"低内容量对话形式"**（短指令、应答、报告）。两类
共用一个机制：**kev 学的是 surface-form → irrelevant 的先验，而不是
anchor-相关性判断**。

## 3. 与 lora 同 30 行对照

- lora 在 29/30 上正确（noul<0.5），且绝大多数 <0.01——不是擦线而是量级差。
- 唯一共同 FP：row 542（kev .50 / lora .580，opencode 搜索结果就是 lora 5 FP
  之一）。**kev 与 lora 的 FP 集合几乎不相交**（30 vs 5，交集=1）。
- 反向：lora 的另 4 条 FP（空结果 .919、纠错 .615、研究分叉 .602、文献 .552）
  kev 全部正确（.21–.34）。两头是**互补的错误几何**，不是同一困难族。
- 但 fc3e19d63e48 一簇对 lora 也是相对"硬"的（.03–.58 vs 基线 ~1e-5）——
  Claude-harness 的控制面/REMINDER 文案对两头都是分布外表面，只是 lora 的
  keep 侧饱和度高，阈值下不越界。

## 4. 相关因素检验

| 因素 | 结果 |
|---|---|
| kind | FP 率：user_turn 9.8% (6/61) > tool_result 6.7% (20/300) > assistant_text 2.4% (4/164)。绝对数以 tool_result 为主 |
| transcript band | m 档 11.2% (20/178) vs s 3.2% (10/312) vs l/xl 0% (0/35)。中等长度 transcript 更易 FP，但与 group 混杂 |
| 候选字符长 | **无信号**：FP p50=272 字符 vs keep p50=254；分桶 FP 率 <100:6.4% / 100–500:6.5% / 500–2k:2.2% / ≥2k:6.5%；kev noul~log(len) 相关 −0.11 |
| 窗口内位置 | **无信号**：FP frac mean .40/median .47 vs keep .42/.48 |
| seg 序（transcript 内深度） | 弱 U 形：seg 0–2 = 7.3%、6–15 = 0.8% 低谷、≥41 = 11.3%。开场段（任务建立期内容）与深段都略高危 |
| transcript group | **强聚集**：fc3e19d63e48 一家 7/10 scored = 全部 FP 的 23%；946287c815cb 4/8、61587fdcfb74 4/10、59bc83dccef3 3/10、8444684933cc 3/9；top5 group = 21/30 |
| harness 来源 | 聚合无差：claude_projects 10/184=5.4% vs codex_sessions 20/341=5.9%（scored 分母）；但最大单簇（fc3e）恰是 Claude 系特有控制面文案 |
| noul 边缘性 | **全部 30 条落在 [0.50, 0.69]**，mean .56；0.50–0.55 占一半。τ=0.7 即 0 FP——这是 keep 分布整体上移（p50=.33）后的阈值尾部现象，不是高置信错删 |

group 聚集的机制注记：kev 对单个 transcript 有明显的**整体 offset**（fc3e 全部
10 条在 .47–.69；946287 前 4 条全 ≥.52），误删不是 iid 逐条事件，而是
"这条 transcript 的表面域 kev 没见过/整体判低"。

## 5. 结论

**kev 的 FP 特征 = "通用操作样板过度判删"，但要写成两层：**

1. **样板/harness 表面先验**（21/30）：kev 学到"重复的程序化包装
   （exec 信封、搜索壳、rejected/interrupt 控制面、空输出、行号 dump）→ 无关"。
   在真实 transcript 里这些包装是常量、payload 才是变量，而它恰恰把
   **payload=任务答案**的段落（URL 可行性探测、诊断对象源码、验收输出）
   一起删掉。这是主报告"生成器形式偏见"在 kev 侧的对应物——合成集里
   此类形态大概率常与 drop 标签共现。
2. **低内容量对话先验**（9/30）：短指令、应答、中断标记按"信息量低→可删"
   处理；真实分布里它们是任务转向、范围约束和轨迹断点信号。加上 seg0–2
   轻微富集，opening-of-window/任务枢轴附近内容也被波及。

**机制判断**：kev 的打分面是"这段文字像不像一次独立的实质信息"，而不是
"这段对最终 user 锚点是否相关"——anchor-conditioned 判别弱于
surface-conditioned 判别。所有 FP ≤0.69 说明它没有"自信地错"，是 keep 分布
整体上移后 σ(0.5) 截面恰好扫过这批表面样板。

**修复路径（对应 T182/EXIT 式难负例，落到 v5 数据设计）：**

- **同信封异语义对比对**：相同 `Script completed`/`Chunk ID` 包装、payload 一份
  是当前锚点证据（keep）、一份跨任务噪声（drop）——强制模型读 payload 而非
  信封。这是 EXIT 难负例的直接形态：最小表面差、标签相反。
- **控制面难负例**：tool-use rejected / `[Request interrupted]` / fetch 拦截 /
  REMINDER 壳标注为 keep（轨迹边界语义），尤其补 Claude-harness 形态——
  fc3e 单 transcript 贡献 23% FP，合成域里这类文案缺域。
- **空/负结果反捷径族**：与 lora .919 FP 合并入同一族——"无输出/未找到"
  当其本身是答案时必须 keep，两头共享这个盲区（共 3 条实证）。
- **短 turn / opening 段难负例**：3 字开场、单行指令、范围承诺 ack 的
  keep 例；同时给真正的无关短文本 drop 例，防止反向 shortcut。
- **锚点翻转对**：同一候选文本，锚点 A 下 keep、锚点 B 下 drop——直接训练
  anchor-conditioned 判别，拆掉 surface→label 捷径。
- **评审口径提醒**：τ=0.7 可把这 30 条全部压掉（FP=0），但那是对症状不是
  对机制；且真实集 keep 占 523/525，难负例必须来自标注侧增补才有收益。

**caveat**：30 条 FP 全部 label_confidence=high，但 946287 开场段（row 340/341/
342/343）相对"只建会话日志"锚点属于同 transcript 前任务残留——严格
锚点语义下可辩，agent 裁决按非对称规则给了 keep；即便按更严口径剔除
该簇，剩余 26 条的聚类结论不变。
