# T9d FP32/MPS domain-adaptation protocol V2

状态：**已登记版本，待独立 protocol review 与干净语料审查；不授权训练。**

本文件把 W32/W33 的条件接受落实为一个不可歧义的、可执行的协议对象。旧的
`GATE_CONTRASTIVE_PROTOCOL_V1.md` 不被原地改写；新机器可读协议见
[`research/nanojev_v2_t9d_fp32_mps_protocol_v2.json`](../research/nanojev_v2_t9d_fp32_mps_protocol_v2.json)。

## 1. 本次变更

旧 §7 预注册的是 LoRA rank 16、BF16、有效 batch 8、one epoch。当前机器没有 LoRA/PEFT，
MPS 拒绝 BF16，训练器也没有 epoch 参数。因此 owner 批准把候选方案登记为：全骨干
FP32/MPS、fresh AdamW、300 steps、每 50 steps 评估、三 seed（17/18/19）。这是协议变更，
不是独立审核通过。

协议显式写出哪些值只是沿用运行时默认、哪些值是 amendment：`steps=300`、`eval_every=50`、
`backbone_lr=2e-5` 和 `head_lr=2e-4` 与当前 CLI 默认相同；`batch_questions=16`、
`max_length=2048`、`device=mps`、`precision=fp32` 是本协议明确冻结的实验参数，不能再称为
“全部等于运行时默认值”。运行时还支持 `brier`；仅 `paired_brier_pg` 在 MPS 被拒，本协议选用
`gold_distribution + ce`。

## 2. 数据与门禁

V1 工程语料仍然不可用：存在跨 split 的 canonical input 重复、评估来源冲突和仅换 seed 的
伪独立性。必须先产生经过审查的新语料版本，按“对比 lineage + 完全相同的 model-visible input”
连通分量分组，并冻结 adapter、heldout/OOD 来源和 endpoint 身份。旧 V1、失败收据和 split 不
被覆盖或改写。

训练前还必须完成：新协议独立 review、新协议 SHA-256 登记、干净语料/T8g/T9 review、候选前
baseline 测量。任何训练输出必须写入新目录，不能覆盖生产 checkpoint。

## 3. 配对验收

所有 seed 都报告，门禁按三 seed 最差值判断。G4 必须在匹配初始化、同一迷宫 cohort 下同时
报告 accuracy、NLL、Brier、ECE；不允许只看准确率。若最佳点落在最后一个预注册评估点，
必须如实记录预算耗尽/未证明收敛，不得事后追加步数。

机器可读协议中的所有 authorization 字段保持 `false`。该文件完成的是协议登记，不是训练、
量化、serving、部署或生产上下文裁剪许可。
