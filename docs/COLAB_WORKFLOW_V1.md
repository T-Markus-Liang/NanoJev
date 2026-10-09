# COLAB_WORKFLOW_V1 — Colab 计算工作流设计（2026-10-08）

> 目标：逐步取代 L40 算力服务器工作流。状态：设计就绪，首跑待验收。
> 本文件是 `L40_MIGRATION_V1.md` 的 Colab 版对应物；两者并存直到 Colab 链路
> 跑通一次完整训练+验收。

## 0. 关键事实（实测，非文档推测）

**Colab VM 不持久**：`colab new` = 全新 VM，`colab stop` = 释放；磁盘随
session 销毁（2026-10-08 实测：写入文件→stop→新 session→文件不存在）。
**每次开机必须重新部署环境**——这是与 L40 数据盘模式的核心区别。

**实测 L4 session 规格**（2026-10-08，colab CLI 实开）：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA L4，23,034 MiB，CC **8.9**（与 L40 同架构） |
| 驱动/CUDA | 580.82.07 / CUDA 13.0，nvcc 13.0 预装 |
| vCPU / RAM | 12 核 / 56.9 GB |
| 磁盘 | 236G 临时 |
| Python / torch | 3.13.15 / **torch 2.11.0+cu130 预装** |
| 可选 GPU | T4 / L4 / G4 / H100 / A100；TPU v5e1/v6e1 |
| 账户 | 2500 compute units/月 |

**huggingface.co 从 Colab 直连可达**——L40 上的 `hf-mirror.com` 配置
在 Colab 上不需要，模型权重直接 HF 下载（pinned revision）。

## 1. L40 → Colab 映射

| L40 工作流 | Colab 对应 |
|---|---|
| `ssh root@host` | `colab new --gpu L4 -s <name>` |
| `sshpass + authorized_keys` | 不需要（OAuth2，colab CLI 已认证） |
| `/root/gpufree-data/nanojev`（持久盘） | `/content/nanojev`（**临时**，每次重建） |
| `valen-venv`（持久 venv） | 每次 session 重跑 `colab_remote_setup_v1.sh` |
| `tar pipes over ssh` | `colab upload`（tarball）+ `colab download` |
| `tmux` 长驻训练 | VM 内 `nohup` + 本地轮询 + **周期回传 checkpoint** |
| `huggingface.co 不可达→hf-mirror` | 直连，无 mirror |
| 双 L40 并行 | 单卡（无双卡）；JEMM-27B 分片不可用 |
| 按小时计费 | compute units；空闲 VM 烧钱，用完即 `colab stop` |

## 2. 工作流分层

```
本地（canonical，唯一真相）
  ├── scripts/colab_bringup_v1.sh      ← 编排器（唯一入口）
  ├── scripts/colab_remote_setup_v1.sh ← 上传到 VM 执行的环境部署
  ├── scripts/colab_requirements_v1.txt← Colab 适配依赖（torch 不钉版）
  └── research/remote_archive/l40_notes/causal-conv1d-1.7.0-sm89/  ← sm89 补丁源码（唯一副本！）

Colab VM（临时，/content/nanojev）
  ├── 部署期：requirements → conv1d sm89 编译 → HF 模型 → env_receipt.json
  ├── 训练期：valen.train → output/<run>/latest/checkpoint.pt
  └── 回传期：本地每 PULL_EVERY(300s) 轮询，checkpoint sha 变化即 tar+download
```

**数据流方向永远是单向进出**：本地是 canonical，VM 只进（代码/数据 tar包）
只出（output tarball）。任何"远端唯一文件"都必须当天回传——VM 随时会被回收。

## 3. 环境重建要点（来自 INSTALL_NOTES，已本地化）

1. **causal-conv1d sm89 补丁**：源码已归档到
   `research/remote_archive/l40_notes/causal-conv1d-1.7.0-sm89/`（本地唯一副本，
   L40 系统盘上的原版随时会丢）。setup 时上传 slim tarball 并在 VM 上编译。
   Colab 有系统 nvcc 13.0 → 用 `CUDA_HOME=/usr/local/cuda`；fallback 才是
   INSTALL_NOTES 的 pip toolchain 方案（nvcc/crt/nvvm 13.0.88 三件套）。
2. **NCCL**：Colab 单卡无需 NCCL；L40 的 cu12/cu13 冲突不迁移。
3. **torch 版本漂移**：L40=2.14.0+cu130，Colab 预装 2.11.0+cu130——
   requirements 不钉 torch；需要严格对齐时按
   `colab_requirements_v1.txt` 尾部注释升级。fp32 parity 复测照旧（§6）。
4. **fla fast path 验证**：setup 第 4 步强制断言
   `causal_conv1d_fn` 在 cuda 上跑通；失败即停（fallback 慢 ~30×，不接受）。
5. **Mac 路径 symlink**：`mkdir -p /Users/markus/Documents/NanoJev` +
   `data→$R/data` 符号链接，configs 零修改（与 L40 同约定）。

## 4. 用法

```bash
# 训练一个 run（默认 L4，300s 轮询 checkpoint，结束自动停机）：
bash scripts/colab_bringup_v1.sh data/valen_nano_v9 configs/cuda/sft_nano_v9_cuda.json v9sft

# 调试模式（训练完保留 session）：
KEEP=1 bash scripts/colab_bringup_v1.sh data/valen_nano_v9 configs/cuda/sft_nano_v9_cuda.json v9sft

# 查看用量：
colab usage
```

产物落在 `external/valen/output_colab/<RUN>/`（含训练日志 tarball）。
后续步骤不变：MPS fp32 parity → 四层 eval → G1-G6 门禁
（`V5_ACCEPTANCE_RUNBOOK_V1.md`，其中远端段替换为本脚本）。

## 4.5 Drive 资源布局与存放位置（2026-10-08 落地，权威清单）

rclone remote `gdrive:` 已配置（OAuth 完成，实测 30 TiB 额度 / 29.99 TiB 空闲）。
凭据文件 `~/.config/rclone/rclone.conf`（含 refresh token——**永不进 git**；
Colab VM 通过 `STAGE_DRIVE=1` 时随 tarball 上传，VM 临时盘销毁即失效）。

### Drive 远端布局

```
gdrive:nanojev-archive/          冷归档层（一次性全量，增量 push）
├── checkpoints/                 本地 checkpoints/ 全部旧产物（95G，已校验删除本地副本）
│   └── <run_name>/              e.g. t9d_v7_minicpm2b_seed20_cuda/
├── remote_archive/              A100/L40 远端全部回传文件（27G）
│   ├── nanojev_remote_files/    远端 workspace 全量（manifest 校验过）
│   └── l40_notes/               INSTALL_NOTES.md + conv1d sm89 补丁源码（唯一副本）
└── valen_tgz/                   valen output 归档包（v3/v4/a100_backup）

gdrive:nanojev-staging/          热中转层（Colab 管道用）
└── data/<dataset>/              STAGE_DRIVE=1 推上来的数据集，VM 端 rclone 拉回
```

### 本地 ↔ Drive 操作

```bash
bash scripts/drive_sync_v1.sh status              # 看远端用量/布局
bash scripts/drive_sync_v1.sh push                # 全量冷归档 push
bash scripts/drive_sync_v1.sh check               # rclone check 三段校验
bash scripts/drive_sync_v1.sh stage-data <dir>    # 数据集 → staging
bash scripts/drive_sync_v1.sh pull <remote> <local>  # 按需拉回
# 例：恢复某个 checkpoint
bash scripts/drive_sync_v1.sh pull \
  gdrive:nanojev-archive/checkpoints/t9d_v7_minicpm2b_seed20_cuda \
  checkpoints/t9d_v7_minicpm2b_seed20_cuda
```

### 本地瘦身结果（2026-10-08 执行，全部经 `rclone check` 校验）

| 删除项 | 回收 | 状态 |
|---|---:|---|
| `checkpoints/` 除 local_atomic_seed17 外 29 个目录 | ~93G | ✅ 上 Drive |
| `remote_archive` 内 3×9GB safetensors + l40_env_bundle + conv1d build/ | ~25G | ✅ 上 Drive |
| `external/valen/*.tgz`（v3/v4/a100_backup） | ~1.6G | ✅ 上 Drive |
| 公开权重 Qwen2.5-3B / Qwen3-1.7B / Qwen3-0.6B | ~11G | 可重下，未传 |
| **合计释放** | **~123G** | 空闲 229G→352G |

**保留本地**（在用或自训未归档）：`checkpoints/local_atomic_seed17`(:8876)、
Winnow-12B(:8091)、Qwen3.5-9B(clef teacher)、Kev(:8092)、valen output_v3/v4(:8094)、
`agent-jev`/`this-that-model`（自训）、`data/` 全量。

删除日志：`research/remote_archive/LOCAL_CLEANUP_2026-10-08.md`。

### 实测瓶颈与教训

- **本机上行 ~1-12 MiB/s 波动**（代理出口决定），123G 冷归档实际跑了 ~5h；
  8 并发会被代理掐死在 ~47MB 偏移，**4 并发 + 超时重试稳定**。
- **出口会耗尽**：龙猫云 7892 跑一半没流量了，切星岛梦 7894 续完——
  脚本 `DRIVE_PROXY` env 可切换（默认 7894）。
- **看门狗在 `/tmp/drive_watchdog.sh`**：10min 日志无更新自动重启 rclone。
- VM↔Drive 是 Google 内网，大文件拉取在 VM 侧快——**大件优先走 Drive 中转
  而不是本机直连上传**。

## 5. 不做的事 / 边界

- **JEMM-27B 不上 Colab**：bf16 ~54GB 超 L4(23GB)；A100-80G 档太贵。
  需要时保留 L40 分片方案或本地 MPS。
- **无双卡并行**：两个 run 要开两个独立 session（或排队）。
- **Drive 不用原生 `drive.mount`**：需要每 session 交互授权；
  改用 rclone remote（token 复用、VM 端 headless 可用）——布局见 §4.5。
- **不长驻服务**：eval/judge 服务仍跑本机（8876/809x）；Colab 只做训推。
- **Drive 不替代本地 canonical**：归档是副本；删除本机副本前必须
  `rclone check` 校验通过 + owner 确认。
- **成本纪律**：每次 `colab new` 前 `colab usage`；所有失败路径要
  `colab stop`（脚本 ERR trap 提示，KEEP=1 调试用完即停）。
- **断连恢复**：训练中断后 checkpoint 以最后一次轮询为准续训
  （`--initialize output/<run>/latest`），不要指望 VM 状态。

## 6. 首跑验收（2026-10-08 `colab_smoke1`，全链路 PASS）

| 项 | 结果 |
|---|---|
| session 创建 | ✅ ~40s |
| env 重建（deps+conv1d sm89 编译+fla 断言+HF 模型） | ✅ ~20min（conv1d 全架构编译占大头） |
| warmup 训练（6 steps，bf16 cuda:0） | ✅ peak_cuda 2.03GB，~50s |
| 产物回传 | ✅ checkpoint.pt + metrics + run_manifest 落 `output_colab/colab_smoke1/` |
| **成本** | **整场 ~45min session = 0.98 units**（2500/月 ≈ 用不完） |
| run_manifest versions | torch 2.11.0+cu130 / transformers 5.17.0 / peft 0.21.0 |

### 首跑实测发现的坑（已全部修入脚本）

1. **上传目录必须先在 VM 上 mkdir**——contents API 不建父目录（500）。
2. **代理出口**：星岛梦(7894) 对 colab.dev 长连接会 SSL EOF 断流；
   龙猫云(7892) 通畅但额度可能耗尽——**脚本顶部 export 哪个出口是可变的，
   当前默认 7892，断流时改 `DRIVE_PROXY`/`COLAB_PROXY` 或脚本顶部的端口**。
3. **解包布局**：valen repo 展平到 `$R`（pyproject/configs 在 `$R` 下，
   `valen/` 是包目录）——训练 cwd 必须是 `$R` + `PYTHONPATH=$R`；
   `models/` `output/` `logs/` 也都在 `$R` 直下。
4. `uv pip --system` 参数顺序、`python -m valen` 需要 editable
   install 或 PYTHONPATH——均已修。

## 7. 待办

- [x] 首跑后回填：env 部署 ~20min、整场 0.98 units、训练吞吐实测
- [ ] `colab run` 变体（一次性 job 包装：短 eval/小 smoke 不开常驻 session）
- [ ] Drive 挂载交互流程验证（人工跑一次，记步骤）
- [ ] 正式数据集首训（v8/v9/v10 之一）+ fp32 parity vs MPS
- [ ] L40 正式下线前的最后一次全量 manifest 对账
