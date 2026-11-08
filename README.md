# UniPercep-DP：基于多源时空融合与大模型认知引导的端到端视觉感知与可微动态规划系统

本项目基于 `nuScenes` 多模态自动驾驶数据集，完成了从 **`v0` 单目原型**、**`v1` A100 全量基线**、**`v2` Pro256 六相机环视高分 BEV**、**`v3` Ultra 频域去噪与多模态 DP**、**`v3.5` SafetyLoRA 轻量安全微调** 到最终版 **`v4.0` Chronos 时空 4D 动力学与符号证据闭环架构** 的完整六代技术演进。

关于每一轮技术演进的详细架构设计、瓶颈根因剖析与消融实验对比，请直接参阅同目录下的完整技术白皮书：[**`项目说明.md`**](./项目说明.md)。

---

## 一、 新实例“一键环境配置与全量恢复”指南（默认不触发重训）

所有源码（`src/` 与 `src_a100/`）、全量/Mini 预处理数据集压缩包（`processed_unipercep_full.tar.gz` 420MB / `processed_unipercep_mini.tar.gz` 40MB）、全部六代模型权重（`checkpoints_a100/` 与 `checkpoints/`，含最终版 `Full_v4_Chronos_best.pt` 与 `Full_v3_5_SafetyLoRA_best.pt`）、训练日志、评估报告及可视化结果均已 100% 同步并持久化保存在谷歌云盘 `/content/gdrive/MyDrive/colab/UniPercep_DP` 中。

当启动全新的云端 GPU 实例并挂载 Google Drive 后，只需执行以下 **一行命令**，即可自动配好 Python 环境并把代码、数据集、模型权重、日志和 Git 仓库完整拉取恢复到本地 `/root/workspace/ceshi`（**默认 `--restore-only` 模式，仅恢复环境与资产，不重新跑训练**）：

```bash
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh
```

> **默认执行流程（约 15~25 秒）**：
> 1. 初始化本地高速工作区 `/root/workspace/ceshi`；
> 2. 从云盘同步全部源码（`src/` + `src_a100/`）、项目文档（`项目说明.md`、`README.md`、`plan.md`）并解压恢复 `.git` 完整提交历史；
> 3. 自动检查并补齐 Python 依赖环境（`torch`, `timm`, `faiss-cpu`, `nuscenes-devkit`, `albumentations`, `einops` 等）；
> 4. 使用多核 `pigz` 从云盘秒级解压 14,400 样本全量多相机数据集（`dataset_full`）与 Mini 数据集（`dataset`）；
> 5. 恢复全部 23 个已训练模型权重（`checkpoints_a100/` 约 5.1 GB + `checkpoints/` 约 544 MB）、训练日志（`logs_a100/`）与评估可视化结果（`outputs_a100/`），并启动后台 30 秒云盘自动同步守护进程。

### 其他可选运行模式（按需显式指定）
```bash
# 手动将本地最新修改的代码、文档、Git 历史、权重、日志强制备份到谷歌云盘
bash /root/workspace/ceshi/reproduce_all.sh --sync-to-gdrive

# 恢复环境并运行 A100 评测与 6 联可视化大图生成
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh --eval-only

# 如需复现 v4.0 Chronos 或 v3.5 SafetyLoRA 训练
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh --resume-v4-chronos
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh --resume-v3-5-lora
```

---

## 二、 六代技术演进全量验证集核心指标总览（720 验证关键帧）

| 模型版本 | 架构代际与核心特征 | 多标签 F1 / Acc ($\uparrow$) | 小目标 AP ($\uparrow$) | BEV 分割 mIoU ($\uparrow$) | 规划 ADE ($\downarrow$) | 规划 FDE ($\downarrow$) | 碰撞率 Collision ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v0 Mini`** | 单相机 + `64x64` BEV + 404帧原型 | `-` / `70.51%` | `-` (MSE `0.041`) | `59.06%` | `3.058 m` | `5.251 m` | `-` |
| **`v1 Full_A100`** | 单前视相机 + `128x128` BEV + 14,400样本 | `66.16%` / `81.91%` | `20.89%` | `72.73%` | `1.884 m` | `3.597 m` | `1.39%` |
| **`v2 Pro256`** | 6相机环视 + `256x256` BEV + 6项升级 | `76.50%` / `79.64%` | `74.66%` | `64.55%` | `1.894 m` | `3.679 m` | `7.36%` |
| **`v3 Ultra`** | 频域去噪 + 3模态DP + 因果桥接 (10项升级) | `84.00%` / `81.45%` | `90.24%` | `68.93%` | `1.132 m` | `2.288 m` | `5.98%` |
| **`v3.5 SafetyLoRA`** | 冻结主干 + 规划/分类/分割尾端 + 安全屏障微调 | `83.64%` / `81.36%` | `87.28%` | `68.87%` | `1.033 m` *(最低 1.016m)* | `2.103 m` | `1.96%` *(最低 1.31%)* |
| **`v4.0 Chronos` (最终版)** | **3帧时空序列 + 15x物理标定二阶CTRA动力学 + 20D符号证据接地 + 原生LiDAR/Box跨连U-Net + 软阈值背景归零 + 4步可微CBF安全投影** | **`85.30%` / `83.26%`** | **`91.72%`** *(最高 **92.20%**)* | **`71.29%`** *(最高 **71.40%**)* | **`0.974 m`** *(最低 **0.957m**)* | **`1.961 m`** *(最低 **1.914m**)* | **`1.17%`** *(最低 **1.03%**)* |

---

## 三、 目录结构说明（本地 `/root/workspace/ceshi` $\leftrightarrow$ 云盘 `/content/gdrive/MyDrive/colab/UniPercep_DP`）

```text
UniPercep_DP/ (本地 /root/workspace/ceshi 与谷歌云盘 100% 镜像同步)
├── 项目说明.md                       # 完整六代技术演进路线、瓶颈根因剖析与 v4.0 架构白皮书
├── README.md                         # 快速指南与目录索引
├── plan.md                           # 阶段研究计划文档
├── reproduce_all.sh                  # 跨实例一键环境配置、数据/权重恢复与双端同步脚本
├── git_repo_backup.tar.gz            # Git 仓库完整提交历史云盘备份包
├── src_a100/                         # A100 全量与演进模型核心源码 (v1 ~ v4.0)
│   ├── build_full_dataset.py         # 3,600 关键帧 / 14,400 样本 6相机环视+LiDAR 预处理流水线
│   ├── models_a100.py                # v1 A100 基础架构与 4 组消融变体
│   ├── models_pro256.py              # v2 Pro256 (6相机环视 + 256x256 BEV + 6项升级)
│   ├── models_v3_ultra.py            # v3 Ultra (FFT频域去噪 + 3模态DP + 因果桥接等10项升级)
│   ├── models_v4_chronos.py          # v3.5 SafetyLoRA & v4.0 Chronos (时空4D + 12项根因级创新)
│   ├── train_and_ablation.py         # v1 四阶段渐进训练与消融实验脚本
│   ├── train_pro256.py               # v2 Pro256 与 v3 Ultra 训练脚本
│   ├── train_v4_chronos.py           # v3.5 SafetyLoRA 与 v4.0 Chronos 全量/微调训练脚本
│   └── evaluate_and_visualize.py     # 全量验证集评测与 6 联对比大图渲染脚本
├── src/                              # v0 Mini 原型源码
├── dataset_full/                     # A100 全量数据集 (3,600 关键帧 / 14,400 增强样本)
│   ├── processed_unipercep_full.tar.gz # 云盘持久化压缩包 (420 MB)
│   ├── train_full.jsonl              # 全量训练集索引 (2,880 关键帧 / 11,520 样本)
│   └── val_full.jsonl                # 全量验证集索引 (720 关键帧)
├── dataset/                          # v0 Mini 数据集 (404 帧, 40 MB)
├── checkpoints_a100/                 # A100 全部代际模型权重 (共 17 个 .pt 文件, ~5.1 GB)
│   ├── Full_v4_Chronos_best.pt       # [最终版 v4.0] 时空 4D 全量架构最优权重 (300 MB)
│   ├── Full_v3_5_SafetyLoRA_best.pt  # [v3.5] Safety-LoRA 轻量微调最优权重 (299 MB)
│   ├── Full_v3_Ultra_best.pt         # [v3] Ultra 架构最优权重 (299 MB)
│   ├── Full_v3_Ultra_ep5_backup.pt   # [v3] Ultra 初始备份权重 (299 MB)
│   ├── Full_Pro256_6Cam_best.pt      # [v2] Pro256 6相机环视最优权重 (283 MB)
│   └── Full_UniPercep_DP_best.pt     # [v1] A100 全量基线及各消融变体权重
├── logs_a100/                        # 全代际训练日志与指标 CSV
│   ├── pro256_metrics.csv            # v2 / v3 / v3.5-LoRA / v4.0-Chronos 逐轮验证集完整指标表
│   ├── v4_chronos_run.log            # v4.0 Chronos 22 Epochs 完整训练日志
│   ├── v3_5_lora_run.log             # v3.5 SafetyLoRA 4 Epochs 完整训练日志
│   ├── v3_ultra_run.log              # v3 Ultra 训练日志
│   └── pro256_run.log                # v2 Pro256 训练日志
└── outputs_a100/                     # 评测汇总报告与高分辨率可视化图表
    ├── final_ablation_report.md      # 跨代际与消融实验完整量化对比报告
    ├── evaluation_summary_a100.json  # JSON 格式指标汇总
    ├── metrics_comparison_chart.png  # 多指标柱状对比图
    ├── training_convergence_curves.png # 训练收敛曲线图
    └── qualitative_6panel_sample_*.png # 6 联多任务感知与可微规划可视化大图
```
