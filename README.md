# UniPercep-DP：基于多源融合与多模态大模型引导的端到端视觉感知与动态规划模型

本项目基于 `nuScenes v1.0-mini` 多模态传感器数据，构建了一套集 **抗退化多任务密集视觉感知（SSL + LP-FT）、多传感器几何对齐跨模态融合（Cross-Attention）、空间运行风险场建模与可微动态规划（Differentiable DP）、以及基于 RAG + LoRA 的动态代价权重调控** 于一体的端到端深度学习系统。

---

## 一、 新实例“一键恢复与复现”快速指南（推荐）

所有源码（`src/` 与 `src_a100/`）、全量/Mini 预处理数据集压缩包（`processed_unipercep_full.tar.gz` 420MB / `processed_unipercep_mini.tar.gz` 40MB）、全部模型权重（`checkpoints_a100/` 与 `checkpoints/`）、训练日志、评估报告及可视化结果均已实时同步并持久化保存在谷歌云盘 `/content/gdrive/MyDrive/colab/UniPercep_DP` 中。

当启动全新的云端 GPU 实例（A100 / H100 / T4）并挂载 Google Drive 后，只需执行以下 **一行命令** 即可一键恢复完整项目状态：

### 1. 一键恢复完整项目状态（默认模式，约 10~20 秒，恢复全部代码、数据集、权重、日志与 Git 历史）
```bash
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh
# 或显式指定：
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh --restore-only
```
> **执行流程**：自动初始化本地工作区 `/root/workspace/ceshi` $\rightarrow$ 同步云盘全部源码（`src/` + `src_a100/`）、文档（`项目说明.md`、`README.md`、`plan.md`）与 `.git` 仓库历史 $\rightarrow$ 检查并安装 Python 依赖 $\rightarrow$ 使用多核 `pigz` 从云盘秒级解压 14,400 样本全量数据集（`dataset_full`）与 Mini 数据集（`dataset`） $\rightarrow$ 恢复全部已训练模型权重（`checkpoints_a100` 20+ 个 `.pt` 权重文件）、训练日志（`logs_a100`）与可视化结果（`outputs_a100`），并启动后台 30 秒云盘自动同步守护进程。

### 2. 一键恢复并复现 A100 全量评测与 6 联可视化大图
```bash
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh --eval-only
```

### 3. 一键断点续训新一轮升级算法（`v2 Pro256` / `v3 Ultra`）
```bash
# 断点续训 v2 Pro256 (6相机环视 + 256x256 BEV + 6大方向升级)
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh --resume-pro256

# 断点续训 v3 Ultra (FFT频域去噪 + 多模态DP + 因果桥接等10大方向升级)
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh --resume-v3-ultra
```

### 4. 一键全量重训（含四阶段渐进式训练与消融实验）或手动推送同步到云盘
```bash
# 全量四阶段 + 消融实验训练（支持阶段级断点续训）
bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh --full-train

# 随时手动将本地最新代码、Git 历史、权重、日志强制备份到谷歌云盘
bash /root/workspace/ceshi/reproduce_all.sh --sync-to-gdrive
```

---

## 二、 目录结构与双端同步说明

项目严格遵循本地高速工作区与谷歌云盘持久化目录的双端镜像结构：

- **本地高速工作目录**：`/root/workspace/ceshi`
- **谷歌云盘备份目录**：`/content/gdrive/MyDrive/colab/UniPercep_DP`

```text
UniPercep_DP/ (与 /root/workspace/ceshi 结构一致)
├── README.md                         # 项目说明与一键复现文档
├── reproduce_all.sh                  # 跨实例一键恢复、评测与训练脚本
├── src/                              # 核心 Python 源码
│   ├── build_dataset.py              # 数据集自动获取、多任务标签构建、RAG 索引生成与云盘打包
│   ├── models.py                     # UniPercepDPNet、DifferentiableDPPlanner、LoRARagController
│   ├── train_pipeline.py             # 硬件自适应四阶段训练流水线与每 Epoch 云盘实时同步
│   └── infer_and_eval.py             # 验证集指标计算与 1x4 四联对比大图渲染
├── dataset/                          # 数据集与索引文件
│   ├── processed_unipercep_mini.tar.gz # 预处理完成的 404 帧多任务数据集压缩包 (40 MB)
│   ├── train.jsonl                   # 训练集索引 (前 340 帧)
│   └── val.jsonl                     # 验证集索引 (后 64 帧)
├── checkpoints/                      # 各阶段训练权重 (每个约 91 MB)
│   ├── unipercep_dp_best.pt          # 全流程验证集综合最优权重
│   ├── unipercep_dp_latest.pt        # 最新训练轮次权重
│   ├── stage1_ssl_lp_latest.pt       # Stage 1 (SSL + Linear Probing) 阶段权重
│   ├── stage1_lp_ft_latest.pt        # Stage 1 (LP-FT 全量微调) 阶段权重
│   ├── stage2_fusion_dp_latest.pt    # Stage 2 (多源融合 + 可微动态规划) 阶段权重
│   └── stage3_rag_lora_latest.pt     # Stage 3 (RAG + LoRA 协同调控) 阶段权重
├── logs/                             # 训练日志
│   ├── training_metrics.csv          # 逐 Epoch 训练损失、验证集指标与动态权重记录
│   └── full_run.log                  # 全量训练与推理评测完整终端日志
└── outputs/                          # 评估报告与可视化输出
    ├── evaluation_summary.json       # 验证集核心指标汇总 JSON
    └── demo_result_sample_0~4.png    # 5 张 1x4 四联对比可视化大图
```

---

## 三、 核心模型与四阶段训练设计

### 1. 模型核心模块（`src/models.py`）
- **视觉主干与自监督重建分支（SSL + LP-FT）**：采用 `timm` 的 `resnet34` 提取多尺度视觉特征，外接轻量上采样重建解码器将弱网受损图像（JPEG 强压缩伪影 + 随机局部宏块遮挡）恢复为干净图像（计算 L1 重建损失 `loss_ssl`）。
- **多任务密集视觉感知三头**：
  - `multi_label_head`：预测 8 类复杂场景属性（夜间、雨雾、路口、行人、远距小目标、高密度拥堵、车辆、巡航状态）；
  - `seg_head`：$64\times 64$ BEV 空间下的 3 类语义分割（`0-可通行区域`、`1-静态背景障碍`、`2-动态高危障碍物`）；
  - `small_det_head`：$64\times 64$ 远距（$>25\text{m}$）及行人/锥桶小目标高斯中心热力图预测。
- **多传感器 Cross-Attention 融合层**：将多尺度视觉 Token、`LIDAR_TOP` BEV 空间编码特征与 14 维几何状态向量（5 维自车位姿速度状态 + 9 维 ORB+RANSAC 帧间单应矩阵 $H$）通过 `nn.MultiheadAttention` 进行跨模态空间对齐融合。
- **可微动态规划层（`DifferentiableDPPlanner`）**：基于融合特征回归 $64\times 64$ 空间风险代价图 `pred_risk`，沿前进方向（Y 轴）执行基于 Softmin（`-temp * logsumexp(-neighbors / temp)`）的可微动态规划状态转移递推，计算累积最小能量价值场 `dp_value_map` 并回归未来 6 步平滑避障轨迹 `pred_traj`（`Bx6x2`）。
- **RAG + LoRA 动态权重调控模块（`LoRARagController`）**：冻结基础线性投影层 `base_proj`，训练低秩矩阵参数 `lora_A`（`rank=16`）与 `lora_B`，结合预测的多标签属性与 FAISS RAG 规则先验动态输出规划器代价权重 `[w_obs, w_smooth]`。

### 2. 硬件自适应与四阶段训练配置（`src/train_pipeline.py`）

| 训练阶段 | 阶段训练策略 | T4 配置 (`fp16`, `bs=4`, `accum=4`) | H100/A100 配置 (`bf16`, `bs=32`, `accum=1`) |
| :--- | :--- | :---: | :---: |
| **`Stage1_SSL_LP`** | 冻结主干浅层，仅训高层、SSL 重建与三感知头 | 10 Epochs | 15 Epochs |
| **`Stage1_LP_FT`** | 解冻全量视觉主干进行 LP-FT 鲁棒微调 | 15 Epochs | 25 Epochs |
| **`Stage2_Fusion_DP`** | 联合训练多传感器跨模态融合、风险图与可微 DP 轨迹 | 20 Epochs | 30 Epochs |
| **`Stage3_RAG_LoRA`** | 引入轨迹二阶平滑度惩罚项，微调 LoRA 调控器与规划头 | 15 Epochs | 20 Epochs |

---

## 四、 全量训练验证集评测指标汇总

在验证集（64 帧）上的评测结果（详见 `outputs/evaluation_summary.json`）：

| 评测指标 | 最终评测值 | 指标说明 |
| :--- | :---: | :--- |
| **`MultiLabel_Acc`** | **70.51%** (`0.7051`) | 8 类场景属性多标签分类准确率（训练期最高达 `82.23%`） |
| **`mIoU`** | **59.06%** (`0.5906`) | BEV 三类障碍物语义分割 mIoU（可通行区域 IoU `87.19%`、静态障碍物 IoU `74.06%`，训练后期整体 `mIoU` 最高达 `64.88%`） |
| **`Small_Object_Heatmap_MSE`** | **0.040867** | 远距及小目标中心高斯热力图均方误差（训练后期达 `0.0205`） |
| **`Risk_Map_MSE`** | **0.06351** | $64\times 64$ 空间运行风险代价场均方误差 |
| **`SSL_Reconstruction_L1`** | **0.10035** | 弱网退化图像自监督重建 L1 误差 |
| **`Traj_ADE`** | **3.0580 m** | 未来 6 步规划避障轨迹平均位移误差（物理米） |
| **`Traj_FDE`** | **5.2508 m** | 规划轨迹终点位移误差（物理米） |
| **`LoRA_Weights_Mean`** | `w_obs=2.0151`, `w_smooth=1.9114` | RAG + LoRA 动态预测的避障与平滑代价权重均值 |

---

## 五、 分步手动运行命令参考

如果需要单独运行各个模块，可按以下命令依次执行：

```bash
# 1. 构建或从云盘恢复数据集
python3 /root/workspace/ceshi/src/build_dataset.py

# 2. 运行冒烟测试 (每阶段 1 Epoch, 最多 3 Batches)
python3 /root/workspace/ceshi/src/train_pipeline.py --smoke_test

# 3. 运行全量四阶段训练
python3 /root/workspace/ceshi/src/train_pipeline.py

# 4. 运行验证集推理评测并生成 5 张 1x4 可视化大图
python3 /root/workspace/ceshi/src/infer_and_eval.py
```
