# 《UniPercep-DP》顶刊补充纯推理实验数据汇总（对标 IEEE TPAMI TransFuser）

> **评估数据集**：`/root/workspace/ceshi/dataset_full/val_full.jsonl`（全量 2,140 个验证集样本 = 535 个物理关键帧 × 4 种传感器退化模式）  
> **硬件平台**：NVIDIA A100-SXM4-80GB GPU（推理延迟与显存采用 `Batch Size = 1`、AMP `BF16/FP16` 预热 50 次后连续测量 200 次统计均值与标准差）

---

## 实验 1：多时间域规划误差（1s/2s/3s）、运动学舒适度与推理效率表（对标 TransFuser Table 2）

### 表 1A：分时间域规划位移误差（L2 Error, `T=12`, `dt=0.25s`, 3.0s 规划域）与碰撞率

| 模型 (Model) | BEV 网格 | 整体感知 Cls F1 / Det AP / Seg mIoU (%) | ADE@1.0s (m) ↓ | ADE@2.0s (m) ↓ | **ADE@3.0s (Overall, m) ↓** | FDE@1.0s (m) ↓ | FDE@2.0s (m) ↓ | **FDE@3.0s (Overall, m) ↓** | **Collision Rate (%) ↓** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Latent TransFuser | `128×128` | 76.73 / 4.73 / 31.08 | 1.799 | 3.035 | 4.307 | 2.411 | 4.893 | 7.506 | 11.82% |
| Latent TransFuser + SSL & LP-FT | `128×128` | 75.30 / 7.85 / 31.08 | 1.902 | 3.175 | 4.465 | 2.534 | 5.095 | 7.718 | 10.47% |
| TransFuser | `128×128` | 76.57 / 45.87 / 57.93 | 1.022 | 1.749 | 2.527 | 1.364 | 2.852 | 4.511 | 7.94% |
| UniPercep-DP (Safety-LoRA) | `256×256` | 83.64 / 87.27 / 68.87 | 0.356 | 0.635 | 1.033 | 0.446 | 1.091 | 2.103 | 1.73% |
| **UniPercep-DP (Ours, Full v4)** | `256×256` | 85.33 / 91.77 / 71.29 | **0.332** | **0.604** | **0.974** | **0.433** | **1.050** | **1.961** | **1.21%** |


### 表 1B：运动学平滑舒适度指标（Kinematic Comfort）与 A100 实时推理效率指标

| 模型 / 基准 (Model / Reference) | Mean Accel ($\text{m/s}^2$) | Mean Lat Accel ($\text{m/s}^2$) | **Mean Jerk ($\text{m/s}^3$) ↓** | P95 Jerk ($\text{m/s}^3$) ↓ | Mean Curvature ($\text{m}^{-1}$) | 总参数量 Params (M) | 单帧峰值显存 Peak VRAM (MB) | **单帧延迟 Latency (ms) ↓** | **吞吐帧率 FPS (Hz) ↑** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Latent TransFuser | 0.98 | 0.41 | 3.45 | 5.09 | 0.0098 | 31.35M | 1560.4 MB | 15.98 ± 1.39 ms | 62.6 Hz |
| Latent TransFuser + SSL & LP-FT | 0.89 | 0.53 | 3.91 | 7.38 | 0.0149 | 31.35M | 482.5 MB | 16.59 ± 0.47 ms | 60.3 Hz |
| TransFuser | 0.88 | 0.65 | 3.87 | 7.25 | 0.0273 | 31.35M | 1255.7 MB | 20.10 ± 1.06 ms | 49.7 Hz |
| UniPercep-DP (Safety-LoRA) | 1.43 | 0.88 | 5.28 | 22.04 | 0.0386 | 37.66M | 5949.1 MB | 71.60 ± 1.96 ms | 14.0 Hz |
| **UniPercep-DP (Ours, Full v4)** | 1.05 | 0.53 | **3.81** | 10.61 | 0.0243 | 37.77M | 1478.9 MB | **70.80 ± 5.47 ms** | **14.1 Hz** |
| Expert GT (Physical Reference) | 0.73 | 0.16 | 2.36 | 9.21 | 0.0065 | - | - | - | - |


#### `UniPercep-DP (Ours, Full v4)` 子模块推理耗时精确分解（Batch Size = 1, A100 BF16）：

- **(a) 视觉与多模态编码器 + FPN + 4D 时空 BEV 融合**：`28.75 ± 0.95 ms`（占比 `40.2%`）
- **(b) 三任务感知解码头（小目标热力增强 + UNet BEV 分割 + 跨任务先验多标签精修）**：`6.77 ± 0.79 ms`（占比 `9.5%`）
- **(c) RAG-LoRA 路由 + 二阶 CTRA 物理锚点 + 时空流交互 Transformer + 4 步可微 CBF 规划器**：`35.97 ± 1.11 ms`（占比 `50.3%`）

---

## 实验 2：相机传感器退化模式（Mode 0~3 & Clean）与昼夜光照鲁棒性切片实验

| 测试条件 / 退化分桶 (Condition / Subset) | 样本数 $N$ | 模型 (Model) | Cls F1 (%) ↑ | Det AP (%) ↑ | Seg mIoU (%) ↑ | **ADE (m) ↓** | **FDE (m) ↓** | **Col Rate (%) ↓** |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Overall Validation (N=2140) | 2140 | Latent TransFuser | 76.73 | 4.73 | 31.08 | 4.307 | 7.506 | 11.82% |
| Overall Validation (N=2140) | 2140 | TransFuser | 76.57 | 45.87 | 57.93 | 2.527 | 4.511 | 7.94% |
| Overall Validation (N=2140) | 2140 | UniPercep-DP (Safety-LoRA) | 83.64 | 87.27 | 68.87 | 1.033 | 2.103 | 1.73% |
| Overall Validation (N=2140) | 2140 | **UniPercep-DP (Ours, Full v4)** | **85.33** | **91.77** | **71.29** | **0.974** | **1.961** | **1.21%** |
| Clean Input (Ideal Sensor, N=2140) | 2140 | Latent TransFuser | 71.77 | 5.03 | 31.08 | 4.295 | 7.494 | 11.59% |
| Clean Input (Ideal Sensor, N=2140) | 2140 | TransFuser | 76.77 | 45.22 | 57.88 | 2.659 | 4.711 | 9.16% |
| Clean Input (Ideal Sensor, N=2140) | 2140 | UniPercep-DP (Safety-LoRA) | 81.73 | 86.60 | 68.99 | 1.043 | 2.116 | 1.68% |
| Clean Input (Ideal Sensor, N=2140) | 2140 | **UniPercep-DP (Ours, Full v4)** | **82.80** | **91.61** | **71.29** | **0.977** | **1.967** | **1.31%** |
| Mode 0: Mild JPEG & Speckle (N=535) | 535 | Latent TransFuser | 71.56 | 4.66 | 31.08 | 4.300 | 7.497 | 11.78% |
| Mode 0: Mild JPEG & Speckle (N=535) | 535 | TransFuser | 76.08 | 46.05 | 57.90 | 2.566 | 4.566 | 8.22% |
| Mode 0: Mild JPEG & Speckle (N=535) | 535 | UniPercep-DP (Safety-LoRA) | 81.92 | 86.95 | 68.94 | 1.036 | 2.103 | 1.87% |
| Mode 0: Mild JPEG & Speckle (N=535) | 535 | **UniPercep-DP (Ours, Full v4)** | **83.05** | **91.59** | **71.31** | **0.977** | **1.970** | **0.75%** |
| Mode 1: Color Shift & Occ. (N=535) | 535 | Latent TransFuser | 72.26 | 4.73 | 31.08 | 4.307 | 7.506 | 11.96% |
| Mode 1: Color Shift & Occ. (N=535) | 535 | TransFuser | 73.55 | 45.62 | 57.89 | 2.531 | 4.518 | 7.85% |
| Mode 1: Color Shift & Occ. (N=535) | 535 | UniPercep-DP (Safety-LoRA) | 82.86 | 87.30 | 68.83 | 1.034 | 2.106 | 1.68% |
| Mode 1: Color Shift & Occ. (N=535) | 535 | **UniPercep-DP (Ours, Full v4)** | **84.67** | **91.72** | **71.29** | **0.974** | **1.960** | **1.31%** |
| Mode 2: Dense Atmospheric Fog (N=535) | 535 | Latent TransFuser | 80.66 | 4.96 | 31.08 | 4.315 | 7.516 | 11.59% |
| Mode 2: Dense Atmospheric Fog (N=535) | 535 | TransFuser | 78.76 | 46.01 | 58.02 | 2.491 | 4.460 | 7.85% |
| Mode 2: Dense Atmospheric Fog (N=535) | 535 | UniPercep-DP (Safety-LoRA) | 84.23 | 87.38 | 68.85 | 1.035 | 2.110 | 1.31% |
| Mode 2: Dense Atmospheric Fog (N=535) | 535 | **UniPercep-DP (Ours, Full v4)** | **85.77** | **91.93** | **71.28** | **0.972** | **1.956** | **1.12%** |
| Mode 3: Severe Packet-Loss (N=535) | 535 | Latent TransFuser | 81.66 | 4.56 | 31.08 | 4.306 | 7.504 | 11.96% |
| Mode 3: Severe Packet-Loss (N=535) | 535 | TransFuser | 77.59 | 45.82 | 57.90 | 2.519 | 4.501 | 7.85% |
| Mode 3: Severe Packet-Loss (N=535) | 535 | UniPercep-DP (Safety-LoRA) | 85.16 | 87.46 | 68.86 | 1.027 | 2.092 | 2.06% |
| Mode 3: Severe Packet-Loss (N=535) | 535 | **UniPercep-DP (Ours, Full v4)** | **87.34** | **91.84** | **71.29** | **0.973** | **1.959** | **1.68%** |
| Daytime Illumination (N=1976) | 1976 | Latent TransFuser | 77.55 | 4.36 | 31.11 | 4.009 | 6.973 | 10.53% |
| Daytime Illumination (N=1976) | 1976 | TransFuser | 78.06 | 46.13 | 57.58 | 2.346 | 4.180 | 6.07% |
| Daytime Illumination (N=1976) | 1976 | UniPercep-DP (Safety-LoRA) | 84.65 | 88.46 | 68.60 | 0.997 | 2.041 | 0.51% |
| Daytime Illumination (N=1976) | 1976 | **UniPercep-DP (Ours, Full v4)** | **85.94** | **92.58** | **71.06** | **0.925** | **1.873** | **0.61%** |
| Nighttime Low-Light (N=164) | 164 | Latent TransFuser | 67.20 | 9.07 | 30.79 | 7.890 | 13.928 | 27.44% |
| Nighttime Low-Light (N=164) | 164 | TransFuser | 58.38 | 42.81 | 60.30 | 4.698 | 8.498 | 30.49% |
| Nighttime Low-Light (N=164) | 164 | UniPercep-DP (Safety-LoRA) | 72.71 | 73.10 | 70.19 | 1.463 | 2.849 | 16.46% |
| Nighttime Low-Light (N=164) | 164 | **UniPercep-DP (Ours, Full v4)** | **79.01** | **82.04** | **72.16** | **1.559** | **3.022** | **8.54%** |

---

## 实验 3：10 类复杂交通场景细粒度切片对比表（Scenario-Wise Breakdown）

| 场景维度 (Scenario Category) | 样本数 $N_k$ | 模型 (Model) | 类别 F1-Score (%) ↑ | Det AP (%) ↑ | Seg mIoU (%) ↑ | **Traj ADE (m) ↓** | **Traj FDE (m) ↓** | **Collision Rate (%) ↓** |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0: Night (夜间低照度) | 164 | Latent TransFuser | 67.20 | 9.07 | 30.79 | 7.890 | 13.928 | 27.44% |
| 0: Night (夜间低照度) | 164 | TransFuser | 58.38 | 42.81 | 60.30 | 4.698 | 8.498 | 30.49% |
| 0: Night (夜间低照度) | 164 | **UniPercep-DP (Ours, Full v4)** | **82.80** | **82.04** | **72.16** | **1.559** | **3.022** | **8.54%** |
| 1: Rain/Fog (雨雾散射) | 1098 | Latent TransFuser | 74.36 | 4.92 | 31.08 | 4.334 | 7.563 | 12.39% |
| 1: Rain/Fog (雨雾散射) | 1098 | TransFuser | 74.91 | 45.76 | 58.05 | 2.522 | 4.521 | 8.56% |
| 1: Rain/Fog (雨雾散射) | 1098 | **UniPercep-DP (Ours, Full v4)** | **91.30** | **92.08** | **71.36** | **0.998** | **2.012** | **1.55%** |
| 2: Intersection (无保护路口) | 1216 | Latent TransFuser | 77.03 | 3.93 | 31.13 | 4.550 | 8.056 | 15.95% |
| 2: Intersection (无保护路口) | 1216 | TransFuser | 70.65 | 46.47 | 57.31 | 2.358 | 4.161 | 8.63% |
| 2: Intersection (无保护路口) | 1216 | **UniPercep-DP (Ours, Full v4)** | **79.76** | **91.55** | **69.95** | **0.798** | **1.622** | **0.99%** |
| 3: Construction (施工锥桶) | 1132 | Latent TransFuser | 74.94 | 5.55 | 31.02 | 4.482 | 7.759 | 8.48% |
| 3: Construction (施工锥桶) | 1132 | TransFuser | 75.17 | 43.32 | 55.61 | 2.744 | 4.862 | 7.24% |
| 3: Construction (施工锥桶) | 1132 | **UniPercep-DP (Ours, Full v4)** | **81.73** | **93.64** | **68.37** | **1.007** | **2.009** | **1.59%** |
| 4: Congestion (拥堵跟车) | 1760 | Latent TransFuser | 84.92 | 4.87 | 30.95 | 4.193 | 7.341 | 13.47% |
| 4: Congestion (拥堵跟车) | 1760 | TransFuser | 84.52 | 46.59 | 57.14 | 2.436 | 4.353 | 9.38% |
| 4: Congestion (拥堵跟车) | 1760 | **UniPercep-DP (Ours, Full v4)** | **89.73** | **95.57** | **70.34** | **0.910** | **1.854** | **1.25%** |
| 5: PedCross (行人横穿) | 1112 | Latent TransFuser | 75.41 | 4.53 | 31.19 | 4.279 | 7.608 | 18.79% |
| 5: PedCross (行人横穿) | 1112 | TransFuser | 74.05 | 48.85 | 58.40 | 2.228 | 3.963 | 10.61% |
| 5: PedCross (行人横穿) | 1112 | **UniPercep-DP (Ours, Full v4)** | **74.13** | **94.73** | **71.58** | **0.838** | **1.754** | **1.17%** |
| 6: FarSmallObj (远距小目标) | 2124 | Latent TransFuser | 88.28 | 4.73 | 31.08 | 4.299 | 7.498 | 11.91% |
| 6: FarSmallObj (远距小目标) | 2124 | TransFuser | 88.19 | 45.87 | 57.89 | 2.516 | 4.496 | 8.00% |
| 6: FarSmallObj (远距小目标) | 2124 | **UniPercep-DP (Ours, Full v4)** | **92.48** | **91.77** | **71.25** | **0.969** | **1.953** | **1.22%** |
| 7: FastApproach (高速逼近) | 1472 | Latent TransFuser | 78.96 | 5.00 | 31.04 | 4.699 | 8.194 | 13.65% |
| 7: FastApproach (高速逼近) | 1472 | TransFuser | 80.30 | 44.42 | 57.26 | 2.586 | 4.606 | 9.44% |
| 7: FastApproach (高速逼近) | 1472 | **UniPercep-DP (Ours, Full v4)** | **83.31** | **90.68** | **70.17** | **0.910** | **1.840** | **1.43%** |
| 8: Occlusion (视野遮挡) | 2028 | Latent TransFuser | 87.48 | 4.80 | 31.05 | 4.264 | 7.431 | 12.38% |
| 8: Occlusion (视野遮挡) | 2028 | TransFuser | 87.41 | 45.73 | 57.86 | 2.504 | 4.472 | 8.19% |
| 8: Occlusion (视野遮挡) | 2028 | **UniPercep-DP (Ours, Full v4)** | **91.67** | **92.27** | **71.24** | **0.961** | **1.941** | **1.18%** |
| 9: Curve (急弯道) | 232 | Latent TransFuser | 66.38 | 4.07 | 30.90 | 4.021 | 7.517 | 19.83% |
| 9: Curve (急弯道) | 232 | TransFuser | 68.01 | 44.58 | 58.40 | 2.762 | 5.362 | 6.03% |
| 9: Curve (急弯道) | 232 | **UniPercep-DP (Ours, Full v4)** | **76.59** | **91.81** | **71.57** | **1.317** | **2.845** | **0.00%** |

---

## 实验 4：规划器内部机制与安全投影消融实验（对标 TransFuser Table 5 & Table 10，纯推理剥离）

| 规划器内部阶段与安全消融变体 (Planner Stage / Safety Variant) | ADE@1.0s (m) ↓ | ADE@2.0s (m) ↓ | **ADE@3.0s (m) ↓** | **FDE@3.0s (m) ↓** | **Collision Rate (%) ↓** | Mean Accel ($\text{m/s}^2$) | **Mean Jerk ($\text{m/s}^3$) ↓** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. 1st-Order Linear Anchor (一阶常速度外推锚点) | 0.473 | 1.086 | 1.788 | 3.598 | 5.79% | 0.21 | 0.80 |
| 2. 2nd-Order CTRA Physical Anchor (二阶物理运动学先验 `ctra_base_128`) | 0.447 | 1.011 | 1.657 | 3.330 | 5.05% | 0.48 | 0.79 |
| 3. CTRA + RAG-LoRA Residual (`blended_traj`, Before Transformer) | 0.497 | 0.711 | 1.046 | 1.957 | 6.17% | 0.78 | 2.63 |
| 4. Before CBF Projection (`refined_traj`, w/o CBF) | 0.330 | 0.596 | 0.954 | 1.909 | 4.91% | 0.68 | 2.27 |
| 5. TransFuser Hard Safety Box Heuristic (规则硬减速安全框) | 2.097 | 3.524 | 4.970 | 8.598 | 0.75% | 0.37 | 1.25 |
| **6. Full UniPercep-DP (`safe_traj`, with 4-Step Differentiable CBF)** | **0.332** | **0.604** | **0.974** | **1.961** | **1.21%** | 1.05 | **3.81** |

