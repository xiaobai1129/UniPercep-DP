# UniPercep-DP (A100-80GB): 全量训练、消融实验与三代架构演进 (`v1 -> v2 -> v3`) 定量评估报告

## 一、 全量验证集定量对比表 (`val_full.jsonl`)

| 模型变体 (Model Variant) | 核心配置说明 | MultiLabel F1 / Acc (%) $\uparrow$ | Small Object AP (%) $\uparrow$ | Seg mIoU (%) $\uparrow$ | Traj ADE (m) $\downarrow$ | Traj FDE (m) $\downarrow$ | Collision Rate (%) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Ablation_1_Baseline` | Monocular Only, No SSL/LP-FT, Direct MLP Traj | 79.66 / 73.59 | 3.68 | 31.17 | 4.095 | 7.007 | 6.25 |
| `Ablation_2_With_SSL_LPFT` | + SSL Anti-Degradation & LP-FT Tuning | 80.37 / 74.79 | 6.10 | 31.17 | 3.916 | 6.812 | 2.73 |
| `Ablation_3_With_MultiSensor` | + 3x128x128 LiDAR BEV & 14D Homography Cross-Attn | 79.85 / 74.82 | 42.46 | 57.61 | 2.981 | 5.348 | 3.71 |
| `Full_UniPercep_DP` (v1.0-Base128) | + 128x128 Differentiable DP & RAG-LoRA Controller | 83.87 / 81.19 | 54.73 | 64.76 | 1.500 | 3.031 | 3.12 |
| **`Full_Pro256_6Cam` (v2.0-Pro256, Ep 21)** | **+ 6-Cam Surround + 256x256 BEV + Polar Lift + Kinematic GridSample DP + ClassQuery ML + ConvNeXt/EMA** | **84.12 / 81.26** | **91.27** | **71.00** | **1.062** | **2.083** | **5.84** |
| **`Full_v3_Ultra` (v3.0-Ultra256 Full, Ep 5)** | **+ All 21 Innovations (2D-FFT + FiLM + Homography Motion + Strip-Polar + LabelGraph + OHEM/Boundary + Repulsive Potential DP + WarmStart)** | **84.00 / 81.45** | **89.92** | **71.32** | **1.039** | **2.052** | **5.98** |
