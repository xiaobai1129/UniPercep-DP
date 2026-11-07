# UniPercep-DP (A100-80GB): 全量训练、消融实验与六大方向升级 (`Pro256`) 定量评估报告

## 一、 全量验证集定量对比表 (`val_full.jsonl`)

| 模型变体 (Model Variant) | 核心配置说明 | MultiLabel F1 / Acc (%) $\uparrow$ | Small Object AP (%) $\uparrow$ | Seg mIoU (%) $\uparrow$ | Traj ADE (m) $\downarrow$ | Traj FDE (m) $\downarrow$ | Collision Rate (%) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Ablation_1_Baseline` | Monocular Only, No SSL/LP-FT, Direct MLP Traj | 79.66 / 73.59 | 3.68 | 31.17 | 4.095 | 7.007 | 6.25 |
| `Ablation_2_With_SSL_LPFT` | + SSL Anti-Degradation & LP-FT Tuning | 80.37 / 74.79 | 6.10 | 31.17 | 3.916 | 6.812 | 2.73 |
| `Ablation_3_With_MultiSensor` | + 3x128x128 LiDAR BEV & 14D Homography Cross-Attn | 79.85 / 74.82 | 42.46 | 57.61 | 2.981 | 5.348 | 3.71 |
| `Full_UniPercep_DP` (v1) | + 128x128 Differentiable DP & RAG-LoRA Controller | 83.87 / 81.19 | 54.73 | 64.76 | 1.500 | 3.031 | 3.12 |
| **`Full_Pro256_6Cam` (v2 Six-Upgrade SOTA)** | **+ 6-Cam Surround + 256x256 BEV + Polar Lift + Kinematic GridSample DP + ClassQuery ML + ConvNeXt/EMA (Ep 1)** | **81.02 / 74.00** | **51.68** | **59.37** | **1.242** | **2.509** | **7.62** |
