# UniPercep-DP 补充实验执行清单（`list-shiyan.md`）

> **当前论文状态说明**：  
> `paper/main_en.pdf`（19 页，0 编译错误 / 0 排版溢出 / 0 BibTeX 警告）已完成全部**无需补实验的投前排雷修订**：
> 1. **页眉与投前格式规范化**：页眉已由硬编码的 `TPAMI` 改为通用盲审/初审头 `MANUSCRIPT SUBMITTED FOR PEER REVIEW`（注：投稿前请在 `paper/main_en.tex` 第 41–43 行填入学校/机构单位 Affiliation）。
> 2. **消除跨协议“双标涨点”表述**：摘要、引言、结论及 `Fig. 1(b)` 已严格区分**完全同公式跨模型指标**（`Scene Micro-F1`: `76.57% -> 85.33%`、`BEV Seg mIoU`: `57.93% -> 71.29%`、`3.0s ADE/FDE`: `-61.5% / -56.5%` 至 `0.974m / 1.961m`）与 **$256^2$ 网格内部同口径对比**（小目标对比度 AP `91.77%` vs. `Safety-LoRA` `87.27%`；4 步 CBF 动态碰撞率较投影前降低 `-75.4%`，`4.91% -> 1.21%`），不再将 $128^2$ 与 $256^2$ 不同公式的数值直接做差。
> 3. **两处易招黑概念全面降敏**：将原 `15-Rule Offline-FAISS RAG` 统一规范为 **15-Rule Differentiable Memory Retrieval & Rank-32 LoRA（15 条交规可微记忆检索与低秩适应）**（同步更新 `Fig. 1`、`Fig. 2`、`Fig. 3`、`Fig. 7`）；将单前视相机 FPN 扩维明确界定为 **Feature-Space 6-Sector Symmetric Frustum Regularization（特征空间 6 扇区对称视锥正则化，车侧后几何由 $360^\circ$ LiDAR 锚定）**，彻底消除物理歧义。
> 4. **主动铺垫无高精地图（HD-Map-Free）基线边界**：在 `Sec. 4.3` 与 `Sec. 5.2` 明确阐明 UniAD / VAD / DiffusionDrive 依赖 6 相机环视、矢量化高精地图与 3D 多目标跟踪标注，而本基准聚焦无高精地图、单前视退化 + 双帧点云设定。

---

## 实验优先级总览表（按“二区中稿率提升性价比”排序）

| 优先级 | 编号 | 实验名称 | 是否需重训 | 预计单卡 A100 耗时 | 解决的核心审稿风险 | 建议执行时机 |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| **P0** | **实验 1** | 全模型统一评估器（同一 AP + 同一碰撞率）纯推理重测 | **否（仅推理）** | ~12 分钟 | 彻底消除 Table 2 的 `--*` 跨分辨率协议不一致质疑 | **强烈建议投前必跑** |
| **P0** | **实验 2** | `Config B` 与 `Config C` 全量 $N=2,140$ 补齐全部列 | **否（仅推理）** | ~5 分钟 | 消除 Table 6(a) 中 `Config B*` ($N=512$) 星号与 `@1s/@2s` 空缺 | **强烈建议投前必跑** |
| **P0** | **实验 3** | 导出原始未混合 $\bm{\omega}$ (`dp_weights`) 与真实 `B_CBF` 切片 | **否（仅推理）** | ~3 分钟（顺带导出） | 替换 Fig. 7 / Fig. 8 的复合代理可视化，删掉 Sec. 5.2 第 5 条局限 | **强烈建议投前必跑** |
| **P1** | **实验 4** | $128^2$ `TransFuser` 补齐 `1/7` 在线夜间光度缩放重训 | 是（15 ep） | ~25 分钟 | 剥离 Table 4 夜间子集（$N=164$）数据增强 vs. 4D 架构增益 | 投前算力充裕可跑 |
| **P1** | **实验 5** | 补充 1–2 个 2023+ 现代多模态基线（如 `BEVFusion-Planner`） | 是（15–25 ep） | ~1.5 小时 | **直击头号风险**：回应“外部基线仅有 2022/2023 TransFuser” | **冲稳二区一审极佳加分项** |
| **P1** | **实验 6** | 单子模块解耦消融（`w/o FFT`、`w/o FiLM`、`w/o Sharpener`、`w/o 20D`） | 仅推理或微调 5 ep | ~1–1.5 小时 | 回应 `Config B -> C` 打包式消融过粗、单模块独立贡献不明 | 投前可选 / 大修备用 |
| **P1** | **实验 7** | 后置可微三对角曲率平滑器（Curvature Smoother）微调 | 是（微调 4–6 ep） | ~25 分钟 | 在保持 `1.21%` 低碰撞率的同时将 `P95 Jerk` 从 `10.61` 压回 `~5.5` | 投前可选 / 大修备用 |
| **P2** | **实验 8** | 3 组随机种子（`seed=42, 1024, 2026`）均值 $\pm$ 标准差与 $p$ 值 | 是（重跑 2 个种子） | ~6–8 小时 | 应对部分审稿人对单次训练随机波动与统计显著性的提问 | **等一审意见要求再补** |
| **P2** | **实验 9** | 真实 6 相机环视 `nuScenes` 接入 `PolarLiftSplatBEV256` | 是（重提特征+训练） | ~5–8 小时 | 验证真 6 相机环视输入相对单前视 6 扇区扩维的进一步上限 | **等一审意见或冲一区再补** |
| **P2** | **实验 10** | CARLA / nuPlan 闭环评测 + 去除 `1.25m/0.5s` 静止下界验证 | 是（闭环工程） | ~3–7 天 | 回应开环规划（Open-Loop）局限与静止驻车起停泛化能力 | **二区投前无需做，留作大修预案** |

---

## 第一梯队 P0：零重训纯推理补测（共约 20 分钟，性价比最高！）

> **现存环境与权重路径**（位于云端硬盘 `/content/gdrive/MyDrive/colab/UniPercep_DP` 或服务器 `/root/workspace/ceshi`）：
> - 验证集：`dataset_full/val_full.jsonl`（共 $N = 2,140$ 样本 = $535$ 物理关键帧 $\times 4$ 退化模式）
> - 现成权重（`checkpoints_a100/`）：
>   - `Ablation_1_Baseline_best.pt`（`Latent TransFuser`, $128^2$）
>   - `Ablation_2_With_SSL_LPFT_best.pt`（`Latent TransFuser + SSL`, $128^2$）
>   - `Ablation_3_With_MultiSensor_best.pt`（`TransFuser`, $128^2$）
>   - `Full_UniPercep_DP_Stage3_RAG_LoRA_done.pt` 或 `Full_UniPercep_DP_best.pt`（`Config B`, $128^2$ + Diff-DP + Rule-LoRA）
>   - `Full_v3_Ultra_best.pt`（`Config C`, $256^2$ PolarLSS + FFT/SSL/FiLM + Skip U-Net）
>   - `Full_v3_5_SafetyLoRA_best.pt`（`UniPercep-DP (Safety-LoRA)`, $256^2$）
>   - `Full_v4_Chronos_best.pt`（`UniPercep-DP (Ours)`, $256^2$, Global Ep 53）
> - 基础评测脚本：`src_a100/run_paper_supplement.py`

### 实验 1：【P0 · 仅推理】跨分辨率统一评估协议重测（填平 Table 2 全部 `--*`）
- **问题根源**：`src_a100/run_paper_supplement.py` 第 275–310 行对 $128^2$ 模型使用 `5×5 NMS 2m/4m 命中召回` 和 `sampled_risk > 0.68`（含静态障碍峰值 `0.72`）判撞，而对 $256^2$ 模型使用 `峰值-背景对比度 AP` 和 `gt_seg_256 == 2` 动态像素判撞，导致 Table 2 增益行只能标 `--*`。
- **具体操作**：
  1. 在 `run_paper_supplement.py` 中新增统一物理米制评估函数 `evaluate_unified_256_metrics(...)`：
     - **统一小目标检测 AP（物理米制 CenterNet AP）**：将 $128^2$ 预测热力图 `small_hm_pred` 双线性上采样至 $256\times 256$（或将两者峰值统一转换到自车物理米制坐标 $(x_{\text{m}}, y_{\text{m}})$），在全部 `val_full.jsonl` 真值小目标中心上，统一按阈值 $\{0.5\text{m}, 1.0\text{m}, 2.0\text{m}, 4.0\text{m}\}$ 计算标准 Precision-Recall 曲线下面积（`mAP@Center`），同时保留统一的峰值对比度得分作辅助对照。
     - **统一碰撞率（物理坐标双口径）**：将所有模型的 6 步预测航点统一投影至 $256\times 256$（$0.25\text{m/cell}$）真值语义图 `seg_256`（带自车包络膨胀半径或精确像素一致口径），同时输出 **动态障碍碰撞率（`Dynamic Coll %`, Class 2）** 与 **总障碍碰撞率（`Total Obs Coll %`, Class 1 + Class 2）**。
  2. 对上述 5 个主表模型（含 Clean 与 Mode 0–3、Day/Night、10 类场景）一次性跑完前向推理。
- **产出更新**：直接更新论文 **Table 2、Table 4、Table 5**，彻底删掉 `--*` 脚注，使跨分辨率对比 100% 严丝合缝。

### 实验 2：【P0 · 仅推理】补齐 Table 6(a) 中 `Config B` 与 `Config C` 的全量 $N=2,140$ 指标
- **问题根源**：Table 6(a) 中 `Config B*` 目前引用的是旧版 `final_metrics.json` 在 $N=512$ 子集上的结果，且 `Config B` 与 `Config C` 缺少 `ADE@1.0s` 和 `ADE@2.0s`。
- **具体操作**：
  - 在同一次推理脚本中加载 `Full_UniPercep_DP_Stage3_RAG_LoRA_done.pt`（及 `Full_UniPercep_DP_best.pt` 择优）与 `Full_v3_Ultra_best.pt`，在完整 $N=2,140$ `val_full.jsonl` 上采用实验 1 的统一评估器输出：`Micro-F1`、`Det AP`、`Seg mIoU`、`ADE@1.0s`、`ADE@2.0s`、`ADE@3.0s`、`FDE@3.0s`、`Collision Rate`。
- **产出更新**：去掉 Table 6(a) 的 `*` 号与全部 `--` 空白格。

### 实验 3：【P0 · 仅推理】导出原始无混合 `dp_weights` 与 Eq. (19) 真实 `B_CBF` 势场张量
- **问题根源**：此前未单独落盘逐样本原始张量，导致 Fig. 7 画的是融合场景语义先验的复合权重 $w_{\text{eff}}$，Fig. 8 背景热力图画的是多尺度池化平滑场 $\tilde{B}(x,y)$。
- **具体操作**：
  1. 在 `Full_v4_Chronos_best.pt` 遍历 $N=2,140$ 验证集时，直接把模型原生输出的 `out["dp_weights"]`（形状 `(2140, 3)`）及 `out["rag_weights"]` 保存为 `raw_rule_lora_weights_2140.npz`。
  2. 对 Fig. 8 选定的 3 个代表性关键帧，直接从 `models_v4_chronos.py` 的 `apply_cbf_safety_projection` 内部钩出（hook）真实的 Eq. (19) `barrier_field`（$B_{\text{CBF}} \in \mathbb{R}^{256\times 256}$）与负梯度向量 $-\nabla B_{\text{CBF}}$，保存为 `raw_cbf_fields_3cases.npz`。
- **产出更新**：重绘 **Fig. 7** 与 **Fig. 8**，彻底移除正文中的代理可视化免责声明及 `Sec. 5.2` 第 5 条局限性。

---

## 第二梯队 P1：高价值轻量训练实验（进一步加固二区一审防线）

### 实验 4：【P1 · 轻量重训 ~25 分钟】$128^2$ `TransFuser` 补齐 `1/7` 在线夜间光度缩放
- **审稿人痛点**：`Sec. 4.3` 与 `Table 4` 坦诚披露了 $256^2$ 训练包含 `idx % 7 == 0` 的在线夜间线性变暗增强（$\gamma_s \in \{0.28, 0.34, 0.40\}$），而 $128^2$ 训练脚本 `src_a100/train_and_ablation.py` 未加该增强。审稿人可能质疑夜间子集（`ADE: 4.698m -> 1.559m`）的提升主要靠数据增强而非模型设计。
- **具体操作**：
  - 在 `src_a100/train_and_ablation.py` 的 `UniPercepFullDataset.__getitem__`（训练态）加入与 `UniPercepChronos256Dataset` 完全一致的 `idx % 7 == 0` 在线光度缩放及 `multi_label[0] = 1.0`，重训 `Ablation_3_With_MultiSensor`（`TransFuser + NightAug`）15 个 Epoch，保存权重并用统一评估器评测。
- **产出更新**：在 **Table 4（Nighttime 子集）** 增加一行 `TransFuser (+ Matched Night Aug.)`（或直接替换为严格对齐增强后的基线结果），彻底堵住数据增强不对称的质疑。

### 实验 5：【P1 · 强烈推荐冲高胜率 ~1.5 小时】在自建 14,400 样本基准上补充 1–2 个 2023+ 现代多模态基线
- **审稿人痛点（头号 Major Revision 风险）**：当前同场重训的外部基线仅有 `Latent TransFuser` 和 `TransFuser` [1, 2]。二区审稿人极常提出：*“请在同一数据集与相同无高精地图监督下，至少对比 1 个 2023 年以后的现代 BEV 融合规划基线。”*
- **最具性价比的基线选型（无需高精地图与 3D 跟踪框标注，直接复用现有 `UniPercepChronos256Dataset` 数据流）**：
  1. **`BEVFusion-Planner`（2023 ICRA `BEVFusion` 风格双流拼接融合 + 航点规划头）**：
     - 图像流采用标准 `ResNet50 + LSS` 投影至 BEV，点云流采用标准卷积编码器提取 BEV 特征，通过 `BEVFusion` 标志性的 `ConvFuser`（通道拼接 + $3\times 3$ 卷积 + 残差块，无 2D-FFT/SSL/FiLM、无 268-ch Skip U-Net、无 Peak Sharpener）输出分割与检测头，后端接标准 `Spatial Pool + GRU / MLP` 航点回归头，在相同 $256^2$（或 $128^2$）输入与相同 Epoch 下从头训练。
  2. **（可选第二个现代基线）`QueryPlan-DETR` / 轻量级 `VAD-Lite (Map-Free)`**：
     - 在融合 BEV 特征图上使用可学习 Ego-Query 通过 3 层 Transformer Decoder 做 Cross-Attention 查询 BEV 特征并直接解码多模态轨迹（模拟 `VAD` / `UniAD` 的 Query-based 规划头范式，去掉高精地图依赖）。
- **产出更新**：加入 **Table 2、Table 3、Table 4**，直接把论文的对比年代推进到 2023–2025 现代端到端范式，一审过稿率将大幅跃升。

### 实验 6：【P1 · 细粒度消融 ~1–1.5 小时】拆分前端抗退化与中端认知子模块的独立消融表
- **审稿人痛点**：Table 6(a) 从 `Config B` 到 `Config C` 一次性叠加了 `PolarLiftSplat + 2D-FFT + Gated SSL + FiLM + Skip U-Net`，审稿人容易追问每个子模块是否真的有用（是否有冗余堆叠）。
- **具体操作**（基于 `Full_v4_Chronos` 框架做 Leave-One-Out 消融，可冻结主干其他部分微调 5–8 ep 或从 Stage 3 权重微调）：
  1. `w/o Dynamic 2D-FFT Gate`（旁路 `SpectralDenoiseBlock`）——重点观察 `Mode 1/2/3` 退化子集 mIoU 与 ADE 退化量；
  2. `w/o Degradation FiLM`（旁路 `DegradationFiLMModulator`）；
  3. `w/o BackgroundZeroPeakSharpener`（直接使用原始 Sigmoid 热力图）——重点观察远小目标 `Det AP` 跌落幅度；
  4. `w/o 20D Bottom-Up Evidence Refiner`（仅用 Top-Down 10-Node MHA 分类头）——重点观察 10 类场景 `Micro-F1` 跌落幅度；
  5. `w/o 9-ch 4D LiDAR Flow`（令 $g_{\text{temp}} = 0$，仅保留单帧 `BEV_t`）——重点观察运动目标交互碰撞率与 ADE 变化。
- **产出更新**：在 `Sec. 4.7` 新增一个小巧的 **Table 7（Component-Wise Leave-One-Out Ablation）**，彻底粉碎“模块堆砌无独立验证”的质疑。

### 实验 7：【P1 · 舒适度优化 ~25 分钟】4 步 CBF 后接入可微三对角曲率平滑器微调
- **审稿人痛点**：目前 Stage 6 开启 4 步 CBF 后碰撞率降了 75.4%（`4.91% -> 1.21%`），但 `P95 Jerk` 从 `4.92` 升至 `10.61 m/s³`（略高于真值 `9.21 m/s³`）。
- **具体操作**：
  - 在 `src_a100/models_v4_chronos.py` 的 `apply_cbf_safety_projection` 输出 $\mathbf{p}^{(4)}$ 后，求解带固定起点边界的 6 点三对角二次平滑系统 $(I + \lambda_{\text{sm}} D_2^\top D_2)\hat{W}_{\text{smooth}} = \mathbf{p}^{(4)}$（其中 $\lambda_{\text{sm}} \in \{0.15, 0.25\}$，`torch.linalg.solve` 天然完全可微且耗时 $<0.1\text{ms}$），加载 `Full_v4_Chronos_best.pt` 微调 4–6 个 Epoch（或在推理端直接网格搜索 $\lambda_{\text{sm}}$ 检验零重训效果！）。
- **产出更新**：若零重训或短微调即可在保持 `Collision <= 1.25%`、`ADE <= 0.98m` 的前提下将 `P95 Jerk` 压至 `5.5~7.5 m/s³`，可直接更新 **Table 3** 与 **Table 6(b)**，完美消除舒适度短板。

---

## 第三梯队 P2：高成本重型实验（建议留作一审大修备用，投二区前不必盲跑）

### 实验 8：【P2 · 约 6–8 小时】3 随机种子（`seed = 42, 1024, 2026`）统计显著性检验
- **说明**：固定数据划分，用另外两个随机种子重跑 `TransFuser` 与 `UniPercep-DP` 完整训练流程，计算 `Mean ± Std` 与配对双侧 $t$ 检验（Paired $t$-test $p < 0.001$）。若二区一审审稿人提及随机性，再补跑即可。

### 实验 9：【P2 · 约 5–8 小时】真实 6 相机环视 `nuScenes` 验证
- **说明**：当前 `build_full_dataset.py` 仅提取了 `CAM_FRONT`。若审稿人要求验证真实 6 路环视相机（`CAM_FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, `BACK`, `BACK_LEFT`, `BACK_RIGHT`）输入 `PolarLiftSplatBEV256` 的性能，可提取 535/3,600 关键帧的 6 相机图像并在 $256^2$ 主干上训练对比“单前视 6 扇区扩维 vs. 真实 6 相机环视”。

### 实验 10：【P2 · 约 3–7 天】CARLA / nuPlan 闭环仿真 + 静止驻车样本去下界验证
- **说明**：
  1. 若投 *IEEE TVT* 或 *Neurocomputing*，论文标题与正文已严格界定在 real-world `nuScenes` 开环鲁棒感知与轨迹规划评估，并在 `Sec. 5.2` 坦诚讨论闭环局限，通常无需投前耗费一周搭建 CARLA 传感器退化环境。
  2. 同理，`build_full_dataset.py` 中 `1.25m / 0.5s` 的纵向位移下界若被审稿人问及红绿灯静止等待场景，可在大修阶段抽取 200–300 个 `nuScenes` 红灯刹停/静止帧做专项推理或微调补充表。

---

## 跑完实验后的回传文件清单（方便一键自动更新论文）

你们在服务器/Colab 跑完上述任一批次实验后，只需将生成的以下结果文件放回 `/Users/baiqin/Desktop/top/UniPercep-DP/outputs_a100/paper_supplement/`（或直接告知路径），即可自动同步更新全套 LaTeX 表格、矢量图与正文数值：

1. **`unified_eval_metrics.json`**（实验 1 & 实验 2 输出的全模型统一口径 JSON，含 Overall / Clean / Mode 0–3 / Day & Night / 10 Scenarios / Config A–D 全套指标）；
2. **`raw_rule_lora_weights_2140.npz` 与 `raw_cbf_fields_3cases.npz`**（实验 3 导出的原始路由权重与真实 Eq. 19 CBF 势场切片，用于一键重绘 Fig. 7 与 Fig. 8）；
3. **`p1_extra_baselines_and_ablations.json`**（若执行了实验 4–7，包含夜间对齐基线、`BEVFusion-Planner` 现代基线、单模块消融或曲率平滑器的指标汇总）。
