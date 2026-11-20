#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate all P1 experiment outputs (Exp 4, 5, 6, 7) into:
  - outputs_a100/paper_supplement/p1_extra_baselines_and_ablations.json
  - outputs_a100/paper_supplement/p1_summary.md
"""

import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = SRC_DIR.parent
SUPP_DIR = LOCAL_ROOT / "outputs_a100" / "paper_supplement"


def load_if_exists(name: str):
    p = SUPP_DIR / name
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    p0 = load_if_exists("unified_eval_metrics.json")
    exp4 = load_if_exists("p1_exp4_transfuser_nightaug.json")
    exp5 = load_if_exists("p1_exp5_modern_baselines.json")
    exp6 = load_if_exists("p1_exp6_component_ablation.json")
    exp7 = load_if_exists("p1_exp7_curvature_smoother.json")

    payload = {
        "Exp4_TransFuser_Matched_NightAug": exp4,
        "Exp5_Modern_2023Plus_Baselines": exp5,
        "Exp6_Component_Wise_LOO_Ablation": exp6,
        "Exp7_Differentiable_Curvature_Smoother": exp7,
    }
    out_json = SUPP_DIR / "p1_extra_baselines_and_ablations.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = ["# UniPercep-DP P1 补充训练与消融实验汇总（Exp 4 ~ Exp 7）\n"]

    v4_all = p0["Overall_Unified"]["UniPercep-DP (Ours, Full v4)"] if p0 else None
    v4_night = p0["Conditions_Unified"]["UniPercep-DP (Ours, Full v4)"]["Nighttime"] if p0 else None
    tf_all = p0["Overall_Unified"]["TransFuser"] if p0 else None
    tf_night = p0["Conditions_Unified"]["TransFuser"]["Nighttime"] if p0 else None

    if exp4:
        md.append("## 实验 4：$128^2$ TransFuser 补齐 1/7 在线夜间光度缩放对比（Table 4 Nighttime 对齐）\n")
        md.append("| 模型 (Model) | 子集 | Cls F1 (%) | mAP@Center (%) | Contrast AP (%) | Seg mIoU (%) | ADE (m) | FDE (m) | Dyn Coll (%) | Tot Obs Coll (%) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        rows4 = []
        if tf_all and tf_night:
            rows4.append(("TransFuser (Orig)", "Overall", tf_all))
            rows4.append(("TransFuser (Orig)", "Nighttime (N=164)", tf_night))
        rows4.append(("TransFuser (+ Matched Night Aug.)", "Overall", exp4["Overall_Unified"]))
        rows4.append(("TransFuser (+ Matched Night Aug.)", "Nighttime (N=164)", exp4["Conditions_Unified"]["Nighttime"]))
        if v4_all and v4_night:
            rows4.append(("**UniPercep-DP (Ours, Full v4)**", "Overall", v4_all))
            rows4.append(("**UniPercep-DP (Ours, Full v4)**", "Nighttime (N=164)", v4_night))
        for mname, sub, r in rows4:
            md.append(
                f"| {mname} | {sub} | {r['Cls_F1']:.2f} | {r['Det_mAP_Center']:.2f} | {r['Det_Contrast_AP']:.2f} | "
                f"{r['Seg_mIoU']:.2f} | {r['ADE']:.3f} | {r['FDE']:.3f} | {r['Dynamic_Coll_%']:.2f}% | {r['Total_Obs_Coll_%']:.2f}% |"
            )
        md.append("")

    if exp5:
        md.append("## 实验 5：2023+ 现代无高精地图多模态基线对比（统一 $256^2$ 协议）\n")
        md.append("| 模型 (Model) | Cls F1 (%) | mAP@Center (%) | Contrast AP (%) | Seg mIoU (%) | ADE@1/2/3s (m) | FDE@3s (m) | Dyn Coll (%) | Tot Obs Coll (%) | P95 Jerk | Params (M) | Latency (ms) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for mname, info in exp5.items():
            r = info["Overall_Unified"]
            md.append(
                f"| {mname} | {r['Cls_F1']:.2f} | {r['Det_mAP_Center']:.2f} | {r['Det_Contrast_AP']:.2f} | {r['Seg_mIoU']:.2f} | "
                f"{r['ADE@1.0s']:.3f}/{r['ADE@2.0s']:.3f}/{r['ADE@3.0s']:.3f} | {r['FDE@3.0s']:.3f} | "
                f"{r['Dynamic_Coll_%']:.2f}% | {r['Total_Obs_Coll_%']:.2f}% | {r['P95_Jerk']:.2f} | "
                f"{r.get('Total_Params_M', '-') } | {r.get('Latency_Mean_ms', '-')} |"
            )
        if v4_all:
            r = v4_all
            md.append(
                f"| **UniPercep-DP (Ours, Full v4)** | **{r['Cls_F1']:.2f}** | **{r['Det_mAP_Center']:.2f}** | **{r['Det_Contrast_AP']:.2f}** | **{r['Seg_mIoU']:.2f}** | "
                f"**{r['ADE@1.0s']:.3f}/{r['ADE@2.0s']:.3f}/{r['ADE@3.0s']:.3f}** | **{r['FDE@3.0s']:.3f}** | "
                f"**{r['Dynamic_Coll_%']:.2f}%** | **{r['Total_Obs_Coll_%']:.2f}%** | {r['P95_Jerk']:.2f} | 37.77 | 70.80 |"
            )
        md.append("")

    if exp6:
        md.append("## 实验 6：单子模块 Leave-One-Out 消融表（Table 7，5-Epoch 适配微调）\n")
        md.append("| 消融变体 (Variant) | Cls F1 (%) | mAP@Center (%) | Contrast AP (%) | Seg mIoU (%) | Deg(M1-3) mIoU (%) | ADE@3s (m) | FDE@3s (m) | Dyn Coll (%) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        if v4_all and p0:
            c = p0["Conditions_Unified"]["UniPercep-DP (Ours, Full v4)"]
            m13_miou = round((c["Mode_1_Color_Corrupt"]["Seg_mIoU"] + c["Mode_2_Dense_Fog"]["Seg_mIoU"] + c["Mode_3_Packet_Loss"]["Seg_mIoU"]) / 3.0, 2)
            md.append(
                f"| **Full UniPercep-DP (v4 Chronos)** | **{v4_all['Cls_F1']:.2f}** | **{v4_all['Det_mAP_Center']:.2f}** | **{v4_all['Det_Contrast_AP']:.2f}** | "
                f"**{v4_all['Seg_mIoU']:.2f}** | **{m13_miou:.2f}** | **{v4_all['ADE']:.3f}** | **{v4_all['FDE']:.3f}** | **{v4_all['Dynamic_Coll_%']:.2f}%** |"
            )
        for ab_key, info in exp6.items():
            r = info["adapted_5ep_overall"]
            d13 = info["adapted_5ep_conditions"]["Degraded_Modes_1_2_3"]
            md.append(
                f"| {info['label']} | {r['Cls_F1']:.2f} | {r['Det_mAP_Center']:.2f} | {r['Det_Contrast_AP']:.2f} | "
                f"{r['Seg_mIoU']:.2f} | {d13['Seg_mIoU']:.2f} | {r['ADE']:.3f} | {r['FDE']:.3f} | {r['Dynamic_Coll_%']:.2f}% |"
            )
        md.append("")

    if exp7:
        md.append("## 实验 7：后置可微三对角曲率平滑器（Curvature Smoother，Zero-Retrain 网格搜索 + 5-Epoch 联合微调）\n")
        md.append(f"> 求解器单帧延迟开销（BS=1）：`{exp7['solver_overhead_ms_bs1']:.4f} ms`\n")
        md.append("| 配置 (Config) | ADE@1s (m) | ADE@2s (m) | ADE@3s (m) | FDE@3s (m) | Dyn Coll (%) | Mean Accel | Mean Jerk | **P95 Jerk** | Mean Curv |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for k, r in exp7["zero_retrain_grid_search"].items():
            md.append(
                f"| Zero-Retrain `{k}` | {r['ADE@1.0s']:.3f} | {r['ADE@2.0s']:.3f} | {r['ADE@3.0s']:.3f} | {r['FDE@3.0s']:.3f} | "
                f"{r['Dynamic_Coll_%']:.2f}% | {r['Mean_Accel']:.2f} | {r['Mean_Jerk']:.2f} | **{r['P95_Jerk']:.2f}** | {r['Mean_Curvature']:.4f} |"
            )
        rf = exp7["finetuned_best_metrics"]
        md.append(
            f"| **Fine-Tuned 5ep (`lam={exp7['finetuned_lambda']:.2f}`)** | **{rf['ADE@1.0s']:.3f}** | **{rf['ADE@2.0s']:.3f}** | **{rf['ADE@3.0s']:.3f}** | **{rf['FDE@3.0s']:.3f}** | "
            f"**{rf['Dynamic_Coll_%']:.2f}%** | {rf['Mean_Accel']:.2f} | **{rf['Mean_Jerk']:.2f}** | **{rf['P95_Jerk']:.2f}** | {rf['Mean_Curvature']:.4f} |"
        )
        md.append("")

    out_md = SUPP_DIR / "p1_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[Aggregate] Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
