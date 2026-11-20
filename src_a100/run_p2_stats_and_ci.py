#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniPercep-DP Statistical Significance & Bootstrap 95% CI Evaluation (Exp 8 Supplement):
  Computes exact paired two-sided t-test, Wilcoxon signed-rank test, and 1,000-resample
  bootstrap 95% confidence intervals across all N=2,140 validation samples comparing
  `UniPercep-DP (Ours, Full v4)` and `UniPercep-DP (Full v4 + Curvature Smoother)` against:
    - `TransFuser` (Orig)
    - `TransFuser (+ Matched Night Aug.)`
    - `BEVFusion-Planner (2023)`
    - `VAD-Lite / QueryPlan-DETR (2023)`
  Outputs `outputs_a100/paper_supplement/p2_statistical_significance.json`.
"""

import sys
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SRC_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from models_full import UniPercepDPNet_Full
from models_v4_chronos import UniPercepDPNet_v4_Chronos
from models_modern_baselines import BEVFusionPlanner256, VADLiteQueryPlanner256
from p1_datasets import LowRAM128Dataset, LowRAM256Dataset
from run_p0_unified_eval import enforce_vram_cap, collect_unified_128, collect_unified_256, compute_kinematic_comfort
from run_p1_exp7_curvature_smoother import SmoothChronosWrapper


def bootstrap_ci(arr: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(arr.mean()), 4), round(float(lo), 4), round(float(hi), 4)


def paired_tests(ours: np.ndarray, base: np.ndarray) -> dict:
    diff = ours - base
    t_stat, p_t = stats.ttest_rel(ours, base)
    try:
        w_stat, p_w = stats.wilcoxon(diff, zero_method="wilcox")
    except Exception:
        w_stat, p_w = 0.0, 1.0
    cohen_d = float(diff.mean() / (diff.std(ddof=1) + 1e-9))
    return {
        "mean_diff": round(float(diff.mean()), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value_ttest": float(f"{p_t:.3e}"),
        "p_value_wilcoxon": float(f"{p_w:.3e}"),
        "cohens_d": round(cohen_d, 3),
        "significant_p001": bool(p_t < 0.001),
    }


def main():
    t0 = time.time()
    enforce_vram_cap(24.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = LOCAL_ROOT / "dataset_full" / "processed"
    val_jsonl = LOCAL_ROOT / "dataset_full" / "val_full.jsonl"
    ckpt_dir = LOCAL_ROOT / "checkpoints_a100"
    supp_dir = LOCAL_ROOT / "outputs_a100" / "paper_supplement"

    ds_256 = LowRAM256Dataset(val_jsonl, processed_dir, is_train=False, cam_hw=(192, 320))
    ds_128 = LowRAM128Dataset(val_jsonl, processed_dir, is_train=False)
    N = len(ds_256)

    gt_traj = np.zeros((N, 6, 2), dtype=np.float32)
    gt_seg_256_all = np.zeros((N, 256, 256), dtype=np.uint8)
    gt_hm_256_all = np.zeros((N, 1, 256, 256), dtype=np.float32)
    for kf_i in range(0, N, 4):
        it = ds_256[kf_i]
        seg_u8 = it["seg_256"].numpy().astype(np.uint8)
        hm_f32 = it["small_hm_256"].numpy().astype(np.float32)
        for j in range(kf_i, min(kf_i + 4, N)):
            gt_seg_256_all[j] = seg_u8
            gt_hm_256_all[j] = hm_f32
    for idx in range(N):
        gt_traj[idx] = np.array(ds_256.items[idx]["gt_traj"], dtype=np.float32)

    per_sample = {}

    def _extract_arrays(name: str, p: dict):
        err_6 = np.linalg.norm(p["pred_traj"].astype(np.float32) - gt_traj, axis=-1) * 0.5
        ade_s = err_6.mean(axis=-1)
        fde_s = err_6[:, -1]
        # Per-sample mIoU across present classes
        inter = p["seg_inter"]
        union = p["seg_union"]
        miou_s = np.mean(inter / np.maximum(union, 1.0), axis=-1) * 100.0
        per_sample[name] = {
            "ade": ade_s,
            "fde": fde_s,
            "dyn_coll": p["dyn_coll"],
            "miou": miou_s,
        }

    # 128x128 baselines
    for name, ck_name in [
        ("TransFuser (Orig)", "Ablation_3_With_MultiSensor_best.pt"),
        ("TransFuser (+ Matched Night Aug.)", "Ablation_3_TransFuser_NightAug_best.pt"),
    ]:
        p_ck = ckpt_dir / ck_name
        if not p_ck.exists():
            continue
        m = UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=False).to(device)
        m.load_state_dict(torch.load(p_ck, map_location=device, weights_only=False)["model_state_dict"])
        p = collect_unified_128(m, ds_128, gt_seg_256_all, gt_hm_256_all, device, batch_size=32)
        _extract_arrays(name, p)
        del m, p
        torch.cuda.empty_cache()

    # 256x256 baselines & Ours
    specs_256 = [
        ("BEVFusion-Planner (2023)", "Baseline_BEVFusion_Planner_best.pt", lambda: BEVFusionPlanner256(pretrained_backbone=False, dim=256)),
        ("VAD-Lite / QueryPlan-DETR (2023)", "Baseline_VAD_Lite_QueryPlan_best.pt", lambda: VADLiteQueryPlanner256(pretrained_backbone=False, dim=256)),
        ("UniPercep-DP (Ours, Full v4)", "Full_v4_Chronos_best.pt", lambda: UniPercepDPNet_v4_Chronos()),
    ]
    for name, ck_name, builder in specs_256:
        p_ck = ckpt_dir / ck_name
        if not p_ck.exists():
            continue
        ck = torch.load(p_ck, map_location="cpu", weights_only=False)
        m = builder()
        m.load_state_dict(ck.get("ema", ck.get("model", ck)), strict=True)
        m = m.to(device).eval()
        p = collect_unified_256(m, ds_256, device, batch_size=16, pass_temporal=True)
        _extract_arrays(name, p)
        del m, ck, p
        torch.cuda.empty_cache()

    # Smooth v4
    p_sm = ckpt_dir / "Full_v4_Chronos_Smooth_best.pt"
    if p_sm.exists():
        ck = torch.load(p_sm, map_location="cpu", weights_only=False)
        m_sm = SmoothChronosWrapper(UniPercepDPNet_v4_Chronos(), lam=float(ck.get("lam", 0.15)), frozen_trunk=False)
        m_sm.load_state_dict(ck.get("ema", ck.get("model", ck)), strict=True)
        m_sm = m_sm.to(device).eval()
        p = collect_unified_256(m_sm, ds_256, device, batch_size=16, pass_temporal=True)
        _extract_arrays("UniPercep-DP (Ours + Curvature Smoother)", p)
        del m_sm, ck, p
        torch.cuda.empty_cache()

    # Bootstrap 95% CIs & Paired Tests against Full v4
    ci_summary = {}
    for m_name, d in per_sample.items():
        ci_summary[m_name] = {
            "ADE_mean_CI95": bootstrap_ci(d["ade"]),
            "FDE_mean_CI95": bootstrap_ci(d["fde"]),
            "DynColl_mean_CI95": bootstrap_ci(d["dyn_coll"]),
            "mIoU_mean_CI95": bootstrap_ci(d["miou"]),
        }

    paired_vs_v4 = {}
    if "UniPercep-DP (Ours, Full v4)" in per_sample:
        ours = per_sample["UniPercep-DP (Ours, Full v4)"]
        for m_name, base in per_sample.items():
            if m_name == "UniPercep-DP (Ours, Full v4)":
                continue
            paired_vs_v4[m_name] = {
                "ADE": paired_tests(ours["ade"], base["ade"]),
                "FDE": paired_tests(ours["fde"], base["fde"]),
                "DynColl": paired_tests(ours["dyn_coll"], base["dyn_coll"]),
                "mIoU": paired_tests(ours["miou"], base["miou"]),
            }

    out_payload = {
        "N_samples": N,
        "bootstrap_resamples": 1000,
        "Bootstrap_95_CI": ci_summary,
        "Paired_Tests_Full_v4_vs_Baselines": paired_vs_v4,
    }
    out_p = supp_dir / "p2_statistical_significance.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"[Stats DONE] Saved {out_p} in {time.time() - t0:.1f}s!")
    for k, v in paired_vs_v4.items():
        print(f"  Ours vs {k}: ADE diff={v['ADE']['mean_diff']}m (p={v['ADE']['p_value_ttest']}), DynColl diff={v['DynColl']['mean_diff']}% (p={v['DynColl']['p_value_ttest']})")


if __name__ == "__main__":
    main()
