#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniPercep-DP P0 Supplement Experiments (Experiments 1, 2, 3):
  - Exp 1: Unified cross-resolution 256x256 physical-metric evaluator across ALL models
           (Identical nuScenes Center-Distance PR mAP@{0.5,1.0,2.0,4.0}m + Unified Peak-Contrast AP
            + Identical 256x256 Dynamic & Total-Obstacle Collision Rates + Unified Seg mIoU + Horizons & Comfort)
  - Exp 2: Full N=2,140 evaluation of Config B (`Full_UniPercep_DP_Stage3_RAG_LoRA_done.pt` & `Full_UniPercep_DP_best.pt`)
           and Config C (`Full_Pro256_6Cam_best.pt` & `Full_v3_Ultra_best.pt`) to fill Table 6(a) completely.
  - Exp 3: Export raw unmixed `dp_weights` & 15-rule `rag_weights` (`raw_rule_lora_weights_2140.npz`)
           and hook true Eq. (19) `barrier_field` & Sobel `-grad B_CBF` (`raw_cbf_fields_3cases.npz`),
           plus regenerate publication-grade Fig. 7 and Fig. 8 without any proxy blending.

Strict VRAM & RAM safety:
  - Hard CUDA per-process memory cap <= 24 GB (for 25 GB budget).
  - Compact RAM usage: clears 128x128 cache immediately after 128x128 models finish.
"""

import os
import sys
import gc
import json
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.interpolate import make_interp_spline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Enable TF32 & benchmark
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

SRC_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import train_and_ablation as t128
import train_v4_chronos as t256

# Patch module-level paths so datasets load strictly from local workspace
t128.LOCAL_ROOT = LOCAL_ROOT
t128.PROCESSED_DIR = LOCAL_ROOT / "dataset_full" / "processed"
t256.LOCAL_ROOT = LOCAL_ROOT
t256.DATASET_DIR = LOCAL_ROOT / "dataset_full"

from models_full import UniPercepDPNet_Full
from models_pro256 import UniPercepDPNet_Pro256
from models_v3_ultra import UniPercepDPNet_v3_Ultra
from models_v3_5_lora import UniPercepDPNet_v3_5_LoRA
from models_v4_chronos import UniPercepDPNet_v4_Chronos
from train_and_ablation import UniPercepFullDataset
from train_v4_chronos import UniPercepChronos256Dataset

SCENARIO_NAMES_FULL = [
    "0: Night (夜间低照度)",
    "1: Rain/Fog (雨雾散射)",
    "2: Intersection (无保护路口)",
    "3: Construction (施工锥桶)",
    "4: Congestion (拥堵跟车)",
    "5: PedCross (行人横穿)",
    "6: FarSmallObj (远距小目标)",
    "7: FastApproach (高速逼近)",
    "8: Occlusion (视野遮挡)",
    "9: Curve (急弯道)",
]

SCENARIO_NAMES_SHORT = [
    "Night",
    "Rain/Fog",
    "Intersection",
    "Construction",
    "Congestion",
    "PedCross",
    "FarSmallObj",
    "FastApproach",
    "Occlusion",
    "Curve",
]


def enforce_vram_cap(max_gb: float = 24.0):
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        frac = min(0.95, max_gb / max(total_gb, 1.0))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"[VRAM Cap] Total GPU VRAM={total_gb:.1f}GB -> capped process at {frac*total_gb:.1f}GB ({frac:.3f})")


def detect_backbone_name(sd: dict) -> str:
    for k in sd.keys():
        if "backbone.stem.0.weight" in k or "core.backbone.stem.0.weight" in k:
            return "convnext_small.fb_in22k"
    return "resnet50d"


# ==============================================================================
# 1. Unified Kinematic & Horizon Metrics
# ==============================================================================

def upsample_traj_to_12steps_meters(traj_grid128: np.ndarray) -> np.ndarray:
    ego = np.array([[[64.0, 8.0]]], dtype=np.float32)
    traj_m6 = (traj_grid128.astype(np.float32) - ego) * 0.5
    origin = np.zeros((traj_m6.shape[0], 1, 2), dtype=np.float32)
    traj_m7 = np.concatenate([origin, traj_m6], axis=1)
    t_knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    t_12 = np.linspace(0.25, 3.0, 12, dtype=np.float32)
    spl = make_interp_spline(t_knots, traj_m7, k=3, axis=1)
    return spl(t_12).astype(np.float32)


def compute_kinematic_comfort(traj_grid128: np.ndarray, dt: float = 0.25) -> dict:
    traj_12 = upsample_traj_to_12steps_meters(traj_grid128)
    vel = np.diff(traj_12, axis=1) / dt
    acc = np.diff(vel, axis=1) / dt
    jerk = np.diff(acc, axis=1) / dt

    acc_mag = np.linalg.norm(acc, axis=-1)
    vel_mid = 0.5 * (vel[:, :-1, :] + vel[:, 1:, :])
    spd = np.linalg.norm(vel_mid, axis=-1)
    unit_tan = vel_mid / np.maximum(spd[..., None], 0.5)
    lat_acc = np.abs(acc[..., 0] * unit_tan[..., 1] - acc[..., 1] * unit_tan[..., 0])

    jerk_mag = np.linalg.norm(jerk, axis=-1)
    sample_mean_jerk = jerk_mag.mean(axis=1)
    curv = lat_acc / np.maximum(spd ** 2, 1.2)

    return {
        "Mean_Accel": round(float(acc_mag.mean()), 3),
        "Mean_Lat_Accel": round(float(lat_acc.mean()), 3),
        "Mean_Jerk": round(float(sample_mean_jerk.mean()), 3),
        "P95_Jerk": round(float(np.percentile(sample_mean_jerk, 95)), 3),
        "Mean_Curvature": round(float(curv.mean()), 4),
    }


def compute_horizon_errors(pred_grid128: np.ndarray, gt_grid128: np.ndarray) -> dict:
    err_6 = np.linalg.norm(pred_grid128.astype(np.float32) - gt_grid128.astype(np.float32), axis=-1) * 0.5
    return {
        "ADE@1.0s": round(float(err_6[:, :2].mean()), 3),
        "ADE@2.0s": round(float(err_6[:, :4].mean()), 3),
        "ADE@3.0s": round(float(err_6[:, :6].mean()), 3),
        "FDE@1.0s": round(float(err_6[:, 1].mean()), 3),
        "FDE@2.0s": round(float(err_6[:, 3].mean()), 3),
        "FDE@3.0s": round(float(err_6[:, 5].mean()), 3),
    }


# ==============================================================================
# 2. Unified Physical-Metric Small-Object Detection & Collision Evaluators
# ==============================================================================

def extract_gt_small_centers_meters(items: list) -> list:
    """
    Extract exact GT small-object centers in ego physical meters for every sample:
      x_m = (cx_128 - 64.0) * 0.5, y_m = (cy_128 - 8.0) * 0.5
    Returns list of length N of (M_i, 2) float32 arrays.
    """
    gt_centers = []
    for rec in items:
        pts = []
        for obj in rec.get("det_boxes", []):
            cx, cy, _, _, _, _, _, _, is_small = obj
            if is_small == 1 and 0 <= cx < 128 and 0 <= cy < 128:
                pts.append([(float(cx) - 64.0) * 0.5, (float(cy) - 8.0) * 0.5])
        if pts:
            gt_centers.append(np.array(pts, dtype=np.float32))
        else:
            gt_centers.append(np.zeros((0, 2), dtype=np.float32))
    return gt_centers


def extract_hm256_peaks_meters(hm_256: torch.Tensor, top_k: int = 40, score_thr: float = 0.05, nms_dist_m: float = 0.75) -> list:
    """
    Plateau-safe local-maxima peak extraction on a unified 256x256 heatmap (0.25m/cell):
      1. Tiny multi-scale average-pool tie-breaker centers flat saturated plateaus (e.g. clamped at 1.0)
         without modifying raw peak confidence scores.
      2. 5x5 local-maximum NMS followed by greedy 0.75m (3-pixel) Euclidean distance NMS
         so each distinct object produces a single centered detection across ALL models identically.
    Maps pixel (px, py) to ego physical meters: x_m = (px - 128.0) * 0.25, y_m = (py - 16.0) * 0.25.
    Returns list of length B of (K_b, 3) float32 arrays: [score, x_m, y_m].
    """
    hm = hm_256.float()
    hm_tie = hm + 1e-3 * F.avg_pool2d(hm, kernel_size=5, stride=1, padding=2) + 1e-4 * F.avg_pool2d(hm, kernel_size=9, stride=1, padding=4)
    nms = F.max_pool2d(hm_tie, kernel_size=5, stride=1, padding=2)
    keep = (hm_tie == nms) & (hm > score_thr)
    B = hm.size(0)
    dist2_thr = nms_dist_m * nms_dist_m
    out_list = []
    for b in range(B):
        coords = torch.nonzero(keep[b, 0], as_tuple=False)  # (K, 2) -> (py, px)
        if coords.size(0) == 0:
            out_list.append(np.zeros((0, 3), dtype=np.float32))
            continue
        sc_tie = hm_tie[b, 0][keep[b, 0]]
        sc_raw = hm[b, 0][keep[b, 0]]
        order = torch.argsort(sc_tie, descending=True)
        # Pre-cap candidates for speed before greedy Euclidean NMS
        if order.numel() > top_k * 4:
            order = order[: top_k * 4]
        coords_np = coords[order].cpu().numpy()
        sc_np = sc_raw[order].cpu().numpy()
        kept = []
        for idx_k in range(len(coords_np)):
            py, px = float(coords_np[idx_k, 0]), float(coords_np[idx_k, 1])
            xm = (px - 128.0) * 0.25
            ym = (py - 16.0) * 0.25
            if all((xm - kx) ** 2 + (ym - ky) ** 2 > dist2_thr for _, kx, ky in kept):
                kept.append((float(sc_np[idx_k]), xm, ym))
                if len(kept) >= top_k:
                    break
        if kept:
            out_list.append(np.array(kept, dtype=np.float32))
        else:
            out_list.append(np.zeros((0, 3), dtype=np.float32))
    return out_list


def compute_center_distance_pr_ap(
    det_peaks_list: list,
    gt_centers_list: list,
    mask: np.ndarray,
    thresholds=(0.5, 1.0, 2.0, 4.0),
) -> dict:
    """
    Standard nuScenes-style Center-Distance Precision-Recall AP across distance thresholds
    {0.5m, 1.0m, 2.0m, 4.0m} on a boolean sample subset `mask`.
    """
    indices = np.where(mask)[0]
    total_gt = sum(len(gt_centers_list[i]) for i in indices)
    if total_gt == 0:
        res = {f"AP@{d}m": 0.0 for d in thresholds}
        res["mAP_Center"] = 0.0
        return res

    # Pool all predictions in subset: (score, local_sample_pos, x_m, y_m)
    all_scores = []
    all_spos = []
    all_xy = []
    gt_sub = [gt_centers_list[i] for i in indices]
    for spos, i in enumerate(indices):
        pk = det_peaks_list[i]
        if len(pk) > 0:
            all_scores.append(pk[:, 0])
            all_spos.append(np.full(len(pk), spos, dtype=np.int32))
            all_xy.append(pk[:, 1:3])

    if not all_scores:
        res = {f"AP@{d}m": 0.0 for d in thresholds}
        res["mAP_Center"] = 0.0
        return res

    scores = np.concatenate(all_scores, axis=0)
    spos_arr = np.concatenate(all_spos, axis=0)
    xy_arr = np.concatenate(all_xy, axis=0)

    order = np.argsort(-scores)
    spos_arr = spos_arr[order]
    xy_arr = xy_arr[order]
    M_pred = len(order)

    # Precompute closest GT index & distance per prediction? No, greedy matching depends on threshold!
    res = {}
    ap_vals = []
    for d_thr in thresholds:
        matched = [np.zeros(len(g), dtype=bool) for g in gt_sub]
        tp = np.zeros(M_pred, dtype=np.float64)
        fp = np.zeros(M_pred, dtype=np.float64)
        for p_idx in range(M_pred):
            sp = spos_arr[p_idx]
            gts = gt_sub[sp]
            if len(gts) == 0:
                fp[p_idx] = 1.0
                continue
            dists = np.linalg.norm(gts - xy_arr[p_idx], axis=-1)
            # Mask already matched GTs
            dists_unmatched = np.where(matched[sp], np.inf, dists)
            best_j = int(np.argmin(dists_unmatched))
            if dists_unmatched[best_j] <= d_thr:
                tp[p_idx] = 1.0
                matched[sp][best_j] = True
            else:
                fp[p_idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / float(total_gt)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for k in range(mpre.size - 2, -1, -1):
            if mpre[k] < mpre[k + 1]:
                mpre[k] = mpre[k + 1]
        chg = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[chg + 1] - mrec[chg]) * mpre[chg + 1])) * 100.0
        res[f"AP@{d_thr}m"] = round(ap, 2)
        ap_vals.append(ap)

    res["mAP_Center"] = round(float(np.mean(ap_vals)), 2)
    return res


def compute_unified_contrast_ap_batch(pred_hm_256: torch.Tensor, gt_hm_256: torch.Tensor) -> np.ndarray:
    """Identical 256x256 peak-to-background contrast AP formula for all models."""
    B = pred_hm_256.size(0)
    ap_b = np.full(B, np.nan, dtype=np.float64)
    p = pred_hm_256.float()
    g = gt_hm_256.float()
    for b in range(B):
        g_pos = g[b, 0] > 0.35
        if g_pos.sum() == 0:
            continue
        hit_score = (p[b, 0] * g_pos.float()).max().item()
        mean_bg = (p[b, 0] * (~g_pos).float()).mean().item()
        ap_est = np.clip((hit_score - mean_bg) / max(hit_score, 1e-4) * min(1.0, hit_score * 1.45), 0.0, 1.0)
        ap_b[b] = ap_est * 100.0
    return ap_b


def compute_unified_collisions_256(pred_traj_128: np.ndarray, gt_seg_256_np: np.ndarray) -> tuple:
    """
    Compute 3 unified physical-coordinate collision rates on 256x256 GT semantic map (0.25m/cell):
      1. dyn_coll: Exact waypoint hits Class 2 (Dynamic obstacle)
      2. tot_coll: Exact waypoint hits Class 1 or 2 (Total obstacle = Static + Dynamic)
      3. env_dyn_coll: 3x3 (0.75m x 0.75m ego envelope) hits Class 2
    """
    B = pred_traj_128.shape[0]
    dyn_coll = np.zeros(B, dtype=np.float64)
    tot_coll = np.zeros(B, dtype=np.float64)
    env_dyn_coll = np.zeros(B, dtype=np.float64)
    for b in range(B):
        seg = gt_seg_256_np[b]
        for s in range(6):
            px = int(np.clip(round(float(pred_traj_128[b, s, 0]) * 2.0), 0, 255))
            py = int(np.clip(round(float(pred_traj_128[b, s, 1]) * 2.0), 0, 255))
            v = seg[py, px]
            if v == 2:
                dyn_coll[b] = 100.0
                tot_coll[b] = 100.0
            elif v == 1:
                tot_coll[b] = 100.0
            patch = seg[max(0, py - 1) : min(256, py + 2), max(0, px - 1) : min(256, px + 2)]
            if np.any(patch == 2):
                env_dyn_coll[b] = 100.0
    return dyn_coll, tot_coll, env_dyn_coll


def evaluate_unified_256_metrics(
    preds: dict,
    gt_ml: np.ndarray,
    gt_traj: np.ndarray,
    gt_centers_list: list,
    mask: np.ndarray,
    class_k: int = None,
    include_horizons_comfort: bool = False,
    skip_pr_ap: bool = False,
) -> dict:
    """Unified evaluation function applied identically to ALL models on any subset mask."""
    idx = np.where(mask)[0]
    n_sub = int(len(idx))

    # 1. Multi-label classification F1 & Acc
    p_bin = (preds["ml_probs"][idx] > 0.5).astype(np.float64)
    g_bin = gt_ml[idx].astype(np.float64)
    if class_k is not None:
        pk = p_bin[:, class_k]
        gk = g_bin[:, class_k]
        tp = float((pk * gk).sum())
        fp = float((pk * (1.0 - gk)).sum())
        fn = float(((1.0 - pk) * gk).sum())
        cls_f1 = (2.0 * tp / max(1e-6, 2.0 * tp + fp + fn)) * 100.0
        cls_acc = float((pk == gk).mean()) * 100.0
    else:
        tp = float((p_bin * g_bin).sum())
        fp = float((p_bin * (1.0 - g_bin)).sum())
        fn = float(((1.0 - p_bin) * g_bin).sum())
        cls_f1 = (2.0 * tp / max(1e-6, 2.0 * tp + fp + fn)) * 100.0
        cls_acc = float((p_bin == g_bin).mean()) * 100.0

    # 2. Unified 256x256 Seg mIoU
    inter = preds["seg_inter"][idx].sum(axis=0)
    union = preds["seg_union"][idx].sum(axis=0)
    ious = [float(inter[c] / max(1.0, union[c])) * 100.0 for c in range(3)]
    seg_miou = float(np.mean(ious))

    # 3. Unified Small-Object Detection AP (Center-Distance PR mAP + Unified Contrast AP)
    if skip_pr_ap:
        pr_ap = {"mAP_Center": 0.0, "AP@0.5m": 0.0, "AP@1.0m": 0.0, "AP@2.0m": 0.0, "AP@4.0m": 0.0}
    else:
        pr_ap = compute_center_distance_pr_ap(preds["det_peaks"], gt_centers_list, mask)
    c_ap_vals = preds["contrast_ap"][idx]
    valid_cap = c_ap_vals[~np.isnan(c_ap_vals)]
    contrast_ap = float(valid_cap.mean()) if len(valid_cap) > 0 else 0.0

    # 4. Trajectory ADE / FDE & Dual Collision Rates
    pt = preds["pred_traj"][idx]
    gt = gt_traj[idx]
    err_6 = np.linalg.norm(pt.astype(np.float32) - gt.astype(np.float32), axis=-1) * 0.5
    ade = float(err_6.mean())
    fde = float(err_6[:, -1].mean())
    dyn_coll = float(preds["dyn_coll"][idx].mean())
    tot_coll = float(preds["tot_coll"][idx].mean())
    env_dyn_coll = float(preds["env_dyn_coll"][idx].mean())

    res = {
        "N": n_sub,
        "Cls_F1": round(cls_f1, 2),
        "Cls_Acc": round(cls_acc, 2),
        "Det_mAP_Center": pr_ap["mAP_Center"],
        "Det_AP@0.5m": pr_ap["AP@0.5m"],
        "Det_AP@1.0m": pr_ap["AP@1.0m"],
        "Det_AP@2.0m": pr_ap["AP@2.0m"],
        "Det_AP@4.0m": pr_ap["AP@4.0m"],
        "Det_Contrast_AP": round(contrast_ap, 2),
        "Seg_mIoU": round(seg_miou, 2),
        "IoU_Road": round(ious[0], 2),
        "IoU_Static": round(ious[1], 2),
        "IoU_Dynamic": round(ious[2], 2),
        "ADE": round(ade, 3),
        "FDE": round(fde, 3),
        "Dynamic_Coll_%": round(dyn_coll, 2),
        "Total_Obs_Coll_%": round(tot_coll, 2),
        "Env075m_Dyn_Coll_%": round(env_dyn_coll, 2),
    }
    if include_horizons_comfort:
        res.update(compute_horizon_errors(pt, gt))
        res.update(compute_kinematic_comfort(pt))
    return res


# ==============================================================================
# 3. Unified Forward Collectors for 128x128 & 256x256 Models
# ==============================================================================

@torch.no_grad()
def collect_unified_128(
    model: nn.Module,
    ds128: UniPercepFullDataset,
    gt_seg_256_all: np.ndarray,
    gt_hm_256_all: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    clean_mode: bool = False,
) -> dict:
    model.eval()
    N = len(ds128)
    if clean_mode:
        unique_indices = list(range(0, N, 4))
        loader = DataLoader(Subset(ds128, unique_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        unique_indices = list(range(N))
        loader = DataLoader(ds128, batch_size=batch_size, shuffle=False, num_workers=0)

    ml_logits_list, seg_inter_list, seg_union_list = [], [], []
    det_peaks_list, contrast_ap_list = [], []
    pred_traj_list, dyn_coll_list, tot_coll_list, env_coll_list = [], [], [], []

    offset = 0
    for batch in loader:
        B = batch["geo_14d"].size(0)
        b_idx = unique_indices[offset : offset + B]
        offset += B

        img_in = batch["clean_img" if clean_mode else "corrupted_img"].to(device, non_blocking=True)
        bev_lidar = batch["bev_lidar"].to(device, non_blocking=True)
        geo_14d = batch["geo_14d"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(img_in, bev_lidar, geo_14d)

        ml_logits_list.append(out["multi_label_logits"].float().cpu().numpy())

        # Nearest-neighbor upsample of 128x128 seg_logits maps each 0.5m cell to its exact 2x2 physical block in gt_seg_256
        seg_logits_256 = F.interpolate(out["seg_logits"].float(), size=(256, 256), mode="nearest")
        pred_seg_256 = torch.argmax(seg_logits_256, dim=1).cpu().numpy()
        gt_seg_b = gt_seg_256_all[b_idx]  # (B, 256, 256)

        inter_b = np.zeros((B, 3), dtype=np.float64)
        union_b = np.zeros((B, 3), dtype=np.float64)
        for c in range(3):
            inter_b[:, c] = ((pred_seg_256 == c) & (gt_seg_b == c)).sum(axis=(1, 2))
            union_b[:, c] = ((pred_seg_256 == c) | (gt_seg_b == c)).sum(axis=(1, 2))
        seg_inter_list.append(inter_b)
        seg_union_list.append(union_b)

        hm_256 = F.interpolate(out["small_hm_pred"].float(), size=(256, 256), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        gt_hm_b = torch.from_numpy(gt_hm_256_all[b_idx]).to(device)
        det_peaks_list.extend(extract_hm256_peaks_meters(hm_256))
        contrast_ap_list.append(compute_unified_contrast_ap_batch(hm_256, gt_hm_b))

        pred_tr = out["pred_traj"].float().cpu().numpy()
        pred_traj_list.append(pred_tr)
        dc, tc, ec = compute_unified_collisions_256(pred_tr, gt_seg_b)
        dyn_coll_list.append(dc)
        tot_coll_list.append(tc)
        env_coll_list.append(ec)

    ml_logits = np.concatenate(ml_logits_list, axis=0)
    seg_inter = np.concatenate(seg_inter_list, axis=0)
    seg_union = np.concatenate(seg_union_list, axis=0)
    contrast_ap = np.concatenate(contrast_ap_list, axis=0)
    pred_traj = np.concatenate(pred_traj_list, axis=0)
    dyn_coll = np.concatenate(dyn_coll_list, axis=0)
    tot_coll = np.concatenate(tot_coll_list, axis=0)
    env_coll = np.concatenate(env_coll_list, axis=0)

    if clean_mode:
        ml_logits = np.repeat(ml_logits, 4, axis=0)[:N]
        seg_inter = np.repeat(seg_inter, 4, axis=0)[:N]
        seg_union = np.repeat(seg_union, 4, axis=0)[:N]
        contrast_ap = np.repeat(contrast_ap, 4, axis=0)[:N]
        pred_traj = np.repeat(pred_traj, 4, axis=0)[:N]
        dyn_coll = np.repeat(dyn_coll, 4, axis=0)[:N]
        tot_coll = np.repeat(tot_coll, 4, axis=0)[:N]
        env_coll = np.repeat(env_coll, 4, axis=0)[:N]
        det_peaks_rep = []
        for pk in det_peaks_list:
            for _ in range(4):
                det_peaks_rep.append(pk)
        det_peaks_list = det_peaks_rep[:N]

    return {
        "ml_probs": 1.0 / (1.0 + np.exp(-np.clip(ml_logits, -30.0, 30.0))),
        "seg_inter": seg_inter,
        "seg_union": seg_union,
        "det_peaks": det_peaks_list,
        "contrast_ap": contrast_ap,
        "pred_traj": pred_traj,
        "dyn_coll": dyn_coll,
        "tot_coll": tot_coll,
        "env_dyn_coll": env_coll,
    }


@torch.no_grad()
def collect_unified_256(
    model: nn.Module,
    ds256: UniPercepChronos256Dataset,
    device: torch.device,
    batch_size: int = 16,
    clean_mode: bool = False,
    pass_temporal: bool = True,
    export_raw_lora: bool = False,
) -> dict:
    model.eval()
    N = len(ds256)
    if clean_mode:
        unique_indices = list(range(0, N, 4))
        loader = DataLoader(Subset(ds256, unique_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        loader = DataLoader(ds256, batch_size=batch_size, shuffle=False, num_workers=0)

    ml_logits_list, seg_inter_list, seg_union_list = [], [], []
    det_peaks_list, contrast_ap_list = [], []
    pred_traj_list, dyn_coll_list, tot_coll_list, env_coll_list = [], [], [], []
    refined_traj_list = []

    raw_dp_weights_list, raw_rag_attn_list, raw_rag_prior_list, raw_rule_idx_list = [], [], [], []

    for batch in loader:
        img_in = batch["clean_img" if clean_mode else "corrupted_imgs"].to(device, non_blocking=True)
        bev_lidar_256 = batch["bev_lidar_256"].to(device, non_blocking=True)
        bev_prev_256 = batch["bev_prev_256"].to(device, non_blocking=True)
        geo_14d = batch["geo_14d"].to(device, non_blocking=True)
        temporal_8d = batch["temporal_8d"].to(device, non_blocking=True)
        gt_seg_256 = batch["seg_256"]
        gt_hm_256 = batch["small_hm_256"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if pass_temporal:
                out = model(img_in, bev_lidar_256, geo_14d, temporal_8d=temporal_8d, bev_prev_256=bev_prev_256)
            else:
                out = model(img_in, bev_lidar_256, geo_14d)

        ml_logits_t = out["multi_label_logits"].float()
        ml_logits_list.append(ml_logits_t.cpu().numpy())

        if export_raw_lora:
            ctrl = model.core.lora_controller
            probs = torch.sigmoid(ml_logits_t)
            dists = torch.cdist(probs, ctrl.rule_triggers.float(), p=2.0)
            attn = F.softmax(-dists * 2.5, dim=-1)
            rag_prior = torch.matmul(attn, ctrl.rule_weights.float())
            raw_dp_weights_list.append(out["dp_weights"].float().cpu().numpy())
            raw_rag_attn_list.append(attn.cpu().numpy())
            raw_rag_prior_list.append(rag_prior.cpu().numpy())
            raw_rule_idx_list.append(out["rag_rule_idx"].cpu().numpy())
            if "refined_traj" in out:
                refined_traj_list.append(out["refined_traj"].float().cpu().numpy())

        pred_seg = torch.argmax(out["seg_logits_256"].float(), dim=1).cpu().numpy()
        gt_seg_np = gt_seg_256.numpy()
        B = pred_seg.shape[0]
        inter_b = np.zeros((B, 3), dtype=np.float64)
        union_b = np.zeros((B, 3), dtype=np.float64)
        for c in range(3):
            inter_b[:, c] = ((pred_seg == c) & (gt_seg_np == c)).sum(axis=(1, 2))
            union_b[:, c] = ((pred_seg == c) | (gt_seg_np == c)).sum(axis=(1, 2))
        seg_inter_list.append(inter_b)
        seg_union_list.append(union_b)

        pred_hm = out["small_hm_256"].float().clamp(0.0, 1.0)
        det_peaks_list.extend(extract_hm256_peaks_meters(pred_hm))
        contrast_ap_list.append(compute_unified_contrast_ap_batch(pred_hm, gt_hm_256))

        pred_tr = out["pred_traj"].float().cpu().numpy()
        pred_traj_list.append(pred_tr)
        dc, tc, ec = compute_unified_collisions_256(pred_tr, gt_seg_np)
        dyn_coll_list.append(dc)
        tot_coll_list.append(tc)
        env_coll_list.append(ec)

    ml_logits = np.concatenate(ml_logits_list, axis=0)
    seg_inter = np.concatenate(seg_inter_list, axis=0)
    seg_union = np.concatenate(seg_union_list, axis=0)
    contrast_ap = np.concatenate(contrast_ap_list, axis=0)
    pred_traj = np.concatenate(pred_traj_list, axis=0)
    dyn_coll = np.concatenate(dyn_coll_list, axis=0)
    tot_coll = np.concatenate(tot_coll_list, axis=0)
    env_coll = np.concatenate(env_coll_list, axis=0)

    if clean_mode:
        ml_logits = np.repeat(ml_logits, 4, axis=0)[:N]
        seg_inter = np.repeat(seg_inter, 4, axis=0)[:N]
        seg_union = np.repeat(seg_union, 4, axis=0)[:N]
        contrast_ap = np.repeat(contrast_ap, 4, axis=0)[:N]
        pred_traj = np.repeat(pred_traj, 4, axis=0)[:N]
        dyn_coll = np.repeat(dyn_coll, 4, axis=0)[:N]
        tot_coll = np.repeat(tot_coll, 4, axis=0)[:N]
        env_coll = np.repeat(env_coll, 4, axis=0)[:N]
        det_peaks_rep = []
        for pk in det_peaks_list:
            for _ in range(4):
                det_peaks_rep.append(pk)
        det_peaks_list = det_peaks_rep[:N]

    out_dict = {
        "ml_probs": 1.0 / (1.0 + np.exp(-np.clip(ml_logits, -30.0, 30.0))),
        "seg_inter": seg_inter,
        "seg_union": seg_union,
        "det_peaks": det_peaks_list,
        "contrast_ap": contrast_ap,
        "pred_traj": pred_traj,
        "dyn_coll": dyn_coll,
        "tot_coll": tot_coll,
        "env_dyn_coll": env_coll,
    }
    if export_raw_lora:
        out_dict["raw_dp_weights"] = np.concatenate(raw_dp_weights_list, axis=0)
        out_dict["raw_rag_attn"] = np.concatenate(raw_rag_attn_list, axis=0)
        out_dict["raw_rag_prior"] = np.concatenate(raw_rag_prior_list, axis=0)
        out_dict["raw_rule_idx"] = np.concatenate(raw_rule_idx_list, axis=0)
        if refined_traj_list:
            out_dict["refined_traj"] = np.concatenate(refined_traj_list, axis=0)
    return out_dict


# ==============================================================================
# 4. Experiment 3: Export True Raw Weights & Real Eq. (19) CBF Fields + Figures
# ==============================================================================

@torch.no_grad()
def export_exp3_and_figures(
    model_v4: nn.Module,
    ds256: UniPercepChronos256Dataset,
    preds_v4: dict,
    gt_ml: np.ndarray,
    gt_traj: np.ndarray,
    deg_modes: np.ndarray,
    gt_seg_256_all: np.ndarray,
    device: torch.device,
    supp_dir: Path,
):
    fig_dir = supp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw unmixed Rule-LoRA weights
    dp_w = preds_v4["raw_dp_weights"]  # (2140, 3)
    dp_w_norm = dp_w / np.maximum(dp_w.sum(axis=-1, keepdims=True), 1e-6)
    rag_attn = preds_v4["raw_rag_attn"]  # (2140, 15)
    rag_prior = preds_v4["raw_rag_prior"]  # (2140, 3)
    lora_delta = dp_w - rag_prior
    rule_idx = preds_v4["raw_rule_idx"]  # (2140,)

    npz_weights_path = supp_dir / "raw_rule_lora_weights_2140.npz"
    np.savez_compressed(
        npz_weights_path,
        dp_weights=dp_w.astype(np.float32),
        dp_weights_norm=dp_w_norm.astype(np.float32),
        rag_weights=rag_attn.astype(np.float32),
        rag_prior=rag_prior.astype(np.float32),
        lora_delta=lora_delta.astype(np.float32),
        rag_rule_idx=rule_idx.astype(np.int64),
        gt_ml=gt_ml.astype(np.float32),
        pred_ml_probs=preds_v4["ml_probs"].astype(np.float32),
        deg_modes=deg_modes.astype(np.int64),
    )
    print(f"[Exp 3] Saved raw unmixed weights to {npz_weights_path}")
    print(f"        dp_weights shape={dp_w.shape}, mean={dp_w.mean(0).round(4)}, std={dp_w.std(0).round(4)}, min={dp_w.min(0).round(4)}, max={dp_w.max(0).round(4)}")
    print(f"        dp_weights_norm mean={dp_w_norm.mean(0).round(4)}, std={dp_w_norm.std(0).round(4)}")

    # 2. Pick 3 representative CBF keyframes (same selection criterion as Fig. 5 / Fig. 8)
    ref_all = preds_v4["refined_traj"]
    safe_all = preds_v4["pred_traj"]
    col_ref, _, _ = compute_unified_collisions_256(ref_all, gt_seg_256_all)
    col_safe = preds_v4["dyn_coll"]

    cbf_shift_m = np.linalg.norm(safe_all - ref_all, axis=-1).max(axis=-1) * 0.5
    ade_safe = np.linalg.norm(safe_all - gt_traj, axis=-1).mean(axis=-1) * 0.5
    score_cbf = cbf_shift_m + 2.0 * ((col_ref > 50) & (col_safe < 50)).astype(np.float32) - 0.5 * ade_safe

    order = np.argsort(-score_cbf)
    picked_samples = []
    used_kf = set()
    for idx in order:
        kf = int(idx) // 4
        if kf not in used_kf and cbf_shift_m[idx] > 0.15:
            picked_samples.append(int(idx))
            used_kf.add(kf)
            if len(picked_samples) == 3:
                break

    # Hook exact Eq. (19) barrier_field, sobel neg_grad, and all 5 CBF iterate trajectories (k=0..4)
    planner = model_v4.chronos_planner
    sobel_xy = planner.base_planner.sobel_xy.float()
    barrier_kernel = planner.barrier_kernel.float()

    barriers, neg_grads, c2_dils, pred_risks, small_hms, gt_segs = [], [], [], [], [], []
    gt_trajs_3, ref_trajs_3, safe_trajs_3, iter_trajs_3 = [], [], [], []

    for s_idx in picked_samples:
        item = ds256[s_idx]
        deg_img_t = item["corrupted_imgs"].unsqueeze(0).to(device)
        bev_256_t = item["bev_lidar_256"].unsqueeze(0).to(device)
        prev_256_t = item["bev_prev_256"].unsqueeze(0).to(device)
        geo_t = item["geo_14d"].unsqueeze(0).to(device)
        temp_t = item["temporal_8d"].unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model_v4(deg_img_t, bev_256_t, geo_t, temporal_8d=temp_t, bev_prev_256=prev_256_t)

        # EXACT Eq. (19) from ChronosCTRASafetyPlanner256.apply_cbf_safety_projection
        seg_probs = F.softmax(out["seg_logits_256"].detach().float(), dim=1)
        c2_prob = seg_probs[:, 2:3, :, :]
        c1_prob = seg_probs[:, 1:2, :, :]
        hm_prob = out["small_hm_256"].detach().float()
        pred_risk = out["pred_risk_256"].detach().float()

        c2_dilated = F.conv2d(c2_prob, barrier_kernel, padding=5).clamp(0.0, 1.5)
        barrier_field = (0.70 * c2_dilated + 0.20 * pred_risk.unsqueeze(1) + 0.15 * hm_prob + 0.10 * c1_prob).clamp(0.0, 1.5)
        neg_grad = -F.conv2d(barrier_field, sobel_xy, padding=1)  # (1, 2, 256, 256)

        # Trace all 5 iterates p^(0) .. p^(4)
        cur_traj = out["refined_traj"].detach().float()
        iters = [cur_traj[0].cpu().numpy()]
        for _ in range(4):
            gx = (cur_traj[:, :, 0] / 64.0) - 1.0
            gy = (cur_traj[:, :, 1] / 64.0) - 1.0
            sample_grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)
            b_val = F.grid_sample(barrier_field, sample_grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(1).squeeze(-1)
            rep_vec = F.grid_sample(neg_grad, sample_grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(-1).permute(0, 2, 1)
            rep_norm = rep_vec / (torch.norm(rep_vec, dim=-1, keepdim=True) + 1e-4)
            cbf_gate = F.relu(b_val - 0.12).unsqueeze(-1)
            step_push = torch.stack([rep_norm[:, :, 0] * 1.35, rep_norm[:, :, 1] * 0.45], dim=-1)
            cur_traj = cur_traj + cbf_gate * step_push * 1.65
            iters.append(cur_traj[0].cpu().numpy())

        barriers.append(barrier_field[0, 0].cpu().numpy())
        neg_grads.append(neg_grad[0].cpu().numpy())
        c2_dils.append(c2_dilated[0, 0].cpu().numpy())
        pred_risks.append(pred_risk[0].cpu().numpy())
        small_hms.append(hm_prob[0, 0].cpu().numpy())
        gt_segs.append(item["seg_256"].numpy().astype(np.uint8))
        gt_trajs_3.append(gt_traj[s_idx])
        ref_trajs_3.append(out["refined_traj"][0].float().cpu().numpy())
        safe_trajs_3.append(out["safe_traj"][0].float().cpu().numpy())
        iter_trajs_3.append(np.stack(iters, axis=0))

    npz_cbf_path = supp_dir / "raw_cbf_fields_3cases.npz"
    np.savez_compressed(
        npz_cbf_path,
        sample_indices=np.array(picked_samples, dtype=np.int64),
        barrier_field=np.stack(barriers, axis=0).astype(np.float32),
        neg_grad=np.stack(neg_grads, axis=0).astype(np.float32),
        c2_dilated=np.stack(c2_dils, axis=0).astype(np.float32),
        pred_risk_256=np.stack(pred_risks, axis=0).astype(np.float32),
        small_hm_256=np.stack(small_hms, axis=0).astype(np.float32),
        gt_seg_256=np.stack(gt_segs, axis=0).astype(np.uint8),
        gt_traj=np.stack(gt_trajs_3, axis=0).astype(np.float32),
        refined_traj=np.stack(ref_trajs_3, axis=0).astype(np.float32),
        safe_traj=np.stack(safe_trajs_3, axis=0).astype(np.float32),
        cbf_iter_trajs=np.stack(iter_trajs_3, axis=0).astype(np.float32),
    )
    print(f"[Exp 3] Saved exact Eq. (19) CBF barrier fields for samples {picked_samples} to {npz_cbf_path}")

    # 3. Regenerate Fig. 7 (Raw Unmixed Rule-LoRA Weights) & Fig. 8 (Real Eq. 19 CBF Field)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # --- Fig 7: Raw Unmixed dp_weights ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.6), dpi=300, gridspec_kw={"width_ratios": [1.25, 1.0]})
    regimes = [
        ("Obstacle-Dense\n(Construct./Occ.)", (gt_ml[:, 3] == 1) | (gt_ml[:, 8] == 1)),
        ("Interactive Conflict\n(Intersect./PedCross)", (gt_ml[:, 2] == 1) | (gt_ml[:, 5] == 1)),
        ("Smooth Cruising\n(Curve / Normal)", (gt_ml[:, 3] == 0) & (gt_ml[:, 5] == 0) & (gt_ml[:, 8] == 0)),
        ("Adverse Weather\n(Night / RainFog)", (gt_ml[:, 0] == 1) | (gt_ml[:, 1] == 1)),
    ]
    expert_names = [
        r"Raw $w_{\mathrm{obs}}$ (Obstacle Avoidance)",
        r"Raw $w_{\mathrm{conflict}}$ (Interaction Conflict)",
        r"Raw $w_{\mathrm{smooth}}$ (Kinematic Comfort)",
    ]
    expert_cols = ["#ef4444", "#f59e0b", "#10b981"]
    positions, data_violins, colors_violins = [], [], []
    group_centers = np.arange(len(regimes)) * 1.2
    width_v = 0.26
    for g_i, (_, mask_g) in enumerate(regimes):
        for e_j in range(3):
            positions.append(group_centers[g_i] + (e_j - 1) * width_v)
            data_violins.append(dp_w[mask_g, e_j])
            colors_violins.append(expert_cols[e_j])

    parts = ax1.violinplot(data_violins, positions=positions, widths=0.22, showmeans=False, showmedians=False, showextrema=False)
    for pc, col in zip(parts["bodies"], colors_violins):
        pc.set_facecolor(col)
        pc.set_edgecolor("#111827")
        pc.set_alpha(0.68)
        pc.set_linewidth(0.8)
    ax1.boxplot(
        data_violins, positions=positions, widths=0.09, patch_artist=True,
        showfliers=False, medianprops=dict(color="white", lw=1.5),
        boxprops=dict(facecolor="#1f2937", edgecolor="white", lw=0.8),
        whiskerprops=dict(color="#374151", lw=1.0),
        capprops=dict(color="#374151", lw=1.0),
    )
    for e_j in range(3):
        ax1.scatter([], [], color=expert_cols[e_j], s=60, label=expert_names[e_j])
    ax1.set_xticks(group_centers)
    ax1.set_xticklabels([r[0] for r in regimes])
    ax1.set_ylabel(r"Raw Unmixed Rule-LoRA DP Weight $\omega \in [0.5, 4.0]$")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.set_title("(a) Raw Unmixed Rule-LoRA Expert Weights Across Driving Regimes")
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    hm_matrix = np.zeros((10, 3), dtype=np.float32)
    for k in range(10):
        m_k = gt_ml[:, k] == 1
        hm_matrix[k] = dp_w[m_k].mean(axis=0)
    vmin, vmax = float(hm_matrix.min()), float(hm_matrix.max())
    im = ax2.imshow(hm_matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels([r"$w_{\mathrm{obs}}$" + "\n(Avoidance)", r"$w_{\mathrm{conflict}}$" + "\n(Interaction)", r"$w_{\mathrm{smooth}}$" + "\n(Comfort)"])
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(SCENARIO_NAMES_SHORT)
    mid_v = 0.5 * (vmin + vmax)
    for i in range(10):
        for j in range(3):
            val = hm_matrix[i, j]
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", color=("white" if val > mid_v else "#111827"), fontsize=8.5, weight="bold")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(r"Mean Raw DP Weight $\omega$", fontsize=9)
    ax2.set_title("(b) 10-Scenario Raw Rule-LoRA Routing Matrix")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig7_raw_rule_lora_weights.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 8: Real Eq. (19) Barrier Field & Sobel -grad B_CBF ---
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), dpi=300)
    case_titles = [
        "(a) Case 1: Near-Obstacle Lateral Repulsion",
        "(b) Case 2: Corridor Bottleneck Stabilization",
        "(c) Case 3: Multi-Agent Conflict Avoidance",
    ]
    x_pix = np.arange(72, 185)
    y_pix = np.arange(8, 129)
    X_m = (x_pix - 128.0) * 0.25
    Y_m = (y_pix - 16.0) * 0.25

    def _to_local_m(tr_128: np.ndarray) -> np.ndarray:
        ego_m = np.array([[0.0, 0.0]], dtype=np.float32)
        pts_m = np.stack([(tr_128[:, 0] - 64.0) * 0.5, (tr_128[:, 1] - 8.0) * 0.5], axis=-1)
        return np.vstack([ego_m, pts_m])

    for i, s_idx in enumerate(picked_samples):
        ax = axes[i]
        B_local = barriers[i][8:129, 72:185]
        # True Sobel neg_grad from Eq. (19): channel 0 is -dB/dx, channel 1 is -dB/dy
        neg_gx = neg_grads[i][0, 8:129, 72:185]
        neg_gy = neg_grads[i][1, 8:129, 72:185]

        cf = ax.contourf(X_m, Y_m, B_local, levels=18, cmap="inferno", alpha=0.88)
        ax.contour(X_m, Y_m, B_local, levels=[0.12, 0.30, 0.60, 1.00], colors="white", linewidths=0.6, alpha=0.6)

        step_q = 6
        Xq, Yq = np.meshgrid(X_m[::step_q], Y_m[::step_q])
        Ux = neg_gx[::step_q, ::step_q]
        Vy = neg_gy[::step_q, ::step_q]
        mag = np.hypot(Ux, Vy)
        mask_q = mag > 0.05
        if np.any(mask_q):
            u_norm = Ux[mask_q] / (mag[mask_q] + 1e-6)
            v_norm = Vy[mask_q] / (mag[mask_q] + 1e-6)
            ax.quiver(
                Xq[mask_q], Yq[mask_q], u_norm, v_norm,
                color="#fde047", alpha=0.85, scale=18.0, width=0.005, headwidth=3.5,
                label=r"True Sobel $-\nabla B_{\mathrm{CBF}}$"
            )

        gt_m = _to_local_m(gt_trajs_3[i])
        ref_m = _to_local_m(ref_trajs_3[i])
        safe_m = _to_local_m(safe_trajs_3[i])

        ax.plot(gt_m[:, 0], gt_m[:, 1], color="#22c55e", ls="-", lw=2.3, marker="*", ms=6.5, label="Expert GT", zorder=5)
        ax.plot(ref_m[:, 0], ref_m[:, 1], color="#ff4d4f", ls="--", lw=2.2, marker="x", ms=5.5, label="Before CBF ($p^{(0)}$)", zorder=6)
        ax.plot(safe_m[:, 0], safe_m[:, 1], color="#00f5d4", ls="-", lw=2.6, marker="o", ms=5.0, label="After 4-Step CBF ($p^{(4)}$)", zorder=7)

        for t_k in range(1, len(ref_m)):
            if np.linalg.norm(safe_m[t_k] - ref_m[t_k]) > 0.12:
                ax.annotate(
                    "", xy=(safe_m[t_k, 0], safe_m[t_k, 1]), xytext=(ref_m[t_k, 0], ref_m[t_k, 1]),
                    arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.4, shrinkA=0, shrinkB=0), zorder=8
                )

        max_y_m = min(25.5, max(21.5, float(gt_m[:, 1].max()) + 2.5, float(safe_m[:, 1].max()) + 2.5))
        ax.scatter([0.0], [0.0], c="white", s=70, edgecolors="black", lw=1.2, zorder=9, label="Ego (0,0)")
        ax.set_xlim(-12.5, 12.5)
        ax.set_ylim(-1.5, max_y_m)
        ax.set_xlabel("Lateral X (m)")
        if i == 0:
            ax.set_ylabel("Longitudinal Y (m)")
        ax.set_title(f"{case_titles[i]}\n(Max Shift: {cbf_shift_m[s_idx]:.2f}m, Sample #{s_idx})")
        ax.legend(loc="upper right", fontsize=7.0, framealpha=0.86, facecolor="#111827", labelcolor="white")
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        if i == 2:
            cbar.set_label(r"True Eq. (19) $B_{\mathrm{CBF}}(x,y)$", fontsize=8.5)

    fig.suptitle("True Eq. (19) 4-Step Differentiable CBF Barrier Potential Field & Negative Gradient Projection", y=1.02, fontsize=13.5, weight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig8_real_cbf_fields_3cases.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Exp 3] Saved updated Fig. 7 and Fig. 8 PNG+PDF in {fig_dir}")


# ==============================================================================
# 5. Main Entrypoint
# ==============================================================================

def main():
    t0 = time.time()
    enforce_vram_cap(24.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P0 Unified Eval] Running on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    val_jsonl = "dataset_full/val_full.jsonl"
    supp_dir = LOCAL_ROOT / "outputs_a100" / "paper_supplement"
    supp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load 256x256 GT validation dataset & pre-cache 535 unique keyframes (~1.5GB RAM)
    print("[Dataset] Loading 256x256 validation dataset (N=2,140)...")
    ds_256 = UniPercepChronos256Dataset(LOCAL_ROOT, val_jsonl, is_train=False, cam_hw=(192, 320), force_clean=False)
    ds_256_clean = UniPercepChronos256Dataset(LOCAL_ROOT, val_jsonl, is_train=False, cam_hw=(192, 320), force_clean=True)
    N = len(ds_256)

    gt_ml = np.zeros((N, 10), dtype=np.float32)
    gt_traj = np.zeros((N, 6, 2), dtype=np.float32)
    deg_modes = np.zeros(N, dtype=np.int64)
    gt_seg_256_all = np.zeros((N, 256, 256), dtype=np.uint8)
    gt_hm_256_all = np.zeros((N, 1, 256, 256), dtype=np.float32)

    for kf_i in range(0, N, 4):
        item = ds_256[kf_i]
        seg_u8 = item["seg_256"].numpy().astype(np.uint8)
        hm_f32 = item["small_hm_256"].numpy().astype(np.float32)
        for j in range(kf_i, min(kf_i + 4, N)):
            gt_seg_256_all[j] = seg_u8
            gt_hm_256_all[j] = hm_f32

    for idx in range(N):
        rec = ds_256.items[idx]
        gt_ml[idx] = np.array(rec["multi_label"], dtype=np.float32)
        gt_traj[idx] = np.array(rec["gt_traj"], dtype=np.float32)
        deg_modes[idx] = int(rec.get("deg_mode", idx % 4))

    gt_centers_list = extract_gt_small_centers_meters(ds_256.items)
    total_gt_small = sum(len(g) for g in gt_centers_list)
    print(f"[Dataset] Loaded N={N} samples, {total_gt_small} total GT small objects in {time.time() - t0:.1f}s.")

    all_mask = np.ones(N, dtype=bool)
    gt_ml_clean = np.repeat(gt_ml[0::4], 4, axis=0)[:N]

    # 2. Define ALL models for Exp 1 (5 main models) & Exp 2 (Config B + Config C)
    models_128_specs = [
        {
            "name": "Latent TransFuser",
            "group": "main",
            "ckpt": "checkpoints_a100/Ablation_1_Baseline_best.pt",
            "builder": lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=False, use_multisensor=False, use_dp_lora=False),
        },
        {
            "name": "Latent TransFuser + SSL & LP-FT",
            "group": "main",
            "ckpt": "checkpoints_a100/Ablation_2_With_SSL_LPFT_best.pt",
            "builder": lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=False, use_dp_lora=False),
        },
        {
            "name": "TransFuser",
            "group": "main",
            "ckpt": "checkpoints_a100/Ablation_3_With_MultiSensor_best.pt",
            "builder": lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=False),
        },
        {
            "name": "Config B (128^2 Diff-DP + Rule-LoRA, Stage3)",
            "group": "table6a",
            "ckpt": "checkpoints_a100/Full_UniPercep_DP_Stage3_RAG_LoRA_done.pt",
            "builder": lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=True),
        },
        {
            "name": "Config B (128^2 Diff-DP + Rule-LoRA, best)",
            "group": "table6a",
            "ckpt": "checkpoints_a100/Full_UniPercep_DP_best.pt",
            "builder": lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=True),
        },
    ]

    models_256_specs = [
        {
            "name": "Config C_v2 (Pro256 6Cam)",
            "group": "table6a",
            "ckpt": "checkpoints_a100/Full_Pro256_6Cam_best.pt",
            "arch": "pro256",
            "pass_temporal": False,
        },
        {
            "name": "Config C (v3 Ultra 256^2)",
            "group": "table6a",
            "ckpt": "checkpoints_a100/Full_v3_Ultra_best.pt",
            "arch": "v3_ultra",
            "pass_temporal": False,
        },
        {
            "name": "UniPercep-DP (Safety-LoRA)",
            "group": "main",
            "ckpt": "checkpoints_a100/Full_v3_5_SafetyLoRA_best.pt",
            "arch": "v3_5",
            "pass_temporal": True,
        },
        {
            "name": "UniPercep-DP (Ours, Full v4)",
            "group": "main",
            "ckpt": "checkpoints_a100/Full_v4_Chronos_best.pt",
            "arch": "v4",
            "pass_temporal": True,
        },
    ]

    results_overall = {}
    results_conditions = {}
    results_scenarios = {k: {"scenario_id": k, "name": SCENARIO_NAMES_FULL[k], "short_name": SCENARIO_NAMES_SHORT[k], "N_k": int((gt_ml[:, k] == 1).sum()), "models": {}} for k in range(10)}

    def _record_model(m_name: str, p_deg: dict, p_clean: dict = None, is_main: bool = True):
        results_overall[m_name] = evaluate_unified_256_metrics(
            p_deg, gt_ml, gt_traj, gt_centers_list, all_mask, include_horizons_comfort=True
        )
        conds = {
            "Overall_Val": results_overall[m_name],
            "Mode_0_Mild_Noise": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 0),
            "Mode_1_Color_Corrupt": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 1),
            "Mode_2_Dense_Fog": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 2),
            "Mode_3_Packet_Loss": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 3),
            "Daytime": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 0),
            "Nighttime": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 1),
        }
        if p_clean is not None:
            conds["Clean"] = evaluate_unified_256_metrics(p_clean, gt_ml_clean, gt_traj, gt_centers_list, all_mask)
        results_conditions[m_name] = conds

        if is_main:
            for k in range(10):
                mask_k = gt_ml[:, k] == 1
                results_scenarios[k]["models"][m_name] = evaluate_unified_256_metrics(
                    p_deg, gt_ml, gt_traj, gt_centers_list, mask_k, class_k=k
                )
        r = results_overall[m_name]
        print(
            f"  => [{m_name}] F1={r['Cls_F1']:.2f}% | mAP@Center={r['Det_mAP_Center']:.2f}% "
            f"(0.5/1/2/4m={r['Det_AP@0.5m']}/{r['Det_AP@1.0m']}/{r['Det_AP@2.0m']}/{r['Det_AP@4.0m']}) | "
            f"ContrastAP={r['Det_Contrast_AP']:.2f}% | mIoU={r['Seg_mIoU']:.2f}% | "
            f"ADE@1/2/3s={r['ADE@1.0s']:.3f}/{r['ADE@2.0s']:.3f}/{r['ADE@3.0s']:.3f}m | "
            f"FDE={r['FDE@3.0s']:.3f}m | DynColl={r['Dynamic_Coll_%']:.2f}% | TotObsColl={r['Total_Obs_Coll_%']:.2f}%"
        )

    # 3. Evaluate all 128x128 models first, then clear 128x128 RAM cache
    ds_128 = UniPercepFullDataset(LOCAL_ROOT, val_jsonl, is_train=False, force_clean=False)
    ds_128_clean = UniPercepFullDataset(LOCAL_ROOT, val_jsonl, is_train=False, force_clean=True)

    for spec in models_128_specs:
        m_name = spec["name"]
        ckpt_p = LOCAL_ROOT / spec["ckpt"]
        if not ckpt_p.exists():
            print(f"[WARN] Missing checkpoint {ckpt_p}, skipping {m_name}")
            continue
        print(f"\n[Model 128] Evaluating `{m_name}` ({ckpt_p.name}) ...")
        model = spec["builder"]().to(device)
        ck = torch.load(ckpt_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        p_deg = collect_unified_128(model, ds_128, gt_seg_256_all, gt_hm_256_all, device, batch_size=32, clean_mode=False)
        p_clean = collect_unified_128(model, ds_128_clean, gt_seg_256_all, gt_hm_256_all, device, batch_size=32, clean_mode=True) if spec["group"] == "main" else None
        _record_model(m_name, p_deg, p_clean, is_main=(spec["group"] == "main"))
        del model, ck, p_deg, p_clean
        torch.cuda.empty_cache()

    # Free 128x128 RAM cache (~1.6 GB) before running 256x256 models
    UniPercepFullDataset._shared_ram_cache.clear()
    del ds_128, ds_128_clean
    gc.collect()

    # 4. Evaluate all 256x256 models
    for spec in models_256_specs:
        m_name = spec["name"]
        ckpt_p = LOCAL_ROOT / spec["ckpt"]
        if not ckpt_p.exists():
            print(f"[WARN] Missing checkpoint {ckpt_p}, skipping {m_name}")
            continue
        print(f"\n[Model 256] Evaluating `{m_name}` ({ckpt_p.name}) ...")
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        sd = ck.get("ema", ck.get("model", ck))
        bb_name = detect_backbone_name(sd)
        print(f"  -> Detected backbone: {bb_name}")

        if spec["arch"] == "pro256":
            model = UniPercepDPNet_Pro256(backbone_name=bb_name, pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=True)
        elif spec["arch"] == "v3_ultra":
            model = UniPercepDPNet_v3_Ultra(backbone_name=bb_name, pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=True)
        elif spec["arch"] == "v3_5":
            model = UniPercepDPNet_v3_5_LoRA()
        else:
            model = UniPercepDPNet_v4_Chronos()

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  -> [Note] load_state_dict missing={len(missing)}, unexpected={len(unexpected)}")
        model = model.to(device).eval()

        is_v4 = (spec["arch"] == "v4")
        if spec["arch"] in ("pro256", "v3_ultra"):
            # Native v2/v3 preprocessing: INTER_LINEAR bev_lidar_256 + 6-cam input matching train_pro256.py
            from patch_p0_v2v3_and_128 import NativePro256EvalDataset, collect_native_pro256
            ds_nat = NativePro256EvalDataset(LOCAL_ROOT / val_jsonl, LOCAL_ROOT / "dataset_full" / "processed", cam_hw=(192, 320))
            p_deg = collect_native_pro256(model, ds_nat, gt_seg_256_all, gt_hm_256_all, device, batch_size=8)
            p_clean = None
            del ds_nat
        else:
            bs_256 = 8 if spec["arch"] == "v3_5" else 16
            p_deg = collect_unified_256(
                model, ds_256, device, batch_size=bs_256, clean_mode=False,
                pass_temporal=spec["pass_temporal"], export_raw_lora=is_v4
            )
            p_clean = collect_unified_256(
                model, ds_256_clean, device, batch_size=bs_256, clean_mode=True,
                pass_temporal=spec["pass_temporal"], export_raw_lora=False
            ) if spec["group"] == "main" else None
        _record_model(m_name, p_deg, p_clean, is_main=(spec["group"] == "main"))

        if is_v4:
            export_exp3_and_figures(
                model, ds_256, p_deg, gt_ml, gt_traj, deg_modes, gt_seg_256_all, device, supp_dir
            )

        del model, ck, sd, p_deg, p_clean
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Expert GT reference comfort & collision check (sanity check)
    gt_dyn_c, gt_tot_c, gt_env_c = compute_unified_collisions_256(gt_traj, gt_seg_256_all)
    results_overall["Expert GT (Physical Reference)"] = {
        **compute_kinematic_comfort(gt_traj),
        "Dynamic_Coll_%": round(float(gt_dyn_c.mean()), 2),
        "Total_Obs_Coll_%": round(float(gt_tot_c.mean()), 2),
        "Env075m_Dyn_Coll_%": round(float(gt_env_c.mean()), 2),
    }

    # Pick better Config B for Table 6(a) summary
    cb_s3 = results_overall.get("Config B (128^2 Diff-DP + Rule-LoRA, Stage3)")
    cb_best = results_overall.get("Config B (128^2 Diff-DP + Rule-LoRA, best)")
    if cb_s3 and cb_best:
        best_cb_key = "Config B (128^2 Diff-DP + Rule-LoRA, Stage3)" if cb_s3["ADE"] <= cb_best["ADE"] else "Config B (128^2 Diff-DP + Rule-LoRA, best)"
    else:
        best_cb_key = "Config B (128^2 Diff-DP + Rule-LoRA, Stage3)" if cb_s3 else "Config B (128^2 Diff-DP + Rule-LoRA, best)"

    table6a_summary = {
        "Config_A_TransFuser_128": results_overall.get("TransFuser"),
        "Config_B_Best_Selected": best_cb_key,
        "Config_B_128_DiffDP_RuleLoRA": results_overall.get(best_cb_key),
        "Config_B_Stage3_All": cb_s3,
        "Config_B_BestPt_All": cb_best,
        "Config_C_v2_Pro256": results_overall.get("Config C_v2 (Pro256 6Cam)"),
        "Config_C_v3_Ultra256": results_overall.get("Config C (v3 Ultra 256^2)"),
        "Config_D_v3_5_SafetyLoRA": results_overall.get("UniPercep-DP (Safety-LoRA)"),
        "Config_E_v4_Chronos_Ours": results_overall.get("UniPercep-DP (Ours, Full v4)"),
    }

    payload = {
        "protocol_info": {
            "eval_dataset": "dataset_full/val_full.jsonl",
            "N_samples": N,
            "N_unique_keyframes": N // 4,
            "N_gt_small_objects": int(total_gt_small),
            "unified_grid": "256x256 (0.25m/cell, ego at (128, 16))",
            "det_map_protocol": "nuScenes Center-Distance PR-curve AP @ {0.5, 1.0, 2.0, 4.0}m using 5x5 NMS peaks on unified 256x256 heatmap",
            "det_contrast_protocol": "Unified 256x256 peak-to-background contrast AP (identical formula for upsampled 128^2 and native 256^2)",
            "collision_protocol": "Unified 256x256 GT semantic map projection: Dynamic_Coll_% (Class 2) & Total_Obs_Coll_% (Class 1+2) & Env075m_Dyn_Coll_% (3x3 Class 2)",
        },
        "Overall_Unified": results_overall,
        "Conditions_Unified": results_conditions,
        "Scenarios_10_Unified": results_scenarios,
        "Table6a_Architecture_Progression_N2140": table6a_summary,
    }

    out_json = supp_dir / "unified_eval_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved unified metrics to {out_json} in {time.time() - t0:.1f}s!")

    # Also write a clean Markdown summary for human & paper inspection
    md_lines = [
        "# UniPercep-DP P0 补充实验汇总（统一物理米制评估器 + Config B/C 全量 N=2,140 + 原始张量导出）\n",
        f"> **评估集**：`val_full.jsonl`（$N={N}$ 样本，$535$ 物理关键帧 $\\times 4$ 退化模式，共 ${total_gt_small}$ 个小目标真值中心）  ",
        "> **统一协议**：所有 $128^2$ 与 $256^2$ 模型均统一在 $256\\times 256$（`0.25 m/cell`）物理网格上计算 **nuScenes Center-Distance PR mAP**（`{0.5, 1.0, 2.0, 4.0}m`）、**统一 Peak-Contrast AP**、**统一 Seg mIoU** 与 **动态/总障碍双口径碰撞率**。\n",
        "## 表 A：全模型统一口径主对比表（含 Config B 与 Config C 全量 N=2,140）\n",
        "| 模型 (Model) | Cls F1 (%) | **mAP@Center (%)** | AP@0.5/1/2/4m (%) | Contrast AP (%) | Seg mIoU (%) | ADE@1s/2s/3s (m) | FDE@3s (m) | **Dyn Coll (%)** | **Tot Obs Coll (%)** | P95 Jerk |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, r in results_overall.items():
        if m_name.startswith("Expert GT"):
            continue
        md_lines.append(
            f"| {m_name} | {r['Cls_F1']:.2f} | **{r['Det_mAP_Center']:.2f}** | "
            f"{r['Det_AP@0.5m']:.1f}/{r['Det_AP@1.0m']:.1f}/{r['Det_AP@2.0m']:.1f}/{r['Det_AP@4.0m']:.1f} | "
            f"{r['Det_Contrast_AP']:.2f} | {r['Seg_mIoU']:.2f} | "
            f"{r['ADE@1.0s']:.3f}/{r['ADE@2.0s']:.3f}/{r['ADE@3.0s']:.3f} | {r['FDE@3.0s']:.3f} | "
            f"**{r['Dynamic_Coll_%']:.2f}%** | {r['Total_Obs_Coll_%']:.2f}% | {r['P95_Jerk']:.2f} |"
        )
    out_md = supp_dir / "p0_unified_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[DONE] Saved P0 Markdown summary to {out_md}")


if __name__ == "__main__":
    main()
