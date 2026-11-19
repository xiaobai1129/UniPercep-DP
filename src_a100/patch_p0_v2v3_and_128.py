#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted Surgical Patch for P0 `unified_eval_metrics.json`:
  1. Uses `mode="nearest"` for 128x128 -> 256x256 `seg_logits` upsampling (exact 2x2 cell alignment
     with `gt_seg_256` which is `cv2.INTER_NEAREST`-upsampled from `seg_128`, eliminating bilinear blur bias).
  2. Evaluates `Config C_v2 (Pro256 6Cam)` and `Config C (v3 Ultra 256^2)` using their native
     `UniPercepSurround256Dataset` input preprocessing (`cv2.INTER_LINEAR` on `bev_lidar_256` + 6-cam input)
     while scoring ALL outputs on the exact same unified 256x256 physical-metric evaluator!
  3. Updates `unified_eval_metrics.json` and `p0_unified_summary.md` atomically.
"""

import os
import sys
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

SRC_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import train_and_ablation as t128
import train_v4_chronos as t256
t128.LOCAL_ROOT = LOCAL_ROOT
t128.PROCESSED_DIR = LOCAL_ROOT / "dataset_full" / "processed"
t256.LOCAL_ROOT = LOCAL_ROOT
t256.DATASET_DIR = LOCAL_ROOT / "dataset_full"

from models_full import UniPercepDPNet_Full
from models_pro256 import UniPercepDPNet_Pro256
from models_v3_ultra import UniPercepDPNet_v3_Ultra
from p1_datasets import LowRAM128Dataset
from run_p0_unified_eval import (
    enforce_vram_cap,
    detect_backbone_name,
    extract_gt_small_centers_meters,
    extract_hm256_peaks_meters,
    compute_unified_contrast_ap_batch,
    compute_unified_collisions_256,
    evaluate_unified_256_metrics,
    SCENARIO_NAMES_FULL,
    SCENARIO_NAMES_SHORT,
)


class NativePro256EvalDataset(Dataset):
    """Exact `UniPercepSurround256Dataset` preprocessing for v2 Pro256 & v3 Ultra (INTER_LINEAR bev_256)."""

    def __init__(self, jsonl_path: Path, processed_dir: Path, cam_hw=(192, 320)):
        self.processed_dir = Path(processed_dir)
        self.cam_hw = cam_hw
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _get_kf(self, rec):
        kf = rec["keyframe_idx"]
        if kf in self._cache:
            return self._cache[kf]
        H_c, W_c = self.cam_hw
        clean_bgr = cv2.imread(str(self.processed_dir / rec["clean_img"]))
        corrupt_bgr = cv2.imread(str(self.processed_dir / rec["corrupted_img"]))
        clean_res = cv2.resize(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB), (W_c, H_c), interpolation=cv2.INTER_LINEAR)
        corr_res = cv2.resize(cv2.cvtColor(corrupt_bgr, cv2.COLOR_BGR2RGB), (W_c, H_c), interpolation=cv2.INTER_LINEAR)

        # Match `train_pro256.py` lines 113-125 exactly when raw_nuscenes is absent
        other_5 = np.stack([
            cv2.flip(clean_res, 1),
            np.roll(clean_res, shift=W_c // 4, axis=1),
            cv2.flip(clean_res, -1),
            np.roll(cv2.flip(clean_res, 1), shift=W_c // 3, axis=1),
            np.roll(clean_res, shift=-W_c // 3, axis=1),
        ], axis=0)

        bev_128 = np.load(self.processed_dir / rec["bev_lidar"]).astype(np.float32)
        bev_256 = np.zeros((3, 256, 256), dtype=np.float32)
        for c in range(3):
            bev_256[c] = cv2.resize(bev_128[c], (256, 256), interpolation=cv2.INTER_LINEAR)

        entry = {"clean_res": clean_res, "corr_res": corr_res, "other_5": other_5, "bev_256": bev_256}
        self._cache[kf] = entry
        return entry

    def __getitem__(self, idx):
        rec = self.items[idx]
        c = self._get_kf(rec)
        deg_mode = int(rec.get("deg_mode", 1))
        if deg_mode == 0:
            front_deg = (0.7 * c["clean_res"].astype(np.float32) + 0.3 * c["corr_res"].astype(np.float32)).astype(np.uint8)
        elif deg_mode == 1:
            front_deg = c["corr_res"]
        elif deg_mode == 2:
            front_deg = np.clip(0.78 * c["corr_res"].astype(np.float32) + 0.22 * 204.0, 0, 255).astype(np.uint8)
        else:
            front_deg = c["corr_res"].copy()
            h_f, w_f, _ = front_deg.shape
            front_deg[h_f // 3 : h_f // 2, w_f // 3 : w_f // 2, :] = 25

        imgs_6 = np.concatenate([front_deg[None, ...], c["other_5"]], axis=0).astype(np.float32) / 255.0
        sensor_state = np.array(rec["sensor_state"], dtype=np.float32)
        homography = np.array(rec["homography"], dtype=np.float32).reshape(-1)
        geo_14d = np.concatenate([sensor_state, homography], axis=0)
        return {
            "corrupted_imgs": torch.from_numpy(imgs_6).permute(0, 3, 1, 2).contiguous(),
            "bev_lidar_256": torch.from_numpy(c["bev_256"]),
            "geo_14d": torch.from_numpy(geo_14d),
        }


@torch.no_grad()
def collect_128_nearest_seg(model, ds128, gt_seg_256_all, gt_hm_256_all, device, batch_size=32, clean_mode=False):
    model.eval()
    N = len(ds128)
    unique_indices = list(range(0, N, 4)) if clean_mode else list(range(N))
    loader = DataLoader(Subset(ds128, unique_indices) if clean_mode else ds128, batch_size=batch_size, shuffle=False, num_workers=2)

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
        # Nearest-neighbor upsample of 128x128 seg logits matches gt_seg_256 (exact 2x2 physical cell mapping)
        seg_logits_256 = F.interpolate(out["seg_logits"].float(), size=(256, 256), mode="nearest")
        pred_seg_256 = torch.argmax(seg_logits_256, dim=1).cpu().numpy()
        gt_seg_b = gt_seg_256_all[b_idx]
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
        det_peaks_list = [pk for pk in det_peaks_list for _ in range(4)][:N]

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
def collect_native_pro256(model, ds_nat, gt_seg_256_all, gt_hm_256_all, device, batch_size=8):
    model.eval()
    loader = DataLoader(ds_nat, batch_size=batch_size, shuffle=False, num_workers=0)
    ml_logits_list, seg_inter_list, seg_union_list = [], [], []
    det_peaks_list, contrast_ap_list = [], []
    pred_traj_list, dyn_coll_list, tot_coll_list, env_coll_list = [], [], [], []
    offset = 0
    for batch in loader:
        B = batch["geo_14d"].size(0)
        b_idx = list(range(offset, offset + B))
        offset += B
        imgs = batch["corrupted_imgs"].to(device, non_blocking=True)
        bev = batch["bev_lidar_256"].to(device, non_blocking=True)
        geo = batch["geo_14d"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(imgs, bev, geo)
        ml_logits_list.append(out["multi_label_logits"].float().cpu().numpy())
        pred_seg = torch.argmax(out["seg_logits_256"].float(), dim=1).cpu().numpy()
        gt_seg_b = gt_seg_256_all[b_idx]
        inter_b = np.zeros((B, 3), dtype=np.float64)
        union_b = np.zeros((B, 3), dtype=np.float64)
        for c in range(3):
            inter_b[:, c] = ((pred_seg == c) & (gt_seg_b == c)).sum(axis=(1, 2))
            union_b[:, c] = ((pred_seg == c) | (gt_seg_b == c)).sum(axis=(1, 2))
        seg_inter_list.append(inter_b)
        seg_union_list.append(union_b)

        pred_hm = out["small_hm_256"].float().clamp(0.0, 1.0)
        gt_hm_b = torch.from_numpy(gt_hm_256_all[b_idx]).to(device)
        det_peaks_list.extend(extract_hm256_peaks_meters(pred_hm))
        contrast_ap_list.append(compute_unified_contrast_ap_batch(pred_hm, gt_hm_b))
        pred_tr = out["pred_traj"].float().cpu().numpy()
        pred_traj_list.append(pred_tr)
        dc, tc, ec = compute_unified_collisions_256(pred_tr, gt_seg_b)
        dyn_coll_list.append(dc)
        tot_coll_list.append(tc)
        env_coll_list.append(ec)

    ml_logits = np.concatenate(ml_logits_list, axis=0)
    return {
        "ml_probs": 1.0 / (1.0 + np.exp(-np.clip(ml_logits, -30.0, 30.0))),
        "seg_inter": np.concatenate(seg_inter_list, axis=0),
        "seg_union": np.concatenate(seg_union_list, axis=0),
        "det_peaks": det_peaks_list,
        "contrast_ap": np.concatenate(contrast_ap_list, axis=0),
        "pred_traj": np.concatenate(pred_traj_list, axis=0),
        "dyn_coll": np.concatenate(dyn_coll_list, axis=0),
        "tot_coll": np.concatenate(tot_coll_list, axis=0),
        "env_dyn_coll": np.concatenate(env_coll_list, axis=0),
    }


def main():
    t0 = time.time()
    enforce_vram_cap(24.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    supp_dir = LOCAL_ROOT / "outputs_a100" / "paper_supplement"
    json_path = supp_dir / "unified_eval_metrics.json"
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    processed_dir = LOCAL_ROOT / "dataset_full" / "processed"
    val_jsonl = LOCAL_ROOT / "dataset_full" / "val_full.jsonl"

    ds_nat = NativePro256EvalDataset(val_jsonl, processed_dir, cam_hw=(192, 320))
    N = len(ds_nat)
    gt_ml = np.zeros((N, 10), dtype=np.float32)
    gt_traj = np.zeros((N, 6, 2), dtype=np.float32)
    deg_modes = np.zeros(N, dtype=np.int64)
    gt_seg_256_all = np.zeros((N, 256, 256), dtype=np.uint8)
    gt_hm_256_all = np.zeros((N, 1, 256, 256), dtype=np.float32)

    rad, sigma = 8, 3.2
    py_grid, px_grid = np.mgrid[-rad : rad + 1, -rad : rad + 1].astype(np.float32)
    g_kernel = np.exp(-(px_grid ** 2 + py_grid ** 2) / (2.0 * sigma * sigma))

    for kf_i in range(0, N, 4):
        rec = ds_nat.items[kf_i]
        seg_128 = np.load(processed_dir / rec["seg_mask"]).astype(np.uint8)
        seg_256 = cv2.resize(seg_128, (256, 256), interpolation=cv2.INTER_NEAREST)
        hm_256 = np.zeros((1, 256, 256), dtype=np.float32)
        for obj in rec.get("det_boxes", []):
            cx, cy, _, _, _, _, _, _, is_small = obj
            if is_small == 1 and 0 <= cx < 128 and 0 <= cy < 128:
                ix, iy = int(np.clip(round(float(cx) * 2.0), 0, 255)), int(np.clip(round(float(cy) * 2.0), 0, 255))
                y0, y1 = max(0, iy - rad), min(256, iy + rad + 1)
                x0, x1 = max(0, ix - rad), min(256, ix + rad + 1)
                ky0, kx0 = y0 - (iy - rad), x0 - (ix - rad)
                hm_256[0, y0:y1, x0:x1] = np.maximum(hm_256[0, y0:y1, x0:x1], g_kernel[ky0 : ky0 + (y1 - y0), kx0 : kx0 + (x1 - x0)])
        for j in range(kf_i, min(kf_i + 4, N)):
            gt_seg_256_all[j] = seg_256
            gt_hm_256_all[j] = hm_256

    for idx in range(N):
        rec = ds_nat.items[idx]
        gt_ml[idx] = np.array(rec["multi_label"], dtype=np.float32)
        gt_traj[idx] = np.array(rec["gt_traj"], dtype=np.float32)
        deg_modes[idx] = int(rec.get("deg_mode", idx % 4))

    gt_centers_list = extract_gt_small_centers_meters(ds_nat.items)
    all_mask = np.ones(N, dtype=bool)
    gt_ml_clean = np.repeat(gt_ml[0::4], 4, axis=0)[:N]

    def _update(m_name, p_deg, p_clean=None, is_main=False):
        payload["Overall_Unified"][m_name] = evaluate_unified_256_metrics(
            p_deg, gt_ml, gt_traj, gt_centers_list, all_mask, include_horizons_comfort=True
        )
        conds = {
            "Overall_Val": payload["Overall_Unified"][m_name],
            "Mode_0_Mild_Noise": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 0),
            "Mode_1_Color_Corrupt": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 1),
            "Mode_2_Dense_Fog": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 2),
            "Mode_3_Packet_Loss": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 3),
            "Daytime": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 0),
            "Nighttime": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 1),
        }
        if p_clean is not None:
            conds["Clean"] = evaluate_unified_256_metrics(p_clean, gt_ml_clean, gt_traj, gt_centers_list, all_mask)
        payload["Conditions_Unified"][m_name] = conds
        if is_main:
            for k in range(10):
                mask_k = gt_ml[:, k] == 1
                payload["Scenarios_10_Unified"][str(k)]["models"][m_name] = evaluate_unified_256_metrics(
                    p_deg, gt_ml, gt_traj, gt_centers_list, mask_k, class_k=k
                )
        r = payload["Overall_Unified"][m_name]
        print(
            f"  => [PATCHED {m_name}] F1={r['Cls_F1']:.2f}% | mAP@Center={r['Det_mAP_Center']:.2f}% | "
            f"ContrastAP={r['Det_Contrast_AP']:.2f}% | mIoU={r['Seg_mIoU']:.2f}% | "
            f"ADE@1/2/3s={r['ADE@1.0s']:.3f}/{r['ADE@2.0s']:.3f}/{r['ADE@3.0s']:.3f}m | FDE={r['FDE@3.0s']:.3f}m | "
            f"DynColl={r['Dynamic_Coll_%']:.2f}% | TotObsColl={r['Total_Obs_Coll_%']:.2f}%"
        )

    # 1. Patch 128x128 models with exact nearest seg upsampling
    ds_128 = LowRAM128Dataset(val_jsonl, processed_dir, is_train=False, force_clean=False)
    ds_128_clean = LowRAM128Dataset(val_jsonl, processed_dir, is_train=False, force_clean=True)
    specs_128 = [
        ("Latent TransFuser", True, "checkpoints_a100/Ablation_1_Baseline_best.pt", lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=False, use_multisensor=False, use_dp_lora=False)),
        ("Latent TransFuser + SSL & LP-FT", True, "checkpoints_a100/Ablation_2_With_SSL_LPFT_best.pt", lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=False, use_dp_lora=False)),
        ("TransFuser", True, "checkpoints_a100/Ablation_3_With_MultiSensor_best.pt", lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=False)),
        ("Config B (128^2 Diff-DP + Rule-LoRA, Stage3)", False, "checkpoints_a100/Full_UniPercep_DP_Stage3_RAG_LoRA_done.pt", lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=True)),
        ("Config B (128^2 Diff-DP + Rule-LoRA, best)", False, "checkpoints_a100/Full_UniPercep_DP_best.pt", lambda: UniPercepDPNet_Full(pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=True)),
    ]
    for m_name, is_main, ckpt_rel, builder in specs_128:
        model = builder().to(device)
        ck = torch.load(LOCAL_ROOT / ckpt_rel, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        p_deg = collect_128_nearest_seg(model, ds_128, gt_seg_256_all, gt_hm_256_all, device, batch_size=32, clean_mode=False)
        p_clean = collect_128_nearest_seg(model, ds_128_clean, gt_seg_256_all, gt_hm_256_all, device, batch_size=32, clean_mode=True) if is_main else None
        _update(m_name, p_deg, p_clean, is_main=is_main)
        del model, ck, p_deg, p_clean
        torch.cuda.empty_cache()

    # 2. Patch Config C_v2 (Pro256) and Config C (v3 Ultra) with native INTER_LINEAR bev_lidar_256 & 6-cam input
    specs_v2v3 = [
        ("Config C_v2 (Pro256 6Cam)", "checkpoints_a100/Full_Pro256_6Cam_best.pt", UniPercepDPNet_Pro256),
        ("Config C (v3 Ultra 256^2)", "checkpoints_a100/Full_v3_Ultra_best.pt", UniPercepDPNet_v3_Ultra),
    ]
    for m_name, ckpt_rel, Cls in specs_v2v3:
        ck = torch.load(LOCAL_ROOT / ckpt_rel, map_location="cpu", weights_only=False)
        sd = ck.get("ema", ck.get("model", ck))
        bb = detect_backbone_name(sd)
        model = Cls(backbone_name=bb, pretrained_backbone=False, use_ssl=True, use_multisensor=True, use_dp_lora=True)
        model.load_state_dict(sd, strict=True)
        model = model.to(device).eval()
        p_deg = collect_native_pro256(model, ds_nat, gt_seg_256_all, gt_hm_256_all, device, batch_size=8)
        _update(m_name, p_deg, None, is_main=False)
        del model, ck, sd, p_deg
        torch.cuda.empty_cache()

    # Refresh Table 6(a) summary
    cb_s3 = payload["Overall_Unified"]["Config B (128^2 Diff-DP + Rule-LoRA, Stage3)"]
    cb_best = payload["Overall_Unified"]["Config B (128^2 Diff-DP + Rule-LoRA, best)"]
    best_cb_key = "Config B (128^2 Diff-DP + Rule-LoRA, Stage3)" if cb_s3["ADE"] <= cb_best["ADE"] else "Config B (128^2 Diff-DP + Rule-LoRA, best)"
    payload["Table6a_Architecture_Progression_N2140"] = {
        "Config_A_TransFuser_128": payload["Overall_Unified"]["TransFuser"],
        "Config_B_Best_Selected": best_cb_key,
        "Config_B_128_DiffDP_RuleLoRA": payload["Overall_Unified"][best_cb_key],
        "Config_B_Stage3_All": cb_s3,
        "Config_B_BestPt_All": cb_best,
        "Config_C_v2_Pro256": payload["Overall_Unified"]["Config C_v2 (Pro256 6Cam)"],
        "Config_C_v3_Ultra256": payload["Overall_Unified"]["Config C (v3 Ultra 256^2)"],
        "Config_D_v3_5_SafetyLoRA": payload["Overall_Unified"]["UniPercep-DP (Safety-LoRA)"],
        "Config_E_v4_Chronos_Ours": payload["Overall_Unified"]["UniPercep-DP (Ours, Full v4)"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Rewrite p0_unified_summary.md
    md_lines = [
        "# UniPercep-DP P0 补充实验汇总（统一物理米制评估器 + Config B/C 全量 N=2,140 + 原始张量导出）\n",
        f"> **评估集**：`val_full.jsonl`（$N={N}$ 样本，$535$ 物理关键帧 $\\times 4$ 退化模式）  ",
        "> **统一协议**：所有 $128^2$ 与 $256^2$ 模型均统一在 $256\\times 256$（`0.25 m/cell`）物理网格上计算 **nuScenes Center-Distance PR mAP**（`{0.5, 1.0, 2.0, 4.0}m`）、**统一 Peak-Contrast AP**、**统一 Seg mIoU**（128 分割网格按 2×2 物理块精确对齐 `gt_seg_256`）与 **动态/总障碍双口径碰撞率**。\n",
        "## 表 A：全模型统一口径主对比表（含 Config B 与 Config C 全量 N=2,140）\n",
        "| 模型 (Model) | Cls F1 (%) | **mAP@Center (%)** | AP@0.5/1/2/4m (%) | Contrast AP (%) | Seg mIoU (%) | ADE@1s/2s/3s (m) | FDE@3s (m) | **Dyn Coll (%)** | **Tot Obs Coll (%)** | P95 Jerk |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, r in payload["Overall_Unified"].items():
        if m_name.startswith("Expert GT"):
            continue
        md_lines.append(
            f"| {m_name} | {r['Cls_F1']:.2f} | **{r['Det_mAP_Center']:.2f}** | "
            f"{r['Det_AP@0.5m']:.1f}/{r['Det_AP@1.0m']:.1f}/{r['Det_AP@2.0m']:.1f}/{r['Det_AP@4.0m']:.1f} | "
            f"{r['Det_Contrast_AP']:.2f} | {r['Seg_mIoU']:.2f} | "
            f"{r['ADE@1.0s']:.3f}/{r['ADE@2.0s']:.3f}/{r['ADE@3.0s']:.3f} | {r['FDE@3.0s']:.3f} | "
            f"**{r['Dynamic_Coll_%']:.2f}%** | {r['Total_Obs_Coll_%']:.2f}% | {r['P95_Jerk']:.2f} |"
        )
    with open(supp_dir / "p0_unified_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n[PATCH DONE] Updated {json_path} and p0_unified_summary.md in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
