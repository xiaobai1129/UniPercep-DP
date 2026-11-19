#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniPercep-DP P1 Experiment 4:
  Retrain 128x128 `TransFuser` (`Ablation_3_With_MultiSensor`) for 15 epochs with
  matched 1/7 online nighttime photometric scaling (`idx % 7 == 0`, gamma in {0.28, 0.34, 0.40}, ml[0]=1.0)
  on full `train_full.jsonl` (12,260 samples), then evaluate with the Unified 256x256 Evaluator
  on full `val_full.jsonl` (N=2,140: Overall, Clean, Mode 0-3, Daytime, Nighttime, 10 Scenarios).
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
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

SRC_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from models_full import UniPercepDPNet_Full, AsymmetricLoss
from train_and_ablation import (
    compute_multiclass_dice_loss,
    compute_focal_heatmap_loss,
    compute_trajectory_collision_penalty,
)
from p1_datasets import LowRAM128Dataset, LowRAM256Dataset
from run_p0_unified_eval import (
    enforce_vram_cap,
    extract_gt_small_centers_meters,
    collect_unified_128,
    evaluate_unified_256_metrics,
    SCENARIO_NAMES_FULL,
    SCENARIO_NAMES_SHORT,
)


def main():
    t0 = time.time()
    enforce_vram_cap(24.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = LOCAL_ROOT / "dataset_full" / "processed"
    train_jsonl = LOCAL_ROOT / "dataset_full" / "train_full.jsonl"
    val_jsonl = LOCAL_ROOT / "dataset_full" / "val_full.jsonl"
    ckpt_dir = LOCAL_ROOT / "checkpoints_a100"
    supp_dir = LOCAL_ROOT / "outputs_a100" / "paper_supplement"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    epochs = 15
    batch_size = 32
    base_lr = 6e-4

    print(f"[Exp 4] Training TransFuser (+ Matched 1/7 Night Aug) for {epochs} epochs, bs={batch_size} on full train_full.jsonl...")
    train_ds = LowRAM128Dataset(train_jsonl, processed_dir, is_train=True, night_aug=True)
    val_ds_128 = LowRAM128Dataset(val_jsonl, processed_dir, is_train=False, night_aug=False, force_clean=False)
    val_ds_128_clean = LowRAM128Dataset(val_jsonl, processed_dir, is_train=False, night_aug=False, force_clean=True)

    from torch.utils.data import RandomSampler
    # 4,096 random samples/epoch (2x original train_and_ablation's 2,048 cap) covering all 3,065 keyframes + 1/7 night aug
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=RandomSampler(train_ds, replacement=False, num_samples=4096),
        num_workers=2, pin_memory=True, drop_last=True
    )

    # Preload lightweight val 256 GT arrays for unified evaluation (~150MB RAM)
    val_ds_256 = LowRAM256Dataset(val_jsonl, processed_dir, is_train=False, cam_hw=(192, 320))
    N = len(val_ds_256)
    gt_ml = np.zeros((N, 10), dtype=np.float32)
    gt_traj = np.zeros((N, 6, 2), dtype=np.float32)
    deg_modes = np.zeros(N, dtype=np.int64)
    gt_seg_256_all = np.zeros((N, 256, 256), dtype=np.uint8)
    gt_hm_256_all = np.zeros((N, 1, 256, 256), dtype=np.float32)
    for kf_i in range(0, N, 4):
        item = val_ds_256[kf_i]
        seg_u8 = item["seg_256"].numpy().astype(np.uint8)
        hm_f32 = item["small_hm_256"].numpy().astype(np.float32)
        for j in range(kf_i, min(kf_i + 4, N)):
            gt_seg_256_all[j] = seg_u8
            gt_hm_256_all[j] = hm_f32
    for idx in range(N):
        rec = val_ds_256.items[idx]
        gt_ml[idx] = np.array(rec["multi_label"], dtype=np.float32)
        gt_traj[idx] = np.array(rec["gt_traj"], dtype=np.float32)
        deg_modes[idx] = int(rec.get("deg_mode", idx % 4))
    gt_centers_list = extract_gt_small_centers_meters(val_ds_256.items)
    all_mask = np.ones(N, dtype=bool)
    gt_ml_clean = np.repeat(gt_ml[0::4], 4, axis=0)[:N]

    model = UniPercepDPNet_Full(
        pretrained_backbone=True,
        use_ssl=True,
        use_multisensor=True,
        use_dp_lora=False,
    ).to(device)
    model.set_training_stage("Stage_Ablation_Joint")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.08)
    asl_criterion = AsymmetricLoss().to(device)
    seg_weights = torch.tensor([1.0, 2.5, 4.0], dtype=torch.float32, device=device)

    best_score = float("inf")
    best_ckpt_path = ckpt_dir / "Ablation_3_TransFuser_NightAug_best.pt"
    history = []

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        ep_losses = []
        for batch in train_loader:
            corrupted_img = batch["corrupted_img"].to(device, non_blocking=True)
            clean_img = batch["clean_img"].to(device, non_blocking=True)
            bev_lidar = batch["bev_lidar"].to(device, non_blocking=True)
            geo_14d = batch["geo_14d"].to(device, non_blocking=True)
            ml_gt = batch["multi_label"].to(device, non_blocking=True)
            seg_mask = batch["seg_mask"].to(device, non_blocking=True)
            risk_map = batch["risk_map"].to(device, non_blocking=True)
            small_hm = batch["small_hm"].to(device, non_blocking=True)
            small_reg = batch["small_reg"].to(device, non_blocking=True)
            small_mask = batch["small_mask"].to(device, non_blocking=True)
            gt_tr = batch["gt_traj"].to(device, non_blocking=True)
            dp_w_gt = batch["dp_weights_gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(corrupted_img, bev_lidar, geo_14d)
                loss_ssl = F.l1_loss(out["pred_clean_img"], clean_img)
                loss_ml = 0.5 * asl_criterion(out["multi_label_logits"], ml_gt) + 0.5 * F.binary_cross_entropy_with_logits(
                    out["multi_label_logits"], ml_gt
                )
                loss_seg = F.cross_entropy(out["seg_logits"], seg_mask, weight=seg_weights) + compute_multiclass_dice_loss(
                    out["seg_logits"], seg_mask
                )
                loss_small_hm = compute_focal_heatmap_loss(out["small_hm_pred"], small_hm)
                loss_small_reg = (
                    torch.abs(out["small_reg_pred"] - small_reg) * small_mask
                ).sum() / small_mask.sum().clamp(min=1.0)
                loss_percep = loss_ml + loss_seg + 0.5 * loss_small_hm + 0.5 * loss_small_reg
                loss_risk = F.mse_loss(out["pred_risk_map"], risk_map)
                loss_traj = F.smooth_l1_loss(out["pred_traj"], gt_tr)
                loss_coll = compute_trajectory_collision_penalty(out["pred_traj"], risk_map)
                accel = out["pred_traj"][:, 2:, :] - 2.0 * out["pred_traj"][:, 1:-1, :] + out["pred_traj"][:, :-2, :]
                loss_smooth = torch.mean(accel ** 2)
                loss_lora_w = F.mse_loss(out["dp_weights"], dp_w_gt)

                total_loss = (
                    0.3 * loss_ssl
                    + 1.0 * loss_percep
                    + 1.8 * loss_risk
                    + 1.5 * loss_traj
                    + 1.2 * loss_coll
                    + 0.08 * loss_smooth
                    + 0.5 * loss_lora_w
                )

            if not torch.isfinite(total_loss):
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
            optimizer.step()
            ep_losses.append(total_loss.item())

        scheduler.step()

        # Evaluate every 3 epochs and on final epoch (fast val without slow PR-curve loop)
        if ep % 3 == 0 or ep == epochs:
            p_deg = collect_unified_128(model, val_ds_128, gt_seg_256_all, gt_hm_256_all, device, batch_size=32, clean_mode=False)
            m_all = evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, all_mask, skip_pr_ap=True)
            m_night = evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 1, skip_pr_ap=True)
            score = m_all["ADE"] - 0.05 * m_all["Seg_mIoU"] - 0.02 * m_all["Cls_F1"]
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            dt = time.time() - ep_t0
            print(
                f"[Exp4-TransFuser+NightAug][Ep {ep:02d}/{epochs}] Loss={np.mean(ep_losses):.4f} | "
                f"All: F1={m_all['Cls_F1']:.2f}%, Contrast={m_all['Det_Contrast_AP']:.2f}%, mIoU={m_all['Seg_mIoU']:.2f}%, "
                f"ADE={m_all['ADE']:.3f}m, DynColl={m_all['Dynamic_Coll_%']:.2f}% | "
                f"Night: F1={m_night['Cls_F1']:.2f}%, ADE={m_night['ADE']:.3f}m, DynColl={m_night['Dynamic_Coll_%']:.2f}% | "
                f"VRAM={vram_gb:.1f}GB ({dt:.1f}s)"
            )
            history.append({"epoch": ep, "loss": round(float(np.mean(ep_losses)), 4), "overall": m_all, "night": m_night})
            if score < best_score or not best_ckpt_path.exists():
                best_score = score
                torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "metrics": m_all, "night_metrics": m_night}, best_ckpt_path)
            del p_deg
        else:
            dt = time.time() - ep_t0
            print(f"[Exp4-TransFuser+NightAug][Ep {ep:02d}/{epochs}] Loss={np.mean(ep_losses):.4f} ({dt:.1f}s)")

    # Final full unified evaluation from best checkpoint
    print(f"\n[Exp 4] Evaluating best checkpoint {best_ckpt_path.name} on full N={N} val set (Degraded + Clean + Night + 10 Scenarios)...")
    ck = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    p_deg = collect_unified_128(model, val_ds_128, gt_seg_256_all, gt_hm_256_all, device, batch_size=32, clean_mode=False)
    p_clean = collect_unified_128(model, val_ds_128_clean, gt_seg_256_all, gt_hm_256_all, device, batch_size=32, clean_mode=True)

    overall = evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, all_mask, include_horizons_comfort=True)
    conditions = {
        "Overall_Val": overall,
        "Clean": evaluate_unified_256_metrics(p_clean, gt_ml_clean, gt_traj, gt_centers_list, all_mask),
        "Mode_0_Mild_Noise": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 0),
        "Mode_1_Color_Corrupt": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 1),
        "Mode_2_Dense_Fog": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 2),
        "Mode_3_Packet_Loss": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 3),
        "Daytime": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 0),
        "Nighttime": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 1),
    }
    scenarios = {}
    for k in range(10):
        mask_k = gt_ml[:, k] == 1
        scenarios[str(k)] = {
            "name": SCENARIO_NAMES_FULL[k],
            "short_name": SCENARIO_NAMES_SHORT[k],
            "N_k": int(mask_k.sum()),
            "metrics": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, mask_k, class_k=k),
        }

    out_payload = {
        "model_name": "TransFuser (+ Matched Night Aug.)",
        "checkpoint": str(best_ckpt_path.name),
        "best_epoch": int(ck["epoch"]),
        "train_samples": len(train_ds),
        "Overall_Unified": overall,
        "Conditions_Unified": conditions,
        "Scenarios_10_Unified": scenarios,
        "training_history": history,
    }
    out_path = supp_dir / "p1_exp4_transfuser_nightaug.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"[Exp 4 DONE] Saved {out_path} in {time.time() - t0:.1f}s!")
    print(f"  Overall: {overall}")
    print(f"  Nighttime: {conditions['Nighttime']}")


if __name__ == "__main__":
    main()
