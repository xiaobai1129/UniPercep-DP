#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniPercep-DP P1 Experiment 5:
  Train & Evaluate 2 Modern 2023+ Multimodal Map-Free Baselines on the exact same
  12,260-sample 256x256 benchmark (`train_full.jsonl` with matched 1/7 night aug):
    1. `BEVFusion-Planner` (ICRA 2023 BEVFusion ConvFuser + GRU Waypoint Decoder)
    2. `VAD-Lite (Map-Free)` (ICCV 2023 Query-Based 3-Layer Transformer Decoder Planner)
  Evaluates both with the exact Unified 256x256 Physical-Metric Evaluator (Overall, Clean,
  Mode 0-3, Daytime, Nighttime, 10 Scenarios, Horizons, Kinematic Comfort, Params & Latency).
"""

import os
import sys
import gc
import json
import time
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

from models_full import AsymmetricLoss
from models_pro256 import ModelEMA
from models_modern_baselines import BEVFusionPlanner256, VADLiteQueryPlanner256
from train_v4_chronos import soft_miou_loss_256, ring_safe_focal_hm_loss_256
from p1_datasets import LowRAM256Dataset
from run_p0_unified_eval import (
    enforce_vram_cap,
    extract_gt_small_centers_meters,
    collect_unified_256,
    evaluate_unified_256_metrics,
    SCENARIO_NAMES_FULL,
    SCENARIO_NAMES_SHORT,
)


@torch.no_grad()
def benchmark_efficiency_256(model: nn.Module, device: torch.device, warmup: int = 30, iters: int = 100) -> dict:
    model.eval()
    img = torch.randn(1, 3, 192, 320, device=device)
    bev = torch.randn(1, 3, 256, 256, device=device)
    geo = torch.randn(1, 14, device=device)
    temp = torch.randn(1, 8, device=device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(img, bev, geo, temporal_8d=temp, bev_prev_256=bev)
    torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(img, bev, geo, temporal_8d=temp, bev_prev_256=bev)
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    mean_ms = float(np.mean(times_ms))
    return {
        "Total_Params_M": round(total_params, 2),
        "Peak_VRAM_MB": round(peak_mb, 1),
        "Latency_Mean_ms": round(mean_ms, 2),
        "Latency_Std_ms": round(float(np.std(times_ms)), 2),
        "FPS_Hz": round(1000.0 / max(mean_ms, 1e-3), 1),
    }


def train_and_eval_baseline(
    name: str,
    ckpt_name: str,
    model_builder,
    train_loader: DataLoader,
    val_ds: LowRAM256Dataset,
    val_ds_clean: LowRAM256Dataset,
    gt_ml: np.ndarray,
    gt_ml_clean: np.ndarray,
    gt_traj: np.ndarray,
    deg_modes: np.ndarray,
    gt_centers_list: list,
    device: torch.device,
    ckpt_dir: Path,
    epochs: int = 18,
    max_lr: float = 5e-4,
) -> dict:
    t0 = time.time()
    print(f"\n==================== [Exp 5] Training {name} ({epochs} epochs) ====================")
    model = model_builder().to(device)
    ema = ModelEMA(model, decay=0.990)

    asl_criterion = AsymmetricLoss()
    ce_weights = torch.tensor([1.0, 2.8, 4.5], device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=ce_weights)

    accum_steps = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
    total_steps = max(1, epochs * (len(train_loader) // accum_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.15, anneal_strategy="cos", div_factor=5.0, final_div_factor=30.0
    )

    N = len(val_ds)
    all_mask = np.ones(N, dtype=bool)
    best_score = -1e9
    best_ckpt_path = ckpt_dir / ckpt_name
    history = []

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)

        for b_idx, batch in enumerate(train_loader):
            corrupted_imgs = batch["corrupted_imgs"].to(device, non_blocking=True)
            bev_lidar_256 = batch["bev_lidar_256"].to(device, non_blocking=True)
            bev_prev_256 = batch["bev_prev_256"].to(device, non_blocking=True)
            geo_14d = batch["geo_14d"].to(device, non_blocking=True)
            temporal_8d = batch["temporal_8d"].to(device, non_blocking=True)
            gt_ml_b = batch["multi_label"].to(device, non_blocking=True)
            gt_seg_256 = batch["seg_256"].to(device, non_blocking=True)
            gt_hm_256 = batch["small_hm_256"].to(device, non_blocking=True)
            gt_reg_256 = batch["small_reg_256"].to(device, non_blocking=True)
            gt_mask_256 = batch["small_mask_256"].to(device, non_blocking=True)
            gt_tr = batch["gt_traj"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(corrupted_imgs, bev_lidar_256, geo_14d, temporal_8d=temporal_8d, bev_prev_256=bev_prev_256)
                ml_logits = out["multi_label_logits"].float()
                loss_ml = asl_criterion(ml_logits, gt_ml_b) + 0.35 * F.binary_cross_entropy_with_logits(ml_logits, gt_ml_b.float())
                seg_logits = out["seg_logits_256"].float()
                loss_seg = ce_criterion(seg_logits, gt_seg_256) + soft_miou_loss_256(seg_logits, gt_seg_256)
                loss_hm = ring_safe_focal_hm_loss_256(out["small_hm_256"], gt_hm_256)
                loss_reg = (
                    F.smooth_l1_loss(out["small_reg_256"].float() * gt_mask_256, gt_reg_256.float() * gt_mask_256, reduction="sum")
                    / gt_mask_256.sum().clamp(min=1.0)
                )
                pred_tr = out["pred_traj"].float()
                loss_pos = F.smooth_l1_loss(pred_tr, gt_tr.float())
                vel_p = pred_tr[:, 1:] - pred_tr[:, :-1]
                vel_g = gt_tr.float()[:, 1:] - gt_tr.float()[:, :-1]
                loss_traj = loss_pos + 0.4 * F.smooth_l1_loss(vel_p, vel_g)

                total_loss = 1.5 * loss_ml + 2.0 * loss_seg + 1.2 * loss_hm + 0.25 * loss_reg + 2.5 * loss_traj

            if not torch.isfinite(total_loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            (total_loss / accum_steps).backward()
            if (b_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
            running_loss += total_loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        if ep % 3 == 0 or ep == epochs:
            eval_net = model if ep <= 3 else ema.ema
            p_deg = collect_unified_256(eval_net, val_ds, device, batch_size=16, clean_mode=False, pass_temporal=True)
            m_all = evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, all_mask, skip_pr_ap=True)
            score = m_all["Cls_F1"] + m_all["Det_Contrast_AP"] + m_all["Seg_mIoU"] - m_all["ADE"] * 18.0 - m_all["Dynamic_Coll_%"] * 2.0
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            dt = time.time() - ep_t0
            print(
                f"[{name}][Ep {ep:02d}/{epochs}] Loss={avg_loss:.4f} | "
                f"F1={m_all['Cls_F1']:.2f}% | Contrast={m_all['Det_Contrast_AP']:.2f}% | "
                f"mIoU={m_all['Seg_mIoU']:.2f}% | ADE={m_all['ADE']:.3f}m | FDE={m_all['FDE']:.3f}m | "
                f"DynColl={m_all['Dynamic_Coll_%']:.2f}% | VRAM={vram_gb:.1f}GB ({dt:.1f}s)"
            )
            history.append({"epoch": ep, "loss": round(avg_loss, 4), "overall": m_all})
            if score > best_score or not best_ckpt_path.exists():
                best_score = score
                torch.save({"model": model.state_dict(), "ema": ema.ema.state_dict(), "epoch": ep, "metrics": m_all}, best_ckpt_path)
            del p_deg
        else:
            dt = time.time() - ep_t0
            print(f"[{name}][Ep {ep:02d}/{epochs}] Loss={avg_loss:.4f} ({dt:.1f}s)")

    # Final full unified eval from best EMA checkpoint
    ck = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    eval_model = model_builder().to(device)
    eval_model.load_state_dict(ck["ema"])
    eval_model.eval()

    p_deg = collect_unified_256(eval_model, val_ds, device, batch_size=16, clean_mode=False, pass_temporal=True)
    p_clean = collect_unified_256(eval_model, val_ds_clean, device, batch_size=16, clean_mode=True, pass_temporal=True)
    eff = benchmark_efficiency_256(eval_model, device)

    overall = evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, all_mask, include_horizons_comfort=True)
    overall.update(eff)
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

    del model, ema, eval_model, p_deg, p_clean
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[Exp 5] Completed `{name}` in {time.time() - t0:.1f}s! Overall={overall}")
    return {
        "model_name": name,
        "checkpoint": ckpt_name,
        "best_epoch": int(ck["epoch"]),
        "Overall_Unified": overall,
        "Conditions_Unified": conditions,
        "Scenarios_10_Unified": scenarios,
        "training_history": history,
    }


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

    train_ds = LowRAM256Dataset(train_jsonl, processed_dir, is_train=True, cam_hw=(192, 320))
    val_ds = LowRAM256Dataset(val_jsonl, processed_dir, is_train=False, cam_hw=(192, 320), force_clean=False)
    val_ds_clean = LowRAM256Dataset(val_jsonl, processed_dir, is_train=False, cam_hw=(192, 320), force_clean=True)

    from torch.utils.data import RandomSampler
    # Micro-batch 8 with accum_steps=2 -> exact effective batch_size 16 while keeping Peak VRAM ~11-13GB (<< 24GB cap)
    train_loader = DataLoader(
        train_ds, batch_size=8, sampler=RandomSampler(train_ds, replacement=False, num_samples=3072),
        num_workers=2, pin_memory=True, drop_last=True
    )

    N = len(val_ds)
    gt_ml = np.zeros((N, 10), dtype=np.float32)
    gt_traj = np.zeros((N, 6, 2), dtype=np.float32)
    deg_modes = np.zeros(N, dtype=np.int64)
    for idx in range(N):
        rec = val_ds.items[idx]
        gt_ml[idx] = np.array(rec["multi_label"], dtype=np.float32)
        gt_traj[idx] = np.array(rec["gt_traj"], dtype=np.float32)
        deg_modes[idx] = int(rec.get("deg_mode", idx % 4))
    gt_ml_clean = np.repeat(gt_ml[0::4], 4, axis=0)[:N]
    gt_centers_list = extract_gt_small_centers_meters(val_ds.items)

    out_path = supp_dir / "p1_exp5_modern_baselines.json"
    results = {}
    results["BEVFusion-Planner (2023)"] = train_and_eval_baseline(
        name="BEVFusion-Planner (2023)",
        ckpt_name="Baseline_BEVFusion_Planner_best.pt",
        model_builder=lambda: BEVFusionPlanner256(pretrained_backbone=True, dim=256),
        train_loader=train_loader,
        val_ds=val_ds,
        val_ds_clean=val_ds_clean,
        gt_ml=gt_ml,
        gt_ml_clean=gt_ml_clean,
        gt_traj=gt_traj,
        deg_modes=deg_modes,
        gt_centers_list=gt_centers_list,
        device=device,
        ckpt_dir=ckpt_dir,
        epochs=12,
        max_lr=5e-4,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    results["VAD-Lite / QueryPlan-DETR (2023)"] = train_and_eval_baseline(
        name="VAD-Lite / QueryPlan-DETR (2023)",
        ckpt_name="Baseline_VAD_Lite_QueryPlan_best.pt",
        model_builder=lambda: VADLiteQueryPlanner256(pretrained_backbone=True, dim=256),
        train_loader=train_loader,
        val_ds=val_ds,
        val_ds_clean=val_ds_clean,
        gt_ml=gt_ml,
        gt_ml_clean=gt_ml_clean,
        gt_traj=gt_traj,
        deg_modes=deg_modes,
        gt_centers_list=gt_centers_list,
        device=device,
        ckpt_dir=ckpt_dir,
        epochs=12,
        max_lr=5e-4,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Exp 5 DONE] Saved both 2023+ modern baselines to {out_path} in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
