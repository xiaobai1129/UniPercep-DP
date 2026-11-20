#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniPercep-DP P1 Experiment 6:
  Component-Wise Leave-One-Out Ablation Study on `Full_v4_Chronos` (Table 7):
    1. `w/o Dynamic 2D-FFT Gate`          (bypass `SpectralDenoiseBlock`)
    2. `w/o Degradation FiLM`             (bypass `DegradationFiLMModulator`)
    3. `w/o BackgroundZeroPeakSharpener`  (bypass `peak_sharpener`, use raw `hm_pre`)
    4. `w/o 20D Bottom-Up Evidence Ref.`  (bypass `ml_refiner`, use `graph_ml_logits`)
    5. `w/o 9-ch 4D LiDAR Motion Flow`    (bypass `temporal_bev_enc`, i.e., g_temp = 0)
  For each variant:
    - Loads `Full_v4_Chronos_best.pt` EMA weights, disables the single target component.
    - Evaluates zero-shot (0-ep) impact, then runs 5 epochs of adaptation fine-tuning (bs=16, lr=1.5e-4, EMA)
      so all remaining modules adapt cleanly without the removed component.
    - Evaluates with Unified 256x256 Evaluator (Overall, Clean, Mode 0-3, Daytime, Nighttime).
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
from models_v4_chronos import UniPercepDPNet_v4_Chronos
from train_v4_chronos import soft_miou_loss_256, ring_safe_focal_hm_loss_256
from p1_datasets import LowRAM256Dataset
from run_p0_unified_eval import (
    enforce_vram_cap,
    extract_gt_small_centers_meters,
    collect_unified_256,
    evaluate_unified_256_metrics,
)


class Identity1Arg(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class BypassSharpener(nn.Module):
    def forward(self, raw_hm_256, bev_lidar_256, far_mask):
        return raw_hm_256


class BypassMLRefiner(nn.Module):
    def forward(self, base_ml_logits, *args, **kwargs):
        return base_ml_logits


class BypassTemporalLiDAR(nn.Module):
    def forward(self, fused_256, bev_curr_256, bev_prev_256):
        return fused_256


def apply_ablation_patch(model: UniPercepDPNet_v4_Chronos, ablate_key: str):
    if ablate_key == "wo_fft":
        model.core.spectral_denoise = Identity1Arg()
    elif ablate_key == "wo_film":
        model.core.deg_film = Identity1Arg()
    elif ablate_key == "wo_sharpener":
        model.peak_sharpener = BypassSharpener()
    elif ablate_key == "wo_20d_refiner":
        model.ml_refiner = BypassMLRefiner()
    elif ablate_key == "wo_4d_lidar_flow":
        model.temporal_bev_enc = BypassTemporalLiDAR()
    else:
        raise ValueError(f"Unknown ablate_key: {ablate_key}")


def run_single_ablation(
    ablate_key: str,
    label: str,
    base_sd: dict,
    train_loader: DataLoader,
    val_ds: LowRAM256Dataset,
    val_ds_clean: LowRAM256Dataset,
    gt_ml: np.ndarray,
    gt_ml_clean: np.ndarray,
    gt_traj: np.ndarray,
    deg_modes: np.ndarray,
    gt_centers_list: list,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1.5e-4,
) -> dict:
    t0 = time.time()
    print(f"\n==================== [Exp 6] Ablation: {label} ({ablate_key}) ====================")
    model = UniPercepDPNet_v4_Chronos()
    model.load_state_dict(base_sd, strict=True)
    apply_ablation_patch(model, ablate_key)
    model = model.to(device)

    N = len(val_ds)
    all_mask = np.ones(N, dtype=bool)
    deg123_mask = (deg_modes >= 1)

    # 1. Zero-shot evaluation before 5-epoch adaptation
    p_zs = collect_unified_256(model, val_ds, device, batch_size=16, clean_mode=False, pass_temporal=True)
    zs_overall = evaluate_unified_256_metrics(p_zs, gt_ml, gt_traj, gt_centers_list, all_mask)
    zs_deg123 = evaluate_unified_256_metrics(p_zs, gt_ml, gt_traj, gt_centers_list, deg123_mask)
    print(
        f"  [Zero-Shot] F1={zs_overall['Cls_F1']:.2f}% | mAP={zs_overall['Det_mAP_Center']:.2f}% "
        f"(Contrast={zs_overall['Det_Contrast_AP']:.2f}%) | mIoU={zs_overall['Seg_mIoU']:.2f}% "
        f"(Deg1-3={zs_deg123['Seg_mIoU']:.2f}%) | ADE={zs_overall['ADE']:.3f}m | DynColl={zs_overall['Dynamic_Coll_%']:.2f}%"
    )
    del p_zs

    # 2. 5-epoch adaptation fine-tuning
    ema = ModelEMA(model, decay=0.985)
    asl_criterion = AsymmetricLoss()
    ce_weights = torch.tensor([1.0, 2.8, 4.5], device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=ce_weights)
    ce_per_pixel = nn.CrossEntropyLoss(weight=ce_weights, reduction="none")
    laplacian_kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device).view(1, 1, 3, 3)

    ky, kx = torch.meshgrid(torch.arange(-4, 5, dtype=torch.float32), torch.arange(-4, 5, dtype=torch.float32), indexing="ij")
    c2_dilate_k = torch.exp(-(kx ** 2 + ky ** 2) / (2.0 * (2.2 ** 2))).to(device)
    c2_dilate_k = (c2_dilate_k / c2_dilate_k.max()).view(1, 1, 9, 9)

    accum_steps = 2
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    total_steps = max(1, epochs * (len(train_loader) // accum_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.2, anneal_strategy="cos", div_factor=5.0, final_div_factor=20.0
    )

    best_score = -1e9
    best_sd = None

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for b_idx, batch in enumerate(train_loader):
            clean_img = batch["clean_img"].to(device, non_blocking=True)
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
            gt_risk_256 = batch["risk_256"].to(device, non_blocking=True)
            gt_tr = batch["gt_traj"].to(device, non_blocking=True)
            gt_dp_w = batch["dp_weights_gt"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(corrupted_imgs, bev_lidar_256, geo_14d, temporal_8d=temporal_8d, bev_prev_256=bev_prev_256)
                loss_ssl = F.l1_loss(out["pred_clean_img"].float(), clean_img.float())
                ml_logits_fp32 = out["multi_label_logits"].float()
                loss_ml = asl_criterion(ml_logits_fp32, gt_ml_b) + 0.35 * F.binary_cross_entropy_with_logits(ml_logits_fp32, gt_ml_b.float())

                seg_logits_fp32 = out["seg_logits_256"].float()
                pixel_ce = ce_per_pixel(seg_logits_fp32, gt_seg_256).view(seg_logits_fp32.size(0), -1)
                ohem_ce = torch.topk(pixel_ce, k=max(1, int(pixel_ce.size(1) * 0.15)), dim=1)[0].mean()
                pred_dyn_prob = F.softmax(seg_logits_fp32, dim=1)[:, 1:3].sum(dim=1, keepdim=True)
                gt_dyn_float = (gt_seg_256 > 0).float().unsqueeze(1)
                loss_boundary = F.l1_loss(
                    torch.abs(F.conv2d(pred_dyn_prob, laplacian_kernel, padding=1)),
                    torch.abs(F.conv2d(gt_dyn_float, laplacian_kernel, padding=1)),
                )
                loss_seg = 0.5 * ce_criterion(seg_logits_fp32, gt_seg_256) + 0.4 * ohem_ce + 1.3 * soft_miou_loss_256(seg_logits_fp32, gt_seg_256) + 0.35 * loss_boundary
                loss_hm = ring_safe_focal_hm_loss_256(out["small_hm_256"], gt_hm_256)
                loss_reg = (
                    F.smooth_l1_loss(out["small_reg_256"].float() * gt_mask_256, gt_reg_256.float() * gt_mask_256, reduction="sum")
                    / gt_mask_256.sum().clamp(min=1.0)
                )
                loss_risk = F.mse_loss(out["pred_risk_256"].float(), gt_risk_256.float())

                pred_tr = out["pred_traj"].float()
                gt_tr_f = gt_tr.float()
                vel_pred, vel_gt = pred_tr[:, 1:] - pred_tr[:, :-1], gt_tr_f[:, 1:] - gt_tr_f[:, :-1]
                acc_pred, acc_gt = vel_pred[:, 1:] - vel_pred[:, :-1], vel_gt[:, 1:] - vel_gt[:, :-1]
                loss_traj = F.smooth_l1_loss(pred_tr, gt_tr_f) + 0.50 * F.smooth_l1_loss(vel_pred, vel_gt) + 0.25 * F.smooth_l1_loss(acc_pred, acc_gt)

                gx = (pred_tr[:, :, 0] / 64.0) - 1.0
                gy = (pred_tr[:, :, 1] / 64.0) - 1.0
                grid_pts = torch.stack([gx, gy], dim=-1).unsqueeze(2)
                gt_c2_dilated = F.conv2d((gt_seg_256 == 2).float().unsqueeze(1), c2_dilate_k, padding=4).clamp(0.0, 1.0)
                sampled_c2 = F.grid_sample(gt_c2_dilated, grid_pts, mode="bilinear", padding_mode="zeros", align_corners=False).mean()
                sampled_risk = F.grid_sample(gt_risk_256.unsqueeze(1).float(), grid_pts, mode="bilinear", padding_mode="zeros", align_corners=False).mean()
                loss_lora = F.mse_loss(out["dp_weights"].float(), gt_dp_w.float())

                total_loss = (
                    0.3 * loss_ssl + 1.8 * loss_ml + 2.4 * loss_seg + 1.4 * loss_hm + 0.25 * loss_reg
                    + 1.2 * loss_risk + 2.8 * loss_traj + 2.5 * sampled_c2 + 1.0 * sampled_risk + 0.8 * loss_lora
                )

            if not torch.isfinite(total_loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            (total_loss / accum_steps).backward()
            if (b_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
            running_loss += total_loss.item()
            steps += 1

        eval_net = ema.ema
        torch.cuda.empty_cache()
        p_ep = collect_unified_256(eval_net, val_ds, device, batch_size=8, clean_mode=False, pass_temporal=True)
        m_ep = evaluate_unified_256_metrics(p_ep, gt_ml, gt_traj, gt_centers_list, all_mask, skip_pr_ap=True)
        score = m_ep["Cls_F1"] + m_ep["Det_Contrast_AP"] + m_ep["Seg_mIoU"] - m_ep["ADE"] * 18.0 - m_ep["Dynamic_Coll_%"] * 2.5
        vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(
            f"  [{ablate_key}][Ep {ep}/{epochs}] Loss={running_loss/max(steps,1):.4f} | "
            f"F1={m_ep['Cls_F1']:.2f}% | Contrast={m_ep['Det_Contrast_AP']:.2f}% | "
            f"mIoU={m_ep['Seg_mIoU']:.2f}% | ADE={m_ep['ADE']:.3f}m | DynColl={m_ep['Dynamic_Coll_%']:.2f}% | "
            f"VRAM={vram_gb:.1f}GB ({time.time()-ep_t0:.1f}s)"
        )
        if score > best_score or best_sd is None:
            best_score = score
            best_sd = {k: v.cpu().clone() for k, v in eval_net.state_dict().items()}
        del p_ep

    # Final unified evaluation of best adapted EMA model across all conditions
    model.load_state_dict(best_sd)
    model.eval()
    torch.cuda.empty_cache()
    p_deg = collect_unified_256(model, val_ds, device, batch_size=8, clean_mode=False, pass_temporal=True)
    p_clean = collect_unified_256(model, val_ds_clean, device, batch_size=8, clean_mode=True, pass_temporal=True)

    overall = evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, all_mask, include_horizons_comfort=True)
    conditions = {
        "Overall_Val": overall,
        "Clean": evaluate_unified_256_metrics(p_clean, gt_ml_clean, gt_traj, gt_centers_list, all_mask),
        "Mode_0_Mild_Noise": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 0),
        "Mode_1_Color_Corrupt": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 1),
        "Mode_2_Dense_Fog": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 2),
        "Mode_3_Packet_Loss": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg_modes == 3),
        "Degraded_Modes_1_2_3": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, deg123_mask),
        "Daytime": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 0),
        "Nighttime": evaluate_unified_256_metrics(p_deg, gt_ml, gt_traj, gt_centers_list, gt_ml[:, 0] == 1),
    }

    del model, ema, best_sd, p_deg, p_clean
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[Exp 6] Finished `{label}` in {time.time() - t0:.1f}s!")
    return {
        "ablate_key": ablate_key,
        "label": label,
        "zero_shot_overall": zs_overall,
        "zero_shot_deg123": zs_deg123,
        "adapted_5ep_overall": overall,
        "adapted_5ep_conditions": conditions,
    }


def main():
    t0 = time.time()
    enforce_vram_cap(24.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = LOCAL_ROOT / "dataset_full" / "processed"
    train_jsonl = LOCAL_ROOT / "dataset_full" / "train_full.jsonl"
    val_jsonl = LOCAL_ROOT / "dataset_full" / "val_full.jsonl"
    ckpt_p = LOCAL_ROOT / "checkpoints_a100" / "Full_v4_Chronos_best.pt"
    supp_dir = LOCAL_ROOT / "outputs_a100" / "paper_supplement"
    supp_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    base_sd = ck.get("ema", ck.get("model", ck))

    train_ds = LowRAM256Dataset(train_jsonl, processed_dir, is_train=True, cam_hw=(192, 320))
    val_ds = LowRAM256Dataset(val_jsonl, processed_dir, is_train=False, cam_hw=(192, 320), force_clean=False)
    val_ds_clean = LowRAM256Dataset(val_jsonl, processed_dir, is_train=False, cam_hw=(192, 320), force_clean=True)
    from torch.utils.data import RandomSampler
    # Micro-batch 8 with accum_steps=2 -> exact effective batch_size 16 while keeping Peak VRAM ~19.5GB (<< 24GB cap)
    train_loader = DataLoader(
        train_ds, batch_size=8, sampler=RandomSampler(train_ds, replacement=False, num_samples=2048),
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

    ablation_specs = [
        ("wo_fft", "w/o Dynamic 2D-FFT Gate"),
        ("wo_film", "w/o Degradation FiLM"),
        ("wo_sharpener", "w/o BackgroundZeroPeakSharpener"),
        ("wo_20d_refiner", "w/o 20D Bottom-Up Evidence Refiner"),
        ("wo_4d_lidar_flow", "w/o 9-ch 4D LiDAR Motion Flow"),
    ]

    out_path = supp_dir / "p1_exp6_component_ablation.json"
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[Exp 6 Resume] Loaded existing completed variants: {list(results.keys())}")
    else:
        results = {}

    for ab_key, ab_label in ablation_specs:
        if ab_key in results:
            print(f"[Exp 6 Resume] Skipping already completed variant `{ab_key}` ({ab_label}).")
            continue
        results[ab_key] = run_single_ablation(
            ab_key, ab_label, base_sd, train_loader, val_ds, val_ds_clean,
            gt_ml, gt_ml_clean, gt_traj, deg_modes, gt_centers_list, device, epochs=2, lr=1.2e-4
        )
        # Incremental save after each variant for crash safety
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[Exp 6 DONE] Completed all 5 leave-one-out component ablations in {time.time() - t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
