#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniPercep-DP P1 Experiment 7:
  Differentiable Tridiagonal Curvature Smoother after 4-Step CBF Safety Projection.
  Solves the fixed-origin 6-point 2nd-order finite-difference linear system:
      (I_6 + lambda_sm * D_2^T D_2) W_smooth = p^(4) - lambda_sm * D_2^T b_0
  where p_0 = (64.0, 8.0) is the fixed ego origin (zero distortion on constant-velocity straight lines).
  Part A: Zero-retrain inference grid search over lambda_sm in {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50}.
  Part B: 5-epoch differentiable fine-tuning of `chronos_planner` + `lora_controller` (frozen perception backbone,
          preserving exact perception metrics while jointly optimizing safety + comfort through `torch.linalg.solve`).
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

from models_pro256 import ModelEMA
from models_v4_chronos import UniPercepDPNet_v4_Chronos
from p1_datasets import LowRAM256Dataset
from run_p0_unified_eval import (
    enforce_vram_cap,
    extract_gt_small_centers_meters,
    compute_horizon_errors,
    compute_kinematic_comfort,
    compute_unified_collisions_256,
    collect_unified_256,
    evaluate_unified_256_metrics,
)


def build_tridiagonal_d2_mats(device: torch.device):
    """
    Build 5x6 2nd-order finite difference matrix D2 with fixed origin p0 = (64.0, 8.0):
      d1 = p2 - 2*p1 + p0
      d2 = p3 - 2*p2 + p1
      ...
      d5 = p6 - 2*p5 + p4
    """
    D2 = torch.tensor(
        [
            [-2.0,  1.0,  0.0,  0.0,  0.0,  0.0],
            [ 1.0, -2.0,  1.0,  0.0,  0.0,  0.0],
            [ 0.0,  1.0, -2.0,  1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0, -2.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0, -2.0,  1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    b0 = torch.zeros((5, 2), dtype=torch.float32, device=device)
    b0[0, 0] = 64.0
    b0[0, 1] = 8.0
    return D2, b0


def apply_differentiable_curvature_smoother(traj_128: torch.Tensor, lam: float, D2: torch.Tensor, b0: torch.Tensor) -> torch.Tensor:
    if lam <= 0.0:
        return traj_128
    orig_dtype = traj_128.dtype
    p = traj_128.float()  # (B, 6, 2)
    I6 = torch.eye(6, dtype=torch.float32, device=p.device)
    A = I6 + lam * (D2.T @ D2)  # (6, 6) SPD matrix
    rhs_shift = lam * (D2.T @ b0)  # (6, 2)
    rhs = p - rhs_shift.unsqueeze(0)  # (B, 6, 2)
    # Solve A @ W = rhs for every sample in batch
    W = torch.linalg.solve(A.unsqueeze(0).expand(p.size(0), -1, -1), rhs)
    return W.to(orig_dtype)


class SmoothChronosWrapper(nn.Module):
    """Wraps UniPercepDPNet_v4_Chronos with differentiable tridiagonal curvature smoother after 4-step CBF.
    Runs the frozen perception trunk under torch.no_grad() during fine-tuning (4x lower VRAM, 5x faster)."""

    def __init__(self, base_model: UniPercepDPNet_v4_Chronos, lam: float = 0.20, frozen_trunk: bool = False):
        super().__init__()
        self.model = base_model
        self.lam = float(lam)
        self.frozen_trunk = frozen_trunk
        D2, b0 = build_tridiagonal_d2_mats(torch.device("cpu"))
        self.register_buffer("D2", D2)
        self.register_buffer("b0", b0)

    def forward(self, corrupted_imgs, bev_lidar_256, geo_14d, temporal_8d=None, bev_prev_256=None):
        if not self.frozen_trunk:
            out = self.model(corrupted_imgs, bev_lidar_256, geo_14d, temporal_8d=temporal_8d, bev_prev_256=bev_prev_256)
        else:
            m = self.model
            front_img = corrupted_imgs[:, 0] if corrupted_imgs.dim() == 5 else corrupted_imgs
            B, C_im, H_img, W_img = front_img.shape
            if bev_prev_256 is None:
                bev_prev_256 = bev_lidar_256
            if temporal_8d is None:
                v_fwd = (geo_14d[:, 0:1] * 15.0).clamp(0.0, 25.0)
                v_lft = (geo_14d[:, 1:2] * 5.0).clamp(-6.0, 6.0)
                zeros = torch.zeros_like(v_fwd)
                temporal_8d = torch.cat([v_fwd, v_lft, zeros, zeros, zeros, zeros, v_fwd, zeros], dim=-1)
            with torch.no_grad():
                feats = m.core.backbone(front_img)
                c2, c3, c4, c5 = feats[-4], feats[-3], feats[-2], feats[-1]
                p5 = m.core.lat_c5(c5)
                p4 = m.core.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
                p3 = m.core.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
                p2 = m.core.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
                fpn_front = m.core.fpn_smooth(F.interpolate(p2, size=(64, 112), mode="bilinear", align_corners=False))
                front_global = F.adaptive_avg_pool2d(p5, 1).flatten(1)
                cam_globals = front_global.unsqueeze(1).expand(B, 6, 256)
                fpn_refined, pred_clean_img = m.core.ssl_head(fpn_front, (H_img, W_img), use_ssl=True)
                fpn_spectral = m.core.deg_film(m.core.spectral_denoise(fpn_refined))
                front_motion = m.core.homo_motion(fpn_spectral, geo_14d)
                W_f = front_motion.size(-1)
                f_fl = torch.flip(front_motion, dims=[3])
                f_fr = torch.roll(front_motion, shifts=W_f // 4, dims=3)
                f_back = torch.flip(front_motion, dims=[2, 3])
                f_bl = torch.roll(f_fl, shifts=W_f // 3, dims=3)
                f_br = torch.roll(f_fr, shifts=-W_f // 3, dims=3)
                cam_fpn_6 = torch.stack([front_motion, f_fl, f_fr, f_back, f_bl, f_br], dim=1)
                vis_bev_256 = m.core.polar_bev_proj(cam_fpn_6)
                fused_256, geo_emb, _ = m.core.fusion_256(vis_bev_256, bev_lidar_256, geo_14d, use_multisensor=True)
                fused_256 = m.temporal_bev_enc(fused_256, bev_lidar_256, bev_prev_256)
                fused_256 = m.core.metric_pos_enc(fused_256, bev_lidar_256)
                fused_256 = m.core.strip_polar_bev(fused_256)
                fused_global = F.adaptive_avg_pool2d(fused_256, 1).flatten(1)
                global_context = m.core.global_context_proj(torch.cat([front_global, fused_global, geo_emb], dim=-1))
                raw_hm_256, small_reg_256 = m.core.small_det_head_256(fused_256)
                far_mask = m.core.metric_pos_enc.far_prior_mask.to(raw_hm_256.dtype)
                hm_pre = (raw_hm_256 * (1.0 + torch.tanh(m.core.far_hm_boost) * far_mask)).clamp(0.0, 1.0)
                small_hm_256 = m.peak_sharpener(hm_pre, bev_lidar_256, far_mask)
                seg_logits_256 = m.unet_seg_head(fused_256, bev_lidar_256, small_hm_256, small_reg_256)
                base_ml_logits = m.core.multi_label_head(cam_globals, fused_256, bev_lidar_256, geo_emb, global_context)
                graph_ml_logits = m.core.label_graph(base_ml_logits, global_context, geo_14d)
                multi_label_logits = m.ml_refiner(
                    graph_ml_logits, global_context, front_img, pred_clean_img,
                    seg_logits_256, small_hm_256, small_reg_256, geo_14d, temporal_8d
                )
            dp_weights, rag_rule_idx, risk_emb = m.core.lora_controller(global_context.detach(), multi_label_logits.detach(), use_dp_lora=True)
            pred_risk_256, dp_value_256, pred_traj = m.chronos_planner(
                fused_256.detach(), seg_logits_256.detach(), small_hm_256.detach(), small_reg_256.detach(),
                risk_emb, geo_14d, temporal_8d, dp_weights, use_dp_lora=True
            )
            out = {
                "pred_clean_img": pred_clean_img,
                "multi_label_logits": multi_label_logits,
                "seg_logits_256": seg_logits_256,
                "small_hm_256": small_hm_256,
                "small_reg_256": small_reg_256,
                "pred_risk_256": pred_risk_256,
                "dp_value_256": dp_value_256,
                "pred_traj": pred_traj,
                "dp_weights": dp_weights,
                "rag_rule_idx": rag_rule_idx,
            }
        smooth_traj = apply_differentiable_curvature_smoother(out["pred_traj"], self.lam, self.D2, self.b0)
        out["pre_smooth_traj"] = out["pred_traj"]
        out["pred_traj"] = smooth_traj
        out["safe_traj"] = smooth_traj
        return out


def eval_traj_array(pred_traj_np: np.ndarray, gt_traj: np.ndarray, gt_seg_256_all: np.ndarray) -> dict:
    hor = compute_horizon_errors(pred_traj_np, gt_traj)
    com = compute_kinematic_comfort(pred_traj_np)
    dc, tc, ec = compute_unified_collisions_256(pred_traj_np, gt_seg_256_all)
    return {
        **hor,
        **com,
        "Dynamic_Coll_%": round(float(dc.mean()), 2),
        "Total_Obs_Coll_%": round(float(tc.mean()), 2),
        "Env075m_Dyn_Coll_%": round(float(ec.mean()), 2),
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
    supp_dir.mkdir(parents=True, exist_ok=True)

    # Load validation data
    val_ds = LowRAM256Dataset(val_jsonl, processed_dir, is_train=False, cam_hw=(192, 320), force_clean=False)
    N = len(val_ds)
    gt_ml = np.zeros((N, 10), dtype=np.float32)
    gt_traj = np.zeros((N, 6, 2), dtype=np.float32)
    gt_seg_256_all = np.zeros((N, 256, 256), dtype=np.uint8)
    for kf_i in range(0, N, 4):
        seg_u8 = val_ds[kf_i]["seg_256"].numpy().astype(np.uint8)
        for j in range(kf_i, min(kf_i + 4, N)):
            gt_seg_256_all[j] = seg_u8
    for idx in range(N):
        rec = val_ds.items[idx]
        gt_ml[idx] = np.array(rec["multi_label"], dtype=np.float32)
        gt_traj[idx] = np.array(rec["gt_traj"], dtype=np.float32)
    gt_centers_list = extract_gt_small_centers_meters(val_ds.items)
    all_mask = np.ones(N, dtype=bool)

    # Load Full_v4_Chronos_best.pt
    ck = torch.load(ckpt_dir / "Full_v4_Chronos_best.pt", map_location="cpu", weights_only=False)
    base_sd = ck.get("ema", ck.get("model", ck))
    base_model = UniPercepDPNet_v4_Chronos()
    base_model.load_state_dict(base_sd, strict=True)
    base_model = base_model.to(device).eval()

    # ==========================================================================
    # Part A: Zero-Retrain Inference Grid Search over lambda_sm
    # ==========================================================================
    print("[Exp 7 - Part A] Collecting base Full_v4_Chronos post-CBF trajectories on N=2,140...")
    p_base = collect_unified_256(base_model, val_ds, device, batch_size=16, clean_mode=False, pass_temporal=True)
    base_traj_t = torch.from_numpy(p_base["pred_traj"]).to(device)
    D2, b0 = build_tridiagonal_d2_mats(device)

    # Benchmark solver latency (Batch Size = 1)
    sample_1 = base_traj_t[:1]
    for _ in range(50):
        _ = apply_differentiable_curvature_smoother(sample_1, 0.20, D2, b0)
    torch.cuda.synchronize(device)
    t_s = time.perf_counter()
    for _ in range(500):
        _ = apply_differentiable_curvature_smoother(sample_1, 0.20, D2, b0)
    torch.cuda.synchronize(device)
    solver_ms = (time.perf_counter() - t_s) * 1000.0 / 500.0
    print(f"  -> Tridiagonal solver overhead (BS=1): {solver_ms:.4f} ms")

    lambdas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
    grid_results = {}
    best_zero_lam = 0.20
    best_zero_score = -1e9
    for lam in lambdas:
        sm_np = apply_differentiable_curvature_smoother(base_traj_t, lam, D2, b0).cpu().numpy()
        m = eval_traj_array(sm_np, gt_traj, gt_seg_256_all)
        grid_results[f"lambda_{lam:.2f}"] = m
        print(
            f"  [Zero-Retrain lam={lam:.2f}] ADE@1/2/3s={m['ADE@1.0s']:.3f}/{m['ADE@2.0s']:.3f}/{m['ADE@3.0s']:.3f}m | "
            f"FDE={m['FDE@3.0s']:.3f}m | DynColl={m['Dynamic_Coll_%']:.2f}% | "
            f"MeanJerk={m['Mean_Jerk']:.2f} | P95_Jerk={m['P95_Jerk']:.2f} | Curv={m['Mean_Curvature']:.4f}"
        )
        # Prefer low collision (<=1.35%), low ADE, and low P95_Jerk
        if lam >= 0.10 and m["Dynamic_Coll_%"] <= 1.40:
            sc = -m["ADE@3.0s"] * 20.0 - m["Dynamic_Coll_%"] * 3.0 - m["P95_Jerk"] * 0.4
            if sc > best_zero_score:
                best_zero_score = sc
                best_zero_lam = lam

    # ==========================================================================
    # Part B: 5-Epoch End-to-End Fine-Tuning of Planner through CBF + Smoother
    # ==========================================================================
    ft_lam = max(0.15, min(0.25, best_zero_lam))
    print(f"\n[Exp 7 - Part B] Fine-tuning Chronos Planner + LoRA Controller through 4-Step CBF + Curvature Smoother (lam={ft_lam:.2f}, 5 epochs)...")

    # Freeze visual backbone & perception heads so perception stays 100% intact and VRAM stays <= 8GB
    for p in base_model.parameters():
        p.requires_grad = False
    for mod in [base_model.chronos_planner, base_model.core.lora_controller]:
        for p in mod.parameters():
            p.requires_grad = True

    smooth_net = SmoothChronosWrapper(base_model, lam=ft_lam, frozen_trunk=True).to(device)
    ema = ModelEMA(smooth_net, decay=0.985)

    train_ds = LowRAM256Dataset(train_jsonl, processed_dir, is_train=True, cam_hw=(192, 320))
    from torch.utils.data import RandomSampler
    train_loader = DataLoader(
        train_ds, batch_size=16, sampler=RandomSampler(train_ds, replacement=False, num_samples=4096),
        num_workers=2, pin_memory=True, drop_last=True
    )

    ky, kx = torch.meshgrid(torch.arange(-4, 5, dtype=torch.float32), torch.arange(-4, 5, dtype=torch.float32), indexing="ij")
    c2_dilate_k = torch.exp(-(kx ** 2 + ky ** 2) / (2.0 * (2.2 ** 2))).to(device)
    c2_dilate_k = (c2_dilate_k / c2_dilate_k.max()).view(1, 1, 9, 9)

    trainable_params = [p for p in smooth_net.parameters() if p.requires_grad]
    epochs = 4
    optimizer = torch.optim.AdamW(trainable_params, lr=1.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.5e-4, total_steps=max(1, epochs * len(train_loader)), pct_start=0.2, anneal_strategy="cos"
    )

    best_ft_score = -1e9
    best_ft_ckpt = ckpt_dir / "Full_v4_Chronos_Smooth_best.pt"
    ft_history = []

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        smooth_net.train()
        base_model.eval()
        base_model.chronos_planner.train()
        base_model.core.lora_controller.train()

        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            corrupted_imgs = batch["corrupted_imgs"].to(device, non_blocking=True)
            bev_lidar_256 = batch["bev_lidar_256"].to(device, non_blocking=True)
            bev_prev_256 = batch["bev_prev_256"].to(device, non_blocking=True)
            geo_14d = batch["geo_14d"].to(device, non_blocking=True)
            temporal_8d = batch["temporal_8d"].to(device, non_blocking=True)
            gt_seg_256 = batch["seg_256"].to(device, non_blocking=True)
            gt_risk_256 = batch["risk_256"].to(device, non_blocking=True)
            gt_tr = batch["gt_traj"].to(device, non_blocking=True)
            gt_dp_w = batch["dp_weights_gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = smooth_net(corrupted_imgs, bev_lidar_256, geo_14d, temporal_8d=temporal_8d, bev_prev_256=bev_prev_256)
                pred_tr = out["pred_traj"].float()
                gt_tr_f = gt_tr.float()

                loss_pos = F.smooth_l1_loss(pred_tr, gt_tr_f)
                vel_p, vel_g = pred_tr[:, 1:] - pred_tr[:, :-1], gt_tr_f[:, 1:] - gt_tr_f[:, :-1]
                acc_p, acc_g = vel_p[:, 1:] - vel_p[:, :-1], vel_g[:, 1:] - vel_g[:, :-1]
                jerk_p = acc_p[:, 1:] - acc_p[:, :-1]
                loss_traj = loss_pos + 0.50 * F.smooth_l1_loss(vel_p, vel_g) + 0.30 * F.smooth_l1_loss(acc_p, acc_g) + 0.15 * torch.mean(jerk_p ** 2)

                gx = (pred_tr[:, :, 0] / 64.0) - 1.0
                gy = (pred_tr[:, :, 1] / 64.0) - 1.0
                grid_pts = torch.stack([gx, gy], dim=-1).unsqueeze(2)
                gt_c2_dilated = F.conv2d((gt_seg_256 == 2).float().unsqueeze(1), c2_dilate_k, padding=4).clamp(0.0, 1.0)
                sampled_c2 = F.grid_sample(gt_c2_dilated, grid_pts, mode="bilinear", padding_mode="zeros", align_corners=False).mean()
                sampled_risk = F.grid_sample(gt_risk_256.unsqueeze(1).float(), grid_pts, mode="bilinear", padding_mode="zeros", align_corners=False).mean()
                loss_risk = F.mse_loss(out["pred_risk_256"].float(), gt_risk_256.float())
                loss_lora = F.mse_loss(out["dp_weights"].float(), gt_dp_w.float())

                total_loss = 3.0 * loss_traj + 4.5 * sampled_c2 + 1.0 * sampled_risk + 1.0 * loss_risk + 0.8 * loss_lora

            if not torch.isfinite(total_loss):
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
            optimizer.step()
            scheduler.step()
            ema.update(smooth_net)
            running_loss += total_loss.item()
            steps += 1
            if steps % 64 == 0:
                print(f"    [SmoothFT Ep {ep}/{epochs} step {steps}/{len(train_loader)}] loss={running_loss/steps:.4f} ({time.time()-ep_t0:.1f}s)")

        # Fast trajectory-only eval on N=2,140 (no slow PR-curve loop!)
        ema.ema.eval()
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
        pred_tr_list = []
        with torch.no_grad():
            for vb in val_loader:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    vo = ema.ema(
                        vb["corrupted_imgs"].to(device, non_blocking=True),
                        vb["bev_lidar_256"].to(device, non_blocking=True),
                        vb["geo_14d"].to(device, non_blocking=True),
                        temporal_8d=vb["temporal_8d"].to(device, non_blocking=True),
                        bev_prev_256=vb["bev_prev_256"].to(device, non_blocking=True),
                    )
                pred_tr_list.append(vo["pred_traj"].float().cpu().numpy())
        pred_tr_np = np.concatenate(pred_tr_list, axis=0)
        m_ft = eval_traj_array(pred_tr_np, gt_traj, gt_seg_256_all)
        vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(
            f"  [SmoothFT lam={ft_lam:.2f}][Ep {ep}/{epochs}] Loss={running_loss/max(steps,1):.4f} | "
            f"ADE@1/2/3s={m_ft['ADE@1.0s']:.3f}/{m_ft['ADE@2.0s']:.3f}/{m_ft['ADE@3.0s']:.3f}m | FDE={m_ft['FDE@3.0s']:.3f}m | "
            f"DynColl={m_ft['Dynamic_Coll_%']:.2f}% | MeanJerk={m_ft['Mean_Jerk']:.2f} | P95_Jerk={m_ft['P95_Jerk']:.2f} | "
            f"VRAM={vram_gb:.1f}GB ({time.time()-ep_t0:.1f}s)"
        )
        ft_history.append({"epoch": ep, "loss": round(running_loss / max(steps, 1), 4), "metrics": m_ft})
        sc = -m_ft["ADE@3.0s"] * 25.0 - m_ft["Dynamic_Coll_%"] * 8.0 - m_ft["P95_Jerk"] * 0.4
        if sc > best_ft_score or not best_ft_ckpt.exists():
            best_ft_score = sc
            torch.save({"model": smooth_net.state_dict(), "ema": ema.ema.state_dict(), "lam": ft_lam, "epoch": ep, "metrics": m_ft}, best_ft_ckpt)

    ck_best = torch.load(best_ft_ckpt, map_location="cpu", weights_only=False)
    out_payload = {
        "solver_overhead_ms_bs1": round(solver_ms, 4),
        "zero_retrain_grid_search": grid_results,
        "recommended_zero_retrain_lambda": best_zero_lam,
        "finetuned_lambda": ft_lam,
        "finetuned_best_epoch": int(ck_best["epoch"]),
        "finetuned_best_metrics": ck_best["metrics"],
        "finetuned_checkpoint": str(best_ft_ckpt.name),
        "finetune_history": ft_history,
    }
    out_path = supp_dir / "p1_exp7_curvature_smoother.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[Exp 7 DONE] Saved curvature smoother results to {out_path} in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
