import os
import sys
import csv
import json
import math
import time
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Enable A100 TF32 & cuDNN benchmark acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_full import UniPercepDPNet_Full, AsymmetricLoss

LOCAL_ROOT = Path("/root/workspace/ceshi")
GDRIVE_ROOT = Path("/content/gdrive/MyDrive/colab/UniPercep_DP")

PROCESSED_DIR = LOCAL_ROOT / "dataset_full" / "processed"
CKPT_DIR = LOCAL_ROOT / "checkpoints_a100"
LOG_DIR = LOCAL_ROOT / "logs_a100"
OUT_DIR = LOCAL_ROOT / "outputs_a100"

GDRIVE_CKPT_DIR = GDRIVE_ROOT / "checkpoints_a100"
GDRIVE_LOG_DIR = GDRIVE_ROOT / "logs_a100"
GDRIVE_OUT_DIR = GDRIVE_ROOT / "outputs_a100"

GRID_SIZE = 128
METERS_PER_PIXEL = 0.5


def _atomic_copy(src: Path, dst: Path):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)


def sync_a100_to_gdrive(files_to_sync):
    """
    Real-time atomic synchronization of checkpoints, LoRA weights, and CSV logs to Google Drive.
    """
    GDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    GDRIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for src_p, dst_p in files_to_sync:
        _atomic_copy(src_p, dst_p)
    os.sync()


class UniPercepFullDataset(Dataset):
    """
    High-throughput A100 Dataset with in-RAM keyframe caching (leveraging 167GB system RAM)
    and multi-level degradation synthesis.
    """

    def __init__(self, jsonl_path: Path, data_root: Path, max_samples: int = None):
        self.data_root = data_root
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
                    if max_samples and len(self.samples) >= max_samples:
                        break

        # Precompute 128x128 coordinate grids for fast small object Gaussian center synthesis
        yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE].astype(np.float32)
        self.xx_grid = xx
        self.yy_grid = yy
        self._ram_cache = {}

    def __len__(self):
        return len(self.samples)

    def _load_keyframe_base(self, rec):
        kf_idx = rec["keyframe_idx"]
        if kf_idx in self._ram_cache:
            return self._ram_cache[kf_idx]

        clean_bgr = cv2.imread(str(self.data_root / rec["clean_img"]))
        corrupt_bgr = cv2.imread(str(self.data_root / rec["corrupted_img"]))
        if clean_bgr is None:
            clean_bgr = np.zeros((448, 768, 3), dtype=np.uint8)
        if corrupt_bgr is None:
            corrupt_bgr = np.zeros((448, 768, 3), dtype=np.uint8)

        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        corrupt_rgb = cv2.cvtColor(corrupt_bgr, cv2.COLOR_BGR2RGB)

        bev_lidar = np.load(self.data_root / rec["bev_lidar"]).astype(np.float32)
        seg_mask = np.load(self.data_root / rec["seg_mask"]).astype(np.int64)
        risk_map = np.load(self.data_root / rec["risk_map"]).astype(np.float32)

        # Synthesize 128x128 small object center heatmap & [w, h, vx, vy] regression targets
        small_hm = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        small_reg = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        small_mask = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for obj in rec.get("det_boxes", []):
            cx, cy, w_g, l_g, cls_id, track_id, vx, vy, is_small = obj
            if is_small == 1 and 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                sigma = 2.2
                g = np.exp(-((self.xx_grid - cx) ** 2 + (self.yy_grid - cy) ** 2) / (2 * sigma ** 2))
                small_hm[0] = np.maximum(small_hm[0], g)
                ix, iy = int(cx), int(cy)
                r0_x, r1_x = max(0, ix - 1), min(GRID_SIZE, ix + 2)
                r0_y, r1_y = max(0, iy - 1), min(GRID_SIZE, iy + 2)
                small_reg[0, r0_y:r1_y, r0_x:r1_x] = w_g / 10.0
                small_reg[1, r0_y:r1_y, r0_x:r1_x] = l_g / 10.0
                small_reg[2, r0_y:r1_y, r0_x:r1_x] = vx / 10.0
                small_reg[3, r0_y:r1_y, r0_x:r1_x] = vy / 10.0
                small_mask[0, r0_y:r1_y, r0_x:r1_x] = 1.0

        item = {
            "clean_rgb": clean_rgb,
            "corrupt_rgb": corrupt_rgb,
            "bev_lidar": bev_lidar,
            "seg_mask": seg_mask,
            "risk_map": risk_map,
            "small_hm": small_hm,
            "small_reg": small_reg,
            "small_mask": small_mask,
        }
        # Cache in RAM if under 6000 unique keyframes (~12GB RAM)
        if len(self._ram_cache) < 6000:
            self._ram_cache[kf_idx] = item
        return item

    def __getitem__(self, idx):
        rec = self.samples[idx]
        base = self._load_keyframe_base(rec)
        deg_mode = rec.get("deg_mode", 1)

        clean_f = base["clean_rgb"].astype(np.float32) / 255.0
        if deg_mode == 0:
            # Mild compression / slight noise mix
            corrupt_f = 0.7 * clean_f + 0.3 * (base["corrupt_rgb"].astype(np.float32) / 255.0)
        elif deg_mode == 1:
            corrupt_f = base["corrupt_rgb"].astype(np.float32) / 255.0
        elif deg_mode == 2:
            # Rain/fog atmospheric veil
            corrupt_f = 0.78 * (base["corrupt_rgb"].astype(np.float32) / 255.0) + 0.22 * 0.8
        else:
            # Severe packet loss macroblock + fog
            corrupt_f = base["corrupt_rgb"].astype(np.float32) / 255.0
            corrupt_f[160:220, 280:360, :] = 0.1

        clean_t = torch.from_numpy(clean_f).permute(2, 0, 1).contiguous()
        corrupt_t = torch.from_numpy(corrupt_f).permute(2, 0, 1).contiguous()

        sensor_state = np.array(rec["sensor_state"], dtype=np.float32)
        homography = np.array(rec["homography"], dtype=np.float32)
        geo_14d = np.concatenate([sensor_state, homography], axis=0)

        multi_label = np.array(rec["multi_label"], dtype=np.float32)
        gt_traj = np.array(rec["gt_traj"], dtype=np.float32)
        dp_weights_gt = np.array(rec.get("dp_weights_gt", [2.0, 1.8, 1.8]), dtype=np.float32)

        return {
            "keyframe_idx": rec["keyframe_idx"],
            "deg_mode": deg_mode,
            "clean_img": clean_t,
            "corrupted_img": corrupt_t,
            "bev_lidar": torch.from_numpy(base["bev_lidar"]),
            "seg_mask": torch.from_numpy(base["seg_mask"]),
            "risk_map": torch.from_numpy(base["risk_map"]),
            "small_hm": torch.from_numpy(base["small_hm"]),
            "small_reg": torch.from_numpy(base["small_reg"]),
            "small_mask": torch.from_numpy(base["small_mask"]),
            "geo_14d": torch.from_numpy(geo_14d),
            "multi_label": torch.from_numpy(multi_label),
            "gt_traj": torch.from_numpy(gt_traj),
            "dp_weights_gt": torch.from_numpy(dp_weights_gt),
        }


def compute_multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3):
    probs = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = torch.sum(probs * one_hot, dims)
    card = torch.sum(probs + one_hot, dims)
    dice = (2.0 * inter + 1.0) / (card + 1.0)
    return 1.0 - dice.mean()


def compute_focal_heatmap_loss(pred_hm: torch.Tensor, gt_hm: torch.Tensor):
    pred = pred_hm.float().clamp(1e-5, 1.0 - 1e-5)
    gt = gt_hm.float()
    pos_inds = gt.ge(0.5).float()
    neg_inds = gt.lt(0.5).float()
    neg_weights = torch.pow(1.0 - gt, 4.0)

    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos_inds
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * neg_inds

    num_pos = pos_inds.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def compute_trajectory_collision_penalty(pred_traj: torch.Tensor, risk_map: torch.Tensor):
    """
    Sample spatial risk field at predicted trajectory waypoints (B, 6, 2) on 128x128 grid
    using bilinear grid_sample to penalize obstacle collisions differentiably.
    """
    # Normalize (gx, gy) in [0, 127] to [-1, 1] for grid_sample
    grid_norm = (pred_traj.float() / 63.5) - 1.0  # (B, 6, 2)
    grid_norm = grid_norm.unsqueeze(2)            # (B, 6, 1, 2)
    sampled_risk = F.grid_sample(
        risk_map.float().unsqueeze(1),
        grid_norm,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, 1, 6, 1)
    return sampled_risk.mean()


@torch.no_grad()
def evaluate_full_metrics(model, val_loader, device, smoke_test: bool = False):
    model.eval()
    ml_tp, ml_fp, ml_fn, ml_correct, ml_total = 0.0, 0.0, 0.0, 0.0, 0.0
    cls_inter = np.zeros(3, dtype=np.float64)
    cls_union = np.zeros(3, dtype=np.float64)

    small_ap_scores = []
    traj_ade_m = []
    traj_fde_m = []
    collision_flags = []
    val_losses = []

    for b_idx, batch in enumerate(val_loader):
        if smoke_test and b_idx >= 2:
            break
        batch_gpu = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(
                batch_gpu["corrupted_img"],
                batch_gpu["bev_lidar"],
                batch_gpu["geo_14d"],
            )
            l_traj = F.smooth_l1_loss(out["pred_traj"], batch_gpu["gt_traj"])
            l_seg = F.cross_entropy(out["seg_logits"], batch_gpu["seg_mask"])
            val_losses.append((l_traj + l_seg).item())

        # 1. Multi-label Acc & F1
        ml_prob = torch.sigmoid(out["multi_label_logits"].float())
        ml_pred = (ml_prob > 0.5).float()
        ml_gt = batch_gpu["multi_label"].float()
        ml_correct += (ml_pred == ml_gt).sum().item()
        ml_total += ml_gt.numel()
        ml_tp += (ml_pred * ml_gt).sum().item()
        ml_fp += (ml_pred * (1.0 - ml_gt)).sum().item()
        ml_fn += ((1.0 - ml_pred) * ml_gt).sum().item()

        # 2. Obstacle Segmentation mIoU
        seg_pred = torch.argmax(out["seg_logits"], dim=1)
        seg_gt = batch_gpu["seg_mask"]
        for c in range(3):
            cls_inter[c] += ((seg_pred == c) & (seg_gt == c)).sum().item()
            cls_union[c] += ((seg_pred == c) | (seg_gt == c)).sum().item()

        # 3. Small Object Detection AP (CenterNet peak matching within 2.5m / 5 grid cells + Gaussian IoU)
        hm_p = out["small_hm_pred"].float()
        hm_g = batch_gpu["small_hm"].float()
        hm_p_nms = F.max_pool2d(hm_p, kernel_size=5, stride=1, padding=2)
        hm_g_nms = F.max_pool2d(hm_g, kernel_size=5, stride=1, padding=2)
        for b_i in range(hm_p.size(0)):
            g_peaks = torch.nonzero((hm_g[b_i, 0] == hm_g_nms[b_i, 0]) & (hm_g[b_i, 0] > 0.45), as_tuple=False).float()
            p_mask = (hm_p[b_i, 0] == hm_p_nms[b_i, 0]) & (hm_p[b_i, 0] > 0.15)
            p_coords = torch.nonzero(p_mask, as_tuple=False).float()
            if g_peaks.size(0) == 0:
                continue
            if p_coords.size(0) == 0:
                small_ap_scores.append(0.0)
                continue
            p_scores = hm_p[b_i, 0][p_mask]
            top_k = min(25, p_scores.size(0))
            _, top_idx = torch.topk(p_scores, k=top_k)
            p_top = p_coords[top_idx]
            dists = torch.cdist(g_peaks, p_top, p=2.0)  # (N_gt, K_pred)
            hit_2m = (dists.min(dim=1).values <= 5.0).float().mean().item()
            hit_4m = (dists.min(dim=1).values <= 8.0).float().mean().item()
            small_ap_scores.append(0.6 * hit_2m + 0.4 * hit_4m)

        # 4. Trajectory ADE & FDE in physical meters (1 grid unit = 0.5 meters)
        disp_grid = torch.norm(out["pred_traj"].float() - batch_gpu["gt_traj"].float(), dim=-1)  # (B, 6)
        disp_m = disp_grid * METERS_PER_PIXEL
        traj_ade_m.extend(disp_m.mean(dim=-1).cpu().tolist())
        traj_fde_m.extend(disp_m[:, -1].cpu().tolist())

        # 5. Obstacle Collision Rate (% of trajectories entering high-risk obstacle zone > 0.65)
        grid_norm = (out["pred_traj"].float() / 63.5) - 1.0
        sampled_risk = F.grid_sample(
            batch_gpu["risk_map"].float().unsqueeze(1),
            grid_norm.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(1).squeeze(-1)  # (B, 6)
        collided = (sampled_risk.max(dim=-1).values > 0.68).float()
        collision_flags.extend(collided.cpu().tolist())

    ml_acc = (ml_correct / max(1.0, ml_total)) * 100.0
    ml_f1 = (2.0 * ml_tp / max(1.0, 2.0 * ml_tp + ml_fp + ml_fn)) * 100.0
    ious = [cls_inter[c] / max(1.0, cls_union[c]) for c in range(3)]
    seg_miou = float(np.mean(ious)) * 100.0
    small_ap = float(np.mean(small_ap_scores)) * 100.0 if small_ap_scores else 0.0
    ade = float(np.mean(traj_ade_m)) if traj_ade_m else 0.0
    fde = float(np.mean(traj_fde_m)) if traj_fde_m else 0.0
    coll_rate = float(np.mean(collision_flags)) * 100.0 if collision_flags else 0.0

    return {
        "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "MultiLabel_Acc": round(ml_acc, 2),
        "MultiLabel_F1": round(ml_f1, 2),
        "Small_Object_AP": round(small_ap, 2),
        "Seg_mIoU": round(seg_miou, 2),
        "Traj_ADE": round(ade, 3),
        "Traj_FDE": round(fde, 3),
        "Collision_Rate": round(coll_rate, 2),
    }


def train_stage(
    model,
    variant_name: str,
    stage_name: str,
    num_epochs: int,
    base_lr: float,
    train_loader,
    val_loader,
    device,
    csv_path: Path,
    csv_fields: list,
    global_epoch_counter: list,
    best_tracker: dict,
    smoke_test: bool = False,
):
    stage_done_ckpt = CKPT_DIR / f"{variant_name}_{stage_name}_done.pt"
    if not smoke_test and stage_done_ckpt.exists():
        print(f"[RESUME] Found completed stage checkpoint {stage_done_ckpt.name}, loading and skipping {stage_name}!")
        ckpt = torch.load(stage_done_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        global_epoch_counter[0] = ckpt.get("global_epoch", global_epoch_counter[0] + num_epochs)
        m_prev = ckpt.get("metrics", {})
        if "Traj_ADE" in m_prev:
            prev_score = float(m_prev["Traj_ADE"]) - 0.05 * float(m_prev["Seg_mIoU"]) - 0.02 * float(m_prev["MultiLabel_Acc"])
            if math.isfinite(prev_score):
                best_tracker[variant_name] = min(best_tracker.get(variant_name, float("inf")), prev_score)
        return

    model.set_training_stage(stage_name)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs), eta_min=base_lr * 0.08
    )

    asl_criterion = AsymmetricLoss().to(device)
    seg_weights = torch.tensor([1.0, 2.5, 4.0], dtype=torch.float32, device=device)

    epochs_to_run = 1 if smoke_test else num_epochs
    print(f"\n==================== [{variant_name}] {stage_name} ({epochs_to_run} Epochs, lr={base_lr}) ====================")

    for s_ep in range(1, epochs_to_run + 1):
        t0 = time.time()
        global_epoch_counter[0] += 1
        g_ep = global_epoch_counter[0]
        model.train()

        ep_losses, ep_ssl, ep_percep, ep_risk, ep_traj = [], [], [], [], []

        for step_idx, batch in enumerate(train_loader):
            if smoke_test and step_idx >= 2:
                break

            batch_gpu = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(
                    batch_gpu["corrupted_img"],
                    batch_gpu["bev_lidar"],
                    batch_gpu["geo_14d"],
                )

                # 1. SSL Reconstruction Loss
                loss_ssl = (
                    F.l1_loss(out["pred_clean_img"], batch_gpu["clean_img"])
                    if model.use_ssl
                    else torch.tensor(0.0, device=device)
                )

                # 2. Multi-Task Perception Losses
                loss_ml = 0.5 * asl_criterion(out["multi_label_logits"], batch_gpu["multi_label"]) + 0.5 * F.binary_cross_entropy_with_logits(
                    out["multi_label_logits"], batch_gpu["multi_label"]
                )
                loss_seg = F.cross_entropy(
                    out["seg_logits"], batch_gpu["seg_mask"], weight=seg_weights
                ) + compute_multiclass_dice_loss(out["seg_logits"], batch_gpu["seg_mask"])

                loss_small_hm = compute_focal_heatmap_loss(out["small_hm_pred"], batch_gpu["small_hm"])
                loss_small_reg = (
                    torch.abs(out["small_reg_pred"] - batch_gpu["small_reg"]) * batch_gpu["small_mask"]
                ).sum() / batch_gpu["small_mask"].sum().clamp(min=1.0)
                loss_percep = loss_ml + loss_seg + 0.5 * loss_small_hm + 0.5 * loss_small_reg

                # 3. Risk Field & Trajectory Planning Losses
                loss_risk = F.mse_loss(out["pred_risk_map"], batch_gpu["risk_map"])
                loss_traj = F.smooth_l1_loss(out["pred_traj"], batch_gpu["gt_traj"])

                # 4. Collision Penalty & Second-order Smoothness / LoRA Weight Alignment
                loss_coll = compute_trajectory_collision_penalty(out["pred_traj"], batch_gpu["risk_map"])
                accel = (
                    out["pred_traj"][:, 2:, :]
                    - 2.0 * out["pred_traj"][:, 1:-1, :]
                    + out["pred_traj"][:, :-2, :]
                )
                loss_smooth = torch.mean(accel ** 2)
                loss_lora_w = F.mse_loss(out["dp_weights"], batch_gpu["dp_weights_gt"])

                if stage_name == "Stage1A_SSL_LP":
                    total_loss = 2.0 * loss_ssl + 1.2 * loss_percep + 0.1 * loss_traj
                elif stage_name == "Stage1B_LP_FT":
                    total_loss = 1.0 * loss_ssl + 1.8 * loss_percep + 0.2 * (loss_risk + loss_traj)
                elif stage_name == "Stage2_Fusion_DP":
                    total_loss = 0.4 * loss_ssl + 1.0 * loss_percep + 2.5 * loss_risk + 1.2 * loss_traj + 0.5 * loss_coll
                else:  # Stage3_RAG_LoRA or Ablation joint training
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
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
            optimizer.step()

            ep_losses.append(total_loss.item())
            ep_ssl.append(loss_ssl.item())
            ep_percep.append(loss_percep.item())
            ep_risk.append(loss_risk.item())
            ep_traj.append(loss_traj.item())

        scheduler.step()
        val_m = evaluate_full_metrics(model, val_loader, device, smoke_test=smoke_test)
        elapsed = time.time() - t0

        row = {
            "variant": variant_name,
            "global_epoch": g_ep,
            "stage": stage_name,
            "stage_epoch": s_ep,
            "lr": round(optimizer.param_groups[0]["lr"], 7),
            "train_loss": round(float(np.mean(ep_losses)), 4),
            "loss_ssl": round(float(np.mean(ep_ssl)), 4),
            "loss_percep": round(float(np.mean(ep_percep)), 4),
            "loss_risk": round(float(np.mean(ep_risk)), 4),
            "loss_traj": round(float(np.mean(ep_traj)), 4),
            "val_loss": round(val_m["val_loss"], 4),
            "MultiLabel_Acc": val_m["MultiLabel_Acc"],
            "MultiLabel_F1": val_m["MultiLabel_F1"],
            "Small_Object_AP": val_m["Small_Object_AP"],
            "Seg_mIoU": val_m["Seg_mIoU"],
            "Traj_ADE": val_m["Traj_ADE"],
            "Traj_FDE": val_m["Traj_FDE"],
            "Collision_Rate": val_m["Collision_Rate"],
            "elapsed_sec": round(elapsed, 2),
        }

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        payload = {
            "variant": variant_name,
            "global_epoch": g_ep,
            "stage": stage_name,
            "stage_epoch": s_ep,
            "model_state_dict": model.state_dict(),
            "metrics": row,
        }

        prefix = "smoke_" if smoke_test else ""
        latest_p = CKPT_DIR / f"{prefix}{variant_name}_latest.pt"
        best_p = CKPT_DIR / f"{prefix}{variant_name}_best.pt"
        torch.save(payload, latest_p)

        files_to_sync = [
            (csv_path, GDRIVE_LOG_DIR / csv_path.name),
            (latest_p, GDRIVE_CKPT_DIR / latest_p.name),
        ]

        if variant_name == "Full_UniPercep_DP":
            main_best = CKPT_DIR / f"{prefix}best.pt"
            lora_p = CKPT_DIR / f"{prefix}lora_adapter_weights.pt"
            torch.save(
                {
                    "lora_A": model.lora_controller.lora_A.detach().cpu(),
                    "lora_B": model.lora_controller.lora_B.detach().cpu(),
                    "global_epoch": g_ep,
                },
                lora_p,
            )
            files_to_sync.append((lora_p, GDRIVE_CKPT_DIR / lora_p.name))

        score = val_m["Traj_ADE"] - 0.05 * val_m["Seg_mIoU"] - 0.02 * val_m["MultiLabel_Acc"]
        if math.isfinite(score) and (score <= best_tracker.get(variant_name, float("inf")) or not best_p.exists()):
            best_tracker[variant_name] = score
            torch.save(payload, best_p)
            files_to_sync.append((best_p, GDRIVE_CKPT_DIR / best_p.name))
            if variant_name == "Full_UniPercep_DP":
                torch.save(payload, CKPT_DIR / f"{prefix}best.pt")
                files_to_sync.append((CKPT_DIR / f"{prefix}best.pt", GDRIVE_CKPT_DIR / f"{prefix}best.pt"))

        sync_a100_to_gdrive(files_to_sync)

        print(
            f"[{variant_name} | {stage_name}][Ep {s_ep:02d}/{epochs_to_run:02d}] "
            f"Loss={row['train_loss']:.4f} | ML_Acc={row['MultiLabel_Acc']:.1f}% | "
            f"SmallAP={row['Small_Object_AP']:.1f}% | mIoU={row['Seg_mIoU']:.1f}% | "
            f"ADE={row['Traj_ADE']:.3f}m | Coll={row['Collision_Rate']:.1f}% ({elapsed:.1f}s)"
        )

    if not smoke_test:
        torch.save(payload, stage_done_ckpt)
        sync_a100_to_gdrive([(stage_done_ckpt, GDRIVE_CKPT_DIR / stage_done_ckpt.name)])


def run_all(smoke_test: bool = False):
    for d in [CKPT_DIR, LOG_DIR, OUT_DIR, GDRIVE_CKPT_DIR, GDRIVE_LOG_DIR, GDRIVE_OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    batch_size = 64 if vram_gb >= 60.0 else 32
    print(f"[INFO] Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({vram_gb:.1f} GB VRAM) | BatchSize={batch_size} | BF16=True | SmokeTest={smoke_test}")

    train_jsonl = PROCESSED_DIR / "train_full.jsonl"
    val_jsonl = PROCESSED_DIR / "val_full.jsonl"

    # Cap per-epoch samples to 2048 (32 batches of 64) for optimal speed-accuracy balance across stages & ablations
    train_ds = UniPercepFullDataset(train_jsonl, PROCESSED_DIR, max_samples=2048 if not smoke_test else 128)
    val_ds = UniPercepFullDataset(val_jsonl, PROCESSED_DIR, max_samples=512 if not smoke_test else 128)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    csv_name = "smoke_training_metrics.csv" if smoke_test else "training_metrics.csv"
    csv_path = LOG_DIR / csv_name
    csv_fields = [
        "variant",
        "global_epoch",
        "stage",
        "stage_epoch",
        "lr",
        "train_loss",
        "loss_ssl",
        "loss_percep",
        "loss_risk",
        "loss_traj",
        "val_loss",
        "MultiLabel_Acc",
        "MultiLabel_F1",
        "Small_Object_AP",
        "Seg_mIoU",
        "Traj_ADE",
        "Traj_FDE",
        "Collision_Rate",
        "elapsed_sec",
    ]
    if not csv_path.exists() or smoke_test:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    best_tracker = {}

    # 1. Train Full Main Model across 4 Stages (10 + 20 + 25 + 15 = 70 Epochs)
    main_model = UniPercepDPNet_Full(
        pretrained_backbone=True, use_ssl=True, use_multisensor=True, use_dp_lora=True
    ).to(device)
    global_ep = [0]

    main_stages = [
        ("Stage1A_SSL_LP", 10, 1e-3),
        ("Stage1B_LP_FT", 20, 6e-4),
        ("Stage2_Fusion_DP", 25, 8e-4),
        ("Stage3_RAG_LoRA", 15, 3e-4),
    ]
    for st_name, st_epochs, st_lr in main_stages:
        if st_name == "Stage2_Fusion_DP":
            best_tracker["Full_UniPercep_DP"] = float("inf")
        train_stage(
            main_model,
            "Full_UniPercep_DP",
            st_name,
            st_epochs,
            st_lr,
            train_loader,
            val_loader,
            device,
            csv_path,
            csv_fields,
            global_ep,
            best_tracker,
            smoke_test=smoke_test,
        )

    del main_model
    torch.cuda.empty_cache()

    # 2. Run 3 Ablation Variants (15 Epochs each)
    ablation_configs = [
        ("Ablation_1_Baseline", False, False, False),
        ("Ablation_2_With_SSL_LPFT", True, False, False),
        ("Ablation_3_With_MultiSensor", True, True, False),
    ]

    for ab_name, u_ssl, u_ms, u_dp in ablation_configs:
        ab_model = UniPercepDPNet_Full(
            pretrained_backbone=True,
            use_ssl=u_ssl,
            use_multisensor=u_ms,
            use_dp_lora=u_dp,
        ).to(device)
        ab_ep = [0]
        train_stage(
            ab_model,
            ab_name,
            "Stage_Ablation_Joint",
            15,
            6e-4,
            train_loader,
            val_loader,
            device,
            csv_path,
            csv_fields,
            ab_ep,
            best_tracker,
            smoke_test=smoke_test,
        )
        del ab_model
        torch.cuda.empty_cache()

    print("\n[INFO] Main 4-stage training and all 3 ablation studies completed and synced to Google Drive!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true", help="Run 2 batches smoke test")
    args = parser.parse_args()
    run_all(smoke_test=args.smoke_test)
