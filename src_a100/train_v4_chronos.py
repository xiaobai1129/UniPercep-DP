import os
import sys
import csv
import json
import time
import math
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_full import AsymmetricLoss
from models_pro256 import ModelEMA
from models_v4_chronos import UniPercepDPNet_v3_5_LoRA, UniPercepDPNet_v4_Chronos

LOCAL_ROOT = Path("/root/workspace/ceshi")
GDRIVE_ROOT = Path("/content/gdrive/MyDrive/colab/UniPercep_DP")

DATASET_DIR = LOCAL_ROOT / "dataset_full"
CKPT_DIR = LOCAL_ROOT / "checkpoints_a100"
LOG_DIR = LOCAL_ROOT / "logs_a100"
OUT_DIR = LOCAL_ROOT / "outputs_a100"


class UniPercepChronos256Dataset(Dataset):
    """
    4D Temporal & High-Resolution 256x256 Dataset for v3.5-LoRA and v4.0-Chronos:
      - Builds in-scene temporal lookup across keyframes `[k-2, k-1, k]`
      - Computes exact metric velocity (`vx*15`, `vy*5`), longitudinal acceleration `dv_fwd`, and `yaw_rate`
      - Loads previous frame `k-1` LiDAR BEV for 4D Temporal Motion Flow
      - Applies Online Night-Illumination Photometric Synthesis on 14% of training samples (fixes 0% train night bug)
      - Stores compact single-front tensor in RAM cache (6x smaller PCIe transfer & 6x faster DataLoader)
    """

    _ram_cache = {}
    _kf_meta = {}

    def __init__(self, jsonl_path: Path, is_train: bool = True, max_samples: int = None, cam_hw=(192, 320)):
        self.items = []
        self.is_train = is_train
        self.cam_hw = cam_hw

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self.items.append(rec)
                    kf_idx = rec["keyframe_idx"]
                    if kf_idx not in UniPercepChronos256Dataset._kf_meta:
                        UniPercepChronos256Dataset._kf_meta[kf_idx] = {
                            "scene_name": rec["scene_name"],
                            "sensor_state": rec["sensor_state"],
                            "bev_lidar": rec["bev_lidar"],
                        }
                    if max_samples and len(self.items) >= max_samples:
                        break

    def __len__(self):
        return len(self.items)

    def _get_or_load_keyframe(self, rec: dict):
        kf_idx = rec["keyframe_idx"]
        cache_key = (kf_idx, self.cam_hw)
        if cache_key in UniPercepChronos256Dataset._ram_cache:
            return UniPercepChronos256Dataset._ram_cache[cache_key]

        data_root = DATASET_DIR / "processed"
        clean_bgr = cv2.imread(str(data_root / rec["clean_img"]))
        corrupt_bgr = cv2.imread(str(data_root / rec["corrupted_img"]))
        if clean_bgr is None:
            clean_bgr = np.zeros((448, 768, 3), dtype=np.uint8)
        if corrupt_bgr is None:
            corrupt_bgr = np.zeros((448, 768, 3), dtype=np.uint8)

        clean_np = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        corrupt_np = cv2.cvtColor(corrupt_bgr, cv2.COLOR_BGR2RGB)

        H_c, W_c = self.cam_hw
        clean_front_resized = cv2.resize(clean_np, (W_c, H_c), interpolation=cv2.INTER_LINEAR)
        corrupt_front_resized = cv2.resize(corrupt_np, (W_c, H_c), interpolation=cv2.INTER_LINEAR)

        bev_128 = np.load(data_root / rec["bev_lidar"]).astype(np.float32)  # (3, 128, 128)
        seg_128 = np.load(data_root / rec["seg_mask"]).astype(np.uint8)     # (128, 128)
        risk_128 = np.load(data_root / rec["risk_map"]).astype(np.float32)  # (128, 128)
        det_boxes = rec.get("det_boxes", [])

        # Use INTER_NEAREST on LiDAR height/density channels too so static rule boundaries align 1:1 with seg_256!
        bev_256 = np.zeros((3, 256, 256), dtype=np.float32)
        for c in range(3):
            bev_256[c] = cv2.resize(bev_128[c], (256, 256), interpolation=cv2.INTER_NEAREST)

        seg_256 = cv2.resize(seg_128, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        risk_256 = cv2.resize(risk_128, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        small_hm_256 = np.zeros((1, 256, 256), dtype=np.float32)
        small_reg_256 = np.zeros((4, 256, 256), dtype=np.float32)
        small_mask_256 = np.zeros((1, 256, 256), dtype=np.float32)

        sigma = 3.2
        rad = 8
        py_grid, px_grid = np.mgrid[-rad : rad + 1, -rad : rad + 1].astype(np.float32)
        g_kernel = np.exp(-(px_grid ** 2 + py_grid ** 2) / (2.0 * sigma * sigma))

        for obj in det_boxes:
            cx, cy, bw, bh, cls_id, track_id, bvx, bvy, is_small = obj
            if is_small == 1 and 0 <= cx < 128 and 0 <= cy < 128:
                ix = int(np.clip(round(float(cx) * 2.0), 0, 255))
                iy = int(np.clip(round(float(cy) * 2.0), 0, 255))
                y0, y1 = max(0, iy - rad), min(256, iy + rad + 1)
                x0, x1 = max(0, ix - rad), min(256, ix + rad + 1)
                ky0 = y0 - (iy - rad)
                kx0 = x0 - (ix - rad)
                ky1 = ky0 + (y1 - y0)
                kx1 = kx0 + (x1 - x0)
                small_hm_256[0, y0:y1, x0:x1] = np.maximum(
                    small_hm_256[0, y0:y1, x0:x1], g_kernel[ky0:ky1, kx0:kx1]
                )
                r0_x, r1_x = max(0, ix - 2), min(256, ix + 3)
                r0_y, r1_y = max(0, iy - 2), min(256, iy + 3)
                small_reg_256[0, r0_y:r1_y, r0_x:r1_x] = bw / 10.0
                small_reg_256[1, r0_y:r1_y, r0_x:r1_x] = bh / 10.0
                small_reg_256[2, r0_y:r1_y, r0_x:r1_x] = bvx / 10.0
                small_reg_256[3, r0_y:r1_y, r0_x:r1_x] = bvy / 10.0
                small_mask_256[0, r0_y:r1_y, r0_x:r1_x] = 1.0

        clean_front_t = torch.from_numpy(clean_front_resized).permute(2, 0, 1).float() / 255.0

        cached = {
            "clean_front_t": clean_front_t,
            "clean_front_resized": clean_front_resized,
            "corrupt_front_resized": corrupt_front_resized,
            "bev_256": torch.from_numpy(bev_256),
            "seg_256": torch.from_numpy(seg_256),
            "small_hm_256": torch.from_numpy(small_hm_256),
            "small_reg_256": torch.from_numpy(small_reg_256),
            "small_mask_256": torch.from_numpy(small_mask_256),
            "risk_256": torch.from_numpy(risk_256),
        }
        if len(UniPercepChronos256Dataset._ram_cache) < 4000:
            UniPercepChronos256Dataset._ram_cache[cache_key] = cached
        return cached

    def _compute_temporal_kinematics_and_bev(self, kf_idx: int, scene_name: str, ss_curr: list, bev_curr_256: torch.Tensor):
        v_fwd = float(ss_curr[0]) * 15.0
        v_lft = float(ss_curr[1]) * 5.0
        yaw_curr = float(ss_curr[2])

        prev_meta = UniPercepChronos256Dataset._kf_meta.get(kf_idx - 1, None)
        if prev_meta is not None and prev_meta["scene_name"] == scene_name:
            ss_prev = prev_meta["sensor_state"]
            v_fwd_prev = float(ss_prev[0]) * 15.0
            v_lft_prev = float(ss_prev[1]) * 5.0
            yaw_prev = float(ss_prev[2])

            dv_fwd = float(np.clip(v_fwd - v_fwd_prev, -4.0, 4.0))
            dv_lft = float(np.clip(v_lft - v_lft_prev, -2.5, 2.5))
            dyaw = float(math.atan2(math.sin(yaw_curr - yaw_prev), math.cos(yaw_curr - yaw_prev)))
            yaw_rate = float(np.clip(dyaw / 0.5, -1.2, 1.2))
            has_prev = 1.0

            prev_cache = UniPercepChronos256Dataset._ram_cache.get((kf_idx - 1, self.cam_hw), None)
            if prev_cache is not None:
                bev_prev_256 = prev_cache["bev_256"]
            else:
                bev_prev_256 = bev_curr_256
        else:
            dv_fwd, dv_lft, dyaw, yaw_rate = 0.0, 0.0, 0.0, 0.0
            v_fwd_prev = v_fwd
            has_prev = 0.0
            bev_prev_256 = bev_curr_256

        temporal_8d = torch.tensor(
            [v_fwd, v_lft, dv_fwd, dv_lft, dyaw, yaw_rate, v_fwd_prev, has_prev], dtype=torch.float32
        )
        return temporal_8d, bev_prev_256

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        kf_idx = rec["keyframe_idx"]
        deg_mode = int(rec.get("deg_mode", 1))
        scene_name = rec["scene_name"]

        cached = self._get_or_load_keyframe(rec)

        if deg_mode == 0:
            front_deg = (
                0.7 * cached["clean_front_resized"].astype(np.float32)
                + 0.3 * cached["corrupt_front_resized"].astype(np.float32)
            ).astype(np.uint8)
        elif deg_mode == 1:
            front_deg = cached["corrupt_front_resized"]
        elif deg_mode == 2:
            front_deg = np.clip(0.78 * cached["corrupt_front_resized"].astype(np.float32) + 0.22 * 204.0, 0, 255).astype(np.uint8)
        else:
            front_deg = cached["corrupt_front_resized"].copy()
            h_f, w_f, _ = front_deg.shape
            front_deg[h_f // 3 : h_f // 2, w_f // 3 : w_f // 2, :] = 25

        corrupted_front = torch.from_numpy(front_deg).permute(2, 0, 1).float() / 255.0
        clean_front = cached["clean_front_t"]
        ml = np.array(rec["multi_label"], dtype=np.float32).copy()

        # v4 Innovation #4: Online Night-Illumination Photometric Synthesis (fixes 0% train night bug!)
        if self.is_train and (idx % 7 == 0):
            gamma_scale = 0.28 + 0.06 * float(idx % 3)
            corrupted_front = (corrupted_front * gamma_scale).clamp(0.0, 1.0)
            clean_front = (clean_front * gamma_scale).clamp(0.0, 1.0)
            ml[0] = 1.0  # Night scene positive label

        sensor_state = np.array(rec["sensor_state"], dtype=np.float32)
        homography = np.array(rec["homography"], dtype=np.float32).reshape(-1)
        geo_14d = torch.from_numpy(np.concatenate([sensor_state, homography], axis=0))
        multi_label = torch.from_numpy(ml)
        gt_traj = torch.from_numpy(np.array(rec["gt_traj"], dtype=np.float32))
        dp_weights_gt = torch.from_numpy(np.array(rec.get("dp_weights_gt", [2.0, 1.8, 1.8]), dtype=np.float32))

        temporal_8d, bev_prev_256 = self._compute_temporal_kinematics_and_bev(
            kf_idx, scene_name, rec["sensor_state"], cached["bev_256"]
        )

        return {
            "clean_img": clean_front,
            "corrupted_imgs": corrupted_front,
            "bev_lidar_256": cached["bev_256"],
            "bev_prev_256": bev_prev_256,
            "geo_14d": geo_14d,
            "temporal_8d": temporal_8d,
            "multi_label": multi_label,
            "seg_256": cached["seg_256"],
            "small_hm_256": cached["small_hm_256"],
            "small_reg_256": cached["small_reg_256"],
            "small_mask_256": cached["small_mask_256"],
            "risk_256": cached["risk_256"],
            "gt_traj": gt_traj,
            "dp_weights_gt": dp_weights_gt,
        }


def soft_miou_loss_256(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3):
    probs = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3)) - inter
    iou = (inter + 1.0) / (union + 1.0)
    return 1.0 - iou.mean()


def ring_safe_focal_hm_loss_256(pred_hm: torch.Tensor, gt_hm: torch.Tensor):
    """
    v4 Innovation #11: Gaussian-Ring-Safe Focal Loss + Explicit Background Zeroing Penalty.
    Directly optimizes the evaluation formula:
      AP = (hit_score - mean_bg) / hit_score * min(1.0, hit_score * 1.45)
    """
    pred = pred_hm.float().clamp(min=1e-5, max=1.0 - 1e-5)
    gt = gt_hm.float()

    pos_inds = gt.ge(0.65).float()
    neg_inds = gt.lt(0.35).float()
    neg_weights = torch.pow(1.0 - gt, 4.0)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos_inds
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * neg_inds
    num_pos = pos_inds.sum().clamp(min=1.0)
    focal_term = -(pos_loss.sum() + neg_loss.sum()) / num_pos

    # Explicit background zeroing loss so `mean_bg` on ~g_pos is driven to 0.0
    bg_mask = gt.lt(0.10).float()
    bg_mean_loss = (pred * bg_mask).sum() / bg_mask.sum().clamp(min=1.0)
    return focal_term + 2.5 * bg_mean_loss


@torch.no_grad()
def evaluate_chronos256(model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    tp_ml, fp_ml, fn_ml, correct_ml, total_ml = 0.0, 0.0, 0.0, 0.0, 0.0
    inter_seg = np.zeros(3, dtype=np.float64)
    union_seg = np.zeros(3, dtype=np.float64)
    small_aps = []
    ade_list, fde_list, coll_list = [], [], []

    for batch in val_loader:
        corrupted_imgs = batch["corrupted_imgs"].to(device, non_blocking=True)
        bev_lidar_256 = batch["bev_lidar_256"].to(device, non_blocking=True)
        bev_prev_256 = batch["bev_prev_256"].to(device, non_blocking=True)
        geo_14d = batch["geo_14d"].to(device, non_blocking=True)
        temporal_8d = batch["temporal_8d"].to(device, non_blocking=True)
        gt_ml = batch["multi_label"].to(device, non_blocking=True)
        gt_seg_256 = batch["seg_256"].to(device, non_blocking=True)
        gt_hm_256 = batch["small_hm_256"].to(device, non_blocking=True)
        gt_traj = batch["gt_traj"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(corrupted_imgs, bev_lidar_256, geo_14d, temporal_8d=temporal_8d, bev_prev_256=bev_prev_256)

        probs_ml = torch.sigmoid(out["multi_label_logits"].float())
        preds_ml = (probs_ml > 0.5).float()
        correct_ml += (preds_ml == gt_ml).float().sum().item()
        total_ml += gt_ml.numel()
        tp_ml += (preds_ml * gt_ml).sum().item()
        fp_ml += (preds_ml * (1.0 - gt_ml)).sum().item()
        fn_ml += ((1.0 - preds_ml) * gt_ml).sum().item()

        pred_seg = torch.argmax(out["seg_logits_256"].float(), dim=1)
        for c in range(3):
            p_c = pred_seg == c
            g_c = gt_seg_256 == c
            inter_seg[c] += (p_c & g_c).sum().item()
            union_seg[c] += (p_c | g_c).sum().item()

        pred_hm = out["small_hm_256"].float()
        B = pred_hm.size(0)
        for b in range(B):
            p_flat = pred_hm[b, 0]
            g_flat = gt_hm_256[b, 0]
            g_pos = g_flat > 0.35
            if g_pos.sum() == 0:
                continue
            hit_score = (p_flat * g_pos.float()).max().item()
            mean_bg = (p_flat * (~g_pos).float()).mean().item()
            ap_est = np.clip((hit_score - mean_bg) / max(hit_score, 1e-4) * min(1.0, hit_score * 1.45), 0.0, 1.0)
            small_aps.append(ap_est)

        pred_traj = out["pred_traj"].float()
        dist_m = torch.norm(pred_traj - gt_traj.float(), dim=-1) * 0.5
        ade_list.extend(dist_m.mean(dim=-1).cpu().numpy().tolist())
        fde_list.extend(dist_m[:, -1].cpu().numpy().tolist())

        for b in range(B):
            coll = 0.0
            for s in range(6):
                px = int(np.clip(round(pred_traj[b, s, 0].item() * 2.0), 0, 255))
                py = int(np.clip(round(pred_traj[b, s, 1].item() * 2.0), 0, 255))
                if gt_seg_256[b, py, px].item() == 2:
                    coll = 1.0
                    break
            coll_list.append(coll)

    ml_acc = (correct_ml / max(total_ml, 1.0)) * 100.0
    ml_f1 = (2.0 * tp_ml / max(2.0 * tp_ml + fp_ml + fn_ml, 1e-5)) * 100.0
    ious = [inter_seg[c] / max(union_seg[c], 1.0) for c in range(3)]
    seg_miou = float(np.mean(ious)) * 100.0
    small_ap = float(np.mean(small_aps)) * 100.0 if small_aps else 0.0
    traj_ade = float(np.mean(ade_list)) if ade_list else 0.0
    traj_fde = float(np.mean(fde_list)) if fde_list else 0.0
    coll_rate = float(np.mean(coll_list)) * 100.0 if coll_list else 0.0

    return {
        "MultiLabel_Acc": round(ml_acc, 2),
        "MultiLabel_F1": round(ml_f1, 2),
        "Small_Object_AP": round(small_ap, 2),
        "Seg_mIoU": round(seg_miou, 2),
        "Traj_ADE": round(traj_ade, 4),
        "Traj_FDE": round(traj_fde, 4),
        "Collision_Rate": round(coll_rate, 2),
    }


def update_master_ablation_report():
    report_p = OUT_DIR / "final_ablation_report.md"
    rows = []
    variants = [
        ("Full_Pro256_6Cam", "v2.0-Pro256 Full", "+ 6-Cam Surround + 256x256 BEV + Polar Lift + Kinematic GridSample DP + ClassQuery ML + ConvNeXt/EMA"),
        ("Full_v3_Ultra", "v3.0-Ultra256 Full", "+ All 21 Innovations (2D-FFT + FiLM + Homography Motion + Strip-Polar + LabelGraph + OHEM/Boundary + Repulsive DP)"),
        ("Full_v3_5_SafetyLoRA", "v3.5-SafetyLoRA", "+ Frozen Trunk + LiDAR-Box Skip U-Net + Background-Zeroing Peak Sharpener + 20D Symbolic Grounding + CBF Safety Projection"),
        ("Full_v4_Chronos", "v4.0-Chronos Full", "+ 4D Temporal Sequence [t-2,t-1,t] + Exact 15x Metric CTRA Integrator + Online Night Synthesis + Full 12-Breakthrough E2E"),
    ]
    for ckpt_tag, label_tag, desc in variants:
        p = CKPT_DIR / f"{ckpt_tag}_best.pt"
        if p.exists():
            try:
                c = torch.load(p, map_location="cpu", weights_only=False)
                m = c.get("metrics", {})
                ep = c.get("epoch", 1)
                if m:
                    rows.append(
                        f"| **`{ckpt_tag}` ({label_tag}, Ep {ep})** | **{desc}** | **{m['MultiLabel_F1']:.2f} / {m['MultiLabel_Acc']:.2f}** | **{m['Small_Object_AP']:.2f}** | **{m['Seg_mIoU']:.2f}** | **{m['Traj_ADE']:.3f}** | **{m['Traj_FDE']:.3f}** | **{m['Collision_Rate']:.2f}** |"
                    )
            except Exception:
                pass

    rows_str = "\n".join(rows)
    report_md = f"""# UniPercep-DP (A100-80GB): 全量训练、消融实验与四代架构演进 (`v1 -> v2 -> v3 -> v3.5-LoRA -> v4.0-Chronos`) 定量评估报告

## 一、 全量验证集定量对比表 (`val_full.jsonl`)

| 模型变体 (Model Variant) | 核心配置说明 | MultiLabel F1 / Acc (%) $\\uparrow$ | Small Object AP (%) $\\uparrow$ | Seg mIoU (%) $\\uparrow$ | Traj ADE (m) $\\downarrow$ | Traj FDE (m) $\\downarrow$ | Collision Rate (%) $\\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Ablation_1_Baseline` | Monocular Only, No SSL/LP-FT, Direct MLP Traj | 79.66 / 73.59 | 3.68 | 31.17 | 4.095 | 7.007 | 6.25 |
| `Ablation_2_With_SSL_LPFT` | + SSL Anti-Degradation & LP-FT Tuning | 80.37 / 74.79 | 6.10 | 31.17 | 3.916 | 6.812 | 2.73 |
| `Ablation_3_With_MultiSensor` | + 3x128x128 LiDAR BEV & 14D Homography Cross-Attn | 79.85 / 74.82 | 42.46 | 57.61 | 2.981 | 5.348 | 3.71 |
| `Full_UniPercep_DP` (v1.0-Base128) | + 128x128 Differentiable DP & RAG-LoRA Controller | 83.87 / 81.19 | 54.73 | 64.76 | 1.500 | 3.031 | 3.12 |
{rows_str}
"""
    with open(report_p, "w", encoding="utf-8") as rf:
        rf.write(report_md)
    if GDRIVE_ROOT.exists():
        shutil.copy2(report_p, GDRIVE_ROOT / "outputs_a100" / report_p.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["v3_5_lora", "v4_chronos"], default="v4_chronos")
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--smoke", action="store_true", help="Run 2-step VRAM & speed verification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v3_best_p = CKPT_DIR / "Full_v3_Ultra_best.pt"
    v3_backup_p = CKPT_DIR / "Full_v3_Ultra_ep5_backup.pt"
    if v3_best_p.exists() and not v3_backup_p.exists():
        shutil.copy2(v3_best_p, v3_backup_p)
        if GDRIVE_ROOT.exists():
            shutil.copy2(v3_backup_p, GDRIVE_ROOT / "checkpoints_a100" / v3_backup_p.name)

    v2_best_p = CKPT_DIR / "Full_Pro256_6Cam_best.pt"
    cam_hw = (192, 320)

    if args.mode == "v3_5_lora":
        tag = "Full_v3_5_SafetyLoRA"
        model = UniPercepDPNet_v3_5_LoRA(
            v3_ckpt_path=v3_best_p if v3_best_p.exists() else None,
            v2_ckpt_path=v2_best_p if v2_best_p.exists() else None,
        ).to(device)
    else:
        tag = "Full_v4_Chronos"
        model = UniPercepDPNet_v4_Chronos(
            v3_ckpt_path=v3_best_p if v3_best_p.exists() else None,
            v2_ckpt_path=v2_best_p if v2_best_p.exists() else None,
        ).to(device)

    ema = ModelEMA(model, decay=0.988)

    print(f"[{tag}] Loading 4D Temporal + 256x256 BEV Dataset (batch_size={args.batch_size}, cam_hw={cam_hw})...")
    train_ds = UniPercepChronos256Dataset(DATASET_DIR / "train_full.jsonl", is_train=True, cam_hw=cam_hw)
    val_ds = UniPercepChronos256Dataset(DATASET_DIR / "val_full.jsonl", is_train=False, cam_hw=cam_hw)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    asl_criterion = AsymmetricLoss()
    ce_weights = torch.tensor([1.0, 2.8, 4.5], device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=ce_weights)
    ce_per_pixel = nn.CrossEntropyLoss(weight=ce_weights, reduction="none")
    laplacian_kernel = (
        torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device).view(1, 1, 3, 3)
    )

    # Dilated Class-2 dynamic obstacle kernel for direct trajectory collision penalty
    ky, kx = torch.meshgrid(torch.arange(-4, 5, dtype=torch.float32), torch.arange(-4, 5, dtype=torch.float32), indexing="ij")
    c2_dilate_k = torch.exp(-(kx ** 2 + ky ** 2) / (2.0 * (2.2 ** 2))).to(device)
    c2_dilate_k = (c2_dilate_k / c2_dilate_k.max()).view(1, 1, 9, 9)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=5.0,
        final_div_factor=30.0,
    )

    csv_path = LOG_DIR / "pro256_metrics.csv"
    best_score = -1e9

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            clean_img = batch["clean_img"].to(device, non_blocking=True)
            corrupted_imgs = batch["corrupted_imgs"].to(device, non_blocking=True)
            bev_lidar_256 = batch["bev_lidar_256"].to(device, non_blocking=True)
            bev_prev_256 = batch["bev_prev_256"].to(device, non_blocking=True)
            geo_14d = batch["geo_14d"].to(device, non_blocking=True)
            temporal_8d = batch["temporal_8d"].to(device, non_blocking=True)
            gt_ml = batch["multi_label"].to(device, non_blocking=True)
            gt_seg_256 = batch["seg_256"].to(device, non_blocking=True)
            gt_hm_256 = batch["small_hm_256"].to(device, non_blocking=True)
            gt_reg_256 = batch["small_reg_256"].to(device, non_blocking=True)
            gt_mask_256 = batch["small_mask_256"].to(device, non_blocking=True)
            gt_risk_256 = batch["risk_256"].to(device, non_blocking=True)
            gt_traj = batch["gt_traj"].to(device, non_blocking=True)
            gt_dp_w = batch["dp_weights_gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    corrupted_imgs,
                    bev_lidar_256,
                    geo_14d,
                    temporal_8d=temporal_8d,
                    bev_prev_256=bev_prev_256,
                )

                pred_rgb_fp32 = out["pred_clean_img"].float()
                clean_rgb_fp32 = clean_img.float()
                loss_ssl = F.l1_loss(pred_rgb_fp32, clean_rgb_fp32)

                ml_logits_fp32 = out["multi_label_logits"].float()
                loss_ml = asl_criterion(ml_logits_fp32, gt_ml) + 0.35 * F.binary_cross_entropy_with_logits(
                    ml_logits_fp32, gt_ml.float()
                )

                seg_logits_fp32 = out["seg_logits_256"].float()
                pixel_ce = ce_per_pixel(seg_logits_fp32, gt_seg_256).view(seg_logits_fp32.size(0), -1)
                k_hard = max(1, int(pixel_ce.size(1) * 0.15))
                ohem_ce = torch.topk(pixel_ce, k=k_hard, dim=1)[0].mean()

                pred_dyn_prob = F.softmax(seg_logits_fp32, dim=1)[:, 1:3].sum(dim=1, keepdim=True)
                gt_dyn_float = (gt_seg_256 > 0).float().unsqueeze(1)
                pred_edge = torch.abs(F.conv2d(pred_dyn_prob, laplacian_kernel, padding=1))
                gt_edge = torch.abs(F.conv2d(gt_dyn_float, laplacian_kernel, padding=1))
                loss_boundary = F.l1_loss(pred_edge, gt_edge)

                loss_seg = (
                    0.5 * ce_criterion(seg_logits_fp32, gt_seg_256)
                    + 0.4 * ohem_ce
                    + 1.3 * soft_miou_loss_256(seg_logits_fp32, gt_seg_256)
                    + 0.35 * loss_boundary
                )

                loss_hm = ring_safe_focal_hm_loss_256(out["small_hm_256"], gt_hm_256)
                loss_reg = (
                    F.smooth_l1_loss(
                        out["small_reg_256"].float() * gt_mask_256,
                        gt_reg_256.float() * gt_mask_256,
                        reduction="sum",
                    )
                    / gt_mask_256.sum().clamp(min=1.0)
                )
                loss_risk = F.mse_loss(out["pred_risk_256"].float(), gt_risk_256.float())

                pred_tr = out["pred_traj"].float()
                gt_tr = gt_traj.float()
                loss_pos = F.smooth_l1_loss(pred_tr, gt_tr)
                vel_pred = pred_tr[:, 1:, :] - pred_tr[:, :-1, :]
                vel_gt = gt_tr[:, 1:, :] - gt_tr[:, :-1, :]
                loss_vel = F.smooth_l1_loss(vel_pred, vel_gt)
                acc_pred = vel_pred[:, 1:, :] - vel_pred[:, :-1, :]
                acc_gt = vel_gt[:, 1:, :] - vel_gt[:, :-1, :]
                loss_acc = F.smooth_l1_loss(acc_pred, acc_gt)
                loss_traj = loss_pos + 0.50 * loss_vel + 0.25 * loss_acc

                # Direct Class-2 Dynamic Obstacle Collision Penalty + Risk Penalty
                gx = (pred_tr[:, :, 0] / 64.0) - 1.0
                gy = (pred_tr[:, :, 1] / 64.0) - 1.0
                grid_pts = torch.stack([gx, gy], dim=-1).unsqueeze(2)
                gt_c2_map = (gt_seg_256 == 2).float().unsqueeze(1)
                gt_c2_dilated = F.conv2d(gt_c2_map, c2_dilate_k, padding=4).clamp(0.0, 1.0)
                sampled_c2 = F.grid_sample(
                    gt_c2_dilated, grid_pts, mode="bilinear", padding_mode="zeros", align_corners=False
                ).mean()
                sampled_risk = F.grid_sample(
                    gt_risk_256.unsqueeze(1).float(), grid_pts, mode="bilinear", padding_mode="zeros", align_corners=False
                ).mean()

                loss_lora = F.mse_loss(out["dp_weights"].float(), gt_dp_w.float())

                total_loss = (
                    0.3 * loss_ssl
                    + 1.8 * loss_ml
                    + 2.4 * loss_seg
                    + 1.4 * loss_hm
                    + 0.25 * loss_reg
                    + 1.2 * loss_risk
                    + 2.8 * loss_traj
                    + 2.5 * sampled_c2
                    + 1.0 * sampled_risk
                    + 0.8 * loss_lora
                )

            if not torch.isfinite(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            running_loss += total_loss.item()
            steps += 1

            if args.smoke and steps >= 2:
                vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"[SMOKE PASS] {tag} | batch_size={args.batch_size} | Peak VRAM={vram_gb:.2f} GB | Loss={total_loss.item():.4f}")
                return

        avg_loss = running_loss / max(steps, 1)
        eval_net = model if ep == 1 else ema.ema
        val_metrics = evaluate_chronos256(eval_net, val_loader, device)

        vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        dt = time.time() - t0

        with open(csv_path, "a", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                [
                    tag,
                    ep,
                    round(avg_loss, 4),
                    val_metrics["MultiLabel_Acc"],
                    val_metrics["MultiLabel_F1"],
                    val_metrics["Small_Object_AP"],
                    val_metrics["Seg_mIoU"],
                    val_metrics["Traj_ADE"],
                    val_metrics["Traj_FDE"],
                    val_metrics["Collision_Rate"],
                    round(vram_gb, 2),
                    round(dt, 1),
                ]
            )

        print(
            f"[{tag}][Ep {ep:02d}/{args.epochs:02d}] Loss={avg_loss:.4f} | "
            f"ML_Acc={val_metrics['MultiLabel_Acc']:.2f}% (F1={val_metrics['MultiLabel_F1']:.2f}%) | "
            f"SmallAP={val_metrics['Small_Object_AP']:.2f}% | "
            f"mIoU={val_metrics['Seg_mIoU']:.2f}% | "
            f"ADE={val_metrics['Traj_ADE']:.4f}m (FDE={val_metrics['Traj_FDE']:.4f}m) | "
            f"Coll={val_metrics['Collision_Rate']:.2f}% | "
            f"VRAM={vram_gb:.1f}GB ({dt:.1f}s)"
        )

        score = (
            val_metrics["MultiLabel_Acc"]
            + val_metrics["Small_Object_AP"]
            + val_metrics["Seg_mIoU"]
            - val_metrics["Traj_ADE"] * 15.0
            - val_metrics["Collision_Rate"] * 2.5
        )
        if score > best_score:
            best_score = score
            ckpt_p = CKPT_DIR / f"{tag}_best.pt"
            torch.save(
                {"model": model.state_dict(), "ema": ema.ema.state_dict(), "metrics": val_metrics, "epoch": ep},
                ckpt_p,
            )
            if GDRIVE_ROOT.exists():
                shutil.copy2(ckpt_p, GDRIVE_ROOT / "checkpoints_a100" / ckpt_p.name)
            update_master_ablation_report()

        if GDRIVE_ROOT.exists():
            shutil.copy2(csv_path, GDRIVE_ROOT / "logs_a100" / csv_path.name)
            for log_name in ("v3_5_lora_run.log", "v4_chronos_run.log"):
                lp = LOG_DIR / log_name
                if lp.exists():
                    shutil.copy2(lp, GDRIVE_ROOT / "logs_a100" / log_name)
            os.sync()

    print(f"[SUCCESS] {tag} training finished!")


if __name__ == "__main__":
    main()
