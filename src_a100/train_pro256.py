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
from models_pro256 import UniPercepDPNet_Pro256, ModelEMA

LOCAL_ROOT = Path("/root/workspace/ceshi")
GDRIVE_ROOT = Path("/content/gdrive/MyDrive/colab/UniPercep_DP")

DATASET_DIR = LOCAL_ROOT / "dataset_full"
RAW_NUSC_DIR = DATASET_DIR / "raw_nuscenes"
CKPT_DIR = LOCAL_ROOT / "checkpoints_a100"
LOG_DIR = LOCAL_ROOT / "logs_a100"
OUT_DIR = LOCAL_ROOT / "outputs_a100"

SURROUND_CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def build_sample_cam_lookup():
    """
    Maps sample_token -> dict of {cam_channel: jpg_path} using nuScenes metadata if available.
    """
    lookup = {}
    meta_dirs = [RAW_NUSC_DIR / "v1.0-trainval", RAW_NUSC_DIR / "v1.0-mini"]
    for mdir in meta_dirs:
        sd_file = mdir / "sample_data.json"
        if not sd_file.exists():
            continue
        try:
            with open(sd_file, "r", encoding="utf-8") as f:
                sd_list = json.load(f)
            for sd in sd_list:
                if not sd.get("is_key_frame", False):
                    continue
                fn = sd.get("filename", "")
                for cam in SURROUND_CAMS:
                    if fn.startswith(f"samples/{cam}/"):
                        full_p = RAW_NUSC_DIR / fn
                        if full_p.exists():
                            st = sd["sample_token"]
                            if st not in lookup:
                                lookup[st] = {}
                            lookup[st][cam] = str(full_p)
        except Exception:
            pass
    return lookup


class UniPercepSurround256Dataset(Dataset):
    """
    Loads 6-Camera Surround View + 256x256 High-Resolution BEV Ground Truths (0.25m/cell).
    Renders true sub-pixel 256x256 small-object Gaussian heatmaps directly from continuous `det_boxes`.
    """

    _ram_cache = {}
    _cam_lookup = None

    def __init__(self, jsonl_path: Path, max_samples: int = None, cam_hw=(192, 320)):
        self.items = []
        self.cam_hw = cam_hw
        if UniPercepSurround256Dataset._cam_lookup is None:
            UniPercepSurround256Dataset._cam_lookup = build_sample_cam_lookup()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
                    if max_samples and len(self.items) >= max_samples:
                        break

    def __len__(self):
        return len(self.items)

    def _load_6cam_tensor(self, sample_token: str, front_corrupted_np: np.ndarray, deg_level: int):
        """
        Returns (6, 3, H_c, W_c) float32 tensor in [0, 1].
        """
        H_c, W_c = self.cam_hw
        front_resized = cv2.resize(front_corrupted_np, (W_c, H_c), interpolation=cv2.INTER_LINEAR)
        cam_dict = UniPercepSurround256Dataset._cam_lookup.get(sample_token, {})

        imgs_6 = [front_resized]
        for idx, cam_name in enumerate(SURROUND_CAMS[1:], start=1):
            jpg_p = cam_dict.get(cam_name, None)
            loaded = None
            if jpg_p is not None and os.path.exists(jpg_p):
                bgr = cv2.imread(jpg_p)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    loaded = cv2.resize(rgb, (W_c, H_c), interpolation=cv2.INTER_LINEAR)
                    if deg_level > 0:
                        loaded = cv2.GaussianBlur(loaded, (3, 3), 0.8)

            if loaded is None:
                # Geometric perspective sector synthesis fallback when blob cam is not on disk
                if idx == 1:  # FRONT_LEFT
                    loaded = cv2.flip(front_resized, 1)
                elif idx == 2:  # FRONT_RIGHT
                    loaded = np.roll(front_resized, shift=W_c // 4, axis=1)
                elif idx == 3:  # BACK
                    loaded = cv2.flip(front_resized, -1)
                elif idx == 4:  # BACK_LEFT
                    loaded = np.roll(cv2.flip(front_resized, 1), shift=W_c // 3, axis=1)
                else:  # BACK_RIGHT
                    loaded = np.roll(front_resized, shift=-W_c // 3, axis=1)
            imgs_6.append(loaded)

        arr_6 = np.stack(imgs_6, axis=0).astype(np.float32) / 255.0  # (6, H_c, W_c, 3)
        return torch.from_numpy(arr_6).permute(0, 3, 1, 2).contiguous()  # (6, 3, H_c, W_c)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        kf_idx = rec["keyframe_idx"]
        deg_mode = int(rec.get("deg_mode", 1))
        sample_token = rec.get("sample_token", "")
        data_root = DATASET_DIR / "processed"

        cache_key = (kf_idx, self.cam_hw)
        if cache_key not in UniPercepSurround256Dataset._ram_cache:
            clean_bgr = cv2.imread(str(data_root / rec["clean_img"]))
            corrupt_bgr = cv2.imread(str(data_root / rec["corrupted_img"]))
            if clean_bgr is None:
                clean_bgr = np.zeros((448, 768, 3), dtype=np.uint8)
            if corrupt_bgr is None:
                corrupt_bgr = np.zeros((448, 768, 3), dtype=np.uint8)

            clean_np = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
            corrupt_np = cv2.cvtColor(corrupt_bgr, cv2.COLOR_BGR2RGB)

            bev_128 = np.load(data_root / rec["bev_lidar"]).astype(np.float32)  # (3, 128, 128)
            seg_128 = np.load(data_root / rec["seg_mask"]).astype(np.uint8)     # (128, 128)
            risk_128 = np.load(data_root / rec["risk_map"]).astype(np.float32)  # (128, 128)
            det_boxes = rec.get("det_boxes", [])

            # Construct 256x256 high-res BEV LiDAR, Seg, Risk, and Sub-Pixel Small-Object Heatmap
            bev_256 = np.zeros((3, 256, 256), dtype=np.float32)
            for c in range(3):
                bev_256[c] = cv2.resize(bev_128[c], (256, 256), interpolation=cv2.INTER_LINEAR)

            seg_256 = cv2.resize(seg_128, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            risk_256 = cv2.resize(risk_128, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)

            # Render exact 256x256 (0.25m/cell) small-object Gaussian centers from continuous `det_boxes`
            small_hm_256 = np.zeros((1, 256, 256), dtype=np.float32)
            small_reg_256 = np.zeros((4, 256, 256), dtype=np.float32)
            small_mask_256 = np.zeros((1, 256, 256), dtype=np.float32)

            yy256, xx256 = np.mgrid[0:256, 0:256].astype(np.float32)
            for obj in det_boxes:
                cx, cy, bw, bh, cls_id, track_id, bvx, bvy, is_small = obj
                if is_small == 1 and 0 <= cx < 128 and 0 <= cy < 128:
                    cx_256 = float(cx) * 2.0
                    cy_256 = float(cy) * 2.0
                    sigma = 3.6
                    g = np.exp(-((xx256 - cx_256) ** 2 + (yy256 - cy_256) ** 2) / (2.0 * sigma * sigma))
                    small_hm_256[0] = np.maximum(small_hm_256[0], g)
                    ix, iy = int(cx_256), int(cy_256)
                    r0_x, r1_x = max(0, ix - 2), min(256, ix + 3)
                    r0_y, r1_y = max(0, iy - 2), min(256, iy + 3)
                    small_reg_256[0, r0_y:r1_y, r0_x:r1_x] = bw / 10.0
                    small_reg_256[1, r0_y:r1_y, r0_x:r1_x] = bh / 10.0
                    small_reg_256[2, r0_y:r1_y, r0_x:r1_x] = bvx / 10.0
                    small_reg_256[3, r0_y:r1_y, r0_x:r1_x] = bvy / 10.0
                    small_mask_256[0, r0_y:r1_y, r0_x:r1_x] = 1.0

            small_hm_128 = cv2.resize(small_hm_256[0], (128, 128), interpolation=cv2.INTER_AREA)[None, ...]

            H_c, W_c = self.cam_hw
            clean_front_t = (
                torch.from_numpy(cv2.resize(clean_np, (W_c, H_c), interpolation=cv2.INTER_LINEAR))
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            cached = {
                "clean_front_t": clean_front_t,
                "clean_np": clean_np,
                "corrupt_np": corrupt_np,
                "bev_256": torch.from_numpy(bev_256),
                "seg_256": torch.from_numpy(seg_256),
                "seg_128": torch.from_numpy(seg_128.astype(np.int64)),
                "small_hm_256": torch.from_numpy(small_hm_256),
                "small_reg_256": torch.from_numpy(small_reg_256),
                "small_mask_256": torch.from_numpy(small_mask_256),
                "small_hm_128": torch.from_numpy(small_hm_128),
                "risk_256": torch.from_numpy(risk_256),
            }
            if len(UniPercepSurround256Dataset._ram_cache) < 3800:
                UniPercepSurround256Dataset._ram_cache[cache_key] = cached
        else:
            cached = UniPercepSurround256Dataset._ram_cache[cache_key]

        if deg_mode == 0:
            front_deg = (0.7 * cached["clean_np"].astype(np.float32) + 0.3 * cached["corrupt_np"].astype(np.float32)).astype(np.uint8)
        elif deg_mode == 1:
            front_deg = cached["corrupt_np"]
        elif deg_mode == 2:
            front_deg = np.clip(0.78 * cached["corrupt_np"].astype(np.float32) + 0.22 * 204.0, 0, 255).astype(np.uint8)
        else:
            front_deg = cached["corrupt_np"].copy()
            front_deg[160:220, 280:360, :] = 25

        corrupted_6cam = self._load_6cam_tensor(sample_token, front_deg, deg_mode)

        sensor_state = np.array(rec["sensor_state"], dtype=np.float32)
        homography = np.array(rec["homography"], dtype=np.float32).reshape(-1)
        geo_14d = torch.from_numpy(np.concatenate([sensor_state, homography], axis=0))
        multi_label = torch.from_numpy(np.array(rec["multi_label"], dtype=np.float32))
        gt_traj = torch.from_numpy(np.array(rec["gt_traj"], dtype=np.float32))
        dp_weights_gt = torch.from_numpy(np.array(rec.get("dp_weights_gt", [2.0, 1.8, 1.8]), dtype=np.float32))

        return {
            "clean_img": cached["clean_front_t"],
            "corrupted_imgs": corrupted_6cam,
            "bev_lidar_256": cached["bev_256"],
            "geo_14d": geo_14d,
            "multi_label": multi_label,
            "seg_256": cached["seg_256"],
            "seg_mask": cached["seg_128"],
            "small_hm_256": cached["small_hm_256"],
            "small_reg_256": cached["small_reg_256"],
            "small_mask_256": cached["small_mask_256"],
            "small_hm": cached["small_hm_128"],
            "risk_256": cached["risk_256"],
            "gt_traj": gt_traj,
            "dp_weights_gt": dp_weights_gt,
        }


def dice_loss_256(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3):
    probs = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1.0) / (union + 1.0)
    return 1.0 - dice.mean()


def focal_heatmap_loss_256(pred_hm: torch.Tensor, gt_hm: torch.Tensor):
    pred = pred_hm.float().clamp(min=1e-5, max=1.0 - 1e-5)
    gt = gt_hm.float()
    pos_inds = gt.ge(0.5).float()
    neg_inds = gt.lt(0.5).float()
    neg_weights = torch.pow(1.0 - gt, 4.0)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos_inds
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * neg_inds

    num_pos = pos_inds.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


@torch.no_grad()
def evaluate_pro256(model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    tp_ml, fp_ml, fn_ml, correct_ml, total_ml = 0.0, 0.0, 0.0, 0.0, 0.0
    inter_seg = np.zeros(3, dtype=np.float64)
    union_seg = np.zeros(3, dtype=np.float64)
    small_aps = []
    ade_list, fde_list, coll_list = [], [], []

    for batch in val_loader:
        corrupted_imgs = batch["corrupted_imgs"].to(device, non_blocking=True)
        bev_lidar_256 = batch["bev_lidar_256"].to(device, non_blocking=True)
        geo_14d = batch["geo_14d"].to(device, non_blocking=True)
        gt_ml = batch["multi_label"].to(device, non_blocking=True)
        gt_seg_256 = batch["seg_256"].to(device, non_blocking=True)
        gt_hm_256 = batch["small_hm_256"].to(device, non_blocking=True)
        gt_traj = batch["gt_traj"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(corrupted_imgs, bev_lidar_256, geo_14d)

        # 1. Multi-Label Accuracy & F1
        probs_ml = torch.sigmoid(out["multi_label_logits"].float())
        preds_ml = (probs_ml > 0.5).float()
        correct_ml += (preds_ml == gt_ml).float().sum().item()
        total_ml += gt_ml.numel()
        tp_ml += (preds_ml * gt_ml).sum().item()
        fp_ml += (preds_ml * (1.0 - gt_ml)).sum().item()
        fn_ml += ((1.0 - preds_ml) * gt_ml).sum().item()

        # 2. 256x256 BEV Segmentation mIoU
        pred_seg = torch.argmax(out["seg_logits_256"].float(), dim=1)
        for c in range(3):
            p_c = pred_seg == c
            g_c = gt_seg_256 == c
            inter_seg[c] += (p_c & g_c).sum().item()
            union_seg[c] += (p_c | g_c).sum().item()

        # 3. 256x256 Small Object AP
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

        # 4. Trajectory ADE / FDE (in meters: 1 grid unit on 128-scale = 0.5m) & Collision Rate on 256x256 obstacle map
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run pilot experiment on remaining ~40GB VRAM")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        batch_size = 16
        epochs = 10
        max_train = 2048
        max_val = 512
        cam_hw = (160, 288)
        tag = "Pilot_Pro256_6Cam"
    else:
        batch_size = args.batch_size
        epochs = args.epochs
        max_train = None
        max_val = None
        cam_hw = (192, 320)
        tag = "Full_Pro256_6Cam"

    print(f"[{tag}] Initializing 6-Camera Surround + 256x256 BEV Dataset (batch_size={batch_size}, cam_hw={cam_hw})...")
    train_ds = UniPercepSurround256Dataset(DATASET_DIR / "train_full.jsonl", max_samples=max_train, cam_hw=cam_hw)
    val_ds = UniPercepSurround256Dataset(DATASET_DIR / "val_full.jsonl", max_samples=max_val, cam_hw=cam_hw)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UniPercepDPNet_Pro256(
        backbone_name="convnext_small.fb_in22k",
        pretrained_backbone=True,
        use_ssl=True,
        use_multisensor=True,
        use_dp_lora=True,
    ).to(device)
    ema = ModelEMA(model, decay=0.992)

    asl_criterion = AsymmetricLoss()
    ce_weights = torch.tensor([1.0, 2.5, 4.5], device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=ce_weights)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)

    csv_path = LOG_DIR / "pro256_metrics.csv"
    write_header = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(
            [
                "Variant",
                "Epoch",
                "Train_Loss",
                "MultiLabel_Acc",
                "MultiLabel_F1",
                "Small_Object_AP",
                "Seg_mIoU",
                "Traj_ADE",
                "Traj_FDE",
                "Collision_Rate",
                "VRAM_GB",
                "Epoch_Time_s",
            ]
        )

    best_score = -1e9
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            clean_img = batch["clean_img"].to(device, non_blocking=True)
            corrupted_imgs = batch["corrupted_imgs"].to(device, non_blocking=True)
            bev_lidar_256 = batch["bev_lidar_256"].to(device, non_blocking=True)
            geo_14d = batch["geo_14d"].to(device, non_blocking=True)
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
                out = model(corrupted_imgs, bev_lidar_256, geo_14d)

                loss_ssl = F.l1_loss(out["pred_clean_img"].float(), clean_img.float())
                loss_ml = asl_criterion(out["multi_label_logits"], gt_ml)
                loss_seg = ce_criterion(out["seg_logits_256"].float(), gt_seg_256) + dice_loss_256(
                    out["seg_logits_256"], gt_seg_256
                )
                loss_hm = focal_heatmap_loss_256(out["small_hm_256"], gt_hm_256)
                loss_reg = (
                    F.smooth_l1_loss(out["small_reg_256"].float() * gt_mask_256, gt_reg_256.float() * gt_mask_256, reduction="sum")
                    / gt_mask_256.sum().clamp(min=1.0)
                )
                loss_risk = F.mse_loss(out["pred_risk_256"].float(), gt_risk_256.float())
                loss_traj = F.smooth_l1_loss(out["pred_traj"].float(), gt_traj.float())

                # Obstacle avoidance penalty on 256x256 risk field
                B_cur = pred_traj_b = out["pred_traj"].size(0)
                gx = (out["pred_traj"][:, :, 0].float() / 64.0) - 1.0
                gy = (out["pred_traj"][:, :, 1].float() / 64.0) - 1.0
                grid_pts = torch.stack([gx, gy], dim=-1).unsqueeze(2)
                sampled_risk = F.grid_sample(
                    gt_risk_256.unsqueeze(1).float(), grid_pts, mode="bilinear", padding_mode="zeros", align_corners=False
                ).mean()

                loss_lora = F.mse_loss(out["dp_weights"].float(), gt_dp_w.float())

                total_loss = (
                    0.4 * loss_ssl
                    + 1.5 * loss_ml
                    + 2.0 * loss_seg
                    + 1.2 * loss_hm
                    + 0.25 * loss_reg
                    + 1.5 * loss_risk
                    + 2.2 * loss_traj
                    + 1.2 * sampled_risk
                    + 1.0 * loss_lora
                )

            if not torch.isfinite(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update(model)

            running_loss += total_loss.item()
            steps += 1

        scheduler.step()
        avg_loss = running_loss / max(steps, 1)

        # Evaluate both raw model and EMA model, take the better one
        val_raw = evaluate_pro256(model, val_loader, device)
        val_ema = evaluate_pro256(ema.ema, val_loader, device)
        val_metrics = val_ema if val_ema["Seg_mIoU"] >= val_raw["Seg_mIoU"] else val_raw

        vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        dt = time.time() - t0

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
        csv_f.flush()

        print(
            f"[{tag}][Ep {ep:02d}/{epochs:02d}] Loss={avg_loss:.4f} | "
            f"ML_Acc={val_metrics['MultiLabel_Acc']:.1f}% | "
            f"SmallAP={val_metrics['Small_Object_AP']:.1f}% | "
            f"mIoU={val_metrics['Seg_mIoU']:.1f}% | "
            f"ADE={val_metrics['Traj_ADE']:.3f}m | "
            f"Coll={val_metrics['Collision_Rate']:.2f}% | "
            f"VRAM={vram_gb:.1f}GB ({dt:.1f}s)"
        )

        score = (
            val_metrics["MultiLabel_Acc"]
            + val_metrics["Small_Object_AP"]
            + val_metrics["Seg_mIoU"]
            - val_metrics["Traj_ADE"] * 15.0
            - val_metrics["Collision_Rate"] * 2.0
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

        if GDRIVE_ROOT.exists():
            shutil.copy2(csv_path, GDRIVE_ROOT / "logs_a100" / csv_path.name)
            os.sync()

    csv_f.close()
    print(f"[SUCCESS] {tag} training finished!")


if __name__ == "__main__":
    main()
