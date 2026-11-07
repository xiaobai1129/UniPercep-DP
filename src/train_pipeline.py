import os
import sys
import csv
import json
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

# Ensure local src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import UniPercepDPNet

LOCAL_ROOT = Path("/root/workspace/ceshi")
GDRIVE_ROOT = Path("/content/gdrive/MyDrive/colab/UniPercep_DP")

PROCESSED_DIR = LOCAL_ROOT / "dataset" / "processed_unipercep"
CKPT_DIR = LOCAL_ROOT / "checkpoints"
LOG_DIR = LOCAL_ROOT / "logs"

GDRIVE_CKPT_DIR = GDRIVE_ROOT / "checkpoints"
GDRIVE_LOG_DIR = GDRIVE_ROOT / "logs"


class UniPercepDataset(Dataset):
    def __init__(self, jsonl_path: Path, data_root: Path):
        self.data_root = data_root
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        clean_bgr = cv2.imread(str(self.data_root / rec["clean_img"]))
        corrupt_bgr = cv2.imread(str(self.data_root / rec["corrupted_img"]))
        if clean_bgr is None:
            clean_bgr = np.zeros((288, 512, 3), dtype=np.uint8)
        if corrupt_bgr is None:
            corrupt_bgr = np.zeros((288, 512, 3), dtype=np.uint8)

        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        corrupt_rgb = cv2.cvtColor(corrupt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        clean_tensor = torch.from_numpy(clean_rgb).permute(2, 0, 1)      # (3, 288, 512)
        corrupt_tensor = torch.from_numpy(corrupt_rgb).permute(2, 0, 1)  # (3, 288, 512)

        bev_lidar = np.load(self.data_root / rec["depth_map"]).astype(np.float32)   # (2, 64, 64)
        seg_mask = np.load(self.data_root / rec["seg_mask"]).astype(np.int64)       # (64, 64)
        small_hm = np.load(self.data_root / rec["small_hm"]).astype(np.float32)     # (1, 64, 64)
        risk_map = np.load(self.data_root / rec["risk_map"]).astype(np.float32)     # (64, 64)

        sensor_state = np.array(rec["sensor_state"], dtype=np.float32)  # (5,)
        homography = np.array(rec["homography"], dtype=np.float32)      # (9,)
        geo_14d = np.concatenate([sensor_state, homography], axis=0)    # (14,)

        multi_label = np.array(rec["multi_label"], dtype=np.float32)    # (8,)
        gt_traj = np.array(rec["gt_traj"], dtype=np.float32)            # (6, 2)

        return {
            "frame_idx": rec["frame_idx"],
            "clean_img": clean_tensor,
            "corrupted_img": corrupt_tensor,
            "bev_lidar": torch.from_numpy(bev_lidar),
            "seg_mask": torch.from_numpy(seg_mask),
            "small_hm": torch.from_numpy(small_hm),
            "risk_map": torch.from_numpy(risk_map),
            "geo_14d": torch.from_numpy(geo_14d),
            "multi_label": torch.from_numpy(multi_label),
            "gt_traj": torch.from_numpy(gt_traj),
        }


def compute_batch_metrics(out, batch):
    with torch.no_grad():
        # Multi-label accuracy
        ml_preds = (torch.sigmoid(out["multi_label_logits"]) > 0.5).float()
        ml_acc = (ml_preds == batch["multi_label"]).float().mean().item()

        # Segmentation mIoU (3 classes)
        seg_pred = torch.argmax(out["seg_logits"], dim=1)  # (B, 64, 64)
        seg_gt = batch["seg_mask"]
        ious = []
        for cls_id in range(3):
            inter = ((seg_pred == cls_id) & (seg_gt == cls_id)).sum().item()
            union = ((seg_pred == cls_id) | (seg_gt == cls_id)).sum().item()
            if union > 0:
                ious.append(inter / union)
        miou = float(np.mean(ious)) if ious else 0.0

        # Small object heatmap MSE
        small_mse = F.mse_loss(out["small_hm_pred"], batch["small_hm"]).item()

        # Trajectory Average Displacement Error (ADE) in meters (1 grid unit = 1.0 meter)
        traj_dist = torch.norm(out["pred_traj"] - batch["gt_traj"], dim=-1)  # (B, 6)
        traj_ade = traj_dist.mean().item()

    return {
        "ml_acc": ml_acc,
        "miou": miou,
        "small_mse": small_mse,
        "traj_ade": traj_ade,
    }


def detect_hardware_config():
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "gpu_name": "CPU",
            "amp_dtype": torch.float32,
            "use_amp": False,
            "batch_size": 4,
            "grad_accum": 4,
            "stage_epochs": {
                "Stage1_SSL_LP": 10,
                "Stage1_LP_FT": 15,
                "Stage2_Fusion_DP": 20,
                "Stage3_RAG_LoRA": 15,
            },
        }

    gpu_name = torch.cuda.get_device_name(0)
    is_high_end = any(k in gpu_name.upper() for k in ["H100", "A100"])

    if is_high_end:
        return {
            "device": "cuda",
            "gpu_name": gpu_name,
            "amp_dtype": torch.bfloat16,
            "use_amp": True,
            "batch_size": 32,
            "grad_accum": 1,
            "stage_epochs": {
                "Stage1_SSL_LP": 15,
                "Stage1_LP_FT": 25,
                "Stage2_Fusion_DP": 30,
                "Stage3_RAG_LoRA": 20,
            },
        }
    else:
        # T4 or equivalent GPU config
        return {
            "device": "cuda",
            "gpu_name": gpu_name,
            "amp_dtype": torch.float16,
            "use_amp": True,
            "batch_size": 4,
            "grad_accum": 4,
            "stage_epochs": {
                "Stage1_SSL_LP": 10,
                "Stage1_LP_FT": 15,
                "Stage2_Fusion_DP": 20,
                "Stage3_RAG_LoRA": 15,
            },
        }


def _atomic_copy(src: Path, dst: Path):
    if not src.exists():
        return
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)


def sync_to_gdrive(csv_path: Path, latest_ckpt: Path, best_ckpt: Path, stage_ckpt: Path = None):
    """
    Real-time synchronization rule with atomic write protection:
    After every single epoch, copy latest weights, best weights, and training_metrics.csv
    to Google Drive and run os.sync().
    """
    GDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    GDRIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    _atomic_copy(csv_path, GDRIVE_LOG_DIR / csv_path.name)
    _atomic_copy(latest_ckpt, GDRIVE_CKPT_DIR / latest_ckpt.name)
    _atomic_copy(best_ckpt, GDRIVE_CKPT_DIR / best_ckpt.name)
    if stage_ckpt is not None:
        _atomic_copy(stage_ckpt, GDRIVE_CKPT_DIR / stage_ckpt.name)
    os.sync()


def compute_multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice = (2.0 * inter + 1.0) / (cardinality + 1.0)
    return 1.0 - dice.mean()


def evaluate_model(model, val_loader, device, seg_weights, smoke_test=False):
    model.eval()
    losses = []
    metrics_list = []

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if smoke_test and b_idx >= 3:
                break
            batch_gpu = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(
                batch_gpu["corrupted_img"],
                batch_gpu["bev_lidar"],
                batch_gpu["geo_14d"],
            )

            loss_ssl = F.l1_loss(out["pred_clean_img"], batch_gpu["clean_img"])
            loss_ml = F.binary_cross_entropy_with_logits(
                out["multi_label_logits"], batch_gpu["multi_label"]
            )
            loss_seg = F.cross_entropy(out["seg_logits"], batch_gpu["seg_mask"], weight=seg_weights)
            loss_small = F.mse_loss(out["small_hm_pred"], batch_gpu["small_hm"])
            loss_risk = F.mse_loss(out["pred_risk"], batch_gpu["risk_map"])
            loss_traj = F.smooth_l1_loss(out["pred_traj"], batch_gpu["gt_traj"])

            total_val_loss = loss_ssl + loss_ml + loss_seg + 5.0 * loss_small + loss_risk + 0.2 * loss_traj
            losses.append(total_val_loss.item())
            metrics_list.append(compute_batch_metrics(out, batch_gpu))

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_ml_acc = float(np.mean([m["ml_acc"] for m in metrics_list])) if metrics_list else 0.0
    avg_miou = float(np.mean([m["miou"] for m in metrics_list])) if metrics_list else 0.0
    avg_small_mse = float(np.mean([m["small_mse"] for m in metrics_list])) if metrics_list else 0.0
    avg_traj_ade = float(np.mean([m["traj_ade"] for m in metrics_list])) if metrics_list else 0.0

    return {
        "val_loss": avg_loss,
        "val_ml_acc": avg_ml_acc,
        "val_miou": avg_miou,
        "val_small_mse": avg_small_mse,
        "val_traj_ade": avg_traj_ade,
    }


def run_training(smoke_test: bool = False):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    GDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    GDRIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    hw = detect_hardware_config()
    device = torch.device(hw["device"])
    print(
        f"[INFO] Hardware detected: {hw['gpu_name']} | AMP: {hw['amp_dtype']} | "
        f"BatchSize: {hw['batch_size']} | GradAccum: {hw['grad_accum']} | SmokeTest: {smoke_test}"
    )

    train_jsonl = PROCESSED_DIR / "train.jsonl"
    val_jsonl = PROCESSED_DIR / "val.jsonl"
    if not train_jsonl.exists() or not val_jsonl.exists():
        raise FileNotFoundError(
            f"Dataset files not found at {PROCESSED_DIR}. Please run src/build_dataset.py first."
        )

    train_ds = UniPercepDataset(train_jsonl, PROCESSED_DIR)
    val_ds = UniPercepDataset(val_jsonl, PROCESSED_DIR)

    train_loader = DataLoader(
        train_ds,
        batch_size=hw["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hw["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    model = UniPercepDPNet(pretrained_backbone=True).to(device)
    seg_weights = torch.tensor([1.0, 2.5, 4.0], dtype=torch.float32, device=device)

    use_fp16_scaler = hw["use_amp"] and (hw["amp_dtype"] == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    csv_path = LOG_DIR / "training_metrics.csv"
    latest_ckpt_path = CKPT_DIR / "unipercep_dp_latest.pt"
    best_ckpt_path = CKPT_DIR / "unipercep_dp_best.pt"

    csv_fields = [
        "global_epoch",
        "stage",
        "stage_epoch",
        "lr",
        "train_loss",
        "loss_ssl",
        "loss_percep",
        "loss_risk",
        "loss_traj",
        "loss_smooth",
        "val_loss",
        "val_ml_acc",
        "val_miou",
        "val_small_mse",
        "val_traj_ade",
        "w_obs_mean",
        "w_smooth_mean",
        "elapsed_sec",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    stages = [
        ("Stage1_SSL_LP", 1e-3),
        ("Stage1_LP_FT", 5e-4),
        ("Stage2_Fusion_DP", 8e-4),
        ("Stage3_RAG_LoRA", 3e-4),
    ]

    global_epoch = 0
    best_composite_metric = float("inf")

    for stage_name, base_lr in stages:
        num_epochs = 1 if smoke_test else hw["stage_epochs"][stage_name]
        model.set_training_stage(stage_name)

        # Reset best metric when entering Stage2_Fusion_DP so best checkpoint always includes trained DP planner
        if stage_name == "Stage2_Fusion_DP":
            best_composite_metric = float("inf")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_epochs), eta_min=base_lr * 0.1
        )

        print(f"\n==================== Starting {stage_name} ({num_epochs} Epochs, lr={base_lr}) ====================")

        for s_epoch in range(1, num_epochs + 1):
            t0 = time.time()
            global_epoch += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)

            ep_losses = []
            ep_ssl, ep_percep, ep_risk, ep_traj, ep_smooth = [], [], [], [], []
            ep_w_obs, ep_w_smooth = [], []

            for step_idx, batch in enumerate(train_loader):
                if smoke_test and step_idx >= 3:
                    break

                batch_gpu = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with torch.amp.autocast(
                    device_type="cuda" if hw["device"] == "cuda" else "cpu",
                    dtype=hw["amp_dtype"],
                    enabled=hw["use_amp"],
                ):
                    out = model(
                        batch_gpu["corrupted_img"],
                        batch_gpu["bev_lidar"],
                        batch_gpu["geo_14d"],
                    )

                    # 1. Self-Supervised Image Reconstruction L1 Loss
                    loss_ssl = F.l1_loss(out["pred_clean_img"], batch_gpu["clean_img"])

                    # 2. Multi-Task Dense Visual Perception Losses
                    loss_ml = F.binary_cross_entropy_with_logits(
                        out["multi_label_logits"], batch_gpu["multi_label"]
                    )
                    loss_seg = F.cross_entropy(
                        out["seg_logits"], batch_gpu["seg_mask"], weight=seg_weights
                    ) + compute_multiclass_dice_loss(out["seg_logits"], batch_gpu["seg_mask"])
                    # Weighted MSE for small object center heatmap to emphasize foreground peaks
                    pos_weight = 1.0 + 10.0 * batch_gpu["small_hm"]
                    loss_small = torch.mean(
                        pos_weight * (out["small_hm_pred"] - batch_gpu["small_hm"]) ** 2
                    )
                    loss_percep = loss_ml + loss_seg + 5.0 * loss_small

                    # 3. Spatial Risk Map & Differentiable DP Trajectory Losses
                    loss_risk = F.mse_loss(out["pred_risk"], batch_gpu["risk_map"])
                    loss_traj = F.smooth_l1_loss(out["pred_traj"], batch_gpu["gt_traj"])

                    # 4. Second-order trajectory smoothness penalty & RAG weight alignment
                    accel = (
                        out["pred_traj"][:, 2:, :]
                        - 2.0 * out["pred_traj"][:, 1:-1, :]
                        + out["pred_traj"][:, :-2, :]
                    )
                    loss_smooth = torch.mean(accel ** 2)

                    if stage_name in ["Stage1_SSL_LP", "Stage1_LP_FT"]:
                        total_loss = 2.0 * loss_ssl + 1.5 * loss_percep + 0.05 * (loss_risk + loss_traj)
                    elif stage_name == "Stage2_Fusion_DP":
                        total_loss = (
                            0.5 * loss_ssl
                            + 1.0 * loss_percep
                            + 2.5 * loss_risk
                            + 1.0 * loss_traj
                            + 0.05 * loss_smooth
                        )
                    else:  # Stage3_RAG_LoRA
                        # Encourage collision avoidance along predicted trajectory + second-order smoothness
                        w_smooth_factor = out["dp_weights"][:, 1].mean()
                        total_loss = (
                            0.5 * loss_percep
                            + 1.5 * loss_risk
                            + 1.2 * loss_traj
                            + 0.25 * w_smooth_factor * loss_smooth
                        )

                    scaled_loss = total_loss / hw["grad_accum"]

                if use_fp16_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_accum_step = ((step_idx + 1) % hw["grad_accum"] == 0) or (
                    (step_idx + 1) == len(train_loader)
                ) or (smoke_test and step_idx == 2)

                if is_accum_step:
                    if use_fp16_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                ep_losses.append(total_loss.item())
                ep_ssl.append(loss_ssl.item())
                ep_percep.append(loss_percep.item())
                ep_risk.append(loss_risk.item())
                ep_traj.append(loss_traj.item())
                ep_smooth.append(loss_smooth.item())
                ep_w_obs.append(out["dp_weights"][:, 0].mean().item())
                ep_w_smooth.append(out["dp_weights"][:, 1].mean().item())

            scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]

            val_res = evaluate_model(model, val_loader, device, seg_weights, smoke_test=smoke_test)
            elapsed = time.time() - t0

            row = {
                "global_epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": s_epoch,
                "lr": round(curr_lr, 7),
                "train_loss": round(float(np.mean(ep_losses)), 4),
                "loss_ssl": round(float(np.mean(ep_ssl)), 4),
                "loss_percep": round(float(np.mean(ep_percep)), 4),
                "loss_risk": round(float(np.mean(ep_risk)), 4),
                "loss_traj": round(float(np.mean(ep_traj)), 4),
                "loss_smooth": round(float(np.mean(ep_smooth)), 4),
                "val_loss": round(val_res["val_loss"], 4),
                "val_ml_acc": round(val_res["val_ml_acc"], 4),
                "val_miou": round(val_res["val_miou"], 4),
                "val_small_mse": round(val_res["val_small_mse"], 5),
                "val_traj_ade": round(val_res["val_traj_ade"], 4),
                "w_obs_mean": round(float(np.mean(ep_w_obs)), 3),
                "w_smooth_mean": round(float(np.mean(ep_w_smooth)), 3),
                "elapsed_sec": round(elapsed, 2),
            }

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow(row)

            ckpt_payload = {
                "global_epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": s_epoch,
                "model_state_dict": model.state_dict(),
                "metrics": row,
            }
            torch.save(ckpt_payload, latest_ckpt_path)

            stage_ckpt_path = CKPT_DIR / f"{stage_name.lower()}_latest.pt"
            torch.save(ckpt_payload, stage_ckpt_path)

            # Composite score: lower is better
            if stage_name in ["Stage1_SSL_LP", "Stage1_LP_FT"]:
                composite = val_res["val_loss"] - val_res["val_miou"] - val_res["val_ml_acc"]
            else:
                composite = val_res["val_traj_ade"] - 2.0 * val_res["val_miou"] - val_res["val_ml_acc"]

            if composite <= best_composite_metric or not best_ckpt_path.exists():
                best_composite_metric = composite
                torch.save(ckpt_payload, best_ckpt_path)

            # Hard rule #4: Sync latest, best, and training_metrics.csv to Google Drive after every epoch
            sync_to_gdrive(csv_path, latest_ckpt_path, best_ckpt_path, stage_ckpt_path)

            print(
                f"[{stage_name}][Epoch {s_epoch:02d}/{num_epochs:02d} | Global {global_epoch:02d}] "
                f"TrainLoss={row['train_loss']:.4f} | Val_ML_Acc={row['val_ml_acc']:.4f} | "
                f"Val_mIoU={row['val_miou']:.4f} | Val_Traj_ADE={row['val_traj_ade']:.3f}m | "
                f"w_obs={row['w_obs_mean']:.2f}, w_smooth={row['w_smooth_mean']:.2f} ({elapsed:.1f}s)"
            )

    print("\n[INFO] All 4 training stages completed and synced to Google Drive!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniPercep-DP Training Pipeline")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 epoch (max 3 batches) per stage")
    args = parser.parse_args()
    run_training(smoke_test=args.smoke_test)
