import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Ensure local src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import UniPercepDPNet
from train_pipeline import UniPercepDataset

LOCAL_ROOT = Path("/root/workspace/ceshi")
GDRIVE_ROOT = Path("/content/gdrive/MyDrive/colab/UniPercep_DP")

PROCESSED_DIR = LOCAL_ROOT / "dataset" / "processed_unipercep"
CKPT_DIR = LOCAL_ROOT / "checkpoints"
OUT_DIR = LOCAL_ROOT / "outputs"

GDRIVE_OUT_DIR = GDRIVE_ROOT / "outputs"


def load_rag_rules():
    rules_path = PROCESSED_DIR / "rag_db" / "rules.json"
    if rules_path.exists():
        with open(rules_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        {"rule_id": 0, "name": "Night_RainFog_Safety_Rule"},
        {"rule_id": 1, "name": "Dense_Pedestrian_SmallObj_Rule"},
        {"rule_id": 2, "name": "Normal_Road_Cruising_Rule"},
    ]


def render_quad_visualization(sample_idx, batch_item, model_out, rag_rules, save_path: Path):
    """
    Generate 1x4 comparison visualization figure:
      Subplot 1: Degraded input image & LoRA predicted weights [w_obs, w_smooth] + RAG rule
      Subplot 2: BEV Obstacle Semantic Segmentation prediction (64x64)
      Subplot 3: Predicted 64x64 Spatial Risk Cost Heatmap
      Subplot 4: Differentiable DP Energy Field (dp_value_map) with Pred vs GT Trajectory
    """
    corrupt_img = batch_item["corrupted_img"].cpu().permute(1, 2, 0).numpy()
    corrupt_img = np.clip(corrupt_img, 0.0, 1.0)

    seg_pred = torch.argmax(model_out["seg_logits"], dim=0).cpu().numpy()  # (64, 64)
    pred_risk = model_out["pred_risk"].cpu().numpy()                      # (64, 64)
    dp_val_map = model_out["dp_value_map"].cpu().numpy()                  # (64, 64)
    pred_traj = model_out["pred_traj"].cpu().numpy()                      # (6, 2)
    gt_traj = batch_item["gt_traj"].cpu().numpy()                         # (6, 2)
    dp_weights = model_out["dp_weights"].cpu().numpy()                    # (2,)
    rule_id = int(model_out["rag_rule_idx"].cpu().item())

    rule_name = rag_rules[rule_id]["name"] if rule_id < len(rag_rules) else f"Rule_{rule_id}"
    w_obs, w_smooth = float(dp_weights[0]), float(dp_weights[1])
    sample_ade = float(np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1)))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), dpi=150)

    # Subplot 1: Corrupted Input Image & LoRA Predicted Weights
    ax0 = axes[0]
    ax0.imshow(corrupt_img)
    ax0.set_title(
        f"1. Degraded Input & LoRA-RAG Weights\nw_obs={w_obs:.2f}, w_smooth={w_smooth:.2f} ({rule_name})",
        fontsize=11,
        fontweight="bold",
    )
    ax0.text(
        10,
        268,
        f"Frame #{batch_item['frame_idx']} | RAG: {rule_name}\nLoRA Weights: [w_obs={w_obs:.2f}, w_smooth={w_smooth:.2f}]",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.72, edgecolor="cyan", boxstyle="round,pad=0.4"),
    )
    ax0.axis("off")

    # Subplot 2: BEV Obstacle Semantic Segmentation Prediction (64x64)
    ax1 = axes[1]
    seg_cmap = ListedColormap(["#1e293b", "#f59e0b", "#ef4444"])
    ax1.imshow(seg_pred, cmap=seg_cmap, vmin=0, vmax=2, origin="lower")
    ax1.set_title("2. BEV Obstacle Semantic Segmentation", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lateral X (Grid [0..63])")
    ax1.set_ylabel("Forward Y (Grid [0..63])")
    patches = [
        mpatches.Patch(color="#1e293b", label="0: Drivable"),
        mpatches.Patch(color="#f59e0b", label="1: Static Obs"),
        mpatches.Patch(color="#ef4444", label="2: Dynamic High-Risk"),
    ]
    ax1.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.85)

    # Subplot 3: Predicted 64x64 Spatial Risk Heatmap
    ax2 = axes[2]
    im2 = ax2.imshow(pred_risk, cmap="inferno", vmin=0.0, vmax=1.0, origin="lower")
    ax2.set_title("3. Predicted 64x64 Spatial Risk Heatmap", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lateral X (Grid [0..63])")
    ax2.set_ylabel("Forward Y (Grid [0..63])")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Subplot 4: Differentiable DP Energy Field & Trajectory Comparison
    ax3 = axes[3]
    im3 = ax3.imshow(dp_val_map, cmap="viridis", origin="lower")
    # prepend ego start point (32, 4) for smooth trajectory plotting
    ego_pt = np.array([[32.0, 4.0]], dtype=np.float32)
    gt_plot = np.vstack([ego_pt, gt_traj])
    pred_plot = np.vstack([ego_pt, pred_traj])

    ax3.plot(
        gt_plot[:, 0],
        gt_plot[:, 1],
        "g--o",
        linewidth=2.2,
        markersize=5,
        label="GT Trajectory",
    )
    ax3.plot(
        pred_plot[:, 0],
        pred_plot[:, 1],
        "r-*",
        linewidth=2.4,
        markersize=7,
        label=f"DP Pred (ADE={sample_ade:.2f}m)",
    )
    ax3.scatter([32.0], [4.0], c="cyan", s=80, marker="^", zorder=5, label="Ego Vehicle")
    ax3.set_xlim(0, 63)
    ax3.set_ylim(0, 63)
    ax3.set_title("4. Diff-DP Value Field & Trajectory", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Lateral X (Grid [0..63])")
    ax3.set_ylabel("Forward Y (Grid [0..63])")
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_eval():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GDRIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_ckpt = CKPT_DIR / "unipercep_dp_best.pt"
    if not best_ckpt.exists():
        best_ckpt = CKPT_DIR / "unipercep_dp_latest.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found at {CKPT_DIR}")

    print(f"[INFO] Loading best checkpoint from {best_ckpt} ...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)

    model = UniPercepDPNet(pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_jsonl = PROCESSED_DIR / "val.jsonl"
    val_ds = UniPercepDataset(val_jsonl, PROCESSED_DIR)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    ml_correct = 0
    ml_total = 0

    cls_inter = np.zeros(3, dtype=np.float64)
    cls_union = np.zeros(3, dtype=np.float64)

    small_mse_list = []
    risk_mse_list = []
    ssl_l1_list = []
    traj_ade_list = []
    traj_fde_list = []
    w_obs_list = []
    w_smooth_list = []

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(
                batch_gpu["corrupted_img"],
                batch_gpu["bev_lidar"],
                batch_gpu["geo_14d"],
            )

            # 1. MultiLabel_Acc
            ml_pred = (torch.sigmoid(out["multi_label_logits"]) > 0.5).float()
            ml_correct += (ml_pred == batch_gpu["multi_label"]).sum().item()
            ml_total += batch_gpu["multi_label"].numel()

            # 2. Semantic Segmentation mIoU
            seg_pred = torch.argmax(out["seg_logits"], dim=1)
            seg_gt = batch_gpu["seg_mask"]
            for c in range(3):
                cls_inter[c] += ((seg_pred == c) & (seg_gt == c)).sum().item()
                cls_union[c] += ((seg_pred == c) | (seg_gt == c)).sum().item()

            # 3. Small Object Heatmap Error (MSE)
            small_mse_list.append(F.mse_loss(out["small_hm_pred"], batch_gpu["small_hm"]).item())

            # 4. Spatial Risk MSE & SSL L1
            risk_mse_list.append(F.mse_loss(out["pred_risk"], batch_gpu["risk_map"]).item())
            ssl_l1_list.append(F.l1_loss(out["pred_clean_img"], batch_gpu["clean_img"]).item())

            # 5. Trajectory ADE & FDE (in physical meters; 1 grid cell = 1.0m)
            disp = torch.norm(out["pred_traj"] - batch_gpu["gt_traj"], dim=-1)  # (B, 6)
            traj_ade_list.extend(disp.mean(dim=-1).cpu().tolist())
            traj_fde_list.extend(disp[:, -1].cpu().tolist())

            w_obs_list.extend(out["dp_weights"][:, 0].cpu().tolist())
            w_smooth_list.extend(out["dp_weights"][:, 1].cpu().tolist())

    ious = [
        float(cls_inter[c] / cls_union[c]) if cls_union[c] > 0 else 0.0
        for c in range(3)
    ]
    miou = float(np.mean(ious))
    multilabel_acc = float(ml_correct / max(1, ml_total))
    small_hm_mse = float(np.mean(small_mse_list))
    traj_ade = float(np.mean(traj_ade_list))
    traj_fde = float(np.mean(traj_fde_list))

    summary = {
        "checkpoint_used": str(best_ckpt.name),
        "trained_global_epoch": int(ckpt.get("global_epoch", 0)),
        "trained_stage": str(ckpt.get("stage", "unknown")),
        "num_val_samples": len(val_ds),
        "MultiLabel_Acc": round(multilabel_acc, 4),
        "mIoU": round(miou, 4),
        "Class_IoUs": {
            "Drivable_Area": round(ious[0], 4),
            "Static_Obstacle": round(ious[1], 4),
            "Dynamic_HighRisk": round(ious[2], 4),
        },
        "Small_Object_Heatmap_MSE": round(small_hm_mse, 6),
        "Risk_Map_MSE": round(float(np.mean(risk_mse_list)), 5),
        "SSL_Reconstruction_L1": round(float(np.mean(ssl_l1_list)), 5),
        "Traj_ADE": round(traj_ade, 4),
        "Traj_FDE": round(traj_fde, 4),
        "LoRA_Weights_Mean": {
            "w_obs": round(float(np.mean(w_obs_list)), 4),
            "w_smooth": round(float(np.mean(w_smooth_list)), 4),
        },
    }

    eval_json_path = OUT_DIR / "evaluation_summary.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    shutil.copy2(eval_json_path, GDRIVE_OUT_DIR / "evaluation_summary.json")
    print("[INFO] Evaluation Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Generate 5 visualization samples: demo_result_sample_0.png ~ demo_result_sample_4.png
    rag_rules = load_rag_rules()
    sample_indices = [0, 12, 25, 40, 55]
    with torch.no_grad():
        for i, ds_idx in enumerate(sample_indices):
            ds_idx = min(ds_idx, len(val_ds) - 1)
            item = val_ds[ds_idx]
            c_img = item["corrupted_img"].unsqueeze(0).to(device)
            b_lid = item["bev_lidar"].unsqueeze(0).to(device)
            g_14d = item["geo_14d"].unsqueeze(0).to(device)

            out = model(c_img, b_lid, g_14d)
            single_out = {k: v[0] for k, v in out.items()}

            out_png = OUT_DIR / f"demo_result_sample_{i}.png"
            render_quad_visualization(i, item, single_out, rag_rules, out_png)
            shutil.copy2(out_png, GDRIVE_OUT_DIR / out_png.name)
            print(f"[INFO] Saved visualization {out_png} and synced to Google Drive.")

    os.sync()
    print("[INFO] Inference & evaluation completed successfully!")


if __name__ == "__main__":
    run_eval()
