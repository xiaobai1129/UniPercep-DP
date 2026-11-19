#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-RAM NVMe Streaming Datasets for P1 Experiments (strictly < 1 GB CPU RAM).
Supports:
  - LowRAM128Dataset (with optional matched 1/7 online night photometric scaling for Exp 4)
  - LowRAM256Dataset (exact 4D temporal kinematics & prev-frame LiDAR BEV for Exp 5, 6, 7)
"""

import json
import math
from pathlib import Path

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset

GRID_128 = 128


class LowRAM128Dataset(Dataset):
    """Zero unbounded RAM cache 128x128 dataset with optional matched 1/7 online night augmentation."""

    def __init__(
        self,
        jsonl_path: Path,
        processed_dir: Path,
        is_train: bool = True,
        night_aug: bool = False,
        force_clean: bool = False,
        max_samples: int = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.is_train = is_train
        self.night_aug = night_aug
        self.force_clean = force_clean
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
                    if max_samples and len(self.samples) >= max_samples:
                        break
        yy, xx = np.mgrid[0:GRID_128, 0:GRID_128].astype(np.float32)
        self.xx_grid = xx
        self.yy_grid = yy

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        deg_mode = int(rec.get("deg_mode", 1))

        clean_bgr = cv2.imread(str(self.processed_dir / rec["clean_img"]))
        corrupt_bgr = cv2.imread(str(self.processed_dir / rec["corrupted_img"]))
        if clean_bgr is None:
            clean_bgr = np.zeros((448, 768, 3), dtype=np.uint8)
        if corrupt_bgr is None:
            corrupt_bgr = np.zeros((448, 768, 3), dtype=np.uint8)

        clean_f = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        corr_raw_f = cv2.cvtColor(corrupt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.force_clean:
            corrupt_f = clean_f
        elif deg_mode == 0:
            corrupt_f = 0.7 * clean_f + 0.3 * corr_raw_f
        elif deg_mode == 1:
            corrupt_f = corr_raw_f
        elif deg_mode == 2:
            corrupt_f = np.clip(0.78 * corr_raw_f + 0.22 * 0.8, 0.0, 1.0)
        else:
            corrupt_f = corr_raw_f.copy()
            corrupt_f[160:220, 280:360, :] = 0.1

        ml = np.array(rec["multi_label"], dtype=np.float32).copy()

        # Matched 1/7 online nighttime photometric scaling (exact match to UniPercepChronos256Dataset)
        if self.is_train and self.night_aug and (idx % 7 == 0):
            gamma_scale = 0.28 + 0.06 * float(idx % 3)
            corrupt_f = np.clip(corrupt_f * gamma_scale, 0.0, 1.0)
            clean_f = np.clip(clean_f * gamma_scale, 0.0, 1.0)
            ml[0] = 1.0

        bev_lidar = np.load(self.processed_dir / rec["bev_lidar"]).astype(np.float32)
        seg_mask = np.load(self.processed_dir / rec["seg_mask"]).astype(np.int64)
        risk_map = np.load(self.processed_dir / rec["risk_map"]).astype(np.float32)

        small_hm = np.zeros((1, GRID_128, GRID_128), dtype=np.float32)
        small_reg = np.zeros((4, GRID_128, GRID_128), dtype=np.float32)
        small_mask = np.zeros((1, GRID_128, GRID_128), dtype=np.float32)
        for obj in rec.get("det_boxes", []):
            cx, cy, w_g, l_g, cls_id, track_id, vx, vy, is_small = obj
            if is_small == 1 and 0 <= cx < GRID_128 and 0 <= cy < GRID_128:
                sigma = 2.2
                g = np.exp(-((self.xx_grid - cx) ** 2 + (self.yy_grid - cy) ** 2) / (2.0 * sigma ** 2))
                small_hm[0] = np.maximum(small_hm[0], g)
                ix, iy = int(cx), int(cy)
                r0_x, r1_x = max(0, ix - 1), min(GRID_128, ix + 2)
                r0_y, r1_y = max(0, iy - 1), min(GRID_128, iy + 2)
                small_reg[0, r0_y:r1_y, r0_x:r1_x] = w_g / 10.0
                small_reg[1, r0_y:r1_y, r0_x:r1_x] = l_g / 10.0
                small_reg[2, r0_y:r1_y, r0_x:r1_x] = vx / 10.0
                small_reg[3, r0_y:r1_y, r0_x:r1_x] = vy / 10.0
                small_mask[0, r0_y:r1_y, r0_x:r1_x] = 1.0

        sensor_state = np.array(rec["sensor_state"], dtype=np.float32)
        homography = np.array(rec["homography"], dtype=np.float32).reshape(-1)
        geo_14d = np.concatenate([sensor_state, homography], axis=0)
        gt_traj = np.array(rec["gt_traj"], dtype=np.float32)
        dp_weights_gt = np.array(rec.get("dp_weights_gt", [2.0, 1.8, 1.8]), dtype=np.float32)

        return {
            "keyframe_idx": rec["keyframe_idx"],
            "deg_mode": deg_mode,
            "clean_img": torch.from_numpy(clean_f).permute(2, 0, 1).contiguous(),
            "corrupted_img": torch.from_numpy(corrupt_f).permute(2, 0, 1).contiguous(),
            "bev_lidar": torch.from_numpy(bev_lidar),
            "seg_mask": torch.from_numpy(seg_mask),
            "risk_map": torch.from_numpy(risk_map),
            "small_hm": torch.from_numpy(small_hm),
            "small_reg": torch.from_numpy(small_reg),
            "small_mask": torch.from_numpy(small_mask),
            "geo_14d": torch.from_numpy(geo_14d),
            "multi_label": torch.from_numpy(ml),
            "gt_traj": torch.from_numpy(gt_traj),
            "dp_weights_gt": torch.from_numpy(dp_weights_gt),
        }


class LowRAM256Dataset(Dataset):
    """Zero unbounded RAM cache 256x256 4D temporal dataset (streams cleanly from local NVMe)."""

    def __init__(
        self,
        jsonl_path: Path,
        processed_dir: Path,
        is_train: bool = True,
        cam_hw=(192, 320),
        force_clean: bool = False,
        max_samples: int = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.is_train = is_train
        self.cam_hw = cam_hw
        self.force_clean = force_clean
        self.items = []
        self.kf_meta = {}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self.items.append(rec)
                    kf_idx = rec["keyframe_idx"]
                    if kf_idx not in self.kf_meta:
                        self.kf_meta[kf_idx] = {
                            "scene_name": rec["scene_name"],
                            "sensor_state": rec["sensor_state"],
                            "bev_lidar": rec["bev_lidar"],
                        }
                    if max_samples and len(self.items) >= max_samples:
                        break

        rad = 8
        sigma = 3.2
        py_grid, px_grid = np.mgrid[-rad : rad + 1, -rad : rad + 1].astype(np.float32)
        self.g_kernel = np.exp(-(px_grid ** 2 + py_grid ** 2) / (2.0 * sigma * sigma))
        self.rad = rad

    def __len__(self):
        return len(self.items)

    def _load_bev_256(self, rel_path: str) -> np.ndarray:
        bev_128 = np.load(self.processed_dir / rel_path).astype(np.float32)
        bev_256 = np.zeros((3, 256, 256), dtype=np.float32)
        for c in range(3):
            bev_256[c] = cv2.resize(bev_128[c], (256, 256), interpolation=cv2.INTER_NEAREST)
        return bev_256

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        kf_idx = rec["keyframe_idx"]
        deg_mode = int(rec.get("deg_mode", 1))
        scene_name = rec["scene_name"]
        H_c, W_c = self.cam_hw

        clean_bgr = cv2.imread(str(self.processed_dir / rec["clean_img"]))
        corrupt_bgr = cv2.imread(str(self.processed_dir / rec["corrupted_img"]))
        if clean_bgr is None:
            clean_bgr = np.zeros((448, 768, 3), dtype=np.uint8)
        if corrupt_bgr is None:
            corrupt_bgr = np.zeros((448, 768, 3), dtype=np.uint8)

        clean_res = cv2.resize(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB), (W_c, H_c), interpolation=cv2.INTER_LINEAR)
        corr_res = cv2.resize(cv2.cvtColor(corrupt_bgr, cv2.COLOR_BGR2RGB), (W_c, H_c), interpolation=cv2.INTER_LINEAR)

        if self.force_clean:
            front_deg = clean_res
        elif deg_mode == 0:
            front_deg = (0.7 * clean_res.astype(np.float32) + 0.3 * corr_res.astype(np.float32)).astype(np.uint8)
        elif deg_mode == 1:
            front_deg = corr_res
        elif deg_mode == 2:
            front_deg = np.clip(0.78 * corr_res.astype(np.float32) + 0.22 * 204.0, 0, 255).astype(np.uint8)
        else:
            front_deg = corr_res.copy()
            h_f, w_f, _ = front_deg.shape
            front_deg[h_f // 3 : h_f // 2, w_f // 3 : w_f // 2, :] = 25

        corrupted_front = torch.from_numpy(front_deg).permute(2, 0, 1).float() / 255.0
        clean_front = torch.from_numpy(clean_res).permute(2, 0, 1).float() / 255.0
        ml = np.array(rec["multi_label"], dtype=np.float32).copy()

        # Matched 1/7 Online Night-Illumination Photometric Synthesis
        if self.is_train and (idx % 7 == 0):
            gamma_scale = 0.28 + 0.06 * float(idx % 3)
            corrupted_front = (corrupted_front * gamma_scale).clamp(0.0, 1.0)
            clean_front = (clean_front * gamma_scale).clamp(0.0, 1.0)
            ml[0] = 1.0

        bev_256 = self._load_bev_256(rec["bev_lidar"])
        seg_128 = np.load(self.processed_dir / rec["seg_mask"]).astype(np.uint8)
        risk_128 = np.load(self.processed_dir / rec["risk_map"]).astype(np.float32)
        seg_256 = cv2.resize(seg_128, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        risk_256 = cv2.resize(risk_128, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        small_hm_256 = np.zeros((1, 256, 256), dtype=np.float32)
        small_reg_256 = np.zeros((4, 256, 256), dtype=np.float32)
        small_mask_256 = np.zeros((1, 256, 256), dtype=np.float32)
        rad = self.rad
        for obj in rec.get("det_boxes", []):
            cx, cy, bw, bh, cls_id, track_id, bvx, bvy, is_small = obj
            if is_small == 1 and 0 <= cx < 128 and 0 <= cy < 128:
                ix = int(np.clip(round(float(cx) * 2.0), 0, 255))
                iy = int(np.clip(round(float(cy) * 2.0), 0, 255))
                y0, y1 = max(0, iy - rad), min(256, iy + rad + 1)
                x0, x1 = max(0, ix - rad), min(256, ix + rad + 1)
                ky0, kx0 = y0 - (iy - rad), x0 - (ix - rad)
                small_hm_256[0, y0:y1, x0:x1] = np.maximum(
                    small_hm_256[0, y0:y1, x0:x1],
                    self.g_kernel[ky0 : ky0 + (y1 - y0), kx0 : kx0 + (x1 - x0)],
                )
                r0_x, r1_x = max(0, ix - 2), min(256, ix + 3)
                r0_y, r1_y = max(0, iy - 2), min(256, iy + 3)
                small_reg_256[0, r0_y:r1_y, r0_x:r1_x] = bw / 10.0
                small_reg_256[1, r0_y:r1_y, r0_x:r1_x] = bh / 10.0
                small_reg_256[2, r0_y:r1_y, r0_x:r1_x] = bvx / 10.0
                small_reg_256[3, r0_y:r1_y, r0_x:r1_x] = bvy / 10.0
                small_mask_256[0, r0_y:r1_y, r0_x:r1_x] = 1.0

        ss_curr = rec["sensor_state"]
        v_fwd = float(ss_curr[0]) * 15.0
        v_lft = float(ss_curr[1]) * 5.0
        yaw_curr = float(ss_curr[2])
        prev_meta = self.kf_meta.get(kf_idx - 1, None)
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
            bev_prev_256 = self._load_bev_256(prev_meta["bev_lidar"])
        else:
            dv_fwd, dv_lft, dyaw, yaw_rate = 0.0, 0.0, 0.0, 0.0
            v_fwd_prev = v_fwd
            has_prev = 0.0
            bev_prev_256 = bev_256

        temporal_8d = torch.tensor(
            [v_fwd, v_lft, dv_fwd, dv_lft, dyaw, yaw_rate, v_fwd_prev, has_prev], dtype=torch.float32
        )
        sensor_state = np.array(rec["sensor_state"], dtype=np.float32)
        homography = np.array(rec["homography"], dtype=np.float32).reshape(-1)
        geo_14d = torch.from_numpy(np.concatenate([sensor_state, homography], axis=0))

        return {
            "keyframe_idx": kf_idx,
            "deg_mode": deg_mode,
            "clean_img": clean_front,
            "corrupted_imgs": corrupted_front,
            "bev_lidar_256": torch.from_numpy(bev_256),
            "bev_prev_256": torch.from_numpy(bev_prev_256),
            "geo_14d": geo_14d,
            "temporal_8d": temporal_8d,
            "multi_label": torch.from_numpy(ml),
            "seg_256": torch.from_numpy(seg_256),
            "small_hm_256": torch.from_numpy(small_hm_256),
            "small_reg_256": torch.from_numpy(small_reg_256),
            "small_mask_256": torch.from_numpy(small_mask_256),
            "risk_256": torch.from_numpy(risk_256),
            "gt_traj": torch.from_numpy(np.array(rec["gt_traj"], dtype=np.float32)),
            "dp_weights_gt": torch.from_numpy(np.array(rec.get("dp_weights_gt", [2.0, 1.8, 1.8]), dtype=np.float32)),
        }
