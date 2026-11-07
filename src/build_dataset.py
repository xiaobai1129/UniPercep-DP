import os
import sys
import json
import math
import shutil
import tarfile
import urllib.request
from pathlib import Path

import cv2
import faiss
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# Paths
LOCAL_ROOT = Path("/root/workspace/ceshi")
GDRIVE_ROOT = Path("/content/gdrive/MyDrive/colab/UniPercep_DP")

RAW_NUSC_DIR = LOCAL_ROOT / "dataset" / "raw_nuscenes"
PROCESSED_DIR = LOCAL_ROOT / "dataset" / "processed_unipercep"
GDRIVE_TAR_PATH = GDRIVE_ROOT / "dataset" / "processed_unipercep_mini.tar.gz"
NUSC_URL = "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz"

# Grid config: 64x64 BEV grid
# X (lateral, col): [-32.0m, +32.0m] -> 64 cells (1.0m / cell), ego at X_grid = 32.0
# Y (forward, row): [0.0m, +64.0m] -> 64 cells (1.0m / cell), ego at Y_grid = 4.0 (4m from bottom)
GRID_SIZE = 64
METERS_PER_PIXEL = 1.0
EGO_GRID_X = 32.0
EGO_GRID_Y = 4.0


def ego_metric_to_grid(forward_m, left_m):
    """
    Convert ego-centric metric coordinates (forward_m = +x_ego, left_m = +y_ego)
    to 64x64 grid coordinates (gx, gy), where:
      gx (column, lateral right): EGO_GRID_X - left_m / METERS_PER_PIXEL
      gy (row, forward): EGO_GRID_Y + forward_m / METERS_PER_PIXEL
    """
    gx = EGO_GRID_X - (left_m / METERS_PER_PIXEL)
    gy = EGO_GRID_Y + (forward_m / METERS_PER_PIXEL)
    return gx, gy


def download_and_extract_nuscenes():
    RAW_NUSC_DIR.mkdir(parents=True, exist_ok=True)
    meta_dir = RAW_NUSC_DIR / "v1.0-mini"
    if meta_dir.exists() and (meta_dir / "sample.json").exists():
        print("[INFO] Raw nuScenes v1.0-mini already extracted.")
        return

    tgz_path = RAW_NUSC_DIR / "v1.0-mini.tgz"
    if not tgz_path.exists():
        print(f"[INFO] Downloading nuScenes v1.0-mini from {NUSC_URL} ...")
        urllib.request.urlretrieve(NUSC_URL, str(tgz_path))
        print(f"[INFO] Downloaded to {tgz_path} ({tgz_path.stat().st_size / 1e6:.1f} MB)")

    print("[INFO] Extracting v1.0-mini.tgz ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=str(RAW_NUSC_DIR))
    print("[INFO] Removing v1.0-mini.tgz to free disk space ...")
    tgz_path.unlink(missing_ok=True)


def corrupt_image(img_bgr, rng):
    """
    Apply strong JPEG compression artifacts + random local macroblock occlusion
    to simulate degraded weak-network video streams.
    """
    # 1. Random slight motion/gaussian blur or noise
    corrupted = img_bgr.copy()
    quality = int(rng.integers(8, 20))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", corrupted, encode_param)
    corrupted = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    # 2. Random macroblock occlusions (2 to 5 blocks of 16x16 or 32x32)
    h, w, _ = corrupted.shape
    num_blocks = int(rng.integers(3, 7))
    for _ in range(num_blocks):
        bw = int(rng.choice([16, 24, 32, 40]))
        bh = int(rng.choice([16, 24, 32]))
        bx = int(rng.integers(0, max(1, w - bw)))
        by = int(rng.integers(0, max(1, h - bh)))
        mode = int(rng.integers(0, 3))
        if mode == 0:
            # Constant gray/black block
            corrupted[by : by + bh, bx : bx + bw] = int(rng.integers(10, 60))
        elif mode == 1:
            # Pixelated block artifact
            small = cv2.resize(corrupted[by : by + bh, bx : bx + bw], (2, 2))
            corrupted[by : by + bh, bx : bx + bw] = cv2.resize(
                small, (bw, bh), interpolation=cv2.INTER_NEAREST
            )
        else:
            # Random noise block
            noise = rng.integers(0, 255, size=(bh, bw, 3), dtype=np.uint8)
            corrupted[by : by + bh, bx : bx + bw] = noise

    return corrupted


def compute_homography(prev_img_gray, curr_img_gray):
    """
    Use OpenCV ORB feature matching + RANSAC to compute 3x3 inter-frame Homography H.
    """
    if prev_img_gray is None:
        return np.eye(3, dtype=np.float32).flatten().tolist()

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(prev_img_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_img_gray, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.eye(3, dtype=np.float32).flatten().tolist()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 8:
        return np.eye(3, dtype=np.float32).flatten().tolist()

    matches = sorted(matches, key=lambda x: x.distance)[:80]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or not np.all(np.isfinite(H)):
        return np.eye(3, dtype=np.float32).flatten().tolist()

    # Normalize H by H[2,2] and clip extreme values for numerical stability
    if abs(H[2, 2]) > 1e-6:
        H = H / H[2, 2]
    H = np.clip(H, -10.0, 10.0).astype(np.float32)
    return H.flatten().tolist()


def build_rag_knowledge_base(rag_dir: Path):
    """
    Build RAG rules knowledge base (rag_db/rules.json and FAISS index)
    with 3 core traffic safety rules and recommended DP weights [w_obs, w_smooth].
    """
    rag_dir.mkdir(parents=True, exist_ok=True)
    rules = [
        {
            "rule_id": 0,
            "name": "Night_RainFog_Safety_Rule",
            "condition_desc": "夜间或雨雾低能见度复杂气象场景，视觉退化严重，路面湿滑制动距离增加。",
            "trigger_vector": [1, 1, 0, 0, 0, 0, 1, 0],
            "recommended_weights": [2.2, 1.5],
            "advice": "启用高安全裕度避障与高轨迹平滑约束，降低横向突变速度，保持充裕纵向车距。",
        },
        {
            "rule_id": 1,
            "name": "Dense_Pedestrian_SmallObj_Rule",
            "condition_desc": "路口、存在行人、远距小目标或高密度拥堵场景，存在突发横穿动态高危风险。",
            "trigger_vector": [0, 0, 1, 1, 1, 1, 1, 0],
            "recommended_weights": [2.8, 1.2],
            "advice": "最高优先级规避动态行人与小目标障碍物，强化障碍物斥力场权重，提前预留避让通道。",
        },
        {
            "rule_id": 2,
            "name": "Normal_Road_Cruising_Rule",
            "condition_desc": "常规道路正常光照畅通通行场景，视野清晰且周围无密集动态高危冲突。",
            "trigger_vector": [0, 0, 0, 0, 0, 0, 1, 1],
            "recommended_weights": [1.2, 1.8],
            "advice": "维持标准避障权重并提高轨迹平滑巡航权重，保证行驶舒适性与通行效率。",
        },
    ]

    rules_json_path = rag_dir / "rules.json"
    with open(rules_json_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    # Build FAISS L2 index over 8-dim trigger vectors
    vecs = np.array([r["trigger_vector"] for r in rules], dtype=np.float32)
    index = faiss.IndexFlatL2(8)
    index.add(vecs)
    faiss.write_index(index, str(rag_dir / "rules.index"))
    print(f"[INFO] Built RAG knowledge base at {rag_dir} with {index.ntotal} rules.")


def process_nuscenes_mini():
    # Step 2.1: Check Google Drive cache first
    if GDRIVE_TAR_PATH.exists() and GDRIVE_TAR_PATH.stat().st_size > 100_000:
        print(f"[INFO] Found cached dataset archive at {GDRIVE_TAR_PATH}. Extracting...")
        PROCESSED_DIR.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(GDRIVE_TAR_PATH, "r:gz") as tar:
            tar.extractall(path=str(LOCAL_ROOT / "dataset"))
        # Copy train.jsonl and val.jsonl if needed
        for split_name in ["train.jsonl", "val.jsonl"]:
            p_src = PROCESSED_DIR / split_name
            p_dst = LOCAL_ROOT / "dataset" / split_name
            if p_src.exists() and not p_dst.exists():
                shutil.copy2(p_src, p_dst)
        if (PROCESSED_DIR / "train.jsonl").exists() and (PROCESSED_DIR / "val.jsonl").exists():
            print("[INFO] Successfully restored dataset from Google Drive cache!")
            return

    # Step 2.2: Download raw nuScenes v1.0-mini if not cached
    download_and_extract_nuscenes()

    # Create subdirectories
    for sub in ["images_clean", "images_corrupted", "depth_maps", "seg_masks", "small_hm", "risk_maps", "rag_db"]:
        (PROCESSED_DIR / sub).mkdir(parents=True, exist_ok=True)

    build_rag_knowledge_base(PROCESSED_DIR / "rag_db")

    nusc = NuScenes(version="v1.0-mini", dataroot=str(RAW_NUSC_DIR), verbose=False)
    rng = np.random.default_rng(42)

    # Collect all 404 keyframes ordered by scene sequence
    ordered_samples = []
    for scene in nusc.scene:
        tok = scene["first_sample_token"]
        while tok:
            s = nusc.get("sample", tok)
            ordered_samples.append((scene, s))
            tok = s["next"]

    print(f"[INFO] Total keyframes to process: {len(ordered_samples)}")

    records = []
    prev_gray = None
    prev_scene_token = None
    prev_ego_translation = None
    prev_ego_timestamp = None
    prev_ego_yaw = None

    for idx, (scene, sample) in enumerate(tqdm(ordered_samples, desc="Building UniPercep Dataset")):
        scene_token = scene["token"]
        scene_desc = (scene.get("description", "") or "").lower()

        # 1. Camera front image (clean & corrupted, 512x288)
        cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img_path = RAW_NUSC_DIR / cam_data["filename"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((288, 512, 3), dtype=np.uint8)
        else:
            img_bgr = cv2.resize(img_bgr, (512, 288), interpolation=cv2.INTER_AREA)

        corrupted_bgr = corrupt_image(img_bgr, rng)

        clean_rel = f"images_clean/{idx:04d}.jpg"
        corrupt_rel = f"images_corrupted/{idx:04d}.jpg"
        cv2.imwrite(str(PROCESSED_DIR / clean_rel), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(PROCESSED_DIR / corrupt_rel), corrupted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # 2. Inter-frame Homography (3x3 -> 9D)
        curr_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if prev_scene_token != scene_token:
            prev_gray = None
            prev_ego_translation = None
            prev_ego_timestamp = None
            prev_ego_yaw = None

        homography_9d = compute_homography(prev_gray, curr_gray)
        prev_gray = curr_gray
        prev_scene_token = scene_token

        # 3. Ego Pose & Sensor State (5D) + LIDAR_TOP BEV (2x64x64)
        lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
        calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        ego_trans = np.array(ego_pose["translation"], dtype=np.float64)
        ego_quat = Quaternion(ego_pose["rotation"])
        ego_yaw = float(ego_quat.yaw_pitch_roll[0])
        curr_ts = float(ego_pose["timestamp"]) * 1e-6  # seconds

        if prev_ego_translation is not None and prev_ego_timestamp is not None:
            dt = max(0.05, curr_ts - prev_ego_timestamp)
            delta_global = ego_trans - prev_ego_translation
            # Rotate global velocity into ego frame
            cos_y, sin_y = math.cos(-ego_yaw), math.sin(-ego_yaw)
            vx_ego = float(delta_global[0] * cos_y - delta_global[1] * sin_y) / dt
            vy_ego = float(delta_global[0] * sin_y + delta_global[1] * cos_y) / dt
            dyaw = math.atan2(math.sin(ego_yaw - prev_ego_yaw), math.cos(ego_yaw - prev_ego_yaw))
            yaw_rate = float(dyaw / dt)
        else:
            vx_ego, vy_ego, yaw_rate = 5.0, 0.0, 0.0

        prev_ego_translation = ego_trans
        prev_ego_timestamp = curr_ts
        prev_ego_yaw = ego_yaw

        speed = math.sqrt(vx_ego ** 2 + vy_ego ** 2)
        sensor_state = [
            float(np.clip(vx_ego / 15.0, -2.0, 2.0)),
            float(np.clip(vy_ego / 5.0, -2.0, 2.0)),
            float(np.clip(yaw_rate, -2.0, 2.0)),
            float(math.sin(ego_yaw)),
            float(math.cos(ego_yaw)),
        ]

        # Project LIDAR_TOP to ego frame -> 2x64x64 BEV (density & height)
        pcl_path = RAW_NUSC_DIR / lidar_data["filename"]
        bev_map = np.zeros((2, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        if pcl_path.exists():
            pc = LidarPointCloud.from_file(str(pcl_path))
            # Transform from lidar sensor frame to ego vehicle frame
            pc.rotate(Quaternion(calib["rotation"]).rotation_matrix)
            pc.translate(np.array(calib["translation"]))
            pts = pc.points[:3, :]  # (3, N) in ego frame: x=forward, y=left, z=up

            gx, gy = ego_metric_to_grid(pts[0, :], pts[1, :])
            gx_i = np.floor(gx).astype(np.int32)
            gy_i = np.floor(gy).astype(np.int32)
            valid = (gx_i >= 0) & (gx_i < GRID_SIZE) & (gy_i >= 0) & (gy_i < GRID_SIZE)
            gx_v, gy_v, z_v = gx_i[valid], gy_i[valid], pts[2, valid]

            # Channel 0: log point density; Channel 1: normalized max height
            np.add.at(bev_map[0], (gy_v, gx_v), 1.0)
            bev_map[0] = np.clip(np.log1p(bev_map[0]) / 4.0, 0.0, 1.0)

            height_grid = np.full((GRID_SIZE, GRID_SIZE), -3.0, dtype=np.float32)
            np.maximum.at(height_grid, (gy_v, gx_v), z_v)
            mask_has_pts = bev_map[0] > 0
            bev_map[1, mask_has_pts] = np.clip((height_grid[mask_has_pts] + 2.0) / 5.0, 0.0, 1.0)

        depth_rel = f"depth_maps/{idx:04d}.npy"
        np.save(PROCESSED_DIR / depth_rel, bev_map)

        # 4. Annotations in ego frame: Segmentation mask (64x64), Small object heatmap (1x64x64),
        #    Detection boxes, Multi-label (8D), and Spatial Risk Map (64x64)
        seg_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)  # 0: drivable, 1: static, 2: dynamic
        # Mark background non-road regions from high lidar points as static background (1) initially
        static_lidar_mask = (bev_map[1] > 0.55) & (bev_map[0] > 0.25)
        seg_mask[static_lidar_mask] = 1

        small_hm = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        risk_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Coordinate grids for 2D Gaussian kernel overlay
        yy_grid, xx_grid = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE].astype(np.float32)

        # Add weak risk from static lidar obstacles
        if np.any(static_lidar_mask):
            risk_map = np.maximum(risk_map, static_lidar_mask.astype(np.float32) * 0.35)

        det_boxes = []
        has_pedestrian = 0
        has_small_far = 0
        has_vehicle = 0
        num_near_objs = 0

        ego_rot_inv = ego_quat.inverse
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]
            ann_trans = np.array(ann["translation"], dtype=np.float64)
            rel_pos = ego_rot_inv.rotate(ann_trans - ego_trans)
            forward_m, left_m, up_m = float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2])
            dist = math.sqrt(forward_m ** 2 + left_m ** 2)

            gx, gy = ego_metric_to_grid(forward_m, left_m)
            if not (0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE):
                continue

            num_near_objs += 1
            w_m, l_m, h_m = [float(x) for x in ann["size"]]  # width, length, height

            is_ped_or_cone = ("pedestrian" in cat) or ("barrier" in cat) or ("traffic_cone" in cat) or ("bicycle" in cat)
            is_dynamic = ("pedestrian" in cat) or ("vehicle" in cat) or ("cycle" in cat)
            is_small = 1 if (dist > 25.0 or is_ped_or_cone) else 0

            if "pedestrian" in cat:
                has_pedestrian = 1
            if "vehicle" in cat:
                has_vehicle = 1
            if is_small == 1:
                has_small_far = 1

            det_boxes.append(
                {
                    "category": cat,
                    "center_grid": [round(float(gx), 2), round(float(gy), 2)],
                    "size_m": [round(w_m, 2), round(l_m, 2), round(h_m, 2)],
                    "dist_m": round(dist, 2),
                    "is_small": is_small,
                }
            )

            # Fill segmentation mask footprint
            rad_x = max(1, int(round(max(w_m, l_m) * 0.5 / METERS_PER_PIXEL)))
            rad_y = max(1, int(round(max(w_m, l_m) * 0.5 / METERS_PER_PIXEL)))
            x0, x1 = max(0, int(gx) - rad_x), min(GRID_SIZE, int(gx) + rad_x + 1)
            y0, y1 = max(0, int(gy) - rad_y), min(GRID_SIZE, int(gy) + rad_y + 1)
            cls_id = 2 if is_dynamic else 1
            seg_mask[y0:y1, x0:x1] = np.maximum(seg_mask[y0:y1, x0:x1], cls_id)

            # Small object heatmap (Gaussian peak at (gx, gy))
            if is_small == 1:
                sigma_s = 2.0
                g_small = np.exp(-((xx_grid - gx) ** 2 + (yy_grid - gy) ** 2) / (2 * sigma_s ** 2))
                small_hm[0] = np.maximum(small_hm[0], g_small)

            # Spatial risk cost map (2D Gaussian kernel overlay around obstacle)
            sigma_r = 3.2 if is_dynamic else 2.2
            peak_r = 1.0 if is_dynamic else 0.7
            g_risk = peak_r * np.exp(-((xx_grid - gx) ** 2 + (yy_grid - gy) ** 2) / (2 * sigma_r ** 2))
            risk_map = np.maximum(risk_map, g_risk)

        # Add gentle road boundary risk on far left/right edges so DP prefers staying near road center/corridor
        edge_dist = np.minimum(xx_grid, (GRID_SIZE - 1) - xx_grid)
        boundary_risk = np.clip((6.0 - edge_dist) / 6.0, 0.0, 1.0) * 0.3
        risk_map = np.clip(np.maximum(risk_map, boundary_risk), 0.0, 1.0)

        seg_rel = f"seg_masks/{idx:04d}.npy"
        small_rel = f"small_hm/{idx:04d}.npy"
        risk_rel = f"risk_maps/{idx:04d}.npy"
        np.save(PROCESSED_DIR / seg_rel, seg_mask)
        np.save(PROCESSED_DIR / small_rel, small_hm)
        np.save(PROCESSED_DIR / risk_rel, risk_map)

        # 5. Scene 8-dim binary multi-label vector:
        # [is_night, is_rain_fog, is_intersection, has_pedestrian, has_small_far, is_congested, has_vehicle, is_cruising]
        mean_brightness = float(np.mean(img_bgr))
        is_night = 1 if ("night" in scene_desc or mean_brightness < 75.0) else 0
        is_rain_fog = 1 if ("rain" in scene_desc or "wet" in scene_desc or "fog" in scene_desc or (idx % 7 == 0)) else 0
        is_intersection = 1 if ("intersection" in scene_desc or "turn" in scene_desc or "cross" in scene_desc or abs(yaw_rate) > 0.08) else 0
        is_congested = 1 if num_near_objs >= 8 else 0
        is_cruising = 1 if speed > 4.5 else 0

        multi_label = [
            int(is_night),
            int(is_rain_fog),
            int(is_intersection),
            int(has_pedestrian),
            int(has_small_far),
            int(is_congested),
            int(has_vehicle),
            int(is_cruising),
        ]

        # 6. Extract future 6 timesteps ground-truth collision-free trajectory in 64x64 grid (gt_traj: 6x2)
        gt_traj = []
        curr_tok = sample["token"]
        last_valid_pt = [float(EGO_GRID_X), float(EGO_GRID_Y)]
        last_delta = [0.0, 3.5]  # default forward step in grid cells

        for step_i in range(1, 7):
            s_cur = nusc.get("sample", curr_tok)
            if s_cur["next"] != "":
                curr_tok = s_cur["next"]
                s_next = nusc.get("sample", curr_tok)
                sd_next = nusc.get("sample_data", s_next["data"]["LIDAR_TOP"])
                ep_next = nusc.get("ego_pose", sd_next["ego_pose_token"])
                next_trans = np.array(ep_next["translation"], dtype=np.float64)

                rel_next = ego_rot_inv.rotate(next_trans - ego_trans)
                fwd_m, lft_m = float(rel_next[0]), float(rel_next[1])
                gx_t, gy_t = ego_metric_to_grid(fwd_m, lft_m)
                gx_t = float(np.clip(gx_t, 2.0, GRID_SIZE - 3.0))
                gy_t = float(np.clip(gy_t, EGO_GRID_Y + step_i * 1.5, GRID_SIZE - 2.0))
                pt = [round(gx_t, 3), round(gy_t, 3)]
                last_delta = [
                    pt[0] - last_valid_pt[0],
                    max(2.0, pt[1] - last_valid_pt[1]),
                ]
                last_valid_pt = pt
            else:
                # Extrapolate smoothly when reaching the end of a scene
                gx_t = float(np.clip(last_valid_pt[0] + last_delta[0], 2.0, GRID_SIZE - 3.0))
                gy_t = float(np.clip(last_valid_pt[1] + last_delta[1], EGO_GRID_Y + step_i * 1.5, GRID_SIZE - 2.0))
                last_valid_pt = [round(gx_t, 3), round(gy_t, 3)]

            gt_traj.append(last_valid_pt)

        records.append(
            {
                "frame_idx": idx,
                "sample_token": sample["token"],
                "scene_name": scene["name"],
                "clean_img": clean_rel,
                "corrupted_img": corrupt_rel,
                "depth_map": depth_rel,
                "seg_mask": seg_rel,
                "small_hm": small_rel,
                "risk_map": risk_rel,
                "homography": [round(float(x), 5) for x in homography_9d],
                "sensor_state": [round(float(x), 5) for x in sensor_state],
                "multi_label": multi_label,
                "det_boxes": det_boxes,
                "gt_traj": gt_traj,
            }
        )

    # Split: first 340 frames -> train.jsonl, last 64 frames -> val.jsonl
    train_records = records[:340]
    val_records = records[340:404]

    for split_name, split_data in [("train.jsonl", train_records), ("val.jsonl", val_records)]:
        for target_dir in [PROCESSED_DIR, LOCAL_ROOT / "dataset", GDRIVE_ROOT / "dataset"]:
            target_dir.mkdir(parents=True, exist_ok=True)
            out_file = target_dir / split_name
            with open(out_file, "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote train.jsonl ({len(train_records)} frames) and val.jsonl ({len(val_records)} frames).")

    # Step 2.4: Archive processed_unipercep to Google Drive
    GDRIVE_TAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Archiving {PROCESSED_DIR} to {GDRIVE_TAR_PATH} ...")
    with tarfile.open(GDRIVE_TAR_PATH, "w:gz") as tar:
        tar.add(str(PROCESSED_DIR), arcname="processed_unipercep")
    os.sync()
    print(f"[INFO] Dataset archive synced to Google Drive ({GDRIVE_TAR_PATH.stat().st_size / 1e6:.2f} MB).")


if __name__ == "__main__":
    process_nuscenes_mini()
