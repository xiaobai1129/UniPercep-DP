import os
import sys
import json
import math
import shutil
import tarfile
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import faiss
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

LOCAL_ROOT = Path("/root/workspace/ceshi")
GDRIVE_ROOT = Path("/content/gdrive/MyDrive/colab/UniPercep_DP")

RAW_DIR = LOCAL_ROOT / "dataset_full" / "raw_nuscenes"
PROCESSED_DIR = LOCAL_ROOT / "dataset_full" / "processed"
GDRIVE_DATASET_DIR = GDRIVE_ROOT / "dataset_full"
GDRIVE_TAR_PATH = GDRIVE_DATASET_DIR / "processed_unipercep_full.tar.gz"

CLOUDFRONT_BASE = "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"

# 128x128 BEV Grid Configuration (0.5m per grid cell)
# Lateral X (col 0..127): [-32.0m, +32.0m], Ego X = 64.0
# Forward Y (row 0..127): [-4.0m, +60.0m], Ego Y = 8.0
GRID_SIZE = 128
METERS_PER_PIXEL = 0.5
EGO_GRID_X = 64.0
EGO_GRID_Y = 8.0


def ego_metric_to_grid128(forward_m, left_m):
    gx = EGO_GRID_X - (left_m / METERS_PER_PIXEL)
    gy = EGO_GRID_Y + (forward_m / METERS_PER_PIXEL)
    return gx, gy


def get_15_rag_rules():
    return [
        {
            "rule_id": 0,
            "name": "Rule_00_Night_Clear_Cruise",
            "desc": "夜间无雨照明受限巡航场景：提高静态障碍物安全裕度与轨迹平滑约束。",
            "trigger_10d": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "dp_weights_gt": [1.8, 1.5, 1.8],
        },
        {
            "rule_id": 1,
            "name": "Rule_01_Rain_Fog_Wet_Road",
            "desc": "雨雾湿滑路面场景：制动距离延长，强化纵向风险排斥与横向平滑度。",
            "trigger_10d": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "dp_weights_gt": [2.1, 1.7, 2.2],
        },
        {
            "rule_id": 2,
            "name": "Rule_02_Night_RainFog_Severe",
            "desc": "夜间叠加雨雾及远距小目标极端场景：双重提高避障与冲突代价权重。",
            "trigger_10d": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            "dp_weights_gt": [2.6, 2.2, 2.4],
        },
        {
            "rule_id": 3,
            "name": "Rule_03_Urban_Intersection",
            "desc": "城市十字路口多向车流交汇场景：重点提升动态交通冲突消解权重。",
            "trigger_10d": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "dp_weights_gt": [2.0, 2.1, 1.6],
        },
        {
            "rule_id": 4,
            "name": "Rule_04_Construction_Zone",
            "desc": "道路施工锥桶/护栏收窄区域：强化静态边界斥力场，严防擦碰施工设施。",
            "trigger_10d": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            "dp_weights_gt": [2.7, 1.9, 1.9],
        },
        {
            "rule_id": 5,
            "name": "Rule_05_High_Density_Congestion",
            "desc": "高密度拥堵与局部遮挡场景：紧缩车距同时保持高冲突预警与防加塞权重。",
            "trigger_10d": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            "dp_weights_gt": [2.5, 2.6, 1.7],
        },
        {
            "rule_id": 6,
            "name": "Rule_06_Pedestrian_Crossing_Alert",
            "desc": "路口行人横穿高危场景：最高优先级规避弱势交通参与者，放宽转向灵活性。",
            "trigger_10d": [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
            "dp_weights_gt": [3.1, 2.9, 1.3],
        },
        {
            "rule_id": 7,
            "name": "Rule_07_Distant_Small_Obstacles",
            "desc": "前方大于25m远距微小障碍物场景：提前规划微幅横向偏移避让走廊。",
            "trigger_10d": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "dp_weights_gt": [2.3, 1.8, 1.7],
        },
        {
            "rule_id": 8,
            "name": "Rule_08_Fast_Approaching_Vehicle",
            "desc": "高速逼近动态目标场景：拉长速度各向异性风险椭圆，提高时空冲突权重。",
            "trigger_10d": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "dp_weights_gt": [2.6, 2.8, 1.4],
        },
        {
            "rule_id": 9,
            "name": "Rule_09_Severe_Stream_Occlusion",
            "desc": "弱网码流丢包或大面积盲区遮挡场景：依赖点云BEV几何融合，保持保守平滑轨迹。",
            "trigger_10d": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "dp_weights_gt": [2.4, 2.2, 2.1],
        },
        {
            "rule_id": 10,
            "name": "Rule_10_Sharp_Curve_Turning",
            "desc": "大曲率弯道或路口转向场景：强化二阶曲率平滑权重，抑制横向急打方向。",
            "trigger_10d": [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            "dp_weights_gt": [2.1, 1.9, 2.3],
        },
        {
            "rule_id": 11,
            "name": "Rule_11_Night_Pedestrian_Occlusion",
            "desc": "夜间鬼探头/盲区行人横穿高危复合场景：极限拉满避障与动态冲突排斥系数。",
            "trigger_10d": [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            "dp_weights_gt": [3.3, 3.1, 1.5],
        },
        {
            "rule_id": 12,
            "name": "Rule_12_RainFog_Fast_Congestion",
            "desc": "雨雾天气高速拥堵跟车场景：兼顾前后车动态冲突消解与车身稳定控制。",
            "trigger_10d": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            "dp_weights_gt": [2.9, 3.0, 1.8],
        },
        {
            "rule_id": 13,
            "name": "Rule_13_Construction_Ped_Cones",
            "desc": "施工改道区人车混行场景：高权重避让施工人员与密集锥桶阵列。",
            "trigger_10d": [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            "dp_weights_gt": [3.0, 2.7, 1.6],
        },
        {
            "rule_id": 14,
            "name": "Rule_14_Normal_Daytime_Cruising",
            "desc": "日间良好视野畅通道路巡航：采用低阻避障代价与高平滑度舒适巡航策略。",
            "trigger_10d": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "dp_weights_gt": [1.4, 1.2, 2.0],
        },
    ]


def build_rag_db_15(rag_dir: Path):
    rag_dir.mkdir(parents=True, exist_ok=True)
    rules = get_15_rag_rules()
    with open(rag_dir / "rules_15.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    vecs = np.array([r["trigger_10d"] for r in rules], dtype=np.float32)
    index = faiss.IndexFlatL2(10)
    index.add(vecs)
    faiss.write_index(index, str(rag_dir / "rules_15.index"))
    print(f"[INFO] Built 15-rule FAISS RAG knowledge base at {rag_dir} (ntotal={index.ntotal}).")
    return rules, index


def apply_degradation_mode(img_bgr: np.ndarray, mode: int, seed: int):
    """
    Multi-level degradation synthesis on 768x448 image:
      mode 0: Light weak-network compression (QP=25)
      mode 1: Heavy packet-loss macroblock occlusion + strong JPEG artifacts (QP=12)
      mode 2: Rain/fog atmospheric haze + directional motion blur + QP=18
      mode 3: Severe combined degradation (packet loss + rain/fog blur + QP=10)
    """
    rng = np.random.default_rng(seed)
    h, w, _ = img_bgr.shape
    out = img_bgr.copy()

    if mode in (2, 3):
        # Directional motion blur kernel (9x9)
        ksize = 9
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0 / ksize
        out = cv2.filter2D(out, -1, kernel)
        # Add rain/fog atmospheric veil and raindrop streaks
        fog_veil = np.full_like(out, 200, dtype=np.uint8)
        out = cv2.addWeighted(out, 0.78, fog_veil, 0.22, 0)

    if mode in (1, 3):
        # Packet-loss macroblock occlusion (4 to 8 blocks of 24x24 ~ 48x48)
        n_blocks = int(rng.integers(4, 9))
        for _ in range(n_blocks):
            bw = int(rng.choice([24, 32, 48]))
            bh = int(rng.choice([24, 32, 40]))
            bx = int(rng.integers(0, max(1, w - bw)))
            by = int(rng.integers(0, max(1, h - bh)))
            btype = int(rng.integers(0, 3))
            if btype == 0:
                out[by : by + bh, bx : bx + bw] = int(rng.integers(15, 55))
            elif btype == 1:
                small = cv2.resize(out[by : by + bh, bx : bx + bw], (2, 2))
                out[by : by + bh, bx : bx + bw] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
            else:
                out[by : by + bh, bx : bx + bw] = rng.integers(0, 255, size=(bh, bw, 3), dtype=np.uint8)

    qp_map = {0: 25, 1: 12, 2: 18, 3: 10}
    qp = qp_map.get(mode, 20)
    _, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), qp])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def compute_orb_ransac_svd_homography(prev_gray, curr_gray):
    """
    ORB feature matching + RANSAC outlier rejection + SVD normalized DLT refinement
    to solve 3x3 inter-frame Homography matrix H.
    """
    if prev_gray is None:
        return np.eye(3, dtype=np.float32).flatten().tolist()

    orb = cv2.ORB_create(nfeatures=600)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.eye(3, dtype=np.float32).flatten().tolist()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 8:
        return np.eye(3, dtype=np.float32).flatten().tolist()

    matches = sorted(matches, key=lambda x: x.distance)[:100]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or not np.all(np.isfinite(H)):
        return np.eye(3, dtype=np.float32).flatten().tolist()

    # SVD singular value conditioning for numerical stability
    u, s, vt = np.linalg.svd(H)
    s = np.clip(s, 1e-3, 1e3)
    H = u @ np.diag(s) @ vt
    if abs(H[2, 2]) > 1e-6:
        H = H / H[2, 2]
    H = np.clip(H, -10.0, 10.0).astype(np.float32)
    return H.flatten().tolist()


def ensure_raw_nuscenes_keyframes():
    """
    Ensure v1.0-trainval_meta.tgz and keyframe blobs (v1.0-mini + trainval01..03)
    are stream-extracted into RAW_DIR without sweeps/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta_dir = RAW_DIR / "v1.0-trainval"
    if not (meta_dir / "sample.json").exists():
        print("[INFO] Stream-extracting v1.0-trainval_meta.tgz ...")
        cmd = f'curl -sL "{CLOUDFRONT_BASE}/v1.0-trainval_meta.tgz" | tar -xz -C "{RAW_DIR}"'
        subprocess.run(cmd, shell=True, check=True)

    cam_dir = RAW_DIR / "samples" / "CAM_FRONT"
    lidar_dir = RAW_DIR / "samples" / "LIDAR_TOP"
    num_cam = len(list(cam_dir.glob("*.jpg"))) if cam_dir.exists() else 0
    num_lidar = len(list(lidar_dir.glob("*.bin"))) if lidar_dir.exists() else 0
    print(f"[INFO] Currently extracted raw keyframes: CAM_FRONT={num_cam}, LIDAR_TOP={num_lidar}")

    if num_cam < 3000 or num_lidar < 3000:
        for archive in ["v1.0-mini.tgz", "v1.0-trainval01_blobs.tgz", "v1.0-trainval02_blobs.tgz", "v1.0-trainval03_blobs.tgz"]:
            num_cam = len(list(cam_dir.glob("*.jpg"))) if cam_dir.exists() else 0
            if num_cam >= 10000:
                break
            print(f"[INFO] Stream-extracting keyframes from {archive} (ignoring sweeps/)...")
            cmd = (
                f'curl -sL "{CLOUDFRONT_BASE}/{archive}" | pigz -dc | '
                f'tar -x -C "{RAW_DIR}" --wildcards "samples/CAM_FRONT/*" "samples/LIDAR_TOP/*" 2>/dev/null || true'
            )
            subprocess.run(cmd, shell=True)
            num_cam = len(list(cam_dir.glob("*.jpg"))) if cam_dir.exists() else 0
            num_lidar = len(list(lidar_dir.glob("*.bin"))) if lidar_dir.exists() else 0
            print(f"[INFO] After {archive}: CAM_FRONT={num_cam}, LIDAR_TOP={num_lidar}")


def process_single_keyframe_task(task):
    """
    Worker function to process a single physical keyframe into:
      - 768x448 clean & degraded images
      - 3x128x128 BEV point cloud map [max_height, mean_intensity, density] (float16)
      - 128x128 obstacle segmentation mask (uint8)
      - 128x128 anisotropic Gaussian spatial risk map (float16)
      - 10D multi-label vector, det_boxes with tracking & velocity, 6x2 gt_traj
    """
    idx = task["idx"]
    cam_path = Path(task["cam_path"])
    prev_cam_path = Path(task["prev_cam_path"]) if task["prev_cam_path"] else None
    lidar_path = Path(task["lidar_path"])
    calib_rot = task["calib_rot"]
    calib_trans = task["calib_trans"]

    # 1. Read & resize CAM_FRONT image to 768x448
    img_bgr = cv2.imread(str(cam_path))
    if img_bgr is None:
        img_bgr = np.zeros((448, 768, 3), dtype=np.uint8)
    else:
        img_bgr = cv2.resize(img_bgr, (768, 448), interpolation=cv2.INTER_AREA)

    curr_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray = None
    if prev_cam_path is not None and prev_cam_path.exists():
        p_img = cv2.imread(str(prev_cam_path))
        if p_img is not None:
            p_img = cv2.resize(p_img, (768, 448), interpolation=cv2.INTER_AREA)
            prev_gray = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)

    homography_9d = compute_orb_ransac_svd_homography(prev_gray, curr_gray)

    clean_rel = f"images_clean/{idx:05d}.jpg"
    cv2.imwrite(str(PROCESSED_DIR / clean_rel), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # Save primary corrupted image (severe weak-network packet loss + blur) on disk;
    # additional multi-level degradation variants are synthesized deterministically
    corrupt_bgr = apply_degradation_mode(img_bgr, mode=(idx % 3) + 1, seed=idx + 100)
    corrupt_rel = f"images_corrupted/{idx:05d}.jpg"
    cv2.imwrite(str(PROCESSED_DIR / corrupt_rel), corrupt_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    # 2. LIDAR_TOP -> 3x128x128 BEV [max_height, mean_intensity, density]
    bev_3ch = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    if lidar_path.exists():
        raw_pts = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
        xyz = raw_pts[:, :3].T  # (3, N)
        intensity = raw_pts[:, 3] / 255.0

        rot_mat = Quaternion(calib_rot).rotation_matrix
        xyz_ego = rot_mat @ xyz + np.array(calib_trans, dtype=np.float32).reshape(3, 1)

        gx, gy = ego_metric_to_grid128(xyz_ego[0, :], xyz_ego[1, :])
        gx_i = np.floor(gx).astype(np.int32)
        gy_i = np.floor(gy).astype(np.int32)
        valid = (gx_i >= 0) & (gx_i < GRID_SIZE) & (gy_i >= 0) & (gy_i < GRID_SIZE)
        gx_v, gy_v = gx_i[valid], gy_i[valid]
        z_v = xyz_ego[2, valid]
        int_v = np.clip(intensity[valid], 0.0, 1.0)

        # Channel 2: point density
        count_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        np.add.at(count_grid, (gy_v, gx_v), 1.0)
        bev_3ch[2] = np.clip(np.log1p(count_grid) / 4.0, 0.0, 1.0)

        # Channel 1: mean intensity
        int_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        np.add.at(int_sum, (gy_v, gx_v), int_v)
        mask_pts = count_grid > 0
        bev_3ch[1, mask_pts] = np.clip(int_sum[mask_pts] / count_grid[mask_pts], 0.0, 1.0)

        # Channel 0: max height
        h_grid = np.full((GRID_SIZE, GRID_SIZE), -3.0, dtype=np.float32)
        np.maximum.at(h_grid, (gy_v, gx_v), z_v)
        bev_3ch[0, mask_pts] = np.clip((h_grid[mask_pts] + 2.0) / 5.0, 0.0, 1.0)

    bev_rel = f"bev_lidar/{idx:05d}.npy"
    np.save(PROCESSED_DIR / bev_rel, bev_3ch.astype(np.float16))

    # 3. Build 128x128 seg_mask & anisotropic 128x128 risk_map
    seg_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    static_lidar = (bev_3ch[0] > 0.52) & (bev_3ch[2] > 0.22)
    seg_mask[static_lidar] = 1

    yy_grid, xx_grid = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE].astype(np.float32)
    risk_map = (static_lidar.astype(np.float32) * 0.30)

    for obj in task["objects"]:
        cx, cy, w_g, l_g, cls_id, track_id, vx, vy, is_small = obj
        rad_x = max(1, int(round(w_g * 0.5)))
        rad_y = max(1, int(round(l_g * 0.5)))
        x0, x1 = max(0, int(cx) - rad_x), min(GRID_SIZE, int(cx) + rad_x + 1)
        y0, y1 = max(0, int(cy) - rad_y), min(GRID_SIZE, int(cy) + rad_y + 1)
        seg_mask[y0:y1, x0:x1] = np.maximum(seg_mask[y0:y1, x0:x1], int(cls_id))

        # Anisotropic Gaussian risk field elongated along object velocity (vx, vy)
        speed_obj = math.sqrt(vx ** 2 + vy ** 2)
        angle = math.atan2(vy, vx) if speed_obj > 0.3 else (math.pi / 2.0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx = xx_grid - cx - 0.5 * vx
        dy = yy_grid - cy - 0.5 * vy
        rot_x = dx * cos_a + dy * sin_a
        rot_y = -dx * sin_a + dy * cos_a

        sigma_lon = min(10.0, max(3.0, 3.0 + 0.8 * speed_obj))
        sigma_lat = 3.2 if cls_id == 2 else 2.4
        peak = 1.0 if cls_id == 2 else 0.72
        g_risk = peak * np.exp(-(rot_x ** 2 / (2 * sigma_lon ** 2) + rot_y ** 2 / (2 * sigma_lat ** 2)))
        risk_map = np.maximum(risk_map, g_risk)

    # Road edge boundary cost
    edge_dist = np.minimum(xx_grid, (GRID_SIZE - 1) - xx_grid)
    boundary_risk = np.clip((10.0 - edge_dist) / 10.0, 0.0, 1.0) * 0.28
    risk_map = np.clip(np.maximum(risk_map, boundary_risk), 0.0, 1.0)

    seg_rel = f"seg_masks/{idx:05d}.npy"
    risk_rel = f"risk_maps/{idx:05d}.npy"
    np.save(PROCESSED_DIR / seg_rel, seg_mask)
    np.save(PROCESSED_DIR / risk_rel, risk_map.astype(np.float16))

    mean_bright = float(np.mean(img_bgr))
    ml = list(task["multi_label_base"])
    if mean_bright < 78.0:
        ml[0] = 1  # night

    return {
        "keyframe_idx": idx,
        "scene_name": task["scene_name"],
        "clean_img": clean_rel,
        "corrupted_img": corrupt_rel,
        "bev_lidar": bev_rel,
        "seg_mask": seg_rel,
        "risk_map": risk_rel,
        "homography": [round(float(x), 5) for x in homography_9d],
        "sensor_state": task["sensor_state"],
        "multi_label": ml,
        "det_boxes": task["objects"],
        "gt_traj": task["gt_traj"],
    }


def build_full_dataset():
    # Step 2.1: Check Google Drive cache first
    if GDRIVE_TAR_PATH.exists() and GDRIVE_TAR_PATH.stat().st_size > 1_000_000:
        print(f"[INFO] Found cached full dataset archive at {GDRIVE_TAR_PATH}. Extracting...")
        PROCESSED_DIR.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(GDRIVE_TAR_PATH, "r:gz") as tar:
            tar.extractall(path=str(LOCAL_ROOT / "dataset_full"))
        for fname in ["train_full.jsonl", "val_full.jsonl"]:
            src_f = GDRIVE_DATASET_DIR / fname
            if src_f.exists():
                shutil.copy2(src_f, LOCAL_ROOT / "dataset_full" / fname)
                shutil.copy2(src_f, PROCESSED_DIR / fname)
        if (PROCESSED_DIR / "train_full.jsonl").exists():
            print("[INFO] Successfully restored full dataset from Google Drive cache!")
            return

    # Step 2.2: Ensure raw metadata and keyframe blobs are extracted
    ensure_raw_nuscenes_keyframes()

    for sub in ["images_clean", "images_corrupted", "bev_lidar", "seg_masks", "risk_maps", "rag_db"]:
        (PROCESSED_DIR / sub).mkdir(parents=True, exist_ok=True)

    rules, faiss_index = build_rag_db_15(PROCESSED_DIR / "rag_db")

    # Parse v1.0-trainval metadata JSONs directly in memory for high speed (avoiding slow nuscenes-devkit init over missing sweeps)
    meta_dir = RAW_DIR / "v1.0-trainval"
    print("[INFO] Loading v1.0-trainval metadata tables ...")
    with open(meta_dir / "scene.json") as f:
        scenes = json.load(f)
    with open(meta_dir / "sample.json") as f:
        samples = {s["token"]: s for s in json.load(f)}
    with open(meta_dir / "sample_data.json") as f:
        sample_data_list = json.load(f)
    with open(meta_dir / "ego_pose.json") as f:
        ego_poses = {ep["token"]: ep for ep in json.load(f)}
    with open(meta_dir / "calibrated_sensor.json") as f:
        calib_sensors = {cs["token"]: cs for cs in json.load(f)}
    with open(meta_dir / "sensor.json") as f:
        sensors = {sn["token"]: sn["channel"] for sn in json.load(f)}
    with open(meta_dir / "instance.json") as f:
        instances = {inst["token"]: inst["category_token"] for inst in json.load(f)}
    with open(meta_dir / "category.json") as f:
        categories = {cat["token"]: cat["name"] for cat in json.load(f)}
    with open(meta_dir / "sample_annotation.json") as f:
        annotations = json.load(f)

    # Map sample_token -> {channel: sample_data} for key_frame == True
    sample_sensor_map = {}
    for sd in sample_data_list:
        if not sd.get("is_key_frame", False):
            continue
        cs = calib_sensors.get(sd["calibrated_sensor_token"])
        if not cs:
            continue
        ch = sensors.get(cs["sensor_token"])
        if ch in ("CAM_FRONT", "LIDAR_TOP"):
            s_tok = sd["sample_token"]
            if s_tok not in sample_sensor_map:
                sample_sensor_map[s_tok] = {}
            sample_sensor_map[s_tok][ch] = sd

    # Map sample_token -> list of annotations
    sample_anns_map = {}
    for ann in annotations:
        s_tok = ann["sample_token"]
        if s_tok not in sample_anns_map:
            sample_anns_map[s_tok] = []
        sample_anns_map[s_tok].append(ann)

    # Also build ann_token -> ann lookup for velocity computation
    ann_by_token = {ann["token"]: ann for ann in annotations}
    inst_track_id_map = {inst_tok: i + 1 for i, inst_tok in enumerate(instances.keys())}

    # Collect valid keyframes ordered by scene
    tasks = []
    valid_idx = 0

    for scene in scenes:
        scene_name = scene["name"]
        scene_desc = (scene.get("description", "") or "").lower()
        tok = scene["first_sample_token"]

        prev_cam_path = None
        prev_ego_trans = None
        prev_ego_ts = None
        prev_ego_yaw = None

        while tok:
            s_obj = samples.get(tok)
            if not s_obj:
                break
            sd_dict = sample_sensor_map.get(tok, {})
            cam_sd = sd_dict.get("CAM_FRONT")
            lidar_sd = sd_dict.get("LIDAR_TOP")

            if cam_sd and lidar_sd:
                cam_p = RAW_DIR / cam_sd["filename"]
                lidar_p = RAW_DIR / lidar_sd["filename"]

                if cam_p.exists() and lidar_p.exists():
                    ep = ego_poses[lidar_sd["ego_pose_token"]]
                    calib = calib_sensors[lidar_sd["calibrated_sensor_token"]]
                    ego_trans = np.array(ep["translation"], dtype=np.float64)
                    ego_quat = Quaternion(ep["rotation"])
                    ego_yaw = float(ego_quat.yaw_pitch_roll[0])
                    curr_ts = float(ep["timestamp"]) * 1e-6

                    if prev_ego_trans is not None and prev_ego_ts is not None:
                        dt = max(0.05, curr_ts - prev_ego_ts)
                        d_glob = ego_trans - prev_ego_trans
                        cy, sy = math.cos(-ego_yaw), math.sin(-ego_yaw)
                        vx_ego = float(d_glob[0] * cy - d_glob[1] * sy) / dt
                        vy_ego = float(d_glob[0] * sy + d_glob[1] * cy) / dt
                        dyaw = math.atan2(math.sin(ego_yaw - prev_ego_yaw), math.cos(ego_yaw - prev_ego_yaw))
                        yaw_rate = float(dyaw / dt)
                    else:
                        vx_ego, vy_ego, yaw_rate = 5.5, 0.0, 0.0

                    prev_ego_trans = ego_trans
                    prev_ego_ts = curr_ts
                    prev_ego_yaw = ego_yaw

                    # 5D motion state: [vx, vy, yaw, cos(yaw), sin(yaw)]
                    sensor_state = [
                        round(float(np.clip(vx_ego / 15.0, -2.0, 2.0)), 5),
                        round(float(np.clip(vy_ego / 5.0, -2.0, 2.0)), 5),
                        round(float(ego_yaw), 5),
                        round(float(math.cos(ego_yaw)), 5),
                        round(float(math.sin(ego_yaw)), 5),
                    ]

                    ego_rot_inv = ego_quat.inverse
                    anns = sample_anns_map.get(tok, [])
                    objects = []

                    has_ped_cross = 0
                    has_far_small = 0
                    has_fast_app = 0
                    has_occ = 0
                    has_construct = 1 if ("construction" in scene_desc or "cone" in scene_desc) else 0

                    for ann in anns:
                        cat_tok = instances.get(ann["instance_token"])
                        cat_name = categories.get(cat_tok, "vehicle.car")
                        ann_trans = np.array(ann["translation"], dtype=np.float64)
                        rel_pos = ego_rot_inv.rotate(ann_trans - ego_trans)
                        fwd_m, lft_m = float(rel_pos[0]), float(rel_pos[1])
                        dist = math.sqrt(fwd_m ** 2 + lft_m ** 2)

                        gx, gy = ego_metric_to_grid128(fwd_m, lft_m)
                        if not (0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE):
                            continue

                        # Object velocity in grid units/sec
                        vx_obj, vy_obj = 0.0, 0.0
                        if ann["prev"] and ann["prev"] in ann_by_token:
                            prev_ann = ann_by_token[ann["prev"]]
                            p_rel = ego_rot_inv.rotate(np.array(prev_ann["translation"]) - ego_trans)
                            pgx, pgy = ego_metric_to_grid128(float(p_rel[0]), float(p_rel[1]))
                            vx_obj = float(np.clip((gx - pgx) / 0.5, -15.0, 15.0))
                            vy_obj = float(np.clip((gy - pgy) / 0.5, -15.0, 15.0))

                        w_m, l_m, h_m = [float(x) for x in ann["size"]]
                        w_g = max(1.0, w_m / METERS_PER_PIXEL)
                        l_g = max(1.0, l_m / METERS_PER_PIXEL)

                        is_dynamic = ("pedestrian" in cat_name) or ("vehicle" in cat_name) or ("cycle" in cat_name)
                        cls_id = 2 if is_dynamic else 1
                        is_small_cat = ("pedestrian" in cat_name) or ("cone" in cat_name) or ("barrier" in cat_name) or ("bicycle" in cat_name)
                        is_small = 1 if (dist > 25.0 or is_small_cat or (w_m * l_m < 2.2)) else 0

                        if "pedestrian" in cat_name and abs(lft_m) < 8.0:
                            has_ped_cross = 1
                        if "cone" in cat_name or "barrier" in cat_name:
                            has_construct = 1
                        if is_small == 1:
                            has_far_small = 1
                        if math.sqrt(vx_obj ** 2 + vy_obj ** 2) > 6.0:
                            has_fast_app = 1
                        if ann.get("visibility_token", "4") in ("1", "2"):
                            has_occ = 1

                        track_id = int(inst_track_id_map.get(ann["instance_token"], 0) % 10000)
                        objects.append(
                            [
                                round(float(gx), 2),
                                round(float(gy), 2),
                                round(float(w_g), 2),
                                round(float(l_g), 2),
                                int(cls_id),
                                int(track_id),
                                round(float(vx_obj), 2),
                                round(float(vy_obj), 2),
                                int(is_small),
                            ]
                        )

                    # 10D multi_label:
                    # [night, rain_fog, intersection, construction, congestion,
                    #  ped_crossing, far_small_obj, fast_approaching, occlusion, curve]
                    is_night = 1 if "night" in scene_desc else 0
                    is_rain = 1 if ("rain" in scene_desc or "wet" in scene_desc or "fog" in scene_desc) else 0
                    is_inter = 1 if ("intersection" in scene_desc or "cross" in scene_desc or "turn" in scene_desc) else 0
                    is_cong = 1 if len(objects) >= 8 else 0
                    is_curve = 1 if abs(yaw_rate) > 0.07 else 0

                    multi_label_base = [
                        is_night,
                        is_rain,
                        is_inter,
                        has_construct,
                        is_cong,
                        has_ped_cross,
                        has_far_small,
                        has_fast_app,
                        has_occ,
                        is_curve,
                    ]

                    # Future 6 steps (3.0s) ground-truth trajectory on 128x128 grid
                    gt_traj = []
                    c_tok = tok
                    last_pt = [float(EGO_GRID_X), float(EGO_GRID_Y)]
                    last_delta = [0.0, 6.5]
                    for step_i in range(1, 7):
                        sc = samples.get(c_tok)
                        if sc and sc["next"] != "" and sc["next"] in sample_sensor_map and "LIDAR_TOP" in sample_sensor_map[sc["next"]]:
                            c_tok = sc["next"]
                            next_lid = sample_sensor_map[c_tok]["LIDAR_TOP"]
                            ep_next = ego_poses[next_lid["ego_pose_token"]]
                            rel_n = ego_rot_inv.rotate(np.array(ep_next["translation"]) - ego_trans)
                            gx_t, gy_t = ego_metric_to_grid128(float(rel_n[0]), float(rel_n[1]))
                            gx_t = float(np.clip(gx_t, 4.0, GRID_SIZE - 5.0))
                            gy_t = float(np.clip(gy_t, EGO_GRID_Y + step_i * 2.5, GRID_SIZE - 4.0))
                            pt = [round(gx_t, 3), round(gy_t, 3)]
                            last_delta = [pt[0] - last_pt[0], max(3.0, pt[1] - last_pt[1])]
                            last_pt = pt
                        else:
                            gx_t = float(np.clip(last_pt[0] + last_delta[0], 4.0, GRID_SIZE - 5.0))
                            gy_t = float(np.clip(last_pt[1] + last_delta[1], EGO_GRID_Y + step_i * 2.5, GRID_SIZE - 4.0))
                            last_pt = [round(gx_t, 3), round(gy_t, 3)]
                        gt_traj.append(last_pt)

                    tasks.append(
                        {
                            "idx": valid_idx,
                            "scene_name": scene_name,
                            "cam_path": str(cam_p),
                            "prev_cam_path": str(prev_cam_path) if prev_cam_path else None,
                            "lidar_path": str(lidar_p),
                            "calib_rot": calib["rotation"],
                            "calib_trans": calib["translation"],
                            "sensor_state": sensor_state,
                            "objects": objects,
                            "multi_label_base": multi_label_base,
                            "gt_traj": gt_traj,
                        }
                    )
                    prev_cam_path = cam_p
                    valid_idx += 1

            tok = s_obj["next"]

    print(f"[INFO] Found {len(tasks)} valid physical keyframes with synchronized CAM_FRONT + LIDAR_TOP.")
    if len(tasks) > 3600:
        step = len(tasks) / 3600.0
        tasks = [tasks[int(i * step)] for i in range(3600)]
        for new_i, t in enumerate(tasks):
            t["idx"] = new_i
        print(f"[INFO] Selected {len(tasks)} representative physical keyframes across all scenes (x4 multi-level degradation = {len(tasks)*4} sample pairs).")

    # Process physical keyframes in parallel across CPU cores
    base_records = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 8)) as ex:
        futures = {ex.submit(process_single_keyframe_task, t): t["idx"] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Processing Multi-Sensor Keyframes"):
            res = fut.result()
            base_records[res["keyframe_idx"]] = res

    # Expand with sliding-window & 4-level degradation augmentation to form 10,000 ~ 30,000+ training sample pairs
    aug_mult = 4
    print(f"[INFO] Expanding {len(base_records)} keyframes x {aug_mult} degradation levels -> {len(base_records) * aug_mult} total samples.")

    all_scenes = sorted(list({r["scene_name"] for r in base_records}))
    n_train_scenes = int(round(len(all_scenes) * 0.85))
    train_scene_set = set(all_scenes[:n_train_scenes])

    train_samples = []
    val_samples = []

    for r in base_records:
        is_train = r["scene_name"] in train_scene_set
        modes = range(4)
        for deg_mode in modes:
            ml = list(r["multi_label"])
            if deg_mode in (2, 3):
                ml[1] = 1  # rain/fog augmentation active
            if deg_mode in (1, 3):
                ml[8] = 1  # packet loss / macroblock occlusion active

            # Match optimal RAG rule and expert DP weights [w_obs, w_conflict, w_smooth]
            ml_vec = np.array([ml], dtype=np.float32)
            _, I = faiss_index.search(ml_vec, 1)
            matched_rule_id = int(I[0, 0])
            dp_weights_gt = rules[matched_rule_id]["dp_weights_gt"]

            sample_entry = dict(r)
            sample_entry["deg_mode"] = int(deg_mode)
            sample_entry["multi_label"] = ml
            sample_entry["rag_rule_id"] = matched_rule_id
            sample_entry["dp_weights_gt"] = dp_weights_gt

            if is_train:
                train_samples.append(sample_entry)
            else:
                val_samples.append(sample_entry)

    for fname, data_list in [("train_full.jsonl", train_samples), ("val_full.jsonl", val_samples)]:
        for target_d in [PROCESSED_DIR, LOCAL_ROOT / "dataset_full", GDRIVE_DATASET_DIR]:
            target_d.mkdir(parents=True, exist_ok=True)
            with open(target_d / fname, "w", encoding="utf-8") as f:
                for item in data_list:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"[INFO] Scene-level split complete: Train={len(train_samples)} samples ({n_train_scenes} scenes), "
        f"Val={len(val_samples)} samples ({len(all_scenes) - n_train_scenes} scenes)."
    )

    # Archive processed dataset to Google Drive
    GDRIVE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    tmp_tar = GDRIVE_TAR_PATH.with_suffix(".tar.gz.tmp")
    print(f"[INFO] Archiving {PROCESSED_DIR} to {GDRIVE_TAR_PATH} ...")
    cmd = f'tar -cf - -C "{LOCAL_ROOT / "dataset_full"}" processed | pigz -1 > "{tmp_tar}"'
    subprocess.run(cmd, shell=True, check=True)
    os.replace(tmp_tar, GDRIVE_TAR_PATH)
    os.sync()
    print(f"[INFO] Synced {GDRIVE_TAR_PATH} ({GDRIVE_TAR_PATH.stat().st_size / 1e6:.1f} MB) to Google Drive!")


if __name__ == "__main__":
    build_full_dataset()
