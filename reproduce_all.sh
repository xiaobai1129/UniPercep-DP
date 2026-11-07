#!/usr/bin/env bash
set -e

LOCAL_ROOT="/root/workspace/ceshi"
GDRIVE_ROOT="/content/gdrive/MyDrive/colab/UniPercep_DP"
MODE="${1:---eval-only}"

echo "============================================================"
echo " UniPercep-DP: One-Click Cross-Instance Reproduction Script"
echo " Mode: ${MODE} (Options: --eval-only | --smoke-test | --full-train)"
echo "============================================================"

# 1. Initialize local workspace & Google Drive directories
mkdir -p "${LOCAL_ROOT}"/{src,dataset,checkpoints,logs,outputs}
mkdir -p "${GDRIVE_ROOT}"/{src,dataset,checkpoints,logs,outputs}

# 2. Restore source code from Google Drive to local workspace
echo "[Step 1/4] Syncing source code from Google Drive -> ${LOCAL_ROOT}/src/ ..."
cp -rf "${GDRIVE_ROOT}/src/"*.py "${LOCAL_ROOT}/src/"

# 3. Check & install Python dependencies (with fast prebuilt wheels)
echo "[Step 2/4] Checking Python environment and dependencies ..."
python3 -c "import torch, timm, faiss, cv2, pyquaternion; from nuscenes.nuscenes import NuScenes" 2>/dev/null || {
    echo "Installing required Python packages..."
    pip install -q timm albumentations faiss-cpu peft accelerate bitsandbytes transformers einops opencv-python-headless matplotlib tqdm pyquaternion shapely fire cachetools descartes pycocotools
    pip install -q --no-deps nuscenes-devkit
}

# 4. Restore processed dataset from Google Drive cache (or build from scratch if missing)
echo "[Step 3/4] Restoring dataset from Google Drive cache ..."
python3 "${LOCAL_ROOT}/src/build_dataset.py"

# 5. Restore checkpoints from Google Drive if available
if ls "${GDRIVE_ROOT}/checkpoints/"*.pt >/dev/null 2>&1; then
    echo "[Step 3.5/4] Restoring trained checkpoints from Google Drive ..."
    cp -f "${GDRIVE_ROOT}/checkpoints/"*.pt "${LOCAL_ROOT}/checkpoints/"
fi

# 6. Run requested mode
echo "[Step 4/4] Executing pipeline (${MODE}) ..."
if [ "${MODE}" == "--full-train" ]; then
    python3 -u "${LOCAL_ROOT}/src/train_pipeline.py" 2>&1 | tee "${LOCAL_ROOT}/logs/full_run.log"
    python3 -u "${LOCAL_ROOT}/src/infer_and_eval.py" 2>&1 | tee -a "${LOCAL_ROOT}/logs/full_run.log"
    cp -f "${LOCAL_ROOT}/logs/full_run.log" "${GDRIVE_ROOT}/logs/full_run.log"
    sync
elif [ "${MODE}" == "--smoke-test" ]; then
    python3 -u "${LOCAL_ROOT}/src/train_pipeline.py" --smoke_test
    python3 -u "${LOCAL_ROOT}/src/infer_and_eval.py"
    sync
else
    # Default: --eval-only (Directly evaluate & generate visualizations from best checkpoint)
    python3 -u "${LOCAL_ROOT}/src/infer_and_eval.py"
    sync
fi

echo "============================================================"
echo " Reproduction Complete! Outputs available at:"
echo "   Local : ${LOCAL_ROOT}/outputs/"
echo "   GDrive: ${GDRIVE_ROOT}/outputs/"
echo "============================================================"
