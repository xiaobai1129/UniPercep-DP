#!/usr/bin/env bash
# ==============================================================================
# UniPercep-DP: One-Click Cross-Instance State Restoration & Reproduction Script
# ==============================================================================
# Covers all project iterations:
#   - v0 Mini Prototype   : src/, dataset/, checkpoints/, logs/, outputs/
#   - v1 A100 Full & Abl. : src_a100/, dataset_full/, checkpoints_a100/, logs_a100/, outputs_a100/
#   - v2 Pro256 (6-Upgr.) : src_a100/models_pro256.py, train_pro256.py, Full_Pro256_6Cam_best.pt
#   - v3 Ultra (10-Upgr.) : src_a100/models_v3_ultra.py, Full_v3_Ultra_best.pt
#   - Project Docs & Git  : 项目说明.md, README.md, plan.md, .gitignore, .git (git_repo_backup.tar.gz)
#
# Usage in a new instance:
#   bash /content/gdrive/MyDrive/colab/UniPercep_DP/reproduce_all.sh [MODE]
#
# Available Modes:
#   --restore-only    (Default) Restore full project state, datasets, checkpoints, logs & outputs
#   --eval-only       Restore state + run A100 full evaluation & 6-panel visualization
#   --resume-pro256   Restore state + resume/run v2 Full_Pro256_6Cam training (auto-syncs to GDrive)
#   --resume-v3-ultra Restore state + resume/run v3 Full_v3_Ultra training (auto-syncs to GDrive)
#   --full-train      Restore state + run full A100 4-stage & ablation training + evaluation
#   --smoke-test      Restore state + run quick smoke test verification
#   --sync-to-gdrive  Push all current local code, docs, git repo, logs, checkpoints & outputs to GDrive
#   --start-daemon    Start background real-time local -> GDrive sync daemon (every 30s)
# ==============================================================================
set -e

LOCAL_ROOT="/root/workspace/ceshi"
MODE="${1:---restore-only}"

# 0. Auto-detect Google Drive mount point
if [ -d "/content/gdrive/MyDrive/colab/UniPercep_DP" ]; then
    GDRIVE_ROOT="/content/gdrive/MyDrive/colab/UniPercep_DP"
elif [ -d "/content/drive/MyDrive/colab/UniPercep_DP" ]; then
    mkdir -p /content/gdrive
    ln -sfn /content/drive/MyDrive /content/gdrive/MyDrive 2>/dev/null || true
    GDRIVE_ROOT="/content/gdrive/MyDrive/colab/UniPercep_DP"
else
    GDRIVE_ROOT="/content/gdrive/MyDrive/colab/UniPercep_DP"
    echo "[WARN] Google Drive directory ${GDRIVE_ROOT} not found yet. Will create if mounted."
fi

# Function: Sync local workspace -> Google Drive
sync_local_to_gdrive() {
    echo "[Sync] Synchronizing local project state (${LOCAL_ROOT}) -> Google Drive (${GDRIVE_ROOT}) ..."
    mkdir -p "${GDRIVE_ROOT}"/{src,src_a100,dataset,dataset_full,checkpoints,checkpoints_a100,logs,logs_a100,outputs,outputs_a100}

    # 1. Source code & root docs
    cp -uf "${LOCAL_ROOT}/src/"*.py "${GDRIVE_ROOT}/src/" 2>/dev/null || true
    cp -uf "${LOCAL_ROOT}/src_a100/"*.py "${GDRIVE_ROOT}/src_a100/" 2>/dev/null || true
    for f in "README.md" "plan.md" "项目说明.md" ".gitignore" "reproduce_all.sh"; do
        if [ -f "${LOCAL_ROOT}/${f}" ]; then
            cp -uf "${LOCAL_ROOT}/${f}" "${GDRIVE_ROOT}/${f}"
        fi
    done
    if [ -f ~/.gitconfig ]; then
        cp -uf ~/.gitconfig "${GDRIVE_ROOT}/.gitconfig" 2>/dev/null || true
    fi

    # 2. Git repository backup (.git -> git_repo_backup.tar.gz)
    if [ -d "${LOCAL_ROOT}/.git" ]; then
        tar -czf "${GDRIVE_ROOT}/git_repo_backup.tar.gz.tmp" -C "${LOCAL_ROOT}" .git 2>/dev/null && \
            mv -f "${GDRIVE_ROOT}/git_repo_backup.tar.gz.tmp" "${GDRIVE_ROOT}/git_repo_backup.tar.gz"
    fi

    # 3. Dataset JSONL manifests
    for f in train.jsonl val.jsonl; do
        [ -f "${LOCAL_ROOT}/dataset/${f}" ] && cp -uf "${LOCAL_ROOT}/dataset/${f}" "${GDRIVE_ROOT}/dataset/${f}"
    done
    for f in train_full.jsonl val_full.jsonl; do
        [ -f "${LOCAL_ROOT}/dataset_full/${f}" ] && cp -uf "${LOCAL_ROOT}/dataset_full/${f}" "${GDRIVE_ROOT}/dataset_full/${f}"
    done

    # 4. Checkpoints, logs, and outputs (both v0 mini and A100 v1/v2/v3)
    for subdir in checkpoints checkpoints_a100 logs logs_a100 outputs outputs_a100; do
        if [ -d "${LOCAL_ROOT}/${subdir}" ]; then
            if command -v rsync >/dev/null 2>&1; then
                rsync -a --update --inplace "${LOCAL_ROOT}/${subdir}/" "${GDRIVE_ROOT}/${subdir}/" 2>/dev/null || true
            else
                cp -uf "${LOCAL_ROOT}/${subdir}/"* "${GDRIVE_ROOT}/${subdir}/" 2>/dev/null || true
            fi
        fi
    done
    sync
    echo "[Sync] All local artifacts successfully backed up to ${GDRIVE_ROOT}."
}

# Function: Start background sync daemon (every 30s)
start_sync_daemon() {
    if pgrep -f "unipercep_gdrive_sync_daemon" >/dev/null 2>&1; then
        echo "[Daemon] Background Google Drive sync daemon is already running."
        return 0
    fi
    cat > /tmp/unipercep_gdrive_sync_daemon.sh <<'EOS'
#!/usr/bin/env bash
LOCAL_ROOT="/root/workspace/ceshi"
GDRIVE_ROOT="/content/gdrive/MyDrive/colab/UniPercep_DP"
while true; do
    sleep 30
    [ -d "${GDRIVE_ROOT}" ] || continue
    for subdir in src src_a100 logs logs_a100 outputs outputs_a100 checkpoints checkpoints_a100; do
        if [ -d "${LOCAL_ROOT}/${subdir}" ]; then
            cp -uf "${LOCAL_ROOT}/${subdir}/"* "${GDRIVE_ROOT}/${subdir}/" 2>/dev/null || true
        fi
    done
    for f in "README.md" "plan.md" "项目说明.md" ".gitignore" "reproduce_all.sh"; do
        [ -f "${LOCAL_ROOT}/${f}" ] && cp -uf "${LOCAL_ROOT}/${f}" "${GDRIVE_ROOT}/${f}" 2>/dev/null || true
    done
    sync
done
EOS
    chmod +x /tmp/unipercep_gdrive_sync_daemon.sh
    setsid -f /tmp/unipercep_gdrive_sync_daemon.sh </dev/null >/dev/null 2>&1 || true
    echo "[Daemon] Started real-time Google Drive sync daemon (interval=30s)."
}

if [ "${MODE}" == "--sync-to-gdrive" ]; then
    sync_local_to_gdrive
    exit 0
elif [ "${MODE}" == "--start-daemon" ]; then
    start_sync_daemon
    exit 0
fi

echo "======================================================================"
echo " UniPercep-DP: One-Click Cross-Instance Restoration & Reproduction"
echo " Mode: ${MODE}"
echo " Local Workspace : ${LOCAL_ROOT}"
echo " Google Drive    : ${GDRIVE_ROOT}"
echo "======================================================================"

# 1. Initialize local workspace & Google Drive directories
mkdir -p "${LOCAL_ROOT}"/{src,src_a100,dataset,dataset_full,checkpoints,checkpoints_a100,logs,logs_a100,outputs,outputs_a100}
mkdir -p "${GDRIVE_ROOT}"/{src,src_a100,dataset,dataset_full,checkpoints,checkpoints_a100,logs,logs_a100,outputs,outputs_a100}

# 2. Restore source code, project docs & Git repository from Google Drive
echo "[Step 1/5] Restoring source code (src/ & src_a100/), docs & Git repo from Google Drive ..."
cp -uf "${GDRIVE_ROOT}/src/"*.py "${LOCAL_ROOT}/src/" 2>/dev/null || true
cp -uf "${GDRIVE_ROOT}/src_a100/"*.py "${LOCAL_ROOT}/src_a100/" 2>/dev/null || true

for f in "README.md" "plan.md" "项目说明.md" ".gitignore" "reproduce_all.sh"; do
    if [ -f "${GDRIVE_ROOT}/${f}" ]; then
        cp -uf "${GDRIVE_ROOT}/${f}" "${LOCAL_ROOT}/${f}"
    fi
done
chmod +x "${LOCAL_ROOT}/reproduce_all.sh" 2>/dev/null || true

if [ -f "${GDRIVE_ROOT}/.gitconfig" ]; then
    cp -uf "${GDRIVE_ROOT}/.gitconfig" ~/.gitconfig
fi
if [ -f "${GDRIVE_ROOT}/git_repo_backup.tar.gz" ] && [ ! -d "${LOCAL_ROOT}/.git" ]; then
    echo "  -> Extracting Git repository history (git_repo_backup.tar.gz) ..."
    tar -xzf "${GDRIVE_ROOT}/git_repo_backup.tar.gz" -C "${LOCAL_ROOT}"
fi

# 3. Check & install Python dependencies
echo "[Step 2/5] Checking Python environment & dependencies ..."
python3 -c "import torch, timm, faiss, cv2, pyquaternion; from nuscenes.nuscenes import NuScenes" 2>/dev/null || {
    echo "  -> Installing required Python packages ..."
    pip install -q timm albumentations faiss-cpu peft accelerate bitsandbytes transformers einops opencv-python-headless matplotlib tqdm pyquaternion shapely fire cachetools descartes pycocotools
    pip install -q --no-deps nuscenes-devkit
}

# 4. Restore preprocessed datasets (both A100 Full 14,400-sample dataset & Mini dataset)
echo "[Step 3/5] Restoring preprocessed datasets from Google Drive cache ..."

# 4a. Restore dataset_full (3,600 keyframes / 14,400 samples)
for f in train_full.jsonl val_full.jsonl; do
    if [ -f "${GDRIVE_ROOT}/dataset_full/${f}" ] && [ ! -f "${LOCAL_ROOT}/dataset_full/${f}" ]; then
        cp -f "${GDRIVE_ROOT}/dataset_full/${f}" "${LOCAL_ROOT}/dataset_full/${f}"
    fi
done

if [ ! -d "${LOCAL_ROOT}/dataset_full/processed/images_clean" ]; then
    if [ -f "${GDRIVE_ROOT}/dataset_full/processed_unipercep_full.tar.gz" ]; then
        echo "  -> Extracting full dataset archive (processed_unipercep_full.tar.gz, 420MB) ..."
        if command -v pigz >/dev/null 2>&1; then
            pigz -dc "${GDRIVE_ROOT}/dataset_full/processed_unipercep_full.tar.gz" | tar -x -C "${LOCAL_ROOT}/dataset_full"
        else
            tar -xzf "${GDRIVE_ROOT}/dataset_full/processed_unipercep_full.tar.gz" -C "${LOCAL_ROOT}/dataset_full"
        fi
    else
        echo "  -> Building full dataset via src_a100/build_full_dataset.py ..."
        python3 "${LOCAL_ROOT}/src_a100/build_full_dataset.py"
    fi
fi

mkdir -p "${LOCAL_ROOT}/dataset_full/processed"
for f in train_full.jsonl val_full.jsonl; do
    if [ -f "${LOCAL_ROOT}/dataset_full/${f}" ] && [ ! -f "${LOCAL_ROOT}/dataset_full/processed/${f}" ]; then
        cp -f "${LOCAL_ROOT}/dataset_full/${f}" "${LOCAL_ROOT}/dataset_full/processed/${f}"
    fi
done

# 4b. Restore mini dataset (404 keyframes)
for f in train.jsonl val.jsonl; do
    if [ -f "${GDRIVE_ROOT}/dataset/${f}" ] && [ ! -f "${LOCAL_ROOT}/dataset/${f}" ]; then
        cp -f "${GDRIVE_ROOT}/dataset/${f}" "${LOCAL_ROOT}/dataset/${f}"
    fi
done
if [ ! -d "${LOCAL_ROOT}/dataset/processed_unipercep" ] && [ -f "${GDRIVE_ROOT}/dataset/processed_unipercep_mini.tar.gz" ]; then
    echo "  -> Extracting mini dataset archive (processed_unipercep_mini.tar.gz, 40MB) ..."
    tar -xzf "${GDRIVE_ROOT}/dataset/processed_unipercep_mini.tar.gz" -C "${LOCAL_ROOT}/dataset"
fi

# 5. Restore all trained checkpoints, logs, and outputs
echo "[Step 4/5] Restoring trained checkpoints, logs, and evaluation outputs ..."
for subdir in checkpoints checkpoints_a100 logs logs_a100 outputs outputs_a100; do
    if [ -d "${GDRIVE_ROOT}/${subdir}" ]; then
        if command -v rsync >/dev/null 2>&1; then
            rsync -a --update "${GDRIVE_ROOT}/${subdir}/" "${LOCAL_ROOT}/${subdir}/" 2>/dev/null || true
        else
            cp -uf "${GDRIVE_ROOT}/${subdir}/"* "${LOCAL_ROOT}/${subdir}/" 2>/dev/null || true
        fi
    fi
done

# Ensure background sync daemon is active
start_sync_daemon

# 6. Execute requested pipeline mode
echo "[Step 5/5] Executing mode: ${MODE} ..."
if [ "${MODE}" == "--restore-only" ]; then
    echo "----------------------------------------------------------------------"
    echo "[OK] Full project state restored to ${LOCAL_ROOT}!"
    echo "  - Source files (src_a100): $(ls -1 "${LOCAL_ROOT}/src_a100/"*.py 2>/dev/null | wc -l) files"
    echo "  - Full dataset keyframes : $(ls -1 "${LOCAL_ROOT}/dataset_full/processed/images_clean" 2>/dev/null | wc -l) keyframes (14,400 augmented samples)"
    echo "  - A100 Checkpoints       : $(ls -1 "${LOCAL_ROOT}/checkpoints_a100/"*.pt 2>/dev/null | wc -l) checkpoints ($(du -sh "${LOCAL_ROOT}/checkpoints_a100" 2>/dev/null | cut -f1))"
    echo "  - A100 Outputs & Plots   : $(ls -1 "${LOCAL_ROOT}/outputs_a100/"* 2>/dev/null | wc -l) files"
    if [ -f "${LOCAL_ROOT}/logs_a100/pro256_metrics.csv" ]; then
        echo "  - Latest Pro256 / v3 Ultra Metrics (logs_a100/pro256_metrics.csv):"
        tail -n 5 "${LOCAL_ROOT}/logs_a100/pro256_metrics.csv" | sed 's/^/      /'
    fi
    echo "----------------------------------------------------------------------"
elif [ "${MODE}" == "--eval-only" ]; then
    python3 -u "${LOCAL_ROOT}/src_a100/evaluate_and_visualize.py"
    sync_local_to_gdrive
elif [ "${MODE}" == "--resume-pro256" ]; then
    python3 -u "${LOCAL_ROOT}/src_a100/train_pro256.py" --epochs 35 --batch_size 24 2>&1 | tee -a "${LOCAL_ROOT}/logs_a100/pro256_run.log"
    sync_local_to_gdrive
elif [ "${MODE}" == "--resume-v3-ultra" ]; then
    python3 -u "${LOCAL_ROOT}/src_a100/train_pro256.py" --v3_ultra --epochs 35 --batch_size 24 2>&1 | tee -a "${LOCAL_ROOT}/logs_a100/v3_ultra_run.log"
    sync_local_to_gdrive
elif [ "${MODE}" == "--full-train" ]; then
    python3 -u "${LOCAL_ROOT}/src_a100/train_and_ablation.py" 2>&1 | tee -a "${LOCAL_ROOT}/logs_a100/a100_full_run.log"
    python3 -u "${LOCAL_ROOT}/src_a100/evaluate_and_visualize.py" 2>&1 | tee -a "${LOCAL_ROOT}/logs_a100/a100_full_run.log"
    python3 -u "${LOCAL_ROOT}/src_a100/train_pro256.py" --epochs 35 --batch_size 24 2>&1 | tee -a "${LOCAL_ROOT}/logs_a100/pro256_run.log"
    sync_local_to_gdrive
elif [ "${MODE}" == "--smoke-test" ]; then
    python3 -u "${LOCAL_ROOT}/src_a100/train_and_ablation.py" --smoke_test
    python3 -u "${LOCAL_ROOT}/src_a100/evaluate_and_visualize.py" --smoke_test
    sync_local_to_gdrive
else
    echo "[WARN] Unknown mode '${MODE}'. Supported modes:"
    echo "  --restore-only | --eval-only | --resume-pro256 | --resume-v3-ultra | --full-train | --smoke-test | --sync-to-gdrive"
    exit 1
fi

echo "======================================================================"
echo " Done! Workspace ready at: ${LOCAL_ROOT}"
echo "======================================================================"
