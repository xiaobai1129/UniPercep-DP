# Colab 长时训练、实时同步与跨实例迁移指南（Agent & 开发者通用规范）

> 本文档基于 `UniPercep_DP` 仓库（[`reproduce_all.sh`](./reproduce_all.sh)、[`src/train_pipeline.py`](./src/train_pipeline.py)、[`src_a100/train_and_ablation.py`](./src_a100/train_and_ablation.py)、[`src_a100/build_full_dataset.py`](./src_a100/build_full_dataset.py)、[`src_a100/evaluate_and_visualize.py`](./src_a100/evaluate_and_visualize.py)、[`.gitignore`](./.gitignore)）的工程实践提炼，并补充 Colab 通用防断连与新项目迁移规范，适用于将新项目实时同步至 Google Drive 并在后续迁移到新 GPU 实例（如 A100 / H100 / RTX PRO 6000）。

---

## 一、 核心设计原则：本地高速盘计算 + 云盘异步持久化

1. **计算与高并发 I/O 必须在本地磁盘**：
   - 本地工作区放在 `/root/workspace/<project_name>`（或 `/content/<project_name>`）。
   - **严禁直接在 Google Drive FUSE 挂载目录下读写上万张零散图片、频繁保存大模型权重或直接跑 DataLoader**：Drive FUSE 的元操作与随机读极慢，极易卡死训练甚至导致文件截断损坏。
2. **Google Drive 只作为持久化冷备份与跨实例中转站**：
   - 云盘目录统一约定为 `/content/gdrive/MyDrive/colab/<project_name>`。
   - **路径兼容软链接**：Colab 官方 Notebook `drive.mount('/content/drive')` 默认挂载在 `/content/drive/MyDrive`，而脚本常写 `/content/gdrive/MyDrive`。初始化时务必先做兼容检测与软链接（提取自 `reproduce_all.sh`）：
     ```bash
     if [ -d "/content/drive/MyDrive" ] && [ ! -e "/content/gdrive/MyDrive" ]; then
         mkdir -p /content/gdrive
         ln -sfn /content/drive/MyDrive /content/gdrive/MyDrive
     fi
     ```
3. **只放必要资产在本地磁盘，防止系统盘写满崩溃**：
   - 云盘上的 `.tar.gz` 压缩包直接原地流式解压到本地，**不要**先把几 GB 的压缩包复制一份到本地再解压。
4. **Google Drive FUSE 的三大文件系统限制（提炼自 `colab_workspace/init_dsh.sh`）**：
   - **不支持硬链接（`fs.link` 会报 `EPERM`）**：任何会使用硬链接或高频 append 的目录（如工具运行时的 `sessions/`、`cache/`）**必须留在本地磁盘**，不能直接放云盘或软链到云盘，必要时用后台 `rsync` 定时转存到云盘。
   - **不保留可执行权限位（`chmod +x` 失效）**：云盘里的二进制/工具文件在新挂载后常显示为 `-rw-------`，必须先拷贝到本地磁盘（如 `/opt/`）再 `chmod 0755` 执行。
   - **随机小文件读取极慢**：大量小文件（如 Node 依赖、`node_modules` 或上万张图片）若直接放在云盘读取会慢 10~20 倍；应打包为 `.sqsh`（SquashFS 只读镜像）或 `.tar.gz`，启动时先拷贝/解压到本地磁盘再使用。
   - **SSH 登录环境变量陷阱**：非交互 SSH / `bash -lc` 默认只读 `/etc/profile` 与 `/etc/profile.d/*.sh`，**不读 `~/.bashrc`**；需要跨终端生效的环境变量和自恢复钩子必须同时写入 `/etc/profile.d/*.sh` 与 `~/.bashrc`。

---

## 二、 如何防止 Colab 断开与任务被杀

Colab 断开通常来自四个原因：**浏览器前端空闲超时**、**TTY/SSH 断开导致子进程收到 `SIGHUP`**、**系统内存/磁盘 OOM**、**最长运行时间或资源回收**。需从“防断开”与“断开后无损续跑”两手准备。

### 1. 前端与会话层面防空闲断开
- **开启 Colab 后台执行（Background execution）**：使用 Pro / Pro+ / PRO 6000 等付费 GPU 时，在 Colab 菜单开启后台执行，关闭浏览器标签页后运行时仍可继续跑。
- **浏览器防休眠**：把 `colab.research.google.com` 加入浏览器“内存节省程序 / 标签页休眠”白名单；若未开启后台执行，可在浏览器开发者工具 Console 中定时保活，避免长时间无交互被判定为闲置回收。
- **通过 DSH Remote-SSH 操作时注意**：
  - `@zhangfengshun/dsh-remote-ssh` 的 `timeoutMs` 只是本地等待超时，到点不会杀远端进程；但**点击停止按钮**或直接关闭非守护前台进程会向远端进程组发送 `SIGHUP`。
  - **所有长时训练与后台守护进程都必须脱离终端（`setsid -f` 或 `nohup`）**。

### 2. 进程保活与无缓冲日志规范
参考 `reproduce_all.sh`：
- **使用 `setsid -f`（或 `nohup ... &`）脱离控制终端**：
  ```bash
  setsid -f python3 -u src/train.py > logs/train.log 2>&1 </dev/null
  ```
- **Python 必须加 `-u`（禁用 stdout/stderr 缓冲）并用 `tee -a` 落盘**：
  ```bash
  python3 -u src/train.py 2>&1 | tee -a logs/train.log
  ```
  避免实例突然断开时，最后几轮训练日志还在缓冲区没写入文件。

### 3. 防 OOM 与防磁盘打满
- 往内存缓存数据集（RAM Cache）前先评估系统 RAM 上限，避免触发 Linux OOM Killer 直接杀掉训练进程。
- 定期清理临时 `.tmp` 文件或无用中间权重，确保本地磁盘保留充足余量（至少留出 2 倍单次 Checkpoint 体积）。

### 4. 自动断点续训（Auto-Resume，最后一道防线）
即使做足保活，Colab 仍有运行时长上限。训练代码从第一天起就应具备“重启即接续”的能力：
- 每个 Epoch 结束固定写入 `latest.pt`（覆盖写）与 `best.pt`（更优时更新）。
- **建议 `latest.pt` 同时保存**：`epoch`、`model.state_dict()`、`optimizer.state_dict()`、`scheduler.state_dict()`、`scaler.state_dict()`、`best_score`（注：本仓库 `train_pro256.py` 与 `train_v4_chronos.py` 保存了 `model`/`ema`/`metrics`/`epoch` 并自动从上一轮恢复权重，但新项目建议补全优化器与调度器状态以实现完全平滑续训）。

---

## 三、 如何实时把数据同步到 Google Drive

结合本仓库代码，推荐采用 **“原子覆盖同步 + 30 秒后台守护进程 + 大文件打包归档”** 三层同步架构。

### 第 1 层：Python 内部“原子写入 + 强刷盘”（最核心）
**切忌直接 `shutil.copy2(src, gdrive_dst)` 覆盖云盘上的权重或指标文件！** 如果拷贝到一半 Colab 断电/断连，云盘上会留下损坏的半截文件（本仓库部分后期脚本直接调用 `shutil.copy2`，建议新项目统一回归 `train_pipeline.py` 与 `train_and_ablation.py` 的原子拷贝写法）：

```python
import os
import shutil
from pathlib import Path

def atomic_copy(src: Path, dst: Path):
    """先写入 .tmp 临时文件再 os.replace 原子替换，防止断连写坏云盘目标文件"""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)

def sync_epoch_to_gdrive(gdrive_root: Path, files_to_sync: list[tuple[Path, str]]):
    """每个 Epoch 结束调用：原子同步权重、CSV 指标与日志文件，并强制 os.sync() 刷盘"""
    if not gdrive_root.exists():
        return
    for src_path, rel_dst in files_to_sync:
        atomic_copy(src_path, gdrive_root / rel_dst)
    os.sync()  # 强制将 Linux 页缓存刷入 Google Drive FUSE 层
```

**每个 Epoch 至少同步**：
- `checkpoints/latest.pt`（与 `best.pt`）
- `logs/training_metrics.csv` 与 `logs/run.log`
- 实时生成的评估报告/图表（如 `outputs/metrics.json`、`outputs/report.md`）

### 第 2 层：后台 30 秒增量同步守护进程（Daemon）
防止开发者/Agent 在写代码或临时跑脚本时忘记手动同步，在工作区初始化时启动一个后台 Daemon（提取自 `reproduce_all.sh` 的 `start_sync_daemon`）：

- 使用 `pgrep -f` 防止重复启动多个 Daemon。
- 使用 `setsid -f ... </dev/null >/dev/null 2>&1` 脱离当前终端。
- 每 30 秒用 `rsync -a --update --inplace`（或 `cp -ruf`）将 `src/`、`logs/`、`outputs/`、`checkpoints/` 及根目录文档增量同步到云盘并 `sync`。
- **注意避坑**：Daemon 里**绝对不要**同步零散数据集目录（如上万张图片的 `dataset/processed/`），否则 Drive FUSE 会长期阻塞。

### 第 3 层：零散数据集单文件打包归档 & Git 历史同步
1. **数据集归档（提取自 `build_full_dataset.py`）**：
   - 预处理生成的成千上万个图像/标签文件，先用多核 `pigz` 压成单个 `.tar.gz.tmp`，再 `mv` 原子落盘到云盘：
     ```bash
     tar -cf - -C "${LOCAL_ROOT}/dataset" processed | pigz -1 > "${GDRIVE_ROOT}/dataset/processed.tar.gz.tmp"
     mv -f "${GDRIVE_ROOT}/dataset/processed.tar.gz.tmp" "${GDRIVE_ROOT}/dataset/processed.tar.gz"
     sync
     ```
2. **Git 仓库备份（提取自 `reproduce_all.sh`）**：
   - `.gitignore` 严格忽略 `dataset/`、`checkpoints/`、`*.pt`、`*.tar.gz`、`logs/`、`*.png` 等大文件（只用 Git 管代码与文档）。
   - 除了 `git push` 到远程仓库，同步脚本可同时把 `.git` 原子打包为 `git_repo_backup.tar.gz` 存到云盘，确保在无外网权限或未 push 时新实例也能完整恢复提交历史。
3. **版本锁死**：
   - 环境配好后在旧实例执行 `pip freeze > requirements_lock.txt` 并同步到云盘，避免新实例装包版本漂移。

---

## 四、 新建项目：同步到云盘与迁移到新实例的标准模板

> 适用于本次新申请的实例（不迁移 `UniPercep_DP`，专用于**新项目**开发、实时同步云盘、并在后续迁移到下一个新实例）。

### 1. 推荐目录规范（本地与云盘镜像一致）
```text
<PROJECT_NAME>/                 # 本地 /root/workspace/<PROJECT_NAME> <-> 云盘 /content/gdrive/MyDrive/colab/<PROJECT_NAME>
├── README.md                   # 项目说明与启动命令
├── agent-colab.md              # 本规范文件（供后续新实例上的 Agent 直接读取遵守）
├── colab_sync.sh               # 一键恢复 + 同步 + Daemon 脚本（见下方模板）
├── .gitignore                  # 屏蔽数据集、权重、日志、临时文件
├── requirements_lock.txt       # pip freeze 锁定的依赖版本
├── git_repo_backup.tar.gz      # (云盘端) .git 完整历史备份
├── src/                        # 源代码
├── dataset/                    # 预处理后的数据集（云盘端额外存 processed.tar.gz）
├── checkpoints/                # 模型权重（latest.pt / best.pt）
├── logs/                       # 训练日志与 metrics.csv
└── outputs/                    # 评测结果与可视化图表
```

### 2. 通用 `colab_sync.sh` 模板（新建项目直接复制修改顶部变量即可）
```bash
#!/usr/bin/env bash
set -e

PROJECT_NAME="YOUR_NEW_PROJECT"
LOCAL_ROOT="/root/workspace/${PROJECT_NAME}"
MODE="${1:---restore}"

# 1. 自动识别并兼容 Google Drive 挂载路径
if [ -d "/content/gdrive/MyDrive" ]; then
    GDRIVE_BASE="/content/gdrive/MyDrive"
elif [ -d "/content/drive/MyDrive" ]; then
    mkdir -p /content/gdrive
    ln -sfn /content/drive/MyDrive /content/gdrive/MyDrive 2>/dev/null || true
    GDRIVE_BASE="/content/gdrive/MyDrive"
else
    echo "[ERROR] Google Drive 未挂载，请先在 Colab 执行 drive.mount('/content/drive')!"
    exit 1
fi
GDRIVE_ROOT="${GDRIVE_BASE}/colab/${PROJECT_NAME}"
DAEMON_TAG="${PROJECT_NAME}_gdrive_sync_daemon"

sync_to_gdrive() {
    echo "[Sync] ${LOCAL_ROOT} -> ${GDRIVE_ROOT} ..."
    mkdir -p "${GDRIVE_ROOT}"/{src,dataset,checkpoints,logs,outputs}

    # 代码与根目录文档（不覆盖大文件）
    for dir in src; do
        [ -d "${LOCAL_ROOT}/${dir}" ] && rsync -a --update --exclude='__pycache__' "${LOCAL_ROOT}/${dir}/" "${GDRIVE_ROOT}/${dir}/" 2>/dev/null || true
    done
    for f in README.md agent-colab.md colab_sync.sh .gitignore requirements_lock.txt; do
        [ -f "${LOCAL_ROOT}/${f}" ] && cp -uf "${LOCAL_ROOT}/${f}" "${GDRIVE_ROOT}/${f}"
    done

    # Git 历史原子备份
    if [ -d "${LOCAL_ROOT}/.git" ]; then
        tar -czf "${GDRIVE_ROOT}/git_repo_backup.tar.gz.tmp" -C "${LOCAL_ROOT}" .git 2>/dev/null && \
            mv -f "${GDRIVE_ROOT}/git_repo_backup.tar.gz.tmp" "${GDRIVE_ROOT}/git_repo_backup.tar.gz"
    fi

    # 权重、日志、输出增量同步（不删云盘旧文件，使用 --inplace / --update）
    for subdir in checkpoints logs outputs; do
        if [ -d "${LOCAL_ROOT}/${subdir}" ]; then
            rsync -a --update --exclude='*.tmp' "${LOCAL_ROOT}/${subdir}/" "${GDRIVE_ROOT}/${subdir}/" 2>/dev/null || \
                cp -ruf "${LOCAL_ROOT}/${subdir}/"* "${GDRIVE_ROOT}/${subdir}/" 2>/dev/null || true
        fi
    done

    sync
    echo "[Sync] Done."
}

start_daemon() {
    if pgrep -f "${DAEMON_TAG}" >/dev/null 2>&1; then
        echo "[Daemon] 后台同步守护进程已在运行中。"
        return 0
    fi
    cat > "/tmp/${DAEMON_TAG}.sh" <<EOS
#!/usr/bin/env bash
while true; do
    sleep 30
    [ -d "${GDRIVE_ROOT}" ] || continue
    for subdir in src logs outputs checkpoints; do
        if [ -d "${LOCAL_ROOT}/\${subdir}" ]; then
            rsync -a --update --exclude='__pycache__' --exclude='*.tmp' "${LOCAL_ROOT}/\${subdir}/" "${GDRIVE_ROOT}/\${subdir}/" 2>/dev/null || true
        fi
    done
    for f in README.md agent-colab.md colab_sync.sh .gitignore requirements_lock.txt; do
        [ -f "${LOCAL_ROOT}/\${f}" ] && cp -uf "${LOCAL_ROOT}/\${f}" "${GDRIVE_ROOT}/\${f}" 2>/dev/null || true
    done
    sync
done
EOS
    chmod +x "/tmp/${DAEMON_TAG}.sh"
    setsid -f "/tmp/${DAEMON_TAG}.sh" </dev/null >/dev/null 2>&1 || true
    echo "[Daemon] 已启动 30s 后台云盘同步守护进程 (${DAEMON_TAG})。"
}

restore_from_gdrive() {
    echo "[Restore] ${GDRIVE_ROOT} -> ${LOCAL_ROOT} ..."
    mkdir -p "${LOCAL_ROOT}"/{src,dataset,checkpoints,logs,outputs}
    mkdir -p "${GDRIVE_ROOT}"/{src,dataset,checkpoints,logs,outputs}

    # 1. 恢复源码与文档
    rsync -a --update "${GDRIVE_ROOT}/src/" "${LOCAL_ROOT}/src/" 2>/dev/null || true
    for f in README.md agent-colab.md colab_sync.sh .gitignore requirements_lock.txt; do
        [ -f "${GDRIVE_ROOT}/${f}" ] && cp -uf "${GDRIVE_ROOT}/${f}" "${LOCAL_ROOT}/${f}"
    done
    chmod +x "${LOCAL_ROOT}/colab_sync.sh" 2>/dev/null || true

    # 2. 恢复 .git 提交历史
    if [ -f "${GDRIVE_ROOT}/git_repo_backup.tar.gz" ] && [ ! -d "${LOCAL_ROOT}/.git" ]; then
        tar -xzf "${GDRIVE_ROOT}/git_repo_backup.tar.gz" -C "${LOCAL_ROOT}"
    fi

    # 3. 恢复预处理数据集压缩包（如存在）
    if [ -f "${GDRIVE_ROOT}/dataset/processed.tar.gz" ] && [ ! -d "${LOCAL_ROOT}/dataset/processed" ]; then
        echo "  -> 解压 dataset/processed.tar.gz ..."
        if command -v pigz >/dev/null 2>&1; then
            pigz -dc "${GDRIVE_ROOT}/dataset/processed.tar.gz" | tar -x -C "${LOCAL_ROOT}/dataset"
        else
            tar -xzf "${GDRIVE_ROOT}/dataset/processed.tar.gz" -C "${LOCAL_ROOT}/dataset"
        fi
    fi

    # 4. 恢复权重、日志与输出
    for subdir in checkpoints logs outputs; do
        if [ -d "${GDRIVE_ROOT}/${subdir}" ]; then
            rsync -a --update "${GDRIVE_ROOT}/${subdir}/" "${LOCAL_ROOT}/${subdir}/" 2>/dev/null || true
        fi
    done

    # 5. 安装依赖（如有锁定文件则按需安装）
    if [ -f "${LOCAL_ROOT}/requirements_lock.txt" ]; then
        echo "[Env] 检测到 requirements_lock.txt，可按需运行: pip install -r ${LOCAL_ROOT}/requirements_lock.txt"
    fi

    # 6. 启动 30s 实时同步守护进程
    start_daemon
    echo "[OK] 工作区已在 ${LOCAL_ROOT} 恢复就绪！"
}

case "${MODE}" in
    --restore) restore_from_gdrive ;;
    --sync)    sync_to_gdrive ;;
    --daemon)  start_daemon ;;
    *)
        echo "用法: bash colab_sync.sh [--restore | --sync | --daemon]"
        exit 1
        ;;
esac
```

---

## 五、 启动一个全新 Colab 实例时的迁移 SOP

### 阶段 A：旧实例释放/迁移前（确保云盘资产 100% 完整）
1. **停止写入或等当前 Epoch 落盘完成**。
2. 若本地生成了新的零散数据集，先用 `pigz` 打包成单文件 `.tar.gz` 原子写入云盘 `dataset/`。
3. 导出当前环境依赖版本：`pip freeze > requirements_lock.txt`。
4. 执行一次全量同步并等待落盘：
   ```bash
   bash /root/workspace/<PROJECT_NAME>/colab_sync.sh --sync
   ```
5. 在 Google Drive 目录确认 `checkpoints/latest.pt`、`logs/`、`src/`、`git_repo_backup.tar.gz` 的修改时间与文件大小均已更新。

### 阶段 B：全新实例（如 PRO 6000）拉起并恢复内容
1. **在 Colab Notebook 挂载 Google Drive**：
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **（可选）若新项目尚未创建目录，先在云盘新建项目目录并放入 `colab_sync.sh`；若云盘已有项目内容，直接一行命令拉取到本地**：
   ```bash
   bash /content/gdrive/MyDrive/colab/<PROJECT_NAME>/colab_sync.sh --restore
   ```
3. **补齐 Python 依赖**（优先安装项目核心依赖，避免 Colab 预装的 `torch`/`cuda` 被覆盖破坏）：
   - 先用 `python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"` 确认 GPU 与 PyTorch 正常；
   - 缺失的第三方包单独 `pip install`，或维护一份**剔除了基础 CUDA/Colab 系统包**的 `requirements.txt`。
4. **验证续训 / 启动任务**：
   ```bash
   cd /root/workspace/<PROJECT_NAME>
   setsid -f python3 -u src/train.py --resume > logs/run.log 2>&1 </dev/null
   ```

---

## 六、 附：用本机 `colab-dsh` 在全新 Colab 实例上恢复与启动 `dsh`

> 你的 Mac 本机已配置好 [`/Users/baiqin/.local/bin/colab-dsh`](/Users/baiqin/.local/bin/colab-dsh) 与 [`/Users/baiqin/.ssh/config`](/Users/baiqin/.ssh/config)，配合云盘目录 `/content/gdrive/MyDrive/colab_workspace/`（含 [init_dsh.sh](/Users/baiqin/Library/CloudStorage/GoogleDrive-zhhh1129baiqin@gmail.com/我的云端硬盘/colab_workspace/init_dsh.sh) 与 `dsh_home/`）即可秒级拉起远端 `dsh`。

### 方式 1（推荐）：本机终端直接用 `colab-dsh` 连接并启动 Pro 6000 上的 `dsh`
`/Users/baiqin/.ssh/config` 中已经预配好了 **`colab-pro6000`**（即 `colab-g4` 的别名，对应 `colab` CLI 会话名 `test-g4`、`--gpu G4`）：

1. **先确认当前会话情况**（如果你是在 Colab 网页端手动开的 Pro 6000，它的会话名可能不是 `test-g4`）：
   ```bash
   colab sessions
   ```
   - 如果还没建会话、或会话名就是 `test-g4`，直接执行下面第 2~3 步；
   - 如果你在网页端手动开了别的会话名，直接跑 `colab-dsh -s colab-pro6000` 会自动用 `colab new -s test-g4 --gpu G4` **新开第二台实例**；此时建议要么关掉网页端空实例让 `colab-pro6000` 自动创建 `test-g4`，要么把 `~/.ssh/config` 里的会话名改成你在 `colab sessions` 里看到的实际名字。
2. **新实例首次挂载 Google Drive（并自动执行 `init_dsh.sh`）**：
   ```bash
   colab-dsh -s colab-pro6000 mount
   ```
   *(通过 `drive_consent.py` 完成 OAuth 授权后直接走 SSH 拉起 DriveFS 到 `/content/gdrive`，不依赖 Jupyter kernel；若你在网页端已手动挂载过云盘，可跳过此步。)*
3. **在 Pro 6000 上启动 `dsh web` 并打开本地浏览器隧道**：
   ```bash
   colab-dsh -s colab-pro6000 web
   # 若 8877 端口已被 colab-a100 等占用，可指定其他端口：
   colab-dsh -s colab-pro6000 web --port 8879
   ```
   常用辅助命令：
   - 进入已就绪的云端交互终端：`colab-dsh -s colab-pro6000`
   - 手动同步会话历史到云盘：`colab-dsh -s colab-pro6000 sync push`
   - 停止本地隧道与云端 `dsh web`：`colab-dsh -s colab-pro6000 web-stop`（带自定义端口则加 `--port 8879`）

### 方式 2：直接从 Colab Notebook 手动触发 `init_dsh.sh`
```python
from google.colab import drive
drive.mount('/content/gdrive')
```
```bash
bash /content/gdrive/MyDrive/colab_workspace/init_dsh.sh
dsh --version
dsh web --no-open
```

### SSH 防断连关键参数（提炼自 `/Users/baiqin/.ssh/config`）
连接 Colab `colab ssh --proxy-mode` 时，保持长连接与隧道稳定的核心 SSH 配置为：
- `ControlMaster auto` + `ControlPersist 30m`（Colab `sshd` 仅允许单条 SSH 连接，必须复用 ControlMaster 才能同时挂 `-L` 隧道和跑命令）
- `IPQoS none`、`TCPKeepAlive no`、`ServerAliveInterval 15`、`ServerAliveCountMax 20`、`ExitOnForwardFailure no`
