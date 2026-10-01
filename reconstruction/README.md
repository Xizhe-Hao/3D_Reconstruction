# MVTracker 多视角重建复现工程

本仓库只保存项目脚本、说明文档和第三方仓库的 Git submodule 指针。数据集、模型权重、重建输出等大文件不会提交到 GitHub。

## 仓库结构

```text
.
├── scripts/                  # 本项目的运行、安装和可视化脚本
├── docs/                     # 数据适配和重建流程补充说明
├── submodule/
│   ├── mvtracker/            # 官方 MVTracker submodule（固定 commit）
│   └── duster/               # 官方 DUSt3R/DUStER submodule（固定 commit）
├── data/                     # 本地输入数据，Git 忽略
└── outputs/                  # 本地重建结果，Git 忽略
```

不要把项目脚本复制回 `submodule/mvtracker/scripts`。第三方 submodule 应保持干净，所有定制代码都位于根目录 `scripts/`。

## 1. 克隆仓库和 submodules

推荐一次性递归克隆：

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
export PYTHONDONTWRITEBYTECODE=1
```

如果已经普通克隆：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

检查版本和状态：

```bash
git submodule status --recursive
git -C submodule/mvtracker status --short
git -C submodule/duster status --short
```

submodule 的具体 commit 由主仓库固定，不要直接切换到各项目的最新分支，否则可能产生依赖不兼容。

## 2. 创建 Conda 环境和下载模型

需要 Linux、Conda、CUDA 12.1 兼容驱动，以及足够的磁盘空间。运行：

```bash
bash scripts/setup_mvtracker.sh
bash scripts/setup_duster.sh
```

安装脚本会：

- 创建 `mvtracker` Conda 环境并安装 MVTracker 依赖；
- 下载 MVTracker checkpoint 到 `submodule/mvtracker/checkpoints/`；
- 初始化固定版本的 DUStER 及其递归 submodules；
- 下载约 2.1 GB 的 DUSt3R checkpoint 到 `submodule/duster/checkpoints/`；
- 校验 DUSt3R checkpoint 的 MD5。

checkpoint 目录已被 `.gitignore` 排除，不会进入提交。

## 3. 准备数据

将同步四相机数据放在：

```text
data/test/
```

目录至少需要包含导出描述、相机标定、对齐 CSV 和对应视频。详细格式参见：

- `docs/MVTRACKER_DATA_TEST.md`
- `docs/DUSTER_RECONSTRUCTION_TRACKING_ZH.md`

可先检查适配器：

```bash
conda run --no-capture-output -n mvtracker \
  python scripts/test_session_adapter.py \
  --session-dir data/test \
  --target-frames 7 \
  --max-frames 7 \
  --output /tmp/mvtracker_test_clip.npz
```

## 4. 运行 96 帧重建

所有命令从仓库根目录执行：

```bash
conda run --no-capture-output -n mvtracker \
  python scripts/run_test_session.py \
  --session-dir data/test \
  --start 0 \
  --end 1414 \
  --target-frames 96 \
  --max-frames 96 \
  --width 512 \
  --height 384 \
  --depth-backend duster \
  --duster-root submodule/duster \
  --duster-checkpoint submodule/duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --duster-image-size 512 \
  --duster-ga-niter 300 \
  --duster-ga-lr 0.01 \
  --duster-conf-threshold 20 \
  --min-depth-m 0.03 \
  --max-depth-m 2.0 \
  --query-views 0,1,2,3 \
  --query-grid-size 32 \
  --query-voxel-size-m 0.005 \
  --roi 0.2,0.2,0.8,0.8 \
  --world-radius-m 0.25 \
  --pointcloud-pixel-stride 4 \
  --pointcloud-radius-m 0.5 \
  --pointcloud-point-radius-m 0.001 \
  --rerun-pointcloud-mode fused \
  --iterations 6 \
  --device cuda \
  --output-dir outputs/data_test \
  --heartbeat-seconds 15
```

也可以使用封装命令：

```bash
bash scripts/run_duster_tracking_full.sh
```

结果始终写入根目录 `outputs/`。默认 96 帧结果目录为：

```text
outputs/data_test/frames_0_1414_target_96_duster/
```

## 5. 启动可视化界面

```bash
cleanup() {
  fuser -k 7861/tcp 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

fuser -k 7861/tcp 2>/dev/null || true

conda run --no-capture-output -n mvtracker \
  python scripts/mvtracker_visualizer_gradio.py \
  --result-dir outputs/data_test/frames_0_1414_target_96_duster \
  --viewer-url 'http://127.0.0.1:9090/?url=ws://127.0.0.1:9877'
```

默认 Gradio 地址为 `http://127.0.0.1:7861`。

## 6. 提交前检查

```bash
git status --short
git submodule status --recursive
git check-ignore -v data outputs \
git check-ignore -v submodule/mvtracker/checkpoints/mvtracker_200000_june2025.pth \
git check-ignore -v submodule/duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

主仓库应只提交源码、文档、`.gitmodules` 和 submodule gitlink。不要强制添加 `data/`、`outputs/`、checkpoint 或其他模型权重。

## 常见问题

- `ModuleNotFoundError: dust3r`：确认执行了 `git submodule update --init --recursive` 和 `bash scripts/setup_duster.sh`。
- 找不到 checkpoint：重新运行对应 setup 脚本；下载支持断点续传。
- CUDA 内存不足：降低 `--target-frames`、`--query-grid-size` 或输入分辨率。
- submodule 显示修改：不要在第三方目录保存项目文件；用 `git -C submodule/<name> status --short` 定位修改。
