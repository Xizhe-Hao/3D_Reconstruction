# MVTracker 多视角重建复现工程

本仓库只保存项目脚本、说明文档和第三方仓库的 Git submodule 指针。数据集、模型权重、重建输出等大文件不会提交到 GitHub。

## 仓库结构

```text
.
├── scripts/                  # 本项目的运行、安装和可视化脚本
├── docs/                     # 数据适配和重建流程补充说明
├── submodule/
│   ├── mvtracker/            # 官方 MVTracker submodule（固定 commit）
│   ├── duster/               # 官方 DUSt3R/DUStER submodule（固定 commit）
│   ├── FoundationStereo/     # 官方 FoundationStereo submodule（固定 commit）
│   └── depth-anything-3/     # 官方 DA3 submodule（固定 commit）
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
HTTPS_PROXY=http://127.0.0.1:17897 HTTP_PROXY=http://127.0.0.1:17897 \
  bash scripts/setup_foundationstereo.sh
```

安装脚本会：

- 创建 `mvtracker` Conda 环境并安装 MVTracker 依赖；
- 下载 MVTracker checkpoint 到 `submodule/mvtracker/checkpoints/`；
- 初始化固定版本的 DUStER 及其递归 submodules；
- 下载约 2.1 GB 的 DUSt3R checkpoint 到 `submodule/duster/checkpoints/`；
- 校验 DUSt3R checkpoint 的 MD5；
- 在独立 `foundation_stereo` 环境安装 PyTorch 2.4.1、flash-attn 和运行依赖，
  精确下载并加载 ViT-L `23-51-11` checkpoint。

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

## FoundationStereo 全分辨率深度基线

[FoundationStereo 官方仓库](https://github.com/NVlabs/FoundationStereo)及其
[许可证](https://raw.githubusercontent.com/NVlabs/FoundationStereo/master/LICENSE)规定代码和研究权重仅允许非商业研究用途，并使用独立的
`foundation_stereo` Conda 环境。输入会按标定参数校正为水平极线；默认使用
`0-2,1-3` 两组近邻相机，在原始 2048×1536 分辨率运行 hierarchical inference；
不能把未校正的四路原图直接送入模型（参见[官方输入说明](https://github.com/NVlabs/FoundationStereo#run-demo)）。
每个时间点的 float32 深度、视差和有效掩码保存到
`foundationstereo/frame_NNNNN.npz`，96 个时间点预计占用数 GB。低分辨率校正
RGB-D 供 MVTracker 使用，高分辨率深度直接生成稠密 PLY。中断后会逐帧续跑；
传入 `--recompute-depth` 可强制重算。全分辨率 32 次迭代明显慢于下面的 smoke test，
并且 96 帧缓存通常需要数 GB 到十余 GB 空间。

```bash
conda run --no-capture-output -n mvtracker \
  python scripts/run_test_session.py \
  --session-dir data/test_930 \
  --start 12.4 --end 13.5 \
  --target-frames 7 --max-frames 7 \
  --width 512 --height 384 \
  --depth-backend foundationstereo \
  --foundationstereo-pairs 0-2,1-3 \
  --foundationstereo-valid-iters 32 \
  --foundationstereo-hiera \
  --foundationstereo-hiera-small-ratio 0.25 \
  --foundationstereo-lr-consistency-px 1.0 \
  --foundationstereo-scale 1.0 \
  --min-depth-m 0.03 --max-depth-m 0.5 \
  --query-grid-size 32 --query-voxel-size-m 0.001 \
  --pointcloud-pixel-stride 4 \
  --device cuda \
  --output-dir outputs/data_test_foundationstereo
```

独立查看 FoundationStereo 的四视角校正 RGB、过滤深度、原始深度、有效掩码和视差：

```bash
conda run --no-capture-output -n mvtracker \
  python scripts/foundationstereo_depth_viewer_gradio.py \
  --result-dir outputs/data_test_foundationstereo/seconds_12.4_13.5_target_7_foundationstereo \
  --server-name 127.0.0.1 \
  --server-port 7861
```

浏览器通过 SSH 转发访问 http://127.0.0.1:7861。页面中的播放按钮会同步更新四个视角。

## 4. 运行 96 帧重建

所有命令从仓库根目录执行。`--start` 和 `--end` 的单位是视频秒；`--end` 省略时使用视频末尾：

```bash
conda run --no-capture-output -n mvtracker \
  python scripts/run_test_session.py \
  --session-dir data/test \
  --start 0 \
  --end 58.9167 \
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
outputs/data_test/seconds_0_58.9167_target_96_duster/
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
  --result-dir outputs/data_test/seconds_0_58.9167_target_96_duster \
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

## Depth Anything 3 giant 标定多视角深度

官方源码位于 `submodule/depth-anything-3`，固定到
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`。安装：

```bash
bash scripts/setup_da3.sh
```

安装到项目内 `.venv-da3`，在 venv 内安装匹配的 PyTorch 2.4.0 / torchvision 0.19.0，
其余匹配依赖可复用 base，使用 uv 和 `scripts/requirements_da3.lock` 固定版本；
不会改动 MVTracker 环境。权重下载至 `outputs/models/DA3-GIANT`。
脚本自动使用 `/etc/network_turbo`（存在时）。

```bash
conda run --no-capture-output -n mvtracker python scripts/run_test_session.py \
  --session-dir data/test_1001 --start 0 --end 0.5 \
  --target-frames 7 --max-frames 7 --width 512 --height 384 \
  --depth-backend da3 --da3-process-res 504 \
  --da3-confidence-percentile 10 --min-depth-m 0.03 --max-depth-m 2 \
  --query-grid-size 16 --device cuda --output-dir outputs/data_test_da3
```

DA3-GIANT 本身不是单目绝对米制模型：这里每个时间点联合推理四路去畸变
RGB，输入标定内参及米制 world-to-camera 外参，通过官方
`align_to_input_ext_scale=True` 对齐尺度。不同时间点独立处理，避免把形变
序列假设为静态场景。使用 resize 预处理并将深度恢复到原始输入尺寸，
保留管线原有 RGB 和相机标定。米制尺度对齐不代表已验证真实深度精度。

`da3/frame_NNNNN.npz` 保存跟踪尺寸米制深度、模型尺寸原始米制深度和
置信度；`da3/manifest.json` 保存输入配置、有效比例和 GPU 峰值。
默认剔除每个视角置信度最低的 10% 像素，可设为 0。
缓存指纹包含 RGB、标定、帧索引、模型文件及推理配置；中断可逐帧续跑，
`--recompute-depth` 强制重算。子进程退出后释放 DA3 显存，再运行 MVTracker。

安装、真实 7 帧管线验证及当前精度观察见 [DA3 验证记录](docs/DA3_VALIDATION_ZH.md)。

使用原有四视角深度 Web 查看 DA3（支持同步播放、共享自动色标和固定米制范围）：

```bash
conda run --no-capture-output -n mvtracker \
  python scripts/duster_depth_viewer_gradio.py \
  --result-dir outputs/data_test_da3/seconds_0_0.5_target_7_da3 \
  --server-name 127.0.0.1 --server-port 7862
```

在 IDE 中转发 7862 端口后访问 http://127.0.0.1:7862。
`--result-dir` 可以传运行结果目录，也可以直接传其中的 `da3/` 或 `duster/`。
原来的 `--duster-dir` 参数仍然可用。页面显示与 MVTracker 实际使用一致的
过滤后深度；RGB 从 DA3 输入缓存读取，与深度逐帧对齐。
