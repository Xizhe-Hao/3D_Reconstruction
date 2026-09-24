# `data/test`：DUSt3R 重建与 MVTracker 全时序轨迹

本文对应以下两个脚本：

- `scripts/run_duster_tracking_full.sh`：可直接运行的完整流程。
- `scripts/run_test_session.py`：完整 Python CLI，负责同步解码、DUSt3R 重建、MVTracker 跟踪和导出。

处理流程为：

```text
4 路同步 MP4 + calibration.json + alignment.csv
                  │
                  ├─ 去畸变、缩放、同步采样
                  ▼
每个时刻用 DUSt3R 对四视角建立 complete graph
                  │
                  ├─ 固定标定内参和相机位姿
                  ├─ 全局对齐并生成四路一致的米制深度
                  ▼
MVTracker 在所有采样时刻跟踪三维查询点
                  │
                  ├─ 稠密融合点云：ply_dense/
                  ├─ 稀疏轨迹点：ply/
                  └─ 轨迹与点云联合可视化：tracks_4d.rrd
```

DUSt3R 在这里负责逐时刻的多视角几何重建，MVTracker 负责跨时刻建立持久轨迹。DUSt3R 本身不会直接输出全时序轨迹。

## 1. 环境配置

```bash
cd /root/autodl-tmp/nd_project/submodule/mvtracker

bash scripts/setup_mvtracker.sh
bash scripts/setup_duster.sh
conda activate mvtracker
```

`setup_duster.sh` 固定使用：

- ETH-Zurich DUSt3R fork commit：`51baa9c2324b5af6e93c09da13a9ec9168977259`
- CroCo submodule：`743ee71a2a9bf57cea6832a9064a70a0597fcfcb`
- 模型：`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
- 模型 MD5：`c3fab9b455b03f23d20e6bf77f2607bb`
- 额外依赖：`roma==1.5.1`

当前机器没有 CUDA 编译器，因此会显示 cuRoPE2D 回退警告。它使用较慢的 PyTorch 实现，不影响结果格式和几何坐标。

## 2. 先运行短片验证

建议先验证 7 个连续时刻：

```bash
python scripts/run_test_session.py \
  --session-dir ../../data/test \
  --start 660 \
  --end 666 \
  --target-frames 7 \
  --max-frames 7 \
  --depth-backend duster \
  --duster-image-size 512 \
  --duster-ga-niter 50 \
  --duster-conf-threshold 20 \
  --query-views 0,1,2,3 \
  --query-grid-size 16 \
  --query-voxel-size-m 0.005 \
  --roi 0.2,0.2,0.8,0.8 \
  --world-radius-m 0.18 \
  --pointcloud-pixel-stride 4 \
  --pointcloud-radius-m 0.5 \
  --rerun-pointcloud-mode fused
```

短片正确后，再运行完整时序。

## 3. 完整重建和轨迹生成

### 3.1 直接运行完成脚本

```bash
cd /root/autodl-tmp/nd_project/submodule/mvtracker
conda activate mvtracker

bash scripts/run_duster_tracking_full.sh
```

脚本默认将原始 1415 帧从 0 到 1414 均匀采样为 96 个时刻，先完成每个时刻的四视角 DUSt3R 重建，再生成 MVTracker 三维轨迹、融合 PLY 和 Rerun 文件。完整脚本使用偏稠密的预设：24×24 查询网格、2 mm 查询去重、覆盖图像 5%–95% 的 ROI、0.25 m 查询半径和 2 像素点云步长。

可以通过环境变量修改常用设置：

```bash
TARGET_FRAMES=192 \
DUSTER_GA_NITER=300 \
DUSTER_CONFIDENCE=10 \
QUERY_GRID_SIZE=24 \
QUERY_VOXEL_SIZE_M=0.002 \
QUERY_ROI=0.05,0.05,0.95,0.95 \
WORLD_RADIUS_M=0.25 \
POINTCLOUD_PIXEL_STRIDE=2 \
bash scripts/run_duster_tracking_full.sh
```

也可以在命令末尾追加 Python CLI 参数；后出现的参数会覆盖脚本预设。例如：

```bash
bash scripts/run_duster_tracking_full.sh \
  --roi 0.1,0.1,0.9,0.9 \
  --world-radius-m 0.25 \
  --heartbeat-seconds 5
```

### 3.2 等价的完整 Python 命令

```bash
python scripts/run_test_session.py \
  --session-dir ../../data/test \
  --start 0 \
  --end 1414 \
  --target-frames 96 \
  --max-frames 96 \
  --width 512 \
  --height 384 \
  --depth-backend duster \
  --duster-root ../duster \
  --duster-checkpoint ../duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --duster-image-size 512 \
  --duster-ga-niter 300 \
  --duster-ga-lr 0.01 \
  --duster-conf-threshold 20 \
  --min-depth-m 0.03 \
  --max-depth-m 2.0 \
  --query-views 0,1,2,3 \
  --query-grid-size 16 \
  --query-voxel-size-m 0.005 \
  --roi 0.2,0.2,0.8,0.8 \
  --world-radius-m 0.18 \
  --pointcloud-pixel-stride 4 \
  --pointcloud-radius-m 0.5 \
  --pointcloud-point-radius-m 0.001 \
  --rerun-pointcloud-mode fused \
  --iterations 6 \
  --device cuda \
  --output-dir outputs/data_test \
  --heartbeat-seconds 15
```

## 4. 参数说明

### 4.1 输入、帧范围与下采样

| 参数 | 默认值 | 作用 | 如何设置 |
|---|---:|---|---|
| `--session-dir` | `../../data/test` | 会话目录，必须包含导出 JSON、标定 JSON、alignment CSV 和四路视频。 | 换数据时指向同格式目录。 |
| `--start` | `0` | 原视频起始帧，包含该帧。 | 只处理局部动作时设置动作开始帧。 |
| `--end` | 未指定 | 原视频结束帧，包含该帧。 | 对当前数据完整范围显式设为 `1414`。 |
| `--step` | `1` | 固定间隔采样，例如 5 表示每隔 5 帧取一帧。 | 只在没有指定 `--target-frames` 时生效。 |
| `--target-frames` | 未指定 | 在 start/end 范围内均匀选出精确的 N 帧，包含两端。 | 推荐全视频先用 96；快速形变可用 192 或更多。 |
| `--max-frames` | `96` | 安全上限，防止意外载入太多帧。 | 必须不小于 `target-frames`；192 帧时同时设为 192。 |
| `--width` | `512` | 进入 DUSt3R 和 MVTracker 的图像宽度。 | 当前模型推荐 512。减小会损失细节和标定精度。 |
| `--height` | `384` | 输入图像高度。 | 当前 4:3 视频与 512 宽对应 384 高。 |

重要的默认行为：

- 给出 `--target-frames` 但不写 `--end`：自动使用会话最后一帧。
- 不给 `--target-frames` 且不写 `--end`：为了安全只取从 start 开始的最多 24 帧。
- 同时给出 `--target-frames` 和 `--step`：`target-frames` 优先，`step` 被忽略。
- MVTracker 至少需要 7 个采样时刻。

### 4.2 深度和 DUSt3R 重建

| 参数 | 默认值 | 作用 | 如何设置 |
|---|---:|---|---|
| `--depth-backend` | `duster` | 深度来源：`duster`、`moge2` 或外部 `npz`。 | 四视角融合推荐始终使用 `duster`。 |
| `--duster-root` | `../duster` | ETH DUSt3R fork 源码目录。 | 通常不修改。 |
| `--duster-checkpoint` | 512 DPT 权重 | DUSt3R 模型文件。 | 使用自定义权重时修改，并保证与分辨率兼容。 |
| `--duster-image-size` | `512` | DUSt3R 推理长边尺寸，可选 224/512。 | 当前下载的是 512 权重，建议保持 512。 |
| `--duster-ga-niter` | `300` | 每个时刻多视角全局对齐迭代次数。 | 短测用 20–50；正式结果用 300。 |
| `--duster-ga-lr` | `0.01` | 全局对齐学习率。 | 通常保持 0.01；震荡时可降至 0.005。 |
| `--duster-conf-threshold` | `20` | DUSt3R 有效像素置信度阈值。 | 20 较干净；10 较均衡；3 更密但噪声更多。 |
| `--duster-no-clean-depth` | 关闭 | 禁用官方几何深度清理。 | 一般不要加；只有追求最大密度并能接受离群点时使用。 |
| `--recompute-depth` | 关闭 | 忽略汇总缓存并重新生成逐时刻 DUSt3R 场景。 | 修改 GA、置信度或清理参数后需要加。 |
| `--min-depth-m` | `0.03` | 最小有效相机 Z 深度，单位米。 | 过滤镜头附近异常点；当前场景保持默认。 |
| `--max-depth-m` | `2.0` | 最大有效相机 Z 深度，单位米。 | 应大于相机到目标最远距离。 |

以下参数只用于其他深度后端：

| 参数 | 后端 | 说明 |
|---|---|---|
| `--depth-path` | `npz` | 外部深度 NPZ，形状应为 `[V,T,H,W]` 或 `[V,T,1,H,W]`。 |
| `--depth-unit` | `npz` | 外部深度单位，`m` 或 `mm`。 |
| `--moge-model` | `moge2` | Hugging Face MoGe-2 模型名称。 |
| `--depth-batch-size` | `moge2` | MoGe-2 批大小，显存不足时减小。 |

### 4.3 轨迹查询点

| 参数 | 默认值 | 作用 | 如何设置 |
|---|---:|---|---|
| `--query-views` | `0,1,2,3` | 从哪些相机的首时刻表面生成轨迹查询点。 | 完整覆盖物体应使用全部四路。 |
| `--query-view` | 未指定 | 兼容旧接口的单相机覆盖选项。设置后会忽略 query-views。 | 仅用于复现实验，不推荐完整物体跟踪。 |
| `--query-grid-size` | `16` | 每个查询视角在 ROI 中采样 N×N 个候选点。 | 16 较快；24 更密；32 很密且显存、时间明显增加。 |
| `--query-voxel-size-m` | `0.005` | 合并多相机查询后，按世界坐标体素去重，单位米。 | 5 mm 较均衡；降到 0.002–0.003 会保留更多轨迹。 |
| `--roi` | `0.2,0.2,0.8,0.8` | 归一化图像坐标中的查询区域 `x0,y0,x1,y1`。 | 目标更大时扩大；避免把背景纳入 ROI。 |
| `--world-radius-m` | `0.18` | 只保留距标定世界原点该半径内的查询点。 | 目标被截断就增大到 0.25/0.35；背景过多则减小。 |

轨迹密度主要由 `query-grid-size`、`query-views`、`roi` 和
`query-voxel-size-m` 决定。它与稠密点云的 `pointcloud-pixel-stride`
不是同一个概念。

需要注意，当前查询点在第一个采样时刻初始化。四台相机可以覆盖首时刻从不同方向可见的表面，但首时刻仍然完全遮挡的区域不会凭空产生可靠轨迹。应选择物体轮廓完整的 `--start`，并把 ROI 覆盖到物体边缘。

### 4.4 稠密点云与 Rerun

| 参数 | 默认值 | 作用 | 如何设置 |
|---|---:|---|---|
| `--pointcloud-pixel-stride` | `4` | 深度图每隔多少像素导出一点。 | 1 最密且文件最大；2 稠密；4 均衡；8/16 用于预览。 |
| `--pointcloud-radius-m` | `0.5` | 融合点云相对世界原点的裁剪半径。 | 必须覆盖完整物体；若点云截断则增大。 |
| `--pointcloud-point-radius-m` | `0.001` | Rerun 中点的显示半径，只影响显示。 | 点太大就减小，太难看清就增大。 |
| `--rerun-pointcloud-mode` | `fused` | `fused` 将四相机点放入一个实体；`per-view` 分相机显示。 | 最终查看使用 fused；诊断单路深度使用 per-view。 |
| `--no-dense-ply` | 关闭 | 不导出 `ply_dense/`。 | 仅追踪、不需要稠密 PLY 时添加。 |
| `--no-dense-rerun` | 关闭 | RRD 中不记录稠密点云。 | RRD 文件太大时添加。 |

### 4.5 MVTracker、设备和日志

| 参数 | 默认值 | 作用 | 如何设置 |
|---|---:|---|---|
| `--checkpoint` | 官方 MVTracker 权重 | MVTracker checkpoint 路径。 | 通常不修改。 |
| `--device` | `cuda` | 推理设备。 | 实际重建建议 CUDA；CPU 会非常慢。 |
| `--iterations` | `6` | MVTracker 内部迭代次数。 | 默认即可；提高可能更慢且不保证更好。 |
| `--output-dir` | `outputs/data_test` | 所有结果的根目录。 | 可改到空间充足的数据盘。 |
| `--heartbeat-seconds` | `15` | MVTracker 无内部回调时的状态打印间隔。 | 审查卡顿建议设为 5。 |
| `--no-progress` | 关闭 | 关闭阶段日志和 tqdm 进度条。 | 批处理写日志时使用；调试时不要加。 |

## 5. 输出文件

对于完整示例命令，结果目录类似：

```text
outputs/data_test/frames_0_1414_target_96_duster/
├── depths_duster_m.npz
├── duster/
│   ├── manifest.json
│   ├── 3d_model__00000__scene.npz
│   ├── 3d_model__00001__scene.npz
│   └── ...
├── tracks_4d.npz
├── tracks_4d.csv
├── tracks_4d.rrd
├── ply/
│   ├── tracks_000000.ply
│   └── ...
├── ply_dense/
│   ├── pointcloud_000000.ply
│   └── ...
└── run.json
```

各文件含义：

- `duster/3d_model__XXXXX__scene.npz`：某个采样时刻的四视角 DUSt3R 中间结果，包括 `depths`、`confs`、`cleaned_mask` 和官方特征。
- `duster/manifest.json`：局部时刻编号和原视频帧号的对应关系、DUSt3R 参数及有效比例。
- `depths_duster_m.npz`：汇总后的 `[V,T,H,W]` 米制深度，供 MVTracker 和点云导出复用。
- `tracks_4d.npz`：最终轨迹数组、可见性、置信度、原视频帧号和时间戳。
- `tracks_4d.csv`：便于表格分析的逐轨迹、逐时刻记录。
- `ply/`：当前时刻可见的稀疏跟踪点，不代表完整表面。
- `ply_dense/`：四相机深度变换到标定世界坐标后合并的稠密点云。
- `tracks_4d.rrd`：融合点云、固定相机和动态轨迹的联合可视化。
- `run.json`：本次运行的参数摘要与结果统计。

## 6. 查看可视化结果

```bash
conda activate mvtracker
rerun outputs/data_test/frames_0_1414_target_96_duster/tracks_4d.rrd
```

Rerun 中主要实体：

- `/world/pointcloud/fused`：四相机融合后的当前时刻稠密点云。
- `/world/tracks/current`：当前时刻的可见轨迹点。
- `/world/tracks/history`：历史轨迹线。
- `/world/cameras`：四个标定相机。

## 7. 常用调参组合

### 推荐的密集配置

已经在 7 帧测试上验证以下配置：

```bash
python scripts/run_test_session.py \
  --session-dir ../../data/test \
  --start 660 --end 666 \
  --target-frames 7 --max-frames 7 \
  --depth-backend duster \
  --query-views 0,1,2,3 \
  --query-grid-size 24 \
  --query-voxel-size-m 0.002 \
  --roi 0.05,0.05,0.95,0.95 \
  --world-radius-m 0.25 \
  --pointcloud-pixel-stride 2 \
  --pointcloud-radius-m 0.5
```

同一份 DUSt3R 深度下，测试结果由 165 条轨迹、每帧约 1,100 个点，提高到
960 条轨迹、每帧 71,873–73,488 个融合点。若已有
`depths_duster_m.npz`，改变上述查询和 PLY 参数不需要
`--recompute-depth`。

### 时间密度与形变

空间点再密也不能补回被时间下采样跳过的形变。当前 24 FPS、1415 帧视频：

| target-frames | 相邻采样约间隔 | 适用情况 |
|---:|---:|---|
| 96 | 14–15 原始帧，约 0.62 秒 | 初次完整运行、缓慢形变 |
| 192 | 7–8 原始帧，约 0.31 秒 | 推荐的动态形变设置 |
| 288 | 约 5 原始帧，约 0.21 秒 | 较快形变，耗时与存储明显增加 |

要更好体现形变，优先把 `TARGET_FRAMES` 提高到 192；如果显存不足，先降低
`QUERY_GRID_SIZE`，不要继续减少时间点。完整脚本会自动让
`--max-frames` 等于 `TARGET_FRAMES`。

快速排错：

```bash
DUSTER_GA_NITER=20 QUERY_GRID_SIZE=8 POINTCLOUD_PIXEL_STRIDE=16 \
bash scripts/run_duster_tracking_full.sh
```

优先保证轨迹可靠：

```bash
DUSTER_GA_NITER=300 DUSTER_CONFIDENCE=20 QUERY_GRID_SIZE=16 \
bash scripts/run_duster_tracking_full.sh
```

增加轨迹数量：

```bash
QUERY_GRID_SIZE=24 \
bash scripts/run_duster_tracking_full.sh \
  --query-voxel-size-m 0.003
```

增加稠密点云数量：

```bash
POINTCLOUD_PIXEL_STRIDE=2 DUSTER_CONFIDENCE=10 \
bash scripts/run_duster_tracking_full.sh
```

如果修改 DUSt3R 的置信度、清理方式或 GA 参数，并希望重新计算已有结果：

```bash
bash scripts/run_duster_tracking_full.sh --recompute-depth
```

不要在仅想重新导出 PLY/RRD 时添加 `--recompute-depth`；不加该参数会直接复用
`depths_duster_m.npz`，节省绝大部分时间。

## 8. 结果边界

- `ply_dense/` 是每个时刻独立融合的 RGB-D 点云，不含跨时刻点 ID。
- `tracks_4d.*` 才包含跨时刻持久对应关系。
- 当前输出不是封闭网格，也不是带拓扑的动态表面模型。
- 不同视角只看见物体的不同表面，因此分视角点云的质心不必完全相同；判断是否出现“四份物体”应查看融合包围盒、共同可见区域和 Rerun，而不能只比较整组点云质心。

## 9. Gradio 交互页面与形变视频

启动可视化页面：

```bash
conda activate mvtracker
python scripts/mvtracker_visualizer_gradio.py \
  --result-dir outputs/data_test/frames_0_1414_target_96_duster
```

默认服务：

- Gradio：`http://127.0.0.1:7861`
- Rerun Web：`http://127.0.0.1:9090`
- Rerun WebSocket：`127.0.0.1:9877`

页面提供两个独立操作：

- **Load interactive Rerun**：加载已有 `tracks_4d.rrd`。可以旋转、缩放、拖动时间轴，并分别开关融合点云、当前轨迹点、历史轨迹和四个相机。
- **Render deformation MP4**：逐帧读取 `ply_dense/`，叠加 MVTracker 当前点和历史轨迹线，生成 H.264 MP4，并在页面播放和下载。

远程机器建议通过 SSH 同时转发三个端口：

```bash
ssh \
  -L 7861:127.0.0.1:7861 \
  -L 9090:127.0.0.1:9090 \
  -L 9877:127.0.0.1:9877 \
  user@server
```

然后在本机浏览器打开 `http://127.0.0.1:7861`。默认只绑定 localhost，
不会把 RRD 数据直接暴露到公网。


### 9.1 服务器运行时 Rerun 显示“127.0.0.1 拒绝连接”

这是因为 Gradio 页面中的 iframe 由本地浏览器加载。iframe 里的
`127.0.0.1:9090` 指向运行浏览器的本地电脑，而不是远程服务器；如果只转发
Gradio 的 `7861` 端口，Rerun Web 页面和数据连接仍然无法建立。

请在本地电脑上执行 SSH 端口转发，并保持该终端窗口运行：

```bash
ssh -N \
  -L 7861:127.0.0.1:7861 \
  -L 9090:127.0.0.1:9090 \
  -L 9877:127.0.0.1:9877 \
  root@服务器地址
```

如果 AutoDL 提供的 SSH 登录命令包含自定义端口，需要同时传入 `-p`：

```bash
ssh -N -p SSH端口 \
  -L 7861:127.0.0.1:7861 \
  -L 9090:127.0.0.1:9090 \
  -L 9877:127.0.0.1:9877 \
  root@服务器地址
```

服务器端仍使用默认参数启动：

```bash
conda activate mvtracker
python scripts/mvtracker_visualizer_gradio.py \
  --result-dir outputs/data_test/frames_0_1414_target_96_duster
```

随后在本地浏览器访问 `http://127.0.0.1:7861`，再点击
**Load interactive Rerun**。三个端口必须一起转发：

- `7861`：Gradio 页面。
- `9090`：Rerun Web Viewer。
- `9877`：Rerun 数据 WebSocket。

如果使用 AutoDL 的自定义服务或公网端口映射，需要同时映射以上三个端口，
并通过 `--viewer-url` 指定浏览器实际可访问的 Rerun Web 地址。例如：

```bash
python scripts/mvtracker_visualizer_gradio.py \
  --result-dir outputs/data_test/frames_0_1414_target_96_duster \
  --server-name 0.0.0.0 \
  --rerun-bind 0.0.0.0 \
  --viewer-url http://服务器公网地址:9090
```

公网绑定会暴露可视化服务和 RRD 数据，应配合防火墙、平台鉴权或仅允许可信
来源访问。如果 Gradio 页面通过 HTTPS 打开，而 `--viewer-url` 使用 HTTP，浏览器
还可能拦截混合内容；这种情况下应为 Rerun 配置 HTTPS 反向代理，或者改用前述
SSH 隧道方案。

只生成视频而不启动 Gradio：

```bash
python scripts/mvtracker_visualizer_gradio.py \
  --result-dir outputs/data_test/frames_0_1414_target_96_duster \
  --render-only \
  --fps 12 \
  --trail-length 20 \
  --azimuth -60 \
  --elevation 20 \
  --max-render-points 80000 \
  --video-width 960 \
  --video-height 720
```

生成的视频保存在结果目录的 `visualization/` 下。参数含义：

- `--fps`：成片播放帧率，不改变原始轨迹时间戳。
- `--trail-length`：每个点显示最近多少个采样时刻的轨迹。
- `--azimuth` / `--elevation`：离屏渲染相机的水平角和俯仰角。
- `--max-render-points`：每帧最多渲染多少个 PLY 点；只影响视频速度，不修改原 PLY。
- `--video-width` / `--video-height`：MP4 分辨率。
- `--viewer-url`：浏览器实际可访问的 Rerun Web 地址。
- `--rerun-web-port` / `--rerun-ws-port`：Rerun HTTP 与 WebSocket 端口。
- `--server-port`：Gradio 端口。
