# 20261014 RGB4 + D455 重建与浏览器

数据：`data/drive_test/capture_20261014_202456`，四路 2048×1536 RGB 视频，共 871 帧、6fps。1742 张 D455 彩色/uint16 深度图解码验证通过。

## 流程

`scripts/reconstruct_rgbd_web.py` 复用 `test_session_adapter.py` 的标定和去畸变逻辑。原始分辨率去畸变后，以四视角同时输入 DA3-GIANT（固定版本权重），使用相机内外参与 `align_to_input_ext_scale=True` 恢复米制深度。没有先降采样到 512。

四视角深度去除各自置信度最低 20% 的像素，按像素步长 2 反投影到统一世界坐标。D455 使用当前会话的内参和深度单位、最新仓库世界外参及深度校正；过滤有效范围外和 3×3 邻域深度跳变超过 12mm 的边缘点。D455 颜色通过实测色彩相机投影，处理 inverse Brown-Conrady 畸变，无法取色的点标为青色。

实验区域裁剪（米）：x ∈ [-0.05, 0.22]，y ∈ [-0.23, 0.08]，z ∈ [-0.04, 0.20]。每帧独立重建，不跨时间累计，避免把海绵动态形变糊成一体。

导出每路 RGB、RGB4、D455 和融合二进制 PLY。RGB4 与 D455 各做 1mm 体素均值，再合并并再次体素化。这是点云融合，不是封闭网格或 TSDF；不自动修补遮挡面，也不把 D455 输入给 DA3。

## 24GB 显存实测

RTX 4090 D，PyTorch 2.7.1+cu128，DA3 内部 BF16：

| 实际尺寸 | allocated 峰值 GiB | reserved 峰值 GiB |
|---|---:|---:|
| 1008×756 | 12.78 | 18.04 |
| 1176×882 | 15.11 | 22.29 |
| 1344×1008 | 17.80 | 21.87 |
| 1512×1134 | 20.85 | 22.36 |

1512 普通分配器在连续第二帧出现碎片导致的 OOM。启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 并在每帧后清理未用 CUDA 缓存后，代表性连续帧验证通过，采用 **1512×1134** 正式处理。reserved 包含缓存，不能与 allocated 相加。高分辨率不自动意味着更小的几何误差。

## 运行

从 reconstruction 目录：

```bash
.venv-da3/bin/python scripts/run_rgbd_sequence.py \
  --session data/drive_test/capture_20261014_202456 \
  --output outputs/rgbd_20261014 --resolution 1512 \
  --gpus 0,1,2,3,4,5,6,7
python3 scripts/serve_reconstruction.py --root outputs/rgbd_20261014 --port 8080
```

每张卡独立处理一个时间段，四路相机始终在同一张卡联合推理，不合并多卡显存。相同配置可跳过已完成帧。运行前复制 `web/index.html` 到输出目录，并准备 `run.json`（当前结果中已有）。

当前网页：http://10.120.192.13:8080/ 。如果本机无法访问服务器内网，通过 SSH/IDE 转发 8080，再打开 http://127.0.0.1:8080/ 。网页使用原生 WebGL，不依赖 CDN。服务仅公开该次输出目录。

## 分析与限制

`report.json` 动态汇总每帧点数、耗时、显存、D455 时间偏差、重叠像素数量、RGB 重建相对于 D455 的深度差中位数/P90/带符号中位数及对齐的力值。RGB 点云投影到 D455 深度图，用最近深度 z-buffer 比较，不对未覆盖像素计算误差。

D455 不是真值；遮挡、深度传感器噪声、标定、同步及薄膜反光都会影响差异。相隔数毫米的 RGB 和 D455 表面不会被 1mm 体素化强制对齐，因此可能出现双层面。原始数据中有 18 帧 D455 同步偏差超过 40ms，页面对此标注警告。颜色不用于精度评估。

运行配置、环境锁定清单、标定副本和分辨率实测记录保存在输出目录。几何验证：`.venv-da3/bin/python scripts/test_rgbd_geometry.py`。
