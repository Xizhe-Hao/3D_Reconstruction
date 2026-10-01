# DA3 giant 安装与管线验证

验证日期：2026-10-01。源码 commit：
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`。
Hugging Face `depth-anything/DA3-GIANT` revision：
`7cd62ae9315b9dff094d2d300e4ad012640607dd`。
权重 SHA256：`1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001`。
参数量：1,355,674,125。

[官方仓库](https://github.com/ByteDance-Seed/Depth-Anything-3)和
[官方 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/api.py)
说明 giant 支持输入相机标定，API 可以将深度对齐到输入外参尺度。
本项目每个同步时间点输入四视角 RGB、标定 K 和米制 w2c 外参，
采用 `align_to_input_ext_scale=True`；不同时间点独立运行。

## 已完成验证

- 最终 `bash scripts/setup_da3.sh` 重跑成功，实际加载 giant checkpoint。
- 144 个固定依赖及其传递依赖版本约束通过；依赖列表见
  `scripts/requirements_da3.lock`。
- venv 内使用干净的 torch 2.4.0、torchvision 0.19.0、xformers 0.0.27.post2。
  base 环境原有 torch ONNX 文件混杂，通过在 venv 内安装解决。
  DA3 的依赖闭包通过检查；继承的无关 base 包 librosa 与 NumPy 1 存在版本冲突，
  不用于本后端。MVTracker 独立 Conda 环境未改动。
- Open3D wheel 对照 PyPI SHA256 校验通过。
- 真实推理时断言返回内参等于按尺寸缩放后的输入 K、返回外参等于输入 w2c，
  确认 resize 后视场保留。
- 相同输入重复调用后端，帧缓存和汇总深度文件的修改时间均未变化。
- 现有 6 项几何测试、Python 编译检查、安装脚本语法及 diff 格式检查通过。

## 真实 smoke test

输入 `data/test_1001`，视频区间 0–0.5 秒，采样帧
`[0, 2, 4, 6, 8, 10, 12]`，四相机，跟踪尺寸 512×384。
模型处理尺寸为 504×378，深度 resize 回 512×384。

| 指标 | 结果 |
| --- | --- |
| 深度输出形状 | `[4, 7, 384, 512]`，float32 米 |
| DA3 峰值 GPU 分配显存 | 7.93 GiB |
| 首次四视角处理（不含模型加载） | 1.72 秒 |
| 后续四视角处理 | 0.82–1.07 秒/时间点 |
| 深度阶段累计完成时刻 | 31.4 秒（含数据解码、模型加载、保存） |
| 完整管线耗时 | 41.0 秒 |
| 过滤后有效深度比例 | 91.44% |
| 过滤后深度中位数 | 0.14724 米 |
| 轨迹输出形状 | `[7, 858, 3]` |
| 跟踪可见比例 | 96.42% |
| 每帧稠密点云 | 43,657–44,993 点 |

结果目录：`outputs/data_test_da3/seconds_0_0.5_target_7_da3/`。
包含 `depths_da3_m.npz`、逐帧原始深度/置信度、稠密 PLY、
`tracks_4d.npz`、`tracks_4d.csv`、`tracks_4d.rrd` 和 `run.json`。
`da3/depth_preview.png` 为首帧四视角 RGB/深度图；
`da3/validation.json` 保存简单重投影检查。

## 当前精度观察

预览中主体形状可见，但平面和棱角被平滑，微小表面纹理未明显恢复。
首帧按 stride=4 在相机间重投影深度，差值绝对值中位数为：

| 方向 | 差值中位数 | 相对差值中位数 |
| --- | --- | --- |
| 0 → 2 | 3.30 mm | 2.38% |
| 2 → 0 | 5.60 mm | 3.24% |
| 1 → 3 | 21.83 mm | 15.26% |
| 3 → 1 | 5.69 mm | 3.31% |

这些统计包含遮挡，没有真值，不是深度误差测量；1 → 3 的较大差异需要
进一步检查可见性和物体 ROI。0–0.5 秒仅验证完整链路，不足以评价加载期间
的时间一致性和毫米级形变精度。后续应在明确的受力形变片段上与标定双目
深度、表面平面和已知位移做对比。
