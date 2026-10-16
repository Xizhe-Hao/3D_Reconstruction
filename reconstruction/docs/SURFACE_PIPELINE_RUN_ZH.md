# 全帧 AprilTag 外角表面与压头约束：运行入口

这些脚本是针对 `capture_20261014_202456` 的可复现实验流程。标定脚本中的图像搜索范围、初始位移与路径是该次采集的配置，更换采集必须重新标定。需要已有的 `outputs/rgbd_20261014/frames/` 逐相机PLY、depths.npz、metrics.json及RGB预览；原始重建见 `RGBD_WEB_RUN_ZH.md`。

依赖：Python 3.10、NumPy、SciPy、opencv-contrib-python；实验图需要Matplotlib。完整DA3重建还需要相应PyTorch环境和模型。数据、模型、生成点云不随代码上传。

从 `reconstruction` 目录运行：

```bash
# 1. 外角ROI：三个标签多视角三角化，一个单视角/参考平面估计
.venv-da3/bin/python scripts/build_outer_tag_roi.py

# 2. 外角范围下的多视角融合
.venv-da3/bin/python scripts/generate_surface_sequence.py \
  --run outputs/rgbd_20261014 \
  --grid outputs/rgbd_20261014/outer_tag_roi.json \
  --output outputs/rgbd_20261014/surface_outer --workers 8

# 3. 初始压头底面标定（保持原参考坐标系）
.venv-da3/bin/python scripts/calibrate_rigid_indenter.py
mkdir -p outputs/rgbd_20261014/surface_contact_outer
cp outputs/rgbd_20261014/surface_contact/indenter_calibration.json \
  outputs/rgbd_20261014/surface_contact_outer/indenter_calibration.json

# 4. 全帧位移约束补全
CONTACT_OUTPUT=surface_contact_outer SURFACE_BASE=surface_outer \
SURFACE_GRID=outputs/rgbd_20261014/outer_tag_roi.json \
.venv-da3/bin/python scripts/generate_contact_surface.py

# 5. 验证、发布离线网页与启动服务
CONTACT_OUTPUT=surface_contact_outer \
SURFACE_GRID=outputs/rgbd_20261014/outer_tag_roi.json \
.venv-da3/bin/python scripts/validate_contact_surface.py
python3 scripts/publish_surface_web.py
python3 scripts/serve_reconstruction.py --root outputs/rgbd_20261014 --port 8080
```

打开 http://127.0.0.1:8080/surface_contact_outer/viewer.html 。支持1/2/4/8倍播放、同步四路RGB和D455图像、形变热图、观测/估计显示及PLY下载。字体本地加载，许可证见 `web/OFL-NotoSansSC.txt`。

ROI是四个AprilTag黑色边框最外角组成的四边形，不含白纸边缘；不要用旧内角ROI或直接假定120×120mm代替。原参考中心和轴不变。外角ROI的黑色标签点在参考面附近豁免黑色压头过滤。

所有几何以毫米计算，PLY世界坐标以米输出。青色为候选观测，紫色为位移约束接触面，橙色为过渡/插值；估计不能冒充独立测量。全接触假设、初始压头高度/尺寸、曲线时间对齐均有误差。参考 `RIGID_CONTACT_COMPLETION_ZH.md`，其中293帧为原内角版本；最新外角版本291帧启用接触约束，最终以生成的manifest/validation为准。
