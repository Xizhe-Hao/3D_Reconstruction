# 本次数据集与 24 GB DA3 配置

数据来源：https://drive.google.com/drive/folders/1q5oXGcpgKqtptH71nZ8LlqGXtP9GjYZE

当前状态（2026-10-16 更新）：1775 个文件已下载完成，1742 张 D455 图像解码通过。仓库更新到 3668b20，使用该版本的 10 月 14 日标定会话结果。DA3 环境与权重已配置，已实测 1008/1176/1344/1512 分辨率，采用 1512×1134 正式重建，网页运行于 8080。实际流程和命令见 [RGBD_WEB_RUN_ZH.md](RGBD_WEB_RUN_ZH.md)。以下保留初始计划供参考。

## 恢复环境后执行

1. 在可联网环境安装 gdown，使用 `gdown --folder 'https://drive.google.com/drive/folders/1q5oXGcpgKqtptH71nZ8LlqGXtP9GjYZE' -O data/drive_test` 下载。检查文件清单、四路视频、同步 CSV、D455 深度与本次采集的标定；不能仅凭相同设备序列号认定旧外参适用。
2. 初始化仓库子模块，按 reconstruction/README.md 配置模型环境并下载权重。
3. DA3 四视角同时推理，首先以 `--width 1008 --height 756 --da3-process-res 1008` 试跑代表性时间点。这是假定源图为 4:3 的起始测试配置，不是已经验证的 24 GB 上限。实际尺寸以数据为准。
4. 每个候选配置单独启动 worker，测试 1008、1176、1344（显存不足时回退到 840）。保持四路输入与标定不变，记录耗时、输出尺寸、CUDA allocated/reserved 峰值和设备总占用；留出约 2–3 GB 余量，选最高稳定配置。现有 worker 的 manifest 仅记录 allocated 峰值，不能单凭该值判定余量。
5. 用选定配置处理全序列。高分辨率深度重建与 MVTracker 分开安排，不能推断 DA3 能运行的分辨率也适用于跟踪。分块解码以控制主存，避免把四路全视频高分辨率一次性读入内存。
6. 将四视角 DA3 点云和同步 D455 点云变换到相同米制世界坐标；应用对应标定的深度校正、有效范围过滤和体素融合。分别导出 RGB4、D455、RGB4+D455 PLY 供比较。现有 `ply_dense` 仅导出 RGB 重建，不应标为五视角融合结果。

## 离线交互查看器

`web/ply_viewer.html` 不需要第三方库或 CDN，可以直接用浏览器打开，也可从 reconstruction 目录运行：

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

访问 http://127.0.0.1:8080/ply_viewer.html，选择 PLY 文件。支持旋转、平移、缩放、点大小调节；ASCII 和 little-endian 二进制顶点及 RGB。页面不包含模拟结果。远程机器需要将 8080 端口转发到本机。
