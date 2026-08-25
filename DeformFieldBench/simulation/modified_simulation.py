import sys

# 须在 import warp / 大量依赖之前判断，以便 WARP_LOG_LEVEL 等对启动日志生效
_QUIET_EARLY = "--quiet" in sys.argv
if _QUIET_EARLY:
    import os as _os_q

    _os_q.environ.setdefault("WARP_LOG_LEVEL", "error")
    _os_q.environ.setdefault("TI_LOG_LEVEL", "error")

sys.path.append("gaussian-splatting")
import argparse
import contextlib
import ctypes
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import os
import numpy as np
import json
import subprocess
import glob
import shutil
import re
from tqdm import tqdm

def _ensure_maca_runtime_libs() -> None:
    maca_root = os.environ.get("MACA_PATH", "/opt/maca-3.3.0")
    candidate_dirs = [
        os.path.join(maca_root, "tools", "cu-bridge", "lib"),
        os.path.join(maca_root, "lib"),
        os.path.join(maca_root, "ompi", "lib"),
        os.path.join(maca_root, "ucx", "lib"),
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    for d in reversed(candidate_dirs):
        if os.path.exists(d) and d not in parts:
            parts.insert(0, d)
    if parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    preload_libs = [
        os.path.join(maca_root, "ompi", "lib", "libopen-pal.so.40"),
        os.path.join(maca_root, "ompi", "lib", "libopen-rte.so.40"),
        os.path.join(maca_root, "ompi", "lib", "libmpi.so.40"),
        os.path.join(maca_root, "ucx", "lib", "libucs.so.0"),
        os.path.join(maca_root, "ucx", "lib", "libuct.so.0"),
        os.path.join(maca_root, "ucx", "lib", "libucp.so.0"),
    ]
    for lib in preload_libs:
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_ensure_maca_runtime_libs()
import torch

# MetaX/cu-bridge 兼容：确保 Warp 可通过 libcuda.so 兼容链接找到运行库。
def _ensure_maca_cuda_compat_libs() -> None:
    candidate_dirs = [
        "/opt/maca-3.3.0/tools/cu-bridge/lib",
        "/opt/maca-3.3.0/lib",
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    for d in candidate_dirs:
        if os.path.exists(d) and d not in parts:
            parts.insert(0, d)
    if parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)


_ensure_maca_cuda_compat_libs()

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov 

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from my_utils.sim_utils import (
    build_press_boundary_conditions,
    build_drop_boundary_conditions,
    build_shear_boundary_conditions,
    build_stretch_boundary_conditions,
    build_bend_boundary_conditions,
    save_boundary_condition_info,
    save_external_force_info,
    save_initial_force_mask_and_arrow_info,
    setup_field_output_dirs,
    save_fields_for_frame,
    write_stress_pcd_camera_meta_json,
)
from my_utils.view_auxiliary_output import (
    FlowGaussianSplatSharedState,
    TrajectoryGaussianAccumSharedState,
    compute_render_frame_indices,
    count_render_samples_for_sim_rate,
    frame_to_output_index,
    project_world_points_to_screen_means2d,
    splat_stress_heatmap_bgr,
    stress_gaussian_precomp_colors_and_opacity,
    stress_scalars_aligned_to_render_chain,
    write_run_parameters_json,
    write_subsampled_world_tracks_ortho_pngs,
)
from my_utils.filling_cache import (
    build_filling_fingerprint,
    cache_dir_for_checkpoint_model,
    cache_dir_for_ply_model,
    save_filled_positions,
    try_load_filled_positions,
)
from my_utils.arch4_lmdb import write_sample_arch4_lmdb
from my_utils.sample_pack import write_sample_pack
from my_utils.pack_tensors import (
    compress_multiview_render_png_dirs,
    compute_export_resolution,
    downscale_multiview_render_png_dirs,
    pack_sample_arch4_tensors,
)

def _select_best_gpu() -> None:
    """
    自动选择最空闲的 GPU，并通过 CUDA_VISIBLE_DEVICES 屏蔽其它显卡。
    若已通过环境变量指定 CUDA_VISIBLE_DEVICES（如多卡 runner 分配），则不再覆盖。

    策略：
    1. 先打印每块卡的 free_mem 与 util；
    2. 优先在 GPU 利用率 < 10% 的卡中选择 free_mem 最大的；
    3. 如果没有满足利用率条件的卡，则在所有卡中选 free_mem 最大的。
    """
    _gpu_quiet = "--quiet" in sys.argv
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "":
        # 已由外部指定（如 auto_simulation_runner 多卡分配），不再自动选择
        return
    def _query_gpus_from_nvidia_smi():
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        out = []
        for line in result.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            out.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return out, "nvidia-smi"

    def _query_gpus_from_mx_smi():
        # mx-smi 不支持 nvidia-smi 的 query-gpu 参数，改为解析文本输出：
        # GPU#<id> ... Utilization(GPU %) ... Memory(vram total/used, KB)
        result = subprocess.check_output(
            ["mx-smi", "--show-memory", "--show-usage"],
            text=True,
        )
        blocks = re.split(r"\nGPU#(\d+)\s+", result)
        out = []
        for i in range(1, len(blocks), 2):
            gpu_id = int(blocks[i])
            block = blocks[i + 1]
            m_total = re.search(r"vram total\s*:\s*(\d+)\s*KB", block)
            m_used = re.search(r"vram used\s*:\s*(\d+)\s*KB", block)
            m_util = re.search(r"GPU\s*:\s*(\d+)\s*%", block)
            if not (m_total and m_used and m_util):
                continue
            total_kb = int(m_total.group(1))
            used_kb = int(m_used.group(1))
            util = int(m_util.group(1))
            mem_free_mib = max(total_kb - used_kb, 0) // 1024
            out.append((gpu_id, mem_free_mib, util))
        return out, "mx-smi"

    try:
        # 优先适配 MetaX 机器；若没有 mx-smi，再走 nvidia-smi
        candidates = []
        source = None
        try:
            candidates, source = _query_gpus_from_mx_smi()
        except Exception:
            candidates, source = _query_gpus_from_nvidia_smi()

        if not _gpu_quiet:
            print(f"[modified_simulation] GPU status from {source}:")
        for gpu_id, mem_free, util in candidates:
            if not _gpu_quiet:
                print(f"  - GPU {gpu_id}: free_mem={mem_free} MiB, util={util}%")

        if not candidates:
            if not _gpu_quiet:
                print(
                    "[modified_simulation] no GPU info returned, using default CUDA device."
                )
            return

        # 首选利用率较低的 GPU
        low_util = [c for c in candidates if c[2] < 10]
        target_pool = low_util if low_util else candidates

        best_gpu, best_mem, best_util = max(target_pool, key=lambda x: x[1])

        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        if not _gpu_quiet:
            print(
                "[modified_simulation] Auto-selected GPU id "
                f"{best_gpu} with free_mem={best_mem} MiB, util={best_util}%."
            )
    except Exception as e:
        print(f"[modified_simulation] GPU auto-selection failed ({e}), using default CUDA device.")


# 在 Warp / Taichi 初始化之前自动选择显卡
_select_best_gpu()

def _init_warp_taichi_backends() -> None:
    wp.init()
    wp.config.verify_cuda = True
    try:
        ti.init(arch=ti.cuda, device_memory_GB=8.0)
    except Exception as e:
        msg = str(e)
        # MetaX/cu-bridge 常见：cuMemGetInfo_v2 在兼容层返回 NOT_SUPPORTED
        if ("CUDA_ERROR_NOT_SUPPORTED" in msg) or ("cuMemGetInfo_v2" in msg):
            print(
                "[modified_simulation] Taichi CUDA backend unsupported on current driver "
                "(cuMemGetInfo_v2). Fallback to ti.cpu."
            )
            ti.init(arch=ti.cpu)
        else:
            raise


if _QUIET_EARLY:
    # 屏蔽 [Taichi]/Warp 启动横幅、设备列表、部分 JIT 提示（多卡 runner 子进程）
    with open(os.devnull, "w") as _dn:
        with contextlib.redirect_stdout(_dn), contextlib.redirect_stderr(_dn):
            _init_warp_taichi_backends()
else:
    _init_warp_taichi_backends()

@wp.kernel
def _warp_cuda_probe_kernel(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = x[tid] + 0.0


def _select_warp_device() -> str:
    """尝试在 CUDA 上做一次最小 kernel launch；失败则降级 CPU。"""
    try:
        buf = wp.zeros(shape=1, dtype=float, device="cuda:0")
        wp.launch(kernel=_warp_cuda_probe_kernel, dim=1, inputs=[buf], device="cuda:0")
        return "cuda:0"
    except Exception as e:
        print(f"[modified_simulation] Warp CUDA probe failed ({e}); fallback to warp cpu.")
        return "cpu"


_WARP_DEVICE = _select_warp_device()


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):

    # ----------------------------
    # Find checkpoint
    # ----------------------------
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)

    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # ----------------------------
    # 检测是否存在 f_rest
    # ----------------------------
    with open(checkpt_path, "rb") as f:
        header = ""
        while True:
            line = f.readline().decode("utf-8")
            header += line
            if line.strip() == "end_header":
                break

    if "f_rest_" not in header:
        if "--quiet" not in sys.argv:
            print("Detected PhysGaussian format (no SH rest terms).")
            print("Switching sh_degree -> 0")
        sh_degree = 0

    # ----------------------------
    # 原始加载流程（不改）
    # ----------------------------
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)

    return gaussians


def load_gaussians_from_ply(ply_path, sh_degree=3):
    """从单个 PLY 文件加载 Gaussian 模型（用于 ply_dir 批量模式）。"""
    with open(ply_path, "rb") as f:
        header = ""
        while True:
            line = f.readline().decode("utf-8")
            header += line
            if line.strip() == "end_header":
                break
    if "f_rest_" not in header:
        sh_degree = 0
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="单场景目录（含 point_cloud/iteration_*/point_cloud.ply）。与 --ply_dir 二选一。",
    )
    parser.add_argument(
        "--ply_dir",
        type=str,
        default=None,
        help="PLY 文件夹：目录内所有 .ply 依次读取，对每个做自动边界压缩实验；输出到 output_path/各文件名/",
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        default=None,
        help="单个 PLY 文件路径。与 --ply_dir/--model_path 互斥；用于自动化脚本逐个组合调用。",
    )
    parser.add_argument(
        "--ply_flat_output",
        action="store_true",
        help="单任务 --ply_path 时，直接把仿真输出写到 --output_path（不再在其下再建一层 ply 文件名子目录）。"
        "供 auto_simulation_runner 等使用。",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_h5", action="store_true", help="保存仿真粒子为 h5（不保存 simulation ply）")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument(
        "--delete_png_sequences_after_compile_video",
        action="store_true",
        help="须与 --compile_video 同开：某个 PNG 目录被 ffmpeg 成功合成 mp4 后，删除该目录下序列帧"
        "（images/、stress_gaussian/、tracks_gaussian/、flow_gaussian/ 下对应视角子目录；"
        "以及 tracks_subsampled_world/ortho_frames/<轴>）。默认保留 PNG。",
    )
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志：屏蔽 Warp/Taichi 启动横幅、关闭仿真帧 tqdm，并静默 BC/ffmpeg 成功等；错误仍输出。",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no_filling_cache",
        action="store_true",
        help="禁用粒子填充磁盘缓存，强制每次执行 fill_particles（调试或改 PLY 未改 mtime 时可用）。",
    )
    #export deformation
    parser.add_argument("--output_deformation", action="store_true")
    parser.add_argument("--output_stress", action="store_true")
    parser.add_argument("--output_bc_info", action="store_true")
    parser.add_argument("--output_force_info", action="store_true")
    parser.add_argument(
        "--output_initial_force_mask_arrow",
        action="store_true",
        help="在 meta/ 额外导出初始受力区域 mask + 方向箭头（基于速度驱动 cuboid）。",
    )
    _cmp = parser.add_mutually_exclusive_group()
    _cmp.add_argument(
        "--compress_render_pngs",
        action="store_true",
        dest="compress_render_pngs",
        help="渲染序列落盘后、打包 .pt / ffmpeg 前，对 RGB·stress·flow·force_mask 各视角 PNG 做 zlib 无损压缩（默认开启）。",
    )
    _cmp.add_argument(
        "--no_compress_render_pngs",
        action="store_false",
        dest="compress_render_pngs",
        help="关闭 PNG zlib 重写（磁盘占用更大、后续读写略快）。",
    )
    parser.set_defaults(compress_render_pngs=True)
    parser.add_argument(
        "--png_compression_level",
        type=int,
        default=6,
        help="PNG zlib 级别 0-9（默认 6）；仅影响文件大小与编解码耗时，不改变像素。",
    )
    parser.add_argument(
        "--render_export_max_side",
        type=int,
        default=0,
        help=">0 时：将 images/stress_gaussian/flow_gaussian/force_mask 各视角 PNG 统一缩放到最长边≤该值"
        "（保持宽高比，边长为偶数）。0 表示不按最长边限制。",
    )
    parser.add_argument(
        "--render_export_scale",
        type=float,
        default=1.0,
        help="(0,1) 时对上述导出 PNG 整体缩放；1.0 关闭。与 --render_export_max_side>0 同时配置时优先最长边。",
    )
    parser.add_argument(
        "--camera_distance_scale",
        type=float,
        default=1.0,
        help="相机距离缩放（>0）：实际渲染半径=init_radius/scale。scale 越大，镜头越近、物体占画面越大。",
    )
    parser.add_argument(
        "--pack_arch4_tensors",
        action="store_true",
        help="将 images / stress_gaussian / flow_gaussian / force_mask 打成 arch4_tensors/<view>.pt（与 --pack_arch4_lmdb 二选一；默认与 lmdb 互斥见 runner）。",
    )
    parser.add_argument(
        "--pack_arch4_lmdb",
        action="store_true",
        help="将四类 PNG 按 --arch4_lmdb_resize 缩放后写入 LMDB（uint8，单库多键）；不占 float32 .pt；在 ffmpeg 合成视频之前执行，PNG 仍保留供视频。",
    )
    parser.add_argument(
        "--pack_sample_pack",
        action="store_true",
        help="将 images / stress_gaussian / flow_gaussian / force_mask 打成单文件 sample_pack.npz（uint8，多模态同文件）。",
    )
    parser.add_argument(
        "--sample_pack_name",
        type=str,
        default="sample_pack.npz",
        help="样本目录下 sample_pack 文件名（相对 output_path），默认 sample_pack.npz。",
    )
    parser.add_argument(
        "--pack_sample_pack_include_object_mask",
        action="store_true",
        help="写入 sample_pack 时额外包含 object_mask 数组（与 force_mask 分开存储）。",
    )
    parser.add_argument(
        "--arch4_lmdb_resize",
        type=int,
        default=224,
        help="写入 LMDB 前将每帧缩放到 正方形边长（像素），默认 224。",
    )
    parser.add_argument(
        "--arch4_lmdb_map_size_gb",
        type=float,
        default=8.0,
        help="LMDB map_size（GiB），须大于预估库体积；默认 8。",
    )
    parser.add_argument(
        "--arch4_lmdb_name",
        type=str,
        default="arch4_data.lmdb",
        help="样本目录下 LMDB 环境子目录名（相对 output_path），默认 arch4_data.lmdb。",
    )
    parser.add_argument(
        "--pack_arch4_lmdb_include_object_mask",
        action="store_true",
        help="写入 LMDB 时额外写 object_mask 键（与 force_mask 分开存储）。",
    )
    parser.add_argument(
        "--arch4_tensor_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="写入 arch4_tensors/*.pt 时的存储 dtype：float16/bfloat16 约减半磁盘；Dataset 读取时会转为 float32 再训练。",
    )
    parser.add_argument("--field_output_interval", type=int, default=1)
    parser.add_argument(
        "--sim_type",
        type=str,
        default="press",
        choices=["press", "drop", "shear", "stretch", "bend"],
        help="仿真类型，用于自动设置边界条件。目前支持 press/drop 自动推断。",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=4,
        help="渲染视角数量。仅在方位角上均匀分布（俯仰角用 config 的 init_elevation）。",
    )
    parser.add_argument(
        "--num_render_views",
        type=int,
        default=-1,
        help="覆盖渲染视角数；-1 表示使用 --num_views。用于在配置里单独声明「渲染视角数量」。",
    )
    parser.add_argument(
        "--random_render_views",
        action="store_true",
        help="启用随机视角：在 [0,360) 环绕范围随机挑选视角（带最小间隔约束）。",
    )
    parser.add_argument(
        "--random_render_views_min_gap_deg",
        type=float,
        default=20.0,
        help="随机视角最小方位角间隔（度）。",
    )
    parser.add_argument(
        "--random_render_views_seed",
        type=int,
        default=0,
        help="随机视角采样随机种子。",
    )
    parser.add_argument(
        "--num_render_timesteps",
        type=int,
        default=0,
        help="当未使用 --render_outputs_per_sim_second（≤0）时生效：在整个仿真上均匀输出 K 帧"
        "（0 或 ≥仿真帧数表示每仿真步都输出）。与后者同开时以 N 为准。",
    )
    parser.add_argument(
        "--render_outputs_per_sim_second",
        type=float,
        default=0.0,
        help="统一采样率 N（每秒仿真时间输出多少张）：总张数 K≈round(仿真总秒数×N)，"
        "在 [0,frame_num-1] 上均匀取 K 个仿真步做 render/stress/flow 等输出；"
        "compile_video 时 ffmpeg -framerate 也用 N，使视频时长≈仿真总时长。≤0 时改用 --num_render_timesteps。",
    )
    parser.add_argument(
        "--output_view_stress_heatmap",
        action="store_true",
        help="每视角 2D 应力热力图（屏幕空间 splat，非 3DGS）。",
    )
    parser.add_argument(
        "--output_view_stress_gaussian",
        action="store_true",
        help="每视角额外一次 3DGS 光栅化：按 von Mises 调制预计算颜色与不透明度（不改变 MPM/高斯参数，仅多写 PNG）。",
    )
    parser.add_argument(
        "--stress_gaussian_opa_floor",
        type=float,
        default=0.12,
        help="应力 3DGS：低应力端相对原 opacity 缩放下限（与离线 render_stress_gaussian 一致）。",
    )
    parser.add_argument(
        "--stress_gaussian_opa_ceil",
        type=float,
        default=1.0,
        help="应力 3DGS：高应力端相对原 opacity 缩放上限。",
    )
    parser.add_argument(
        "--stress_gaussian_sh_blend",
        type=float,
        default=0.35,
        help="应力 3DGS：SH 底色权重，应力伪彩色权重为 1-该值。",
    )
    parser.add_argument(
        "--stress_gaussian_colormap_steps",
        type=int,
        default=24,
        help="应力 3DGS：von Mises 绝对值线性映射到 JET 阶梯的档数（≥2）。",
    )
    parser.add_argument(
        "--stress_gaussian_vm_pct_low",
        type=float,
        default=1.0,
        help="应力 3DGS：每帧 von Mises 绝对值色标下分位%%（与 vm_pct_high 一起框定，抗离群）。",
    )
    parser.add_argument(
        "--stress_gaussian_vm_pct_high",
        type=float,
        default=99.0,
        help="应力 3DGS：每帧 von Mises 绝对值色标上分位%%；全 min/max 可设 0 与 100。",
    )
    parser.add_argument(
        "--stress_gaussian_single_channel",
        action="store_true",
        help="应力 3DGS：von Mises 线性归一化灰度着色（不与 SH 底色混合）、黑底合成，保存单通道 PNG（uint8 [H,W]）。",
    )
    parser.add_argument(
        "--output_view_flow_gaussian",
        action="store_true",
        help="与主光栅同相机：第二遍 CUDA 3DGS，colors_precomp 为屏幕 (Δu,Δv) 的 Middlebury 伪彩，"
        "opacity 含原不透明度×幅值×距离权重；黑底输出 flow_gaussian/<view>/；不写 npz。",
    )
    parser.add_argument(
        "--output_view_force_mask",
        action="store_true",
        help="每视角输出驱动区域 mask（黑底，仅 mask 渲染）。不影响 images 的 RGB 渲染。",
    )
    parser.add_argument(
        "--force_mask_single_channel",
        action="store_true",
        help="force_mask：仅渲染受力粒子为白、其余高斯不透明度为 0、黑底，保存为单通道 PNG（0/255 风格 mask）。"
        " 默认关闭时为「全场景 + 受力区染红」的 RGB。",
    )
    parser.add_argument(
        "--output_view_object_mask",
        action="store_true",
        help="每视角输出物体 mask（白物体、黑背景）：通过 3DGS 光栅化，colors_precomp 固定白色。",
    )
    parser.add_argument(
        "--flow_gaussian_max_gaussians",
        type=int,
        default=8192,
        help="参与 splat 的高斯数上限（仅从前 gs_num 个 MPM 高斯中随机下采样）。",
    )
    parser.add_argument(
        "--flow_gaussian_seed",
        type=int,
        default=0,
        help="flow 下采样随机种子（各视角共用同一批下标）。",
    )
    parser.add_argument(
        "--flow_gaussian_depth_gamma",
        type=float,
        default=1.0,
        help="距离权重：(1/(dist+eps))^gamma，越大越强调近处 splat。",
    )
    parser.add_argument(
        "--flow_gaussian_depth_eps",
        type=float,
        default=1e-2,
        help="距离权重的 ε，避免除零（与场景尺度同量级时可调）。",
    )
    parser.add_argument(
        "--flow_gaussian_opacity_power",
        type=float,
        default=1.0,
        help="权重中不透明度项：opacity^power。",
    )
    parser.add_argument(
        "--flow_gaussian_vis_max_motion",
        type=float,
        default=0.0,
        help="伪彩饱和位移（像素）；≤0 则每帧按有效像素分位数自动估计。",
    )
    parser.add_argument(
        "--output_view_tracks_gaussian",
        action="store_true",
        help="每视角 3DGS 微型高斯轨迹（黑底）：整物体 MPM 下采样，每输出帧光栅当前（+可选中点）并叠到轨迹图；多视角共用 splat。",
    )
    parser.add_argument(
        "--tracks_gaussian_accum_mode",
        type=str,
        default="max",
        choices=("max", "add"),
        help="轨迹缓冲：max=逐像素取大（线条清晰、后期不糊）；add=逐帧相加（易叠成雾）。",
    )
    parser.add_argument(
        "--tracks_gaussian_accum_frame_weight",
        type=float,
        default=1.0,
        help="轨迹：max 模式下为本帧 splat 峰值缩放；add 模式下为加到累加图的系数。",
    )
    parser.add_argument(
        "--tracks_gaussian_accum_decay",
        type=float,
        default=1.0,
        help="轨迹：每帧先 轨迹图 *= decay 再叠加（1=全程保留；<1 旧迹变淡）。",
    )
    parser.add_argument(
        "--tracks_gaussian_accum_midpoint",
        action="store_true",
        help="轨迹：在上一输出时刻与当前时刻位置中点再 splat（约 2× 点数、折线更连贯）。",
    )
    parser.add_argument(
        "--tracks_gaussian_accum_no_normalize_save",
        action="store_true",
        help="轨迹：写 PNG 时不按 max 归一化，仅用 tracks_gaussian_intensity 线性缩放再 clamp。",
    )
    parser.add_argument(
        "--tracks_gaussian_max_tracks",
        type=int,
        default=2048,
        help="轨迹：整物体 MPM 粒子下采样数量上限。",
    )
    parser.add_argument(
        "--tracks_gaussian_sigma_scale",
        type=float,
        default=0.0012,
        help="轨迹微型高斯各向同性 σ ≈ scale_origin * 该系数（世界系）；越小线越细越利落。",
    )
    parser.add_argument(
        "--tracks_gaussian_point_opacity",
        type=float,
        default=1.0,
        help="轨迹每个 splat 的不透明度 ∈(0,1]；配合较小 sigma 建议 1.0。",
    )
    parser.add_argument(
        "--tracks_gaussian_intensity",
        type=float,
        default=1.0,
        help="轨迹：与 --tracks_gaussian_accum_no_normalize_save 联用时的线性缩放；默认写图前按 acc.max() 归一化时可忽略。",
    )
    parser.add_argument(
        "--tracks_gaussian_seed",
        type=int,
        default=0,
        help="轨迹下采样选粒子的随机种子（各视角共用同一批粒子下标）。",
    )
    parser.add_argument(
        "--output_subsampled_world_tracks",
        action="store_true",
        help="不下光栅化：对仿真粒子下采样，按输出时刻记录世界坐标轨迹 tracks_subsampled_world/tracks_world.npz（与 num_render_timesteps 对齐）。",
    )
    parser.add_argument(
        "--subsampled_tracks_num",
        type=int,
        default=1024,
        help="下采样追踪的粒子数（≤ MPM 粒子数）。",
    )
    parser.add_argument(
        "--subsampled_tracks_seed",
        type=int,
        default=0,
        help="下采样选粒子随机种子。",
    )
    parser.add_argument(
        "--subsampled_tracks_ortho_axes",
        type=str,
        default="xz",
        help="compile_video 时正交轨迹预览 mp4 使用的世界系两维：xy / xz / yz。",
    )
    parser.add_argument(
        "--subsampled_tracks_video_size",
        type=int,
        default=512,
        help="正交轨迹预览视频的边长（正方形，像素）。",
    )
    parser.add_argument(
        "--no_volumetric_stress_deformation",
        action="store_true",
        help="关闭 deformation_field/ 与 stress_field/ 下按帧的大体积 npz（仍可做渲染与视角辅助）。",
    )
    #export deformation
    args = parser.parse_args()

    _vprint = print if not bool(getattr(args, "quiet", False)) else (lambda *a, **k: None)

    want_flow_gaussian = bool(args.output_view_flow_gaussian)
    want_force_mask = bool(args.output_view_force_mask)
    want_object_mask = bool(args.output_view_object_mask)
    want_tracks_gaussian = bool(args.output_view_tracks_gaussian)
    want_subsampled_world_tracks = bool(args.output_subsampled_world_tracks)
    want_stress_heatmap = bool(args.output_view_stress_heatmap)
    want_stress_gaussian = bool(args.output_view_stress_gaussian)
    use_ply_dir = args.ply_dir is not None and os.path.isdir(args.ply_dir)
    use_ply_path = args.ply_path is not None and os.path.isfile(args.ply_path)

    # 默认输出目录：按 sim_type 自动分流
    if args.output_path is None:
        if args.sim_type == "press":
            args.output_path = os.path.join("output", "press_auto_test")
        elif args.sim_type == "drop":
            args.output_path = os.path.join("output", "drop_auto_test")
        else:
            args.output_path = os.path.join("output", f"{args.sim_type}_auto_test")

    if use_ply_path:
        if not os.path.exists(args.config):
            raise AssertionError("Scene config does not exist!")
        os.makedirs(args.output_path, exist_ok=True)
        name = os.path.splitext(os.path.basename(args.ply_path))[0]
        if args.ply_flat_output:
            current_out = args.output_path
        else:
            current_out = os.path.join(args.output_path, name)
        tasks = [("ply", args.ply_path, current_out, os.path.dirname(args.ply_path))]
    elif use_ply_dir:
        if not os.path.exists(args.config):
            raise AssertionError("Scene config does not exist!")
        os.makedirs(args.output_path, exist_ok=True)
        ply_files = sorted(glob.glob(os.path.join(args.ply_dir, "*.ply")))
        if not ply_files:
            raise FileNotFoundError(f"在 --ply_dir 中未找到任何 .ply 文件: {args.ply_dir}")
        tasks = []
        for ply_path in ply_files:
            name = os.path.splitext(os.path.basename(ply_path))[0]
            tasks.append(("ply", ply_path, os.path.join(args.output_path, name), args.ply_dir))
    else:
        if args.model_path is None or not os.path.exists(args.model_path):
            raise AssertionError("请指定 --model_path 或有效的 --ply_dir/--ply_path。Model path does not exist!")
        if not os.path.exists(args.config):
            raise AssertionError("Scene config does not exist!")
        if args.output_path is not None and not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        tasks = [("checkpoint", args.model_path, args.output_path, args.model_path)]

    # load scene config（所有任务共用）
    _vprint("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    for task_idx, (task_type, path_or_ply, current_output_path, model_path_for_camera) in enumerate(tasks):
        if task_type == "ply":
            ply_path = path_or_ply
            _vprint(f"\n[{task_idx+1}/{len(tasks)}] 加载 PLY: {ply_path}")
            gaussians = load_gaussians_from_ply(ply_path)
            model_path = model_path_for_camera
        else:
            _vprint(f"\n[{task_idx+1}/{len(tasks)}] 加载 checkpoint: {path_or_ply}")
            model_path = path_or_ply
            gaussians = load_checkpoint(model_path)
        args.output_path = current_output_path
        if args.output_path is not None:
            os.makedirs(args.output_path, exist_ok=True)
        pipeline = PipelineParamsNoparse()
        pipeline.compute_cov3D_python = True
        background = (
            torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            if args.white_bg
            else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        )

        # init the scene
        _vprint("Initializing scene and pre-processing...")
        params = load_params_from_gs(gaussians, pipeline)

        init_pos = params["pos"]
        init_cov = params["cov3D_precomp"]
        init_screen_points = params["screen_points"]
        init_opacity = params["opacity"]
        init_shs = params["shs"]

        # throw away low opacity kernels
        mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
        init_pos = init_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_screen_points = init_screen_points[mask, :]
        init_shs = init_shs[mask, :]

        # 对 PLY 输入先绕 x 轴旋转 90 度，再走后续边界判断与仿真
        if task_type == "ply":
            device = init_pos.device
            rot_x90 = generate_rotation_matrices(
                torch.tensor([90.0], device=device, dtype=init_pos.dtype),
                [0],
            )  # 0 = x 轴
            init_pos = apply_rotations(init_pos, rot_x90)
            init_cov = apply_cov_rotations(init_cov, rot_x90)

        # rorate and translate object
        if args.debug:
            if not os.path.exists("./log"):
                os.makedirs("./log")
            particle_position_tensor_to_ply(
                init_pos,
                "./log/init_particles.ply",
            )
        rotation_matrices = generate_rotation_matrices(
            torch.tensor(preprocessing_params["rotation_degree"]),
            preprocessing_params["rotation_axis"],
        )
        rotated_pos = apply_rotations(init_pos, rotation_matrices)

        if args.debug:
            particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

        # select a sim area and save params of unslected particles
        unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
            None,
            None,
            None,
            None,
        )
        if preprocessing_params["sim_area"] is not None:
            boundary = preprocessing_params["sim_area"]
            assert len(boundary) == 6
            mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
            for i in range(3):
                mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
                mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

            unselected_pos = init_pos[~mask, :]
            unselected_cov = init_cov[~mask, :]
            unselected_opacity = init_opacity[~mask, :]
            unselected_shs = init_shs[~mask, :]

            rotated_pos = rotated_pos[mask, :]
            init_cov = init_cov[mask, :]
            init_opacity = init_opacity[mask, :]
            init_shs = init_shs[mask, :]

        robust_norm_enable = bool(preprocessing_params.get("robust_norm_enable", True))
        robust_norm_q_low = float(preprocessing_params.get("robust_norm_q_low", 0.01))
        robust_norm_q_high = float(preprocessing_params.get("robust_norm_q_high", 0.99))
        if robust_norm_enable:
            transformed_pos, scale_origin, original_mean_pos = transform2origin_robust(
                rotated_pos,
                preprocessing_params["scale"],
                q_low=robust_norm_q_low,
                q_high=robust_norm_q_high,
            )
        else:
            transformed_pos, scale_origin, original_mean_pos = transform2origin(
                rotated_pos, preprocessing_params["scale"]
            )
        transformed_pos = shift2center111(transformed_pos)

        # 保存归一化后的点云，便于排查离群点导致的构图偏移
        try:
            normalize_root = os.path.join(model_path, "normalize")
            os.makedirs(normalize_root, exist_ok=True)
            if task_type == "ply":
                src_name = os.path.splitext(os.path.basename(ply_path))[0]
            else:
                src_name = os.path.basename(os.path.abspath(model_path))
            normalized_ply = os.path.join(normalize_root, f"{src_name}_normalized.ply")
            particle_position_tensor_to_ply(transformed_pos, normalized_ply)
        except Exception as _e:
            _vprint(f"[normalize] 保存 normalized ply 失败: {_e}")

        # modify covariance matrix accordingly
        init_cov = apply_cov_rotations(init_cov, rotation_matrices)
        init_cov = scale_origin * scale_origin * init_cov

        if args.debug:
            particle_position_tensor_to_ply(
                transformed_pos,
                "./log/transformed_particles.ply",
            )

        # fill particles if needed
        gs_num = transformed_pos.shape[0]
        device = "cuda:0"
        warp_device = _WARP_DEVICE
        filling_params = preprocessing_params["particle_filling"]

        if filling_params is not None:
            use_filling_cache = not bool(args.no_filling_cache)
            cache_dir = None
            ply_for_fp = None
            ply_extra_x90 = False
            fingerprint = None
            mpm_init_pos = None

            if use_filling_cache:
                if task_type == "ply":
                    ply_for_fp = os.path.abspath(ply_path)
                    ply_extra_x90 = True
                    stem = os.path.splitext(os.path.basename(ply_path))[0]
                    cache_dir = cache_dir_for_ply_model(
                        os.path.dirname(ply_for_fp), stem
                    )
                else:
                    _ck_dir = os.path.join(os.path.abspath(model_path), "point_cloud")
                    _ck_iter = searchForMaxIteration(_ck_dir)
                    ply_for_fp = os.path.join(
                        _ck_dir, f"iteration_{_ck_iter}", "point_cloud.ply"
                    )
                    ply_extra_x90 = False
                    cache_dir = cache_dir_for_checkpoint_model(model_path, _ck_iter)

                if os.path.isfile(ply_for_fp):
                    fingerprint = build_filling_fingerprint(
                        ply_path=ply_for_fp,
                        preprocessing_params=preprocessing_params,
                        material_params=material_params,
                        filling_params=filling_params,
                        ply_extra_x90=ply_extra_x90,
                    )
                    mpm_init_pos = try_load_filled_positions(
                        cache_dir, fingerprint, device, gs_num
                    )

            if mpm_init_pos is not None:
                _vprint(
                    f"[particle filling] 使用缓存: {cache_dir} "
                    f"(fingerprint={fingerprint[:16]}...)"
                )
            else:
                _vprint("Filling internal particles...")
                mpm_init_pos = fill_particles(
                    pos=transformed_pos,
                    opacity=init_opacity,
                    cov=init_cov,
                    grid_n=filling_params["n_grid"],
                    max_samples=filling_params["max_particles_num"],
                    grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
                    density_thres=filling_params["density_threshold"],
                    search_thres=filling_params["search_threshold"],
                    max_particles_per_cell=filling_params["max_partciels_per_cell"],
                    search_exclude_dir=filling_params["search_exclude_direction"],
                    ray_cast_dir=filling_params["ray_cast_direction"],
                    boundary=filling_params["boundary"],
                    smooth=filling_params["smooth"],
                ).to(device=device)

                if (
                    use_filling_cache
                    and fingerprint
                    and cache_dir
                    and ply_for_fp
                    and os.path.isfile(ply_for_fp)
                ):
                    save_filled_positions(
                        cache_dir,
                        fingerprint,
                        mpm_init_pos,
                        gs_num,
                        meta_extra={
                            "ply_basename": os.path.basename(ply_for_fp),
                            "task_type": task_type,
                        },
                    )
                    _vprint(f"[particle filling] 已写入缓存: {cache_dir}")

            if args.debug:
                particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
        else:
            mpm_init_pos = transformed_pos.to(device=device)

        # init the mpm solver
        _vprint("Initializing MPM solver and setting up boundary conditions...")
        mpm_init_vol = get_particle_volume(
            mpm_init_pos,
            material_params["n_grid"],
            material_params["grid_lim"] / material_params["n_grid"],
            unifrom=material_params["material"] == "sand",
        ).to(device=device)

        if filling_params is not None and filling_params["visualize"] == True:
            shs, opacity, mpm_init_cov = init_filled_particles(
                mpm_init_pos[:gs_num],
                init_shs,
                init_cov,
                init_opacity,
                mpm_init_pos[gs_num:],
            )
            gs_num = mpm_init_pos.shape[0]
        else:
            mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
            mpm_init_cov[:gs_num] = init_cov
            shs = init_shs
            opacity = init_opacity

        if args.debug:
            print("check *.ply files to see if it's ready for simulation")

        # set up the mpm solver
        mpm_solver = MPM_Simulator_WARP(10, device=warp_device)
        _mpm_tensor_device = device if str(warp_device).startswith("cuda") else "cpu"
        mpm_solver.load_initial_data_from_torch(
            mpm_init_pos.to(device=_mpm_tensor_device),
            mpm_init_vol.to(device=_mpm_tensor_device),
            mpm_init_cov.to(device=_mpm_tensor_device),
            n_grid=material_params["n_grid"],
            grid_lim=material_params["grid_lim"],
            device=warp_device,
        )
        mpm_solver.set_parameters_dict(material_params, device=warp_device)

        # 基于仿真类型自动设置边界条件（当前支持 press/drop 自动推断）
        if args.sim_type == "press":
            _vprint("[modified_simulation] 自动根据粒子位置构造 press 边界条件...")
            bc_params = build_press_boundary_conditions(mpm_init_pos, material_params)
        elif args.sim_type == "drop":
            _vprint("[modified_simulation] 自动构造 drop 边界条件（参考 drop_cube_jelly.json）...")
            bc_params = build_drop_boundary_conditions(mpm_init_pos, material_params)
        elif args.sim_type == "shear":
            _vprint("[modified_simulation] 自动构造 shear 边界条件（上下 1/4 相反方向刚体平面）...")
            bc_params = build_shear_boundary_conditions(mpm_init_pos, material_params)
        elif args.sim_type == "stretch":
            _vprint("[modified_simulation] 自动构造 stretch 边界条件（y 两端夹具拉伸）...")
            bc_params = build_stretch_boundary_conditions(mpm_init_pos, material_params)
        elif args.sim_type == "bend":
            _vprint("[modified_simulation] 自动构造 bend 边界条件（沿最长轴方向底面固定+顶部刚性片推压）...")
            bc_params = build_bend_boundary_conditions(mpm_init_pos, material_params)

            # 同时输出一份带自动 BC 的 config 供记录
            try:
                with open(args.config, "r") as f:
                    sim_cfg = json.load(f)
                sim_cfg["boundary_conditions"] = bc_params
                auto_cfg_path = os.path.join(
                    args.output_path, f"auto_config_{args.sim_type}.json"
                )
                with open(auto_cfg_path, "w") as f:
                    json.dump(sim_cfg, f, indent=4)
                _vprint(f"[modified_simulation] 自动生成 config 已保存到: {auto_cfg_path}")
            except Exception as e:
                print(f"[modified_simulation] 写入自动 config 失败: {e}")
        else:
            _vprint(
                f"[modified_simulation] sim_type={args.sim_type}，暂未实现自动 BC，继续使用模板 config 中的 boundary_conditions。"
            )

        # Note: boundary conditions may depend on mass, so the order cannot be changed!
        set_boundary_conditions(mpm_solver, bc_params, time_params, device=warp_device)

        mpm_solver.finalize_mu_lam(device=warp_device)

        # 输出边界条件和力场信息（如果需要）
        force_overlay_info = None
        # 用于“velocity cuboid 前端截面触碰粒子”可视化（全程累计）
        velocity_cuboids: List[Dict[str, Any]] = []
        touched_mask_gs: Optional[torch.Tensor] = None
        touched_first_frame_gs: Optional[torch.Tensor] = None

        if args.output_path is not None:
            meta_dir = os.path.join(args.output_path, "meta")
            if args.output_bc_info:
                save_boundary_condition_info(bc_params, material_params, meta_dir)
            if args.output_force_info:
                save_external_force_info(bc_params, material_params, meta_dir)
            if args.output_initial_force_mask_arrow:
                force_overlay_info = save_initial_force_mask_and_arrow_info(
                    bc_params, mpm_init_pos, meta_dir
                )
                # 解析速度驱动 cuboid（压力杆）
                for bc in (bc_params or []):
                    bc_item = None
                    if str(bc.get("type", "")).strip().lower() == "cuboid":
                        bc_item = bc
                    elif "set_velocity_on_cuboid" in bc:
                        bc_item = bc["set_velocity_on_cuboid"]
                    elif "enforce_particle_translation" in bc:
                        bc_item = bc["enforce_particle_translation"]
                    if bc_item is None:
                        continue
                    p = np.asarray(bc_item.get("point", [0.0, 0.0, 0.0]), dtype=np.float32)
                    s = np.asarray(bc_item.get("size", [0.0, 0.0, 0.0]), dtype=np.float32)
                    v = np.asarray(bc_item.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
                    v_norm = float(np.linalg.norm(v))
                    if v_norm <= 1e-10:
                        continue
                    axis = int(np.argmax(np.abs(v)))
                    sign = 1.0 if float(v[axis]) >= 0.0 else -1.0
                    velocity_cuboids.append(
                        {
                            "point0": p,
                            "size": s,
                            "velocity": v,
                            "axis": axis,
                            "sign": sign,
                            "start_time": float(bc_item.get("start_time", 0.0)),
                            "end_time": float(bc_item.get("end_time", 1e9)),
                        }
                    )
                if velocity_cuboids:
                    touched_mask_gs = torch.zeros((int(gs_num),), device=device, dtype=torch.bool)
                    touched_first_frame_gs = torch.full(
                        (int(gs_num),), -1, device=device, dtype=torch.long
                    )
                _vprint(f"[force_debug] velocity_cuboids={len(velocity_cuboids)}")
        # camera setting
        mpm_space_viewpoint_center = (
            torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
        )
        mpm_space_vertical_upward_axis = (
            torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
            .reshape((1, 3))
            .cuda()
        )
        (
            viewpoint_center_worldspace,
            observant_coordinates,
        ) = get_center_view_worldspace_and_observant_coordinate(
            mpm_space_viewpoint_center,
            mpm_space_vertical_upward_axis,
            rotation_matrices,
            scale_origin,
            original_mean_pos,
        )

        # run the simulation
        if args.output_h5:
            directory_to_save = os.path.join(args.output_path, "simulation_ply")
            os.makedirs(directory_to_save, exist_ok=True)
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                0,
                save_to_ply=False,
                save_to_h5=True,
            )

        substep_dt = time_params["substep_dt"]
        frame_dt = time_params["frame_dt"]
        frame_num = time_params["frame_num"]

        # press/shear/bend：总时长固定 4 秒；其中前 3 秒施力（BC end_time=3），最后 1 秒撤出力
        if args.sim_type in ("press", "shear", "bend"):
            total_duration_s = 4.0
            frame_num = int(round(total_duration_s / frame_dt))
            frame_num = max(1, frame_num)
        step_per_frame = int(frame_dt / substep_dt)
        opacity_render = opacity
        shs_render = shs
        height = None
        width = None
        fast_track_indices_np: Optional[np.ndarray] = None
        fast_track_xyz_list: List[np.ndarray] = []
        fast_track_sim_frames: List[int] = []
        track_gaussian_accum_shared: Optional[TrajectoryGaussianAccumSharedState] = None
        flow_gaussian_shared: Optional[FlowGaussianSplatSharedState] = None
        save_vol = not bool(args.no_volumetric_stress_deformation)
        eff_output_deformation = bool(args.output_deformation) and save_vol
        eff_output_stress_vol = bool(args.output_stress) and save_vol

        # 准备形变场 / 应力场输出目录（大体积 npz）
        deformation_dir, stress_dir = setup_field_output_dirs(
            args.output_path,
            eff_output_deformation,
            eff_output_stress_vol,
        )

        _n_per_s = float(getattr(args, "render_outputs_per_sim_second", 0.0) or 0.0)
        if _n_per_s > 0.0:
            _K = count_render_samples_for_sim_rate(
                frame_num, float(frame_dt), _n_per_s
            )
            render_frame_indices = compute_render_frame_indices(frame_num, _K)
            video_compile_fps = float(_n_per_s)
        else:
            render_frame_indices = compute_render_frame_indices(
                frame_num, int(args.num_render_timesteps)
            )
            video_compile_fps = float(1.0 / float(frame_dt))
        frame_to_out_idx = frame_to_output_index(render_frame_indices)
        render_frame_set = set(render_frame_indices)
        if _n_per_s > 0.0:
            _vprint(
                f"[modified_simulation] render_outputs_per_sim_second={_n_per_s}: "
                f"仿真约 {float(frame_num) * float(frame_dt):.4g}s，"
                f"均匀输出 {len(render_frame_indices)} 帧，"
                f"compile 视频 -framerate={video_compile_fps}"
            )

        # 输出初始帧场数据
        if eff_output_deformation or eff_output_stress_vol:
            if eff_output_stress_vol:
                mpm_solver.recompute_particle_stress_from_F_trial(
                    float(substep_dt), device=warp_device
                )
            save_fields_for_frame(
                mpm_solver,
                frame_id=0,
                deformation_dir=deformation_dir,
                stress_dir=stress_dir,
                output_deformation=eff_output_deformation,
                output_stress=eff_output_stress_vol,
            )
            # 离线点云应力多视角：与 Gaussian synthetic 轨道相机（方位角环绕）一致
            if args.output_path is not None:
                meta_dir = os.path.join(args.output_path, "meta")
                os.makedirs(meta_dir, exist_ok=True)
                nv_meta = max(
                    1,
                    int(args.num_render_views)
                    if int(args.num_render_views) >= 0
                    else int(args.num_views),
                )
                _mtw = {
                    "rotation_matrices": [
                        R.detach().cpu().numpy().tolist() for R in rotation_matrices
                    ],
                    "scale_origin": float(scale_origin.detach().cpu().item()),
                    "original_mean_pos": original_mean_pos.detach()
                    .cpu()
                    .numpy()
                    .reshape(3)
                    .tolist(),
                }
                write_stress_pcd_camera_meta_json(
                    os.path.join(meta_dir, "stress_pcd_cameras.json"),
                    viewpoint_center_worldspace=np.asarray(
                        viewpoint_center_worldspace, dtype=np.float64
                    ),
                    observant_coordinates=np.asarray(
                        observant_coordinates, dtype=np.float64
                    ),
                    num_views=nv_meta,
                    init_azimuthm=float(camera_params["init_azimuthm"]),
                    init_elevation=float(camera_params["init_elevation"]),
                    init_radius=float(camera_params["init_radius"])
                    / max(1e-6, float(getattr(args, "camera_distance_scale", 1.0))),
                    model_path=model_path,
                    default_camera_index=int(
                        camera_params.get("default_camera_index", -1)
                    ),
                    move_camera=bool(camera_params.get("move_camera", False)),
                    delta_a=float(camera_params.get("delta_a", 0.0)),
                    delta_e=float(camera_params.get("delta_e", 0.0)),
                    delta_r=float(camera_params.get("delta_r", 0.0)),
                    field_output_interval=int(args.field_output_interval),
                    mpm_to_world=_mtw,
                    mpm_space_viewpoint_center=list(
                        camera_params["mpm_space_viewpoint_center"]
                    ),
                )

        # 若需要渲染，准备多视角的图像/视频输出目录
        image_root = None
        video_root = None
        if args.output_path is not None:
            if args.render_img:
                image_root = os.path.join(args.output_path, "images")
                os.makedirs(image_root, exist_ok=True)
            if args.compile_video and (
                args.render_img
                or want_stress_gaussian
                or want_tracks_gaussian
                or want_subsampled_world_tracks
                or want_flow_gaussian
                or want_force_mask
                or want_object_mask
            ):
                video_root = os.path.join(args.output_path, "videos")
                os.makedirs(video_root, exist_ok=True)

        need_view_pass = (
            bool(args.render_img)
            or bool(want_stress_heatmap)
            or bool(want_stress_gaussian)
            or bool(want_flow_gaussian)
            or bool(want_force_mask)
            or bool(want_object_mask)
            or bool(want_tracks_gaussian)
        )

        # 定义多视角：方位角在 [0, 360) 上均匀分布，俯仰角固定为 config
        views: List[Tuple[float, float]] = []
        view_dirs: List[Optional[str]] = []
        stress_view_dirs: List[Optional[str]] = []
        stress_gaussian_view_dirs: List[Optional[str]] = []
        tracks_gaussian_view_dirs: List[Optional[str]] = []
        flow_gaussian_view_dirs: List[Optional[str]] = []
        force_mask_view_dirs: List[Optional[str]] = []
        object_mask_view_dirs: List[Optional[str]] = []
        view_names: List[str] = []
        if need_view_pass and args.output_path is not None:
            nv_cfg = int(args.num_render_views)
            num_views_eff = max(1, nv_cfg if nv_cfg >= 0 else int(args.num_views))
            az_base = camera_params["init_azimuthm"]
            el_base = camera_params["init_elevation"]
            if bool(getattr(args, "random_render_views", False)):
                rng = np.random.RandomState(int(getattr(args, "random_render_views_seed", 0)))
                min_gap = max(0.0, float(getattr(args, "random_render_views_min_gap_deg", 20.0)))
                picked: List[float] = []
                tries = 0
                max_tries = 5000
                while len(picked) < num_views_eff and tries < max_tries:
                    tries += 1
                    cand = float(rng.uniform(0.0, 360.0))
                    ok = True
                    for a in picked:
                        d = abs(cand - a)
                        d = min(d, 360.0 - d)
                        if d < min_gap:
                            ok = False
                            break
                    if ok:
                        picked.append(cand)
                if len(picked) < num_views_eff:
                    # 间隔约束过严时回退到均匀补齐，保证视角数稳定
                    for k in range(num_views_eff):
                        a = (az_base + 360.0 * k / num_views_eff) % 360.0
                        picked.append(a)
                    picked = picked[:num_views_eff]
                az_list = sorted([a % 360.0 for a in picked])
            else:
                az_list = [((az_base + 360.0 * k / num_views_eff) % 360.0) for k in range(num_views_eff)]

            for az in az_list:
                el = el_base
                views.append((az, el))
                name = f"az{int(round(az))}_el{int(round(el))}"
                view_names.append(name)
                if args.render_img and image_root is not None:
                    d = os.path.join(image_root, name)
                    os.makedirs(d, exist_ok=True)
                    view_dirs.append(d)
                else:
                    view_dirs.append(None)
                if want_stress_heatmap:
                    sd = os.path.join(args.output_path, "stress_heatmaps", name)
                    os.makedirs(sd, exist_ok=True)
                    stress_view_dirs.append(sd)
                else:
                    stress_view_dirs.append(None)
                if want_stress_gaussian:
                    sgd = os.path.join(args.output_path, "stress_gaussian", name)
                    os.makedirs(sgd, exist_ok=True)
                    stress_gaussian_view_dirs.append(sgd)
                else:
                    stress_gaussian_view_dirs.append(None)
                if want_tracks_gaussian:
                    tgd = os.path.join(args.output_path, "tracks_gaussian", name)
                    os.makedirs(tgd, exist_ok=True)
                    tracks_gaussian_view_dirs.append(tgd)
                else:
                    tracks_gaussian_view_dirs.append(None)
                if want_flow_gaussian:
                    fgd = os.path.join(args.output_path, "flow_gaussian", name)
                    os.makedirs(fgd, exist_ok=True)
                    flow_gaussian_view_dirs.append(fgd)
                else:
                    flow_gaussian_view_dirs.append(None)
                if want_force_mask:
                    fmd = os.path.join(args.output_path, "force_mask", name)
                    os.makedirs(fmd, exist_ok=True)
                    force_mask_view_dirs.append(fmd)
                else:
                    force_mask_view_dirs.append(None)
                if want_object_mask:
                    omd = os.path.join(args.output_path, "object_mask", name)
                    os.makedirs(omd, exist_ok=True)
                    object_mask_view_dirs.append(omd)
                else:
                    object_mask_view_dirs.append(None)
        else:
            view_dirs = []
            stress_view_dirs = []
            stress_gaussian_view_dirs = []
            tracks_gaussian_view_dirs = []
            flow_gaussian_view_dirs = []
            force_mask_view_dirs = []
            object_mask_view_dirs = []
            view_names = []

        if (
            want_flow_gaussian
            and len(views) > 0
            and args.output_path is not None
        ):
            flow_gaussian_shared = FlowGaussianSplatSharedState(
                max_gaussians=int(args.flow_gaussian_max_gaussians),
                rng_seed=int(args.flow_gaussian_seed),
                depth_gamma=float(args.flow_gaussian_depth_gamma),
                depth_eps=float(args.flow_gaussian_depth_eps),
                opacity_power=float(args.flow_gaussian_opacity_power),
            )
            flow_gaussian_shared.resize_num_views(len(views))

        track_gaussian_accum_rgb: List[Optional[torch.Tensor]] = (
            [None] * len(views) if want_tracks_gaussian else []
        )

        def _match_means2d_to_P(screen_pts: torch.Tensor, n_points: int) -> torch.Tensor:
            """means2D 与 means3D 行数对齐（补零行参与光栅化预处理）。"""
            P = int(n_points)
            sp = screen_pts
            if sp.shape[0] == P:
                return sp
            if sp.shape[0] < P:
                extra = P - sp.shape[0]
                pad = torch.zeros(
                    (extra, sp.shape[1]),
                    device=sp.device,
                    dtype=sp.dtype,
                    requires_grad=True,
                )
                try:
                    pad.retain_grad()
                except Exception:
                    pass
                return torch.cat([sp, pad], dim=0)
            return sp[:P]

        _frame_bar = tqdm(
            range(frame_num),
            desc="仿真帧",
            unit="帧",
            disable=bool(getattr(args, "quiet", False)),
            mininterval=0.5 if bool(getattr(args, "quiet", False)) else 0.1,
            smoothing=0.05,
        )
        for frame in _frame_bar:

            for step in range(step_per_frame):
                mpm_solver.p2g2p(frame, substep_dt, device=warp_device)

            # 循环中输出场（按间隔）
            if (eff_output_deformation or eff_output_stress_vol) and (
                (frame + 1) % args.field_output_interval == 0
            ):
                if eff_output_stress_vol:
                    mpm_solver.recompute_particle_stress_from_F_trial(
                        float(substep_dt), device=warp_device
                    )
                save_fields_for_frame(
                    mpm_solver,
                    frame_id=frame + 1,
                    deformation_dir=deformation_dir,
                    stress_dir=stress_dir,
                    output_deformation=eff_output_deformation,
                    output_stress=eff_output_stress_vol,
                )

            # 循环中输出 ply / h5
            if args.output_h5:
                save_data_at_frame(
                    mpm_solver,
                    directory_to_save,
                    frame + 1,
                    save_to_ply=False,
                    save_to_h5=True,
                )

            if frame in render_frame_set and (
                (need_view_pass and bool(views)) or want_subsampled_world_tracks
            ):
                out_idx = frame_to_out_idx[int(frame)]
                # 与 save_stress_field 一致的三维应力：g2p 已更新 F_trial，须先刷新 τ 再读
                if want_stress_heatmap or want_stress_gaussian:
                    mpm_solver.recompute_particle_stress_from_F_trial(
                        float(substep_dt), device=warp_device
                    )
                # 先导出当前粒子信息一次，供所有视角复用
                pos_base = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                cov3D_base = mpm_solver.export_particle_cov_to_torch(device=warp_device)
                rot_base = mpm_solver.export_particle_R_to_torch(device=warp_device)
                cov3D_base = cov3D_base.view(-1, 6)[:gs_num].to(device)
                rot_base = rot_base.view(-1, 3, 3)[:gs_num].to(device)

                # 还原到世界坐标、Undo transform（仅 MPM 粒子）
                pos_mpm_world = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos_base), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )
                cov3D_world = cov3D_base / (scale_origin * scale_origin)
                cov3D_world = apply_inverse_cov_rotations(cov3D_world, rotation_matrices)

                base_opacity = opacity_render
                base_shs = shs_render
                if preprocessing_params["sim_area"] is not None:
                    pos_world = torch.cat([pos_mpm_world, unselected_pos], dim=0)
                    cov3D_world = torch.cat([cov3D_world, unselected_cov], dim=0)
                    base_opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                    base_shs = torch.cat([shs_render, unselected_shs], dim=0)
                else:
                    pos_world = pos_mpm_world

                # 全仿真累计 touched mask：velocity cuboid 最前端截面触碰到的粒子
                current_selected_idx: Optional[np.ndarray] = None
                if (
                    args.output_initial_force_mask_arrow
                    and touched_mask_gs is not None
                    and velocity_cuboids
                ):
                    t_sec = float((frame + 1) * frame_dt)
                    pos_mpm = pos_base  # [gs_num,3]，与 BC 坐标系一致

                    touched_now = torch.zeros_like(touched_mask_gs)
                    active_any = False
                    for rod in velocity_cuboids:
                        st = float(rod["start_time"])
                        et = float(rod["end_time"])
                        # 超过 end_time 后不再更新受力区域
                        if t_sec < st or t_sec > et:
                            continue
                        active_any = True
                        dt_act = min(max(t_sec - st, 0.0), max(0.0, et - st))
                        p_now = torch.tensor(
                            rod["point0"] + rod["velocity"] * dt_act,
                            device=pos_mpm.device,
                            dtype=pos_mpm.dtype,
                        )
                        s = torch.tensor(rod["size"], device=pos_mpm.device, dtype=pos_mpm.dtype)

                        # 杆体体积内（MPM 原生定义，和 solver 的 cuboid 条件一致）
                        inside = (
                            (torch.abs(pos_mpm[:, 0] - p_now[0]) < s[0])
                            & (torch.abs(pos_mpm[:, 1] - p_now[1]) < s[1])
                            & (torch.abs(pos_mpm[:, 2] - p_now[2]) < s[2])
                        )
                        # 直接以 cuboid 覆盖区域作为受力区域
                        hit = inside
                        touched_now = touched_now | hit

                    touched_mask_gs = touched_mask_gs | touched_now
                    if touched_first_frame_gs is not None:
                        new_hit = touched_now & (touched_first_frame_gs < 0)
                        if torch.any(new_hit):
                            touched_first_frame_gs[new_hit] = int(frame)
                    if int(frame) in render_frame_set:
                        if active_any:
                            # 仅在施力激活窗口内渲染 mask；释放后恢复原始 3DGS
                            sel_t = torch.where(touched_now)[0]
                            current_selected_idx = sel_t.detach().cpu().numpy().astype(np.int64)
                        else:
                            current_selected_idx = np.array([], dtype=np.int64)

                if want_subsampled_world_tracks:
                    if fast_track_indices_np is None:
                        gsn = int(gs_num)
                        if gsn < 1:
                            fast_track_indices_np = np.zeros((0,), dtype=np.int64)
                        else:
                            nt = min(int(args.subsampled_tracks_num), gsn)
                            rng_ft = np.random.RandomState(
                                int(args.subsampled_tracks_seed)
                            )
                            fast_track_indices_np = rng_ft.choice(
                                gsn, size=nt, replace=False
                            )
                            fast_track_indices_np.sort()
                    if fast_track_indices_np.size == 0:
                        samp = np.zeros((0, 3), dtype=np.float32)
                    else:
                        idx_t = torch.as_tensor(
                            fast_track_indices_np,
                            device=pos_mpm_world.device,
                            dtype=torch.long,
                        )
                        samp = pos_mpm_world[idx_t].detach().cpu().numpy().astype(
                            np.float32
                        )
                    fast_track_xyz_list.append(samp)
                    fast_track_sim_frames.append(int(frame))

                vm_frame = None
                P = 0
                means2d_input = None
                extra_tail = 0
                if need_view_pass:
                    P = int(pos_world.shape[0])
                    means2d_input = _match_means2d_to_P(init_screen_points, P)
                    extra_tail = max(0, P - gs_num)
                    if want_stress_heatmap or want_stress_gaussian:
                        vm_frame = stress_scalars_aligned_to_render_chain(
                            mpm_solver, gs_num, extra_tail, pos_world.device
                        )

                track_gaussian_shared_batch = None
                if need_view_pass and views and want_tracks_gaussian:
                    if track_gaussian_accum_shared is None:
                        track_gaussian_accum_shared = TrajectoryGaussianAccumSharedState(
                            num_tracks=int(args.tracks_gaussian_max_tracks),
                            rng_seed=int(args.tracks_gaussian_seed),
                            midpoint_bridge=bool(args.tracks_gaussian_accum_midpoint),
                        )
                    _so_acc = float(scale_origin.detach().cpu().item())
                    _sig_acc = max(
                        1e-8, _so_acc * float(args.tracks_gaussian_sigma_scale)
                    )
                    track_gaussian_shared_batch = (
                        track_gaussian_accum_shared.build_raster_batch(
                            pos_mpm_world,
                            _sig_acc,
                            pos_mpm_world.device,
                            pos_mpm_world.dtype,
                            splat_opacity=float(args.tracks_gaussian_point_opacity),
                        )
                    )

                # 对每个视角分别渲染 / 视角辅助（需光栅化）
                if need_view_pass and views:
                    for v_idx, (az, el) in enumerate(views):
                        current_camera = get_camera_view(
                            model_path,
                            default_camera_index=camera_params["default_camera_index"],
                            center_view_world_space=viewpoint_center_worldspace,
                            observant_coordinates=observant_coordinates,
                            show_hint=camera_params["show_hint"],
                            init_azimuthm=az,
                            init_elevation=el,
                            init_radius=float(camera_params["init_radius"])
                            / max(1e-6, float(getattr(args, "camera_distance_scale", 1.0))),
                            move_camera=camera_params["move_camera"],
                            current_frame=frame,
                            delta_a=camera_params["delta_a"],
                            delta_e=camera_params["delta_e"],
                            delta_r=camera_params["delta_r"],
                        )
                        rasterize = initialize_resterize(
                            current_camera, gaussians, pipeline, background
                        )

                        colors_precomp = convert_SH(
                            base_shs, current_camera, gaussians, pos_world, rot_base
                        )
                        # 光栅 CUDA 前向不会把屏幕坐标写回 Python means2D（仍为全零）；
                        # 热力图 / flow / force debug 等辅助输出必须用与 preprocess 一致的投影。
                        means2d_for_aux = project_world_points_to_screen_means2d(
                            pos_world,
                            current_camera.full_proj_transform,
                            int(width if width is not None else 1024),
                            int(height if height is not None else 1024),
                        )
                        fc_idx: Optional[np.ndarray] = None
                        if (
                            args.output_initial_force_mask_arrow
                            and current_selected_idx is not None
                            and current_selected_idx.size > 0
                        ):
                            _tmp_idx = current_selected_idx[
                                (current_selected_idx >= 0)
                                & (current_selected_idx < int(colors_precomp.shape[0]))
                            ]
                            if _tmp_idx.size > 0:
                                fc_idx = _tmp_idx
                        rendering, raddi = rasterize(
                            means3D=pos_world,
                            means2D=means2d_input,
                            shs=None,
                            colors_precomp=colors_precomp,
                            opacities=base_opacity,
                            scales=None,
                            rotations=None,
                            cov3D_precomp=cov3D_world,
                        )
                        if height is None or width is None:
                            _tmp = rendering.permute(1, 2, 0).detach().cpu().numpy()
                            _tmp = cv2.cvtColor(_tmp, cv2.COLOR_BGR2RGB)
                            height = _tmp.shape[0] // 2 * 2
                            width = _tmp.shape[1] // 2 * 2

                        # 分辨率确定后重算一次投影（用于后续热图/flow）
                        means2d_for_aux = project_world_points_to_screen_means2d(
                            pos_world,
                            current_camera.full_proj_transform,
                            int(width),
                            int(height),
                        )

                        if args.render_img and view_dirs[v_idx] is not None:
                            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(
                                os.path.join(view_dirs[v_idx], f"{out_idx:04d}.png"),
                                255 * cv2_img,
                            )

                        # 独立 mask 渲染
                        if (
                            want_force_mask
                            and v_idx < len(force_mask_view_dirs)
                            and force_mask_view_dirs[v_idx] is not None
                            and fc_idx is not None
                            and width is not None
                            and height is not None
                        ):
                            fc_t = torch.as_tensor(
                                fc_idx, device=base_opacity.device, dtype=torch.long
                            )
                            _fm_sc = bool(getattr(args, "force_mask_single_channel", False))
                            if _fm_sc:
                                # 单通道：仅受力粒子可见、白前景、黑底
                                opa_fm = torch.zeros_like(base_opacity)
                                opa_fm[fc_t] = base_opacity[fc_t]
                                col_fm = torch.zeros_like(colors_precomp)
                                col_fm[fc_t] = 1.0
                                bg_fm = torch.zeros((3,), dtype=torch.float32, device=colors_precomp.device)
                                rasterize_fm = initialize_resterize(
                                    current_camera, gaussians, pipeline, bg_fm
                                )
                                rendering_fm, _ = rasterize_fm(
                                    means3D=pos_world,
                                    means2D=means2d_input,
                                    shs=None,
                                    colors_precomp=col_fm,
                                    opacities=opa_fm,
                                    scales=None,
                                    rotations=None,
                                    cov3D_precomp=cov3D_world,
                                )
                                fm_bgr = rendering_fm.permute(1, 2, 0).detach().cpu().numpy()
                                fm_gray = (
                                    np.clip(255.0 * fm_bgr[:, :, 0], 0, 255).astype(np.uint8)
                                    if fm_bgr.shape[2] >= 1
                                    else np.zeros((int(height), int(width)), dtype=np.uint8)
                                )
                                cv2.imwrite(
                                    os.path.join(force_mask_view_dirs[v_idx], f"{out_idx:04d}.png"),
                                    fm_gray,
                                )
                            else:
                                # RGB：保留完整 3DGS，mask 区域染红，背景与主渲染一致
                                colors_mask = colors_precomp.clone()
                                red_rgb = torch.tensor(
                                    [1.0, 0.0, 0.0],
                                    device=colors_precomp.device,
                                    dtype=colors_precomp.dtype,
                                )
                                colors_mask[fc_t] = red_rgb
                                bg_mask = background
                                rasterize_mask = initialize_resterize(
                                    current_camera, gaussians, pipeline, bg_mask
                                )
                                rendering_mask, _ = rasterize_mask(
                                    means3D=pos_world,
                                    means2D=means2d_input,
                                    shs=None,
                                    colors_precomp=colors_mask,
                                    opacities=base_opacity,
                                    scales=None,
                                    rotations=None,
                                    cov3D_precomp=cov3D_world,
                                )
                                mask_img = rendering_mask.permute(1, 2, 0).detach().cpu().numpy()
                                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                                cv2.imwrite(
                                    os.path.join(force_mask_view_dirs[v_idx], f"{out_idx:04d}.png"),
                                    np.clip(255.0 * mask_img, 0, 255).astype(np.uint8),
                                )

                        # 物体 mask 渲染（白物体、黑背景）
                        if (
                            want_object_mask
                            and v_idx < len(object_mask_view_dirs)
                            and object_mask_view_dirs[v_idx] is not None
                            and width is not None
                            and height is not None
                        ):
                            colors_obj_mask = torch.ones_like(colors_precomp)
                            bg_obj_mask = torch.zeros(
                                (3,), dtype=torch.float32, device=colors_precomp.device
                            )
                            rasterize_obj_mask = initialize_resterize(
                                current_camera, gaussians, pipeline, bg_obj_mask
                            )
                            rendering_obj_mask, _ = rasterize_obj_mask(
                                means3D=pos_world,
                                means2D=means2d_input,
                                shs=None,
                                colors_precomp=colors_obj_mask,
                                opacities=base_opacity,
                                scales=None,
                                rotations=None,
                                cov3D_precomp=cov3D_world,
                            )
                            obj_mask_img = (
                                rendering_obj_mask.permute(1, 2, 0).detach().cpu().numpy()
                            )
                            obj_mask_img = cv2.cvtColor(obj_mask_img, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(
                                os.path.join(object_mask_view_dirs[v_idx], f"{out_idx:04d}.png"),
                                np.clip(255.0 * obj_mask_img, 0, 255).astype(np.uint8),
                            )

                        if want_stress_heatmap and stress_view_dirs[v_idx] is not None:
                            stress_np = vm_frame.detach().cpu().numpy()
                            heat_bgr = splat_stress_heatmap_bgr(
                                means2d_for_aux,
                                raddi,
                                stress_np,
                                int(height),
                                int(width),
                            )
                            cv2.imwrite(
                                os.path.join(
                                    stress_view_dirs[v_idx], f"{out_idx:04d}.png"
                                ),
                                heat_bgr,
                            )

                        if (
                            want_stress_gaussian
                            and stress_gaussian_view_dirs[v_idx] is not None
                            and vm_frame is not None
                        ):
                            _sg_sc = bool(getattr(args, "stress_gaussian_single_channel", False))
                            sg_colors, sg_opacity = stress_gaussian_precomp_colors_and_opacity(
                                vm_frame,
                                colors_precomp,
                                base_opacity,
                                gs_num,
                                opa_floor=float(args.stress_gaussian_opa_floor),
                                opa_ceil=float(args.stress_gaussian_opa_ceil),
                                sh_blend=0.0 if _sg_sc else float(args.stress_gaussian_sh_blend),
                                vm_pct_low=float(args.stress_gaussian_vm_pct_low),
                                vm_pct_high=float(args.stress_gaussian_vm_pct_high),
                                colormap_steps=int(args.stress_gaussian_colormap_steps),
                                colormap_mode="gray" if _sg_sc else "jet",
                            )
                            bg_sg = (
                                torch.zeros((3,), dtype=torch.float32, device=colors_precomp.device)
                                if _sg_sc
                                else background
                            )
                            rasterize_sg = initialize_resterize(
                                current_camera, gaussians, pipeline, bg_sg
                            )
                            rendering_sg, _r_sg = rasterize_sg(
                                means3D=pos_world,
                                means2D=means2d_input,
                                shs=None,
                                colors_precomp=sg_colors,
                                opacities=sg_opacity,
                                scales=None,
                                rotations=None,
                                cov3D_precomp=cov3D_world,
                            )
                            cv2_sg = rendering_sg.permute(1, 2, 0).detach().cpu().numpy()
                            cv2_sg = cv2.cvtColor(cv2_sg, cv2.COLOR_BGR2RGB)
                            u8_rgb = np.clip(255.0 * cv2_sg, 0, 255).astype(np.uint8)
                            sg_path = os.path.join(
                                stress_gaussian_view_dirs[v_idx], f"{out_idx:04d}.png"
                            )
                            if _sg_sc:
                                cv2.imwrite(sg_path, u8_rgb[:, :, 0])
                            else:
                                cv2.imwrite(sg_path, cv2.cvtColor(u8_rgb, cv2.COLOR_RGB2BGR))

                        if want_tracks_gaussian and tracks_gaussian_view_dirs[v_idx] is not None:
                            if track_gaussian_shared_batch is None:
                                traj_img = torch.zeros_like(rendering)
                            else:
                                tpos, tcov, tcol, top = track_gaussian_shared_batch
                                M = int(tpos.shape[0])
                                means2d_t = torch.zeros(
                                    (M, 2),
                                    device=pos_world.device,
                                    dtype=torch.float32,
                                    requires_grad=True,
                                )
                                try:
                                    means2d_t.retain_grad()
                                except Exception:
                                    pass
                                bg_traj = torch.zeros(
                                    (3,),
                                    dtype=torch.float32,
                                    device=pos_world.device,
                                )
                                r_tr = initialize_resterize(
                                    current_camera, gaussians, pipeline, bg_traj
                                )
                                traj_img, _tr = r_tr(
                                    means3D=tpos,
                                    means2D=means2d_t,
                                    shs=None,
                                    colors_precomp=tcol,
                                    opacities=top,
                                    scales=None,
                                    rotations=None,
                                    cov3D_precomp=tcov,
                                )
                            acc = track_gaussian_accum_rgb[v_idx]
                            if acc is None:
                                acc = torch.zeros_like(traj_img)
                            else:
                                acc.mul_(float(args.tracks_gaussian_accum_decay))
                            layer = traj_img * float(args.tracks_gaussian_accum_frame_weight)
                            _tgm = str(args.tracks_gaussian_accum_mode).strip().lower()
                            if _tgm == "add":
                                acc.add_(layer)
                            else:
                                acc = torch.maximum(acc, layer)
                            track_gaussian_accum_rgb[v_idx] = acc
                            if bool(args.tracks_gaussian_accum_no_normalize_save):
                                composed = torch.clamp(
                                    acc * float(args.tracks_gaussian_intensity),
                                    0.0,
                                    1.0,
                                )
                            else:
                                m = acc.max()
                                if float(m.detach().cpu()) <= 1e-12:
                                    composed = torch.zeros_like(acc)
                                else:
                                    composed = torch.clamp(acc / m, 0.0, 1.0)
                            cv2_tg = composed.permute(1, 2, 0).detach().cpu().numpy()
                            cv2_tg = cv2.cvtColor(cv2_tg, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(
                                os.path.join(
                                    tracks_gaussian_view_dirs[v_idx],
                                    f"{out_idx:04d}.png",
                                ),
                                np.clip(255.0 * cv2_tg, 0, 255).astype(np.uint8),
                            )

                        if (
                            want_flow_gaussian
                            and flow_gaussian_shared is not None
                            and v_idx < len(flow_gaussian_view_dirs)
                            and flow_gaussian_view_dirs[v_idx] is not None
                            and height is not None
                            and width is not None
                        ):
                            flow_gaussian_shared.ensure_indices(int(gs_num))
                            _fmm = float(args.flow_gaussian_vis_max_motion)
                            fg_colors, fg_opa = (
                                flow_gaussian_shared.build_flow_precomp_for_raster(
                                    v_idx,
                                    means2d_for_aux,
                                    raddi,
                                    pos_world,
                                    base_opacity,
                                    current_camera.camera_center,
                                    int(width),
                                    int(height),
                                    max_motion=_fmm if _fmm > 0 else None,
                                )
                            )
                            fg_bg = torch.zeros(
                                (3,),
                                dtype=torch.float32,
                                device=pos_world.device,
                            )
                            rasterize_fg = initialize_resterize(
                                current_camera, gaussians, pipeline, fg_bg
                            )
                            rendering_fg, _r_fg = rasterize_fg(
                                means3D=pos_world,
                                means2D=means2d_input,
                                shs=None,
                                colors_precomp=fg_colors,
                                opacities=fg_opa,
                                scales=None,
                                rotations=None,
                                cov3D_precomp=cov3D_world,
                            )
                            cv2_fg = rendering_fg.permute(1, 2, 0).detach().cpu().numpy()
                            cv2_fg = cv2.cvtColor(cv2_fg, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(
                                os.path.join(
                                    flow_gaussian_view_dirs[v_idx],
                                    f"{out_idx:04d}.png",
                                ),
                                np.clip(255.0 * cv2_fg, 0, 255).astype(np.uint8),
                            )

        # 汇总运行参数（供数据集构建与复现）
        if args.output_path is not None:
            meta_dir = os.path.join(args.output_path, "meta")
            os.makedirs(meta_dir, exist_ok=True)
            nv_effective = len(views) if views else 0

            def _json_safe_obj(o):
                if isinstance(o, dict):
                    return {str(k): _json_safe_obj(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_json_safe_obj(v) for v in o]
                if isinstance(o, (np.floating, np.integer)):
                    return o.item()
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return o

            run_payload = {
                "sim_type": args.sim_type,
                "frame_num_simulated": int(frame_num),
                "substep_dt": float(substep_dt),
                "frame_dt": float(frame_dt),
                "step_per_frame": int(step_per_frame),
                "num_views_cli": int(args.num_views),
                "num_render_views_cli": int(args.num_render_views),
                "num_render_views_effective": int(nv_effective),
                "num_render_timesteps_cli": int(args.num_render_timesteps),
                "render_outputs_per_sim_second": float(
                    getattr(args, "render_outputs_per_sim_second", 0.0) or 0.0
                ),
                "video_compile_framerate": float(video_compile_fps),
                "render_frame_indices": [int(x) for x in render_frame_indices],
                "num_render_outputs": int(len(render_frame_indices)),
                "render_img": bool(args.render_img),
                "compile_video": bool(args.compile_video),
                "compress_render_pngs": bool(
                    getattr(args, "compress_render_pngs", True)
                ),
                "png_compression_level": int(
                    getattr(args, "png_compression_level", 6)
                ),
                "render_export_max_side": int(
                    getattr(args, "render_export_max_side", 0) or 0
                ),
                "render_export_scale": float(
                    getattr(args, "render_export_scale", 1.0) or 1.0
                ),
                "camera_distance_scale": float(
                    getattr(args, "camera_distance_scale", 1.0) or 1.0
                ),
                "arch4_tensor_dtype": str(
                    getattr(args, "arch4_tensor_dtype", "float32")
                ),
                "pack_arch4_lmdb": bool(getattr(args, "pack_arch4_lmdb", False)),
                "arch4_lmdb_resize": int(getattr(args, "arch4_lmdb_resize", 224)),
                "arch4_lmdb_map_size_gb": float(
                    getattr(args, "arch4_lmdb_map_size_gb", 8.0)
                ),
                "arch4_lmdb_name": str(getattr(args, "arch4_lmdb_name", "arch4_data.lmdb")),
                "pack_arch4_lmdb_include_object_mask": bool(
                    getattr(args, "pack_arch4_lmdb_include_object_mask", False)
                ),
                "pack_sample_pack": bool(getattr(args, "pack_sample_pack", False)),
                "sample_pack_name": str(getattr(args, "sample_pack_name", "sample_pack.npz")),
                "pack_sample_pack_include_object_mask": bool(
                    getattr(args, "pack_sample_pack_include_object_mask", False)
                ),
                "png_sequence_width": int(width) if width is not None else None,
                "png_sequence_height": int(height) if height is not None else None,
                "delete_png_sequences_after_compile_video": bool(
                    args.delete_png_sequences_after_compile_video
                ),
                "output_view_stress_heatmap_requested": bool(
                    args.output_view_stress_heatmap
                ),
                "output_view_stress_heatmap_effective": bool(want_stress_heatmap),
                "output_view_stress_gaussian": bool(want_stress_gaussian),
                "stress_gaussian_opa_floor": float(args.stress_gaussian_opa_floor),
                "stress_gaussian_opa_ceil": float(args.stress_gaussian_opa_ceil),
                "stress_gaussian_sh_blend": float(args.stress_gaussian_sh_blend),
                "stress_gaussian_colormap_steps": int(args.stress_gaussian_colormap_steps),
                "stress_gaussian_vm_pct_low": float(args.stress_gaussian_vm_pct_low),
                "stress_gaussian_vm_pct_high": float(args.stress_gaussian_vm_pct_high),
                "output_view_flow_gaussian": bool(want_flow_gaussian),
                "output_view_force_mask": bool(want_force_mask),
                "output_view_object_mask": bool(want_object_mask),
                "flow_gaussian_max_gaussians": int(args.flow_gaussian_max_gaussians),
                "flow_gaussian_seed": int(args.flow_gaussian_seed),
                "flow_gaussian_depth_gamma": float(args.flow_gaussian_depth_gamma),
                "flow_gaussian_depth_eps": float(args.flow_gaussian_depth_eps),
                "flow_gaussian_opacity_power": float(args.flow_gaussian_opacity_power),
                "flow_gaussian_vis_max_motion": float(
                    args.flow_gaussian_vis_max_motion
                ),
                "output_view_tracks_gaussian": bool(want_tracks_gaussian),
                "tracks_gaussian_max_tracks": int(args.tracks_gaussian_max_tracks),
                "tracks_gaussian_sigma_scale": float(args.tracks_gaussian_sigma_scale),
                "tracks_gaussian_point_opacity": float(args.tracks_gaussian_point_opacity),
                "tracks_gaussian_intensity": float(args.tracks_gaussian_intensity),
                "tracks_gaussian_seed": int(args.tracks_gaussian_seed),
                "tracks_gaussian_accum_mode": str(args.tracks_gaussian_accum_mode),
                "tracks_gaussian_accum_frame_weight": float(
                    args.tracks_gaussian_accum_frame_weight
                ),
                "tracks_gaussian_accum_decay": float(args.tracks_gaussian_accum_decay),
                "tracks_gaussian_accum_midpoint": bool(
                    args.tracks_gaussian_accum_midpoint
                ),
                "tracks_gaussian_accum_no_normalize_save": bool(
                    args.tracks_gaussian_accum_no_normalize_save
                ),
                "output_subsampled_world_tracks": bool(want_subsampled_world_tracks),
                "subsampled_tracks_num": int(args.subsampled_tracks_num),
                "subsampled_tracks_seed": int(args.subsampled_tracks_seed),
                "subsampled_tracks_ortho_axes": str(
                    args.subsampled_tracks_ortho_axes
                ),
                "subsampled_tracks_video_size": int(
                    args.subsampled_tracks_video_size
                ),
                "view_stress_recompute_before_read": bool(
                    want_stress_heatmap or want_stress_gaussian
                ),
                "output_deformation_volumetric": bool(eff_output_deformation),
                "output_stress_volumetric": bool(eff_output_stress_vol),
                "field_output_interval": int(args.field_output_interval),
                "view_names": list(view_names),
                "material_params": _json_safe_obj(material_params),
                "time_params": _json_safe_obj(time_params),
            }
            write_run_parameters_json(
                os.path.join(meta_dir, "run_parameters.json"), run_payload
            )

        if (
            want_subsampled_world_tracks
            and fast_track_xyz_list
            and args.output_path is not None
        ):
            tw_dir = os.path.join(args.output_path, "tracks_subsampled_world")
            os.makedirs(tw_dir, exist_ok=True)
            xyz_all = np.stack(fast_track_xyz_list, axis=0)
            ax_meta = str(args.subsampled_tracks_ortho_axes).strip().lower()
            np.savez_compressed(
                os.path.join(tw_dir, "tracks_world.npz"),
                xyz_world=xyz_all,
                particle_indices=fast_track_indices_np.astype(np.int64),
                sim_frames=np.array(fast_track_sim_frames, dtype=np.int32),
                ortho_axes_preview=ax_meta,
                description=(
                    "xyz_world: (T_out,N,3) 世界系；particle_indices: MPM 粒子下标；"
                    "sim_frames: 仿真步；时间轴与 render_frame_indices / images 一致。"
                ),
            )
            _vprint(f"[modified_simulation] 已写出下采样世界轨迹: {tw_dir}/tracks_world.npz")

        # 不再输出额外 force overlay/debug 文件，仅保留原 3DGS 渲染结果（通过 SH 着色体现受力区域）

        # 序列 PNG：先按需降分辨率，再 zlib 压缩（可选），再打包 arch4 .pt；与 ffmpeg 是否可用无关
        if args.output_path is not None and views:
            mask_dirs_extra_for_post = object_mask_view_dirs if want_object_mask else []
            if width is not None and height is not None:
                nw, nh = int(width), int(height)
                tw, th = compute_export_resolution(
                    nw,
                    nh,
                    max_side=int(getattr(args, "render_export_max_side", 0) or 0),
                    scale=float(getattr(args, "render_export_scale", 1.0) or 1.0),
                )
                if (tw, th) != (nw, nh):
                    try:
                        n_ds = downscale_multiview_render_png_dirs(
                            view_dirs,
                            stress_gaussian_view_dirs,
                            flow_gaussian_view_dirs,
                            force_mask_view_dirs,
                            target_w=tw,
                            target_h=th,
                        )
                        if mask_dirs_extra_for_post:
                            n_ds += downscale_multiview_render_png_dirs(
                                [],
                                [],
                                [],
                                mask_dirs_extra_for_post,
                                target_w=tw,
                                target_h=th,
                            )
                        width, height = tw, th
                        _vprint(
                            f"[modified_simulation] PNG 导出分辨率 {nw}x{nh} -> {tw}x{th}，重写 {n_ds} 个文件"
                        )
                    except Exception as e:
                        print(f"[modified_simulation] PNG 降分辨率失败（保留原始尺寸）: {e}")
                meta_rp = os.path.join(args.output_path, "meta", "run_parameters.json")
                if os.path.isfile(meta_rp):
                    try:
                        with open(meta_rp, "r", encoding="utf-8") as f:
                            rp = json.load(f)
                        rp["png_sequence_width"] = int(width)
                        rp["png_sequence_height"] = int(height)
                        write_run_parameters_json(meta_rp, rp)
                    except Exception as e:
                        print(
                            f"[modified_simulation] 更新 run_parameters 中 png_sequence 分辨率失败: {e}"
                        )

            if bool(getattr(args, "compress_render_pngs", True)):
                try:
                    n_cmp = compress_multiview_render_png_dirs(
                        view_dirs,
                        stress_gaussian_view_dirs,
                        flow_gaussian_view_dirs,
                        force_mask_view_dirs,
                        compression=int(getattr(args, "png_compression_level", 6)),
                    )
                    if mask_dirs_extra_for_post:
                        n_cmp += compress_multiview_render_png_dirs(
                            [],
                            [],
                            [],
                            mask_dirs_extra_for_post,
                            compression=int(getattr(args, "png_compression_level", 6)),
                        )
                    _vprint(
                        f"[modified_simulation] PNG zlib 压缩完成，重写 {n_cmp} 个文件 "
                        f"(level={int(getattr(args, 'png_compression_level', 6))})"
                    )
                except Exception as e:
                    print(f"[modified_simulation] PNG 压缩失败（保留原图）: {e}")
            if bool(getattr(args, "pack_arch4_lmdb", False)):
                try:
                    lm_info = write_sample_arch4_lmdb(
                        args.output_path,
                        resize=int(getattr(args, "arch4_lmdb_resize", 224)),
                        env_rel=str(getattr(args, "arch4_lmdb_name", "arch4_data.lmdb")),
                        force_mask_subdir="force_mask",
                        object_mask_subdir="object_mask",
                        include_object_mask=bool(
                            getattr(args, "pack_arch4_lmdb_include_object_mask", False)
                        ),
                        num_frames=(
                            len(render_frame_indices) if render_frame_indices else None
                        ),
                        map_size_gb=float(getattr(args, "arch4_lmdb_map_size_gb", 8.0)),
                        overwrite=True,
                    )
                    _vprint(
                        f"[modified_simulation] 已写入 arch4 LMDB: "
                        f"views={lm_info.get('written', 0)} "
                        f"T={lm_info.get('num_frames', 0)} "
                        f"size={lm_info.get('img_size', 0)} "
                        f"path={lm_info.get('lmdb_path', '')}"
                    )
                except Exception as e:
                    print(f"[modified_simulation] 写入 arch4 LMDB 失败: {e}")
            if bool(getattr(args, "pack_sample_pack", False)):
                try:
                    sp_info = write_sample_pack(
                        args.output_path,
                        resize=int(getattr(args, "arch4_lmdb_resize", 224)),
                        sample_pack_name=str(getattr(args, "sample_pack_name", "sample_pack.npz")),
                        force_mask_subdir="force_mask",
                        object_mask_subdir="object_mask",
                        include_object_mask=bool(
                            getattr(args, "pack_sample_pack_include_object_mask", False)
                        ),
                        num_frames=(
                            len(render_frame_indices) if render_frame_indices else None
                        ),
                        overwrite=True,
                        compressed=True,
                    )
                    _vprint(
                        f"[modified_simulation] 已写入 sample_pack: "
                        f"views={sp_info.get('written', 0)} "
                        f"T={sp_info.get('num_frames', 0)} "
                        f"size={sp_info.get('img_size', 0)} "
                        f"path={sp_info.get('sample_pack_path', '')}"
                    )
                except Exception as e:
                    print(f"[modified_simulation] 写入 sample_pack 失败: {e}")
            if args.pack_arch4_tensors and not bool(
                getattr(args, "pack_arch4_lmdb", False)
            ):
                try:
                    pack_info = pack_sample_arch4_tensors(
                        args.output_path,
                        out_subdir="arch4_tensors",
                        force_mask_subdir="force_mask",
                        num_frames=(
                            len(render_frame_indices) if render_frame_indices else None
                        ),
                        img_size=(int(width) if width is not None else None),
                        overwrite=True,
                        tensor_dtype=str(getattr(args, "arch4_tensor_dtype", "float32")),
                    )
                    _vprint(
                        f"[modified_simulation] 已打包 arch4_tensors: "
                        f"written={pack_info.get('written', 0)} "
                        f"skipped={pack_info.get('skipped', 0)}"
                    )
                except Exception as e:
                    print(f"[modified_simulation] 打包 arch4_tensors 失败: {e}")

        # 为每个视角分别合成视频，放在 videos 目录下
        if args.compile_video and video_root is not None:
            # 与上文一致：N 模式用 render_outputs_per_sim_second；否则按仿真 frame_dt
            fps = video_compile_fps
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin is None:
                print(
                    "[modified_simulation] 未找到 ffmpeg，无法合成视频。"
                    f"请先安装 ffmpeg，或关闭 --compile_video。期望输出目录: {video_root}"
                )
            else:
                rendered_png_dirs_for_optional_delete: List[str] = []

                def _ffmpeg_png_dir_to_mp4(
                    in_dir: str, out_mp4: str, vw: int, vh: int
                ) -> bool:
                    """成功返回 True；失败返回 False。"""
                    in_pattern = os.path.join(in_dir, "%04d.png")
                    cmd = [
                        ffmpeg_bin,
                        "-framerate",
                        str(float(fps)),
                        "-i",
                        in_pattern,
                        "-c:v",
                        "libx264",
                        "-s",
                        f"{int(vw)}x{int(vh)}",
                        "-y",
                        "-pix_fmt",
                        "yuv420p",
                        out_mp4,
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        print(f"[modified_simulation] ffmpeg 合成失败: {out_mp4}")
                        print(f"[modified_simulation] cmd: {' '.join(cmd)}")
                        if proc.stderr:
                            print(proc.stderr)
                        return False
                    _vprint(f"[modified_simulation] 已输出视频: {out_mp4}")
                    rendered_png_dirs_for_optional_delete.append(in_dir)
                    return True

                if (
                    want_subsampled_world_tracks
                    and fast_track_xyz_list
                    and args.output_path is not None
                ):
                    xyz_all = np.stack(fast_track_xyz_list, axis=0)
                    tw_dir = os.path.join(args.output_path, "tracks_subsampled_world")
                    ax = str(args.subsampled_tracks_ortho_axes).strip().lower()
                    ortho_d = os.path.join(tw_dir, "ortho_frames", ax)
                    ft_w = max(32, int(args.subsampled_tracks_video_size))
                    ft_h = ft_w
                    write_subsampled_world_tracks_ortho_pngs(
                        xyz_all, ortho_d, ft_w, ft_h, ax
                    )
                    out_mp4 = os.path.join(
                        video_root, f"tracks_subsampled_world_ortho_{ax}.mp4"
                    )
                    _ffmpeg_png_dir_to_mp4(ortho_d, out_mp4, ft_w, ft_h)

                if views and width is not None and height is not None:
                    vw, vh = int(width), int(height)
                    if args.render_img:
                        for v_idx, (az, el) in enumerate(views):
                            view_dir = view_dirs[v_idx]
                            if view_dir is None:
                                continue
                            name = f"az{int(round(az))}_el{int(round(el))}"
                            out_path = os.path.join(video_root, f"{name}.mp4")
                            _ffmpeg_png_dir_to_mp4(view_dir, out_path, vw, vh)

                    if want_stress_gaussian:
                        for v_idx, (az, el) in enumerate(views):
                            sg_dir = stress_gaussian_view_dirs[v_idx]
                            if sg_dir is None:
                                continue
                            name = f"az{int(round(az))}_el{int(round(el))}"
                            out_path = os.path.join(
                                video_root, f"{name}_stress_gaussian.mp4"
                            )
                            _ffmpeg_png_dir_to_mp4(sg_dir, out_path, vw, vh)

                    if want_tracks_gaussian:
                        for v_idx, (az, el) in enumerate(views):
                            tg_dir = tracks_gaussian_view_dirs[v_idx]
                            if tg_dir is None:
                                continue
                            name = f"az{int(round(az))}_el{int(round(el))}"
                            out_path = os.path.join(
                                video_root, f"{name}_tracks_gaussian.mp4"
                            )
                            _ffmpeg_png_dir_to_mp4(tg_dir, out_path, vw, vh)

                    if want_flow_gaussian:
                        for v_idx, (az, el) in enumerate(views):
                            if v_idx >= len(flow_gaussian_view_dirs):
                                break
                            fg_dir = flow_gaussian_view_dirs[v_idx]
                            if fg_dir is None:
                                continue
                            name = f"az{int(round(az))}_el{int(round(el))}"
                            out_path = os.path.join(
                                video_root, f"{name}_flow_gaussian.mp4"
                            )
                            _ffmpeg_png_dir_to_mp4(fg_dir, out_path, vw, vh)

                    if want_force_mask:
                        for v_idx, (az, el) in enumerate(views):
                            if v_idx >= len(force_mask_view_dirs):
                                break
                            fm_dir = force_mask_view_dirs[v_idx]
                            if fm_dir is None:
                                continue
                            name = f"az{int(round(az))}_el{int(round(el))}"
                            out_path = os.path.join(
                                video_root, f"{name}_force_mask.mp4"
                            )
                            _ffmpeg_png_dir_to_mp4(fm_dir, out_path, vw, vh)

                    if want_object_mask:
                        for v_idx, (az, el) in enumerate(views):
                            if v_idx >= len(object_mask_view_dirs):
                                break
                            om_dir = object_mask_view_dirs[v_idx]
                            if om_dir is None:
                                continue
                            name = f"az{int(round(az))}_el{int(round(el))}"
                            out_path = os.path.join(
                                video_root, f"{name}_object_mask.mp4"
                            )
                            _ffmpeg_png_dir_to_mp4(om_dir, out_path, vw, vh)

                if bool(getattr(args, "delete_png_sequences_after_compile_video", False)):
                    for d in sorted(set(rendered_png_dirs_for_optional_delete)):
                        try:
                            shutil.rmtree(d, ignore_errors=False)
                            _vprint(f"[modified_simulation] 已删除 PNG 序列目录: {d}")
                        except OSError as e:
                            print(
                                f"[modified_simulation] 删除 PNG 目录失败（保留文件）: "
                                f"{d} — {e}"
                            )
                elif views:
                    print(
                        "[modified_simulation] 跳过多视角光栅 PNG 的 ffmpeg："
                        "未能确定 width/height（需至少跑过一帧 3DGS 渲染）。"
                    )
