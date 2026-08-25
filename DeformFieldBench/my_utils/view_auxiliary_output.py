"""
渲染视角下的辅助输出：应力热力图（2D）、原生 3DGS 应力着色（预计算颜色+不透明度）、
3DGS 流程下的 **flow 可视化**（第二遍 CUDA 光栅，``colors_precomp`` 编码屏幕 Δu,Δv；仅 PNG），
3DGS 微型高斯轨迹（与 RGB 同相机，整物体下采样 + 帧间图像累加），
以及 **不下光栅化** 的下采样粒子世界坐标轨迹 npz + 可选正交折线预览视频，避免保存完整三维体数据。

与 ``stress_gaussian`` 相同管线：预计算 RGB+opacity，标准 3DGS alpha 合成（不修改 PLY 内 SH 系数）。
屏幕坐标须 ``project_world_points_to_screen_means2d`` 重算（CUDA 前向不会写回 Python means2D）；
3DGS 轨迹为整物体 MPM 下采样 + 累加缓冲；下采样世界轨迹 npz 仅依赖 MPM 世界坐标。

应力标量（Cauchy → von Mises）：
  与 mpm_solver.save_stress_field 相同：τ = particle_stress，J = det(F_trial)，σ = τ/J。
  调用方在「整帧所有子步 p2g2p 结束之后」须先执行
  MPM_Simulator_WARP.recompute_particle_stress_from_F_trial(substep_dt)，
  否则 g2p 已更新 F_trial 而 τ 仍停留在子步初，热力图会与物理状态不一致。
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def project_world_points_to_screen_means2d(
    xyz: torch.Tensor,
    full_proj_transform: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """
    将世界坐标 (P,3) 投影到与 diff-gaussian-rasterization preprocess 一致的像素平面 (x,y)，
    返回 (P,3)（第三维填 0，与历史 means2D 形状对齐）。

    重要：CUDA 光栅前向在内部 geom buffer 里写屏幕坐标，**不会**把结果写回 PyTorch 传入的
    means2D；若仍用 load_params_from_gs 里初始化的全零 screen_points，则热力图 / flow splat
    会全部堆在 (0,0) 附近，帧间位移≈0 → flow 伪彩全黑。
    """
    if xyz.numel() == 0:
        return torch.zeros(0, 3, device=xyz.device, dtype=xyz.dtype)
    p = int(xyz.shape[0])
    w = int(image_width)
    h = int(image_height)
    ones = torch.ones((p, 1), device=xyz.device, dtype=xyz.dtype)
    homog = torch.cat([xyz, ones], dim=1)
    m = full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
    # 与 CUDA transformPoint4x4 后做透视除法一致：clip = homog @ M.T
    clip = homog @ m.T
    pw = 1.0 / (clip[:, 3:4] + 1e-7)
    ndc_x = clip[:, 0:1] * pw
    ndc_y = clip[:, 1:2] * pw
    # ndc2Pix(v, S) = ((v + 1.0) * S - 1.0) * 0.5  （forward.cu）
    u = ((ndc_x + 1.0) * float(w) - 1.0) * 0.5
    v = ((ndc_y + 1.0) * float(h) - 1.0) * 0.5
    out = torch.zeros(p, 3, device=xyz.device, dtype=xyz.dtype)
    out[:, 0] = u.squeeze(-1)
    out[:, 1] = v.squeeze(-1)
    return out


def count_render_samples_for_sim_rate(
    frame_num: int,
    frame_dt: float,
    outputs_per_sim_second: float,
) -> int:
    """
    按「仿真总时长 × 每秒输出张数」得到均匀采样时的目标张数 K，并限制在 [1, frame_num]。

    outputs_per_sim_second：用户记为 N；总时长 T = frame_num * frame_dt（秒），则 K ≈ round(T×N)。
    若 outputs_per_sim_second <= 0，返回 0 表示调用方应改用 num_render_timesteps 逻辑。
    """
    fn = max(0, int(frame_num))
    if fn <= 0 or float(outputs_per_sim_second) <= 0.0:
        return 0
    T = float(fn) * float(frame_dt)
    k = int(round(T * float(outputs_per_sim_second)))
    return max(1, min(k, fn))


def compute_render_frame_indices(
    frame_num: int, num_render_timesteps: int
) -> List[int]:
    """
    在整个仿真时间轴上均匀选取要输出渲染/热力图/轨迹采样的帧（仿真步下标）。

    num_render_timesteps <= 0 或 >= frame_num：每一仿真帧都输出。
    否则在 [0, frame_num-1] 上含端点均匀取 K 个整数帧。
    """
    if frame_num <= 0:
        return []
    if num_render_timesteps <= 0 or num_render_timesteps >= frame_num:
        return list(range(frame_num))
    K = int(num_render_timesteps)
    if K == 1:
        return [0]
    out = sorted(
        {int(round(i * (frame_num - 1) / (K - 1))) for i in range(K)}
    )
    return out


def frame_to_output_index(render_frame_indices: Sequence[int]) -> Dict[int, int]:
    return {int(f): i for i, f in enumerate(render_frame_indices)}


def cauchy_stress_per_particle(mpm_solver) -> torch.Tensor:
    """
    Cauchy 应力 σ，形状 (N, 3, 3)，与 save_stress_field 一致（F_trial 算 J，Kirchhoff τ）。

    须已在当前时刻调用过 recompute_particle_stress_from_F_trial（见模块文档）。
    """
    tau = mpm_solver.export_particle_stress_to_torch()
    F = mpm_solver.export_particle_F_trial_to_torch()
    J = torch.det(F).clamp_min(1e-12)
    sigma = tau / J.view(-1, 1, 1)
    return sigma


def von_mises_cauchy(sigma: torch.Tensor) -> torch.Tensor:
    """
    von Mises 等效应力（基于 Cauchy 张量，单位与 σ 一致）。
    sigma: (N, 3, 3)
    """
    s = sigma
    s11, s12, s13 = s[:, 0, 0], s[:, 0, 1], s[:, 0, 2]
    s22, s23 = s[:, 1, 1], s[:, 1, 2]
    s33 = s[:, 2, 2]
    return torch.sqrt(
        0.5
        * (
            (s11 - s22) ** 2
            + (s22 - s33) ** 2
            + (s33 - s11) ** 2
            + 6.0 * (s12 ** 2 + s23 ** 2 + s13 ** 2)
        )
    )


def stress_scalars_aligned_to_render_chain(
    mpm_solver,
    gs_num: int,
    extra_tail: int,
    device: torch.device,
) -> torch.Tensor:
    """与当前渲染链路的粒子数对齐：前 gs_num 来自 MPM，extra_tail 个补零（如未参与仿真的高斯）。"""
    sigma = cauchy_stress_per_particle(mpm_solver)[:gs_num]
    vm = von_mises_cauchy(sigma)
    if extra_tail > 0:
        z = torch.zeros(extra_tail, device=device, dtype=vm.dtype)
        vm = torch.cat([vm, z], dim=0)
    return vm


def jet_like_palette_linear(n_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    高对比 JET 风格调色板（饱和加强），形状 (n_steps, 3)，RGB∈[0,1]。
    蓝 → 亮蓝 → 青 → 黄绿 → 黄 → 橙 → 红，阶梯上色时档位更易区分。
    """
    n_steps = max(2, int(n_steps))
    anchors = torch.tensor(
        [
            [0.02, 0.05, 0.50],
            [0.00, 0.20, 1.00],
            [0.00, 0.90, 1.00],
            [0.10, 1.00, 0.25],
            [0.50, 1.00, 0.00],
            [1.00, 0.95, 0.00],
            [1.00, 0.45, 0.00],
            [1.00, 0.00, 0.00],
            [0.65, 0.00, 0.15],
        ],
        dtype=dtype,
        device=device,
    )
    n_anch = int(anchors.shape[0])
    out = torch.zeros(n_steps, 3, device=device, dtype=dtype)
    for c in range(3):
        ch = anchors[:, c].view(1, 1, -1)
        out[:, c] = F.interpolate(ch, size=n_steps, mode="linear", align_corners=True).view(
            -1
        )
    return torch.clamp(out, 0.0, 1.0)


def stress_gaussian_precomp_colors_and_opacity(
    vm: torch.Tensor,
    base_sh_rgb: torch.Tensor,
    base_opacity: torch.Tensor,
    gs_num: int,
    opa_floor: float = 0.12,
    opa_ceil: float = 1.0,
    sh_blend: float = 0.35,
    *,
    vm_vmin_fixed: Optional[float] = None,
    vm_vmax_fixed: Optional[float] = None,
    vm_pct_low: Optional[float] = None,
    vm_pct_high: Optional[float] = None,
    colormap_steps: int = 24,
    colormap_mode: str = "jet",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 **von Mises 绝对值** 上线性框定 [vmin, vmax] 后归一化到 [0,1]（不用 log）：
    - **颜色**：默认阶梯式索引高对比 JET 调色板；``colormap_mode="gray"`` 时用与 JET 相同的
      线性归一化标量 ``t`` 作为 RGB 三通道（便于导出单通道 PNG，物理量单调）。
    - **不透明度**：同一线性归一化的 **连续** t，高应力更不透明。

    色标来源（优先级）：
    1. vm_vmin_fixed / vm_vmax_fixed（如离线全序列）
    2. 否则若 vm_pct_low / vm_pct_high 均给出：对 vm[:gs_num] 做分位数框定（抗离群）
    3. 否则 vm[:gs_num] 的 min/max

    colormap_steps：阶梯档数，≥2（仅 jet 模式使用）。
    colormap_mode: ``"jet"`` | ``"gray"``（大小写不敏感）。
    """
    device = base_sh_rgb.device
    dtype = base_sh_rgb.dtype
    vm = vm.to(device=device, dtype=dtype)
    base_opacity = base_opacity.to(device=device, dtype=dtype)
    gsn = max(1, int(gs_num))
    n_stairs = max(2, int(colormap_steps))

    vm_core = torch.clamp(vm[:gsn], min=0.0)

    if vm_vmin_fixed is not None and vm_vmax_fixed is not None:
        v_min = vm.new_tensor(float(vm_vmin_fixed))
        v_max = vm.new_tensor(float(vm_vmax_fixed))
        if float((v_max - v_min).detach().cpu()) <= 1e-12:
            v_max = v_min + vm.new_tensor(1.0)
    elif vm_pct_low is not None and vm_pct_high is not None:
        lo = float(vm_pct_low) / 100.0
        hi = float(vm_pct_high) / 100.0
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if hi <= lo:
            lo, hi = 0.0, 1.0
        v_min = torch.quantile(vm_core, lo)
        v_max = torch.quantile(vm_core, hi)
        if float((v_max - v_min).detach().cpu()) <= 1e-12:
            v_max = v_min + vm.new_tensor(1.0)
    else:
        v_min = vm_core.min()
        v_max = vm_core.max()
        if float((v_max - v_min).detach().cpu()) <= 1e-12:
            v_max = v_min + vm.new_tensor(1.0)

    vm_all = torch.clamp(vm, min=0.0)
    denom = v_max - v_min + 1e-12
    t_cont = torch.clamp((vm_all - v_min) / denom, 0.0, 1.0)

    _cm = str(colormap_mode).strip().lower()
    if _cm in ("gray", "grey", "luminance", "mono", "single", "single_channel"):
        rgb_stress = t_cont.unsqueeze(-1).expand(-1, 3)
    else:
        palette = jet_like_palette_linear(n_stairs, device, dtype)
        bin_idx = (t_cont * float(n_stairs)).long().clamp(0, n_stairs - 1)
        rgb_stress = palette[bin_idx]

    w = float(sh_blend)
    colors_precomp = w * base_sh_rgb + (1.0 - w) * rgb_stress
    colors_precomp = torch.clamp(colors_precomp, 0.0, 1.0)

    t_op = t_cont.unsqueeze(1)
    factor = float(opa_floor) + (float(opa_ceil) - float(opa_floor)) * t_op
    stress_op = torch.clamp(base_opacity * factor, 0.02, 0.99)
    return colors_precomp, stress_op


def splat_stress_heatmap_bgr(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    stress_values: np.ndarray,
    height: int,
    width: int,
    visible_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将标量应力 splat 到图像平面（取每个像素上的最大应力以便重叠区域可见）。

    means2d: (P, 3) 或 (P, 2)，光栅化后的屏幕坐标
    radii: (P,) 整数半径（像素），<=0 表示不可见
    stress_values: (P,) 与 means2d 行对齐
    """
    P = int(means2d.shape[0])
    acc = np.zeros((height, width), dtype=np.float32)
    m2 = means2d.detach().cpu().numpy()
    rad = radii.detach().cpu().numpy().reshape(-1)
    if visible_mask is None:
        visible_mask = rad > 0
    else:
        visible_mask = np.asarray(visible_mask) & (rad > 0)

    smax = float(np.nanmax(stress_values[visible_mask])) if np.any(visible_mask) else 0.0
    smin = float(np.nanmin(stress_values[visible_mask])) if np.any(visible_mask) else 0.0
    if smax <= smin + 1e-12:
        smax = smin + 1.0

    for i in range(P):
        if not visible_mask[i]:
            continue
        x = float(m2[i, 0])
        y = float(m2[i, 1])
        r = int(max(1, rad[i]))
        val = float(stress_values[i])
        if not np.isfinite(val):
            continue
        xi = int(round(x))
        yi = int(round(y))
        if xi < -r or yi < -r or xi >= width + r or yi >= height + r:
            continue
        layer = np.zeros((height, width), dtype=np.float32)
        cv2.circle(layer, (xi, yi), r, 1.0, thickness=-1)
        acc = np.maximum(acc, layer * val)

    norm = ((acc - smin) / (smax - smin) * 255.0).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[acc <= 0] = (0, 0, 0)
    return color


def middlebury_flow_visual_bgr(
    flow_xy: np.ndarray,
    valid_mask: np.ndarray,
    max_motion: Optional[float] = None,
    motion_percentile: float = 99.0,
) -> np.ndarray:
    """
    Middlebury 风格：色调=方向，亮度=幅值（HSV→BGR）。flow_xy: (H,W,2)，无效处可为 nan。
    max_motion 为 None 时由 valid 区域内分位数估计饱和幅值。
    """
    f = np.asarray(flow_xy, dtype=np.float32)
    m = np.asarray(valid_mask, dtype=bool) & np.isfinite(f).all(axis=-1)
    if not np.any(m):
        return np.zeros((f.shape[0], f.shape[1], 3), dtype=np.uint8)
    fx = f[:, :, 0]
    fy = f[:, :, 1]
    mag = np.sqrt(fx * fx + fy * fy)
    if max_motion is None or max_motion <= 0:
        mv = float(np.percentile(mag[m], float(motion_percentile)))
        if mv < 1e-6:
            mv = 1.0
        max_motion = mv
    max_motion = max(float(max_motion), 1e-6)
    ang = (np.arctan2(fy, fx) + np.pi) / (2.0 * np.pi)
    hsv = np.zeros((f.shape[0], f.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (np.clip(ang, 0, 1) * 179.0).astype(np.uint8)
    hsv[:, :, 1] = 255
    vm = np.clip(mag / max_motion, 0.0, 1.0)
    hsv[:, :, 2] = (vm * 255.0).astype(np.uint8)
    hsv[~m] = 0
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def hsv_to_rgb_torch(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    h,s,v ∈ [0,1]（h 为色相一周），返回 (..., 3) RGB，与 Middlebury HSV 伪彩一致的可微近似。
    """
    h6 = (h * 6.0).clamp(0.0, 6.0 - 1e-6)
    hi = torch.remainder(torch.floor(h6).long(), 6)
    f = h6 - torch.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)
    r = torch.where(hi == 0, v, r)
    g = torch.where(hi == 0, t, g)
    b = torch.where(hi == 0, p, b)
    r = torch.where(hi == 1, q, r)
    g = torch.where(hi == 1, v, g)
    b = torch.where(hi == 1, p, b)
    r = torch.where(hi == 2, p, r)
    g = torch.where(hi == 2, v, g)
    b = torch.where(hi == 2, t, b)
    r = torch.where(hi == 3, p, r)
    g = torch.where(hi == 3, q, g)
    b = torch.where(hi == 3, v, b)
    r = torch.where(hi == 4, t, r)
    g = torch.where(hi == 4, p, g)
    b = torch.where(hi == 4, v, b)
    r = torch.where(hi == 5, v, r)
    g = torch.where(hi == 5, p, g)
    b = torch.where(hi == 5, q, b)
    return torch.stack([r, g, b], dim=-1)


class FlowGaussianSplatSharedState:
    """
    与主 3DGS 同一仿真高斯：固定随机下采样下标（仅前 gs_num 个 MPM 高斯内）；
    相邻输出帧之间像平面 (u,v) 差分为 (du,dv)，构造 ``colors_precomp``（HSV→RGB：方向/幅值）
    与调权 opacity，走 **第二遍 CUDA 光栅**（黑底），与 ``stress_gaussian`` 管线一致。

    不透明度缩放仍含 opacity^α · (1/(dist+ε))^γ（与旧 CPU splat 权重语义一致）；合成方式为 3DGS 原生 alpha blending，
    而非圆盘内对位移的加权平均，视觉上接近但非同一数学对象。
    """

    def __init__(
        self,
        max_gaussians: int,
        rng_seed: int,
        depth_gamma: float = 1.0,
        depth_eps: float = 1e-2,
        opacity_power: float = 1.0,
    ) -> None:
        self.max_gaussians = max(1, int(max_gaussians))
        self.rng = np.random.RandomState(int(rng_seed))
        self.depth_gamma = float(depth_gamma)
        self.depth_eps = float(depth_eps)
        self.opacity_power = float(opacity_power)
        self.indices: Optional[np.ndarray] = None
        self.prev_uv_by_view: List[Optional[np.ndarray]] = []

    def ensure_indices(self, gs_num: int) -> None:
        if self.indices is not None:
            return
        g = int(gs_num)
        if g < 1:
            self.indices = np.zeros((0,), dtype=np.int64)
            return
        k = min(self.max_gaussians, g)
        self.indices = self.rng.choice(g, size=k, replace=False)
        self.indices.sort()

    def resize_num_views(self, n_views: int) -> None:
        n = int(n_views)
        while len(self.prev_uv_by_view) < n:
            self.prev_uv_by_view.append(None)
        if len(self.prev_uv_by_view) > n:
            self.prev_uv_by_view = self.prev_uv_by_view[:n]

    def build_flow_precomp_for_raster(
        self,
        view_idx: int,
        means2d: torch.Tensor,
        radii: torch.Tensor,
        pos_world: torch.Tensor,
        base_opacity: torch.Tensor,
        camera_center: torch.Tensor,
        image_width: int,
        image_height: int,
        max_motion: Optional[float] = None,
        motion_percentile: float = 99.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为整幅 P 个高斯生成 ``colors_precomp`` (P,3)、``opacities`` (P,1)。
        首帧或 prev 未就绪：全透明；仍更新 prev_uv。非下采样粒子恒为 0。
        """
        P = int(means2d.shape[0])
        device = means2d.device
        dtype = means2d.dtype
        w_img = int(image_width)
        h_img = int(image_height)
        # 光栅 CUDA 路径通常期望 float32
        colors = torch.zeros(P, 3, device=device, dtype=torch.float32)
        opa = torch.zeros(P, 1, device=device, dtype=torch.float32)
        vi = int(view_idx)
        if (
            self.indices is None
            or self.indices.size == 0
            or vi < 0
            or vi >= len(self.prev_uv_by_view)
        ):
            return colors, opa

        idx = torch.as_tensor(self.indices, device=device, dtype=torch.long)
        u = means2d[idx, 0]
        v = means2d[idx, 1]
        rad = radii[idx].float()

        u_np = u.detach().float().cpu().numpy()
        v_np = v.detach().float().cpu().numpy()
        prev = self.prev_uv_by_view[vi]

        if prev is None or prev.shape[0] != u_np.shape[0]:
            self.prev_uv_by_view[vi] = np.stack([u_np, v_np], axis=1).astype(np.float64)
            return colors, opa

        prev_t = torch.as_tensor(prev, device=device, dtype=dtype)
        u = u.to(dtype=dtype)
        v = v.to(dtype=dtype)
        du = u - prev_t[:, 0]
        dv = v - prev_t[:, 1]
        mag = torch.sqrt(du * du + dv * dv + 1e-20)
        valid = (
            (rad > 0)
            & torch.isfinite(u)
            & torch.isfinite(v)
            & (u >= 0)
            & (v >= 0)
            & (u < float(w_img))
            & (v < float(h_img))
        )
        if max_motion is None or float(max_motion) <= 0:
            if bool(valid.any().item()):
                q = float(motion_percentile) / 100.0
                mm = torch.quantile(mag[valid], q)
                mm = torch.clamp(mm, min=torch.tensor(1e-6, device=device, dtype=dtype))
            else:
                mm = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            mm = torch.tensor(float(max_motion), device=device, dtype=dtype)

        val = torch.clamp(mag / mm, 0.0, 1.0)
        hue = (torch.atan2(dv, du) + math.pi) / (2.0 * math.pi)
        sat = torch.ones_like(val)
        rgb_k = hsv_to_rgb_torch(hue, sat, val)
        valid_f = valid.unsqueeze(-1).to(dtype=dtype)
        rgb_k = (rgb_k * valid_f).float()

        pw = pos_world.index_select(0, idx)
        cc = camera_center.view(1, 3).to(device=device, dtype=pw.dtype)
        dist = torch.linalg.norm(pw - cc, dim=1).clamp_min(0.0)
        op_sub = base_opacity.index_select(0, idx).view(-1).clamp(0.0, 1.0)
        dw = torch.pow(op_sub, self.opacity_power) * torch.pow(
            1.0 / (dist + self.depth_eps), self.depth_gamma
        )
        alpha_k = (
            torch.clamp(op_sub * val * dw, 0.0, 1.0) * valid.to(dtype=dtype)
        ).float()

        colors[idx, :] = rgb_k
        opa[idx, 0] = alpha_k

        self.prev_uv_by_view[vi] = np.stack([u_np, v_np], axis=1).astype(np.float64)
        return colors, opa


def isotropic_cov6_batch(
    sigma: float, n: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """协方差 σ²I 的 strip_symmetric 六元组，形状 (n, 6)。"""
    n = max(0, int(n))
    if n == 0:
        return torch.zeros(0, 6, device=device, dtype=dtype)
    s2 = float(sigma) ** 2
    row = torch.tensor(
        [[s2, 0.0, 0.0, s2, 0.0, s2]], device=device, dtype=dtype
    )
    return row.expand(n, -1).contiguous()


def _track_colors_rgb_distinct_cv2(
    n: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """N 条轨迹的鲜艳 RGB ∈ [0,1]，形状 (N,3)。"""
    n = max(1, int(n))
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    step = max(1, 180 // n)
    hsv[:, 0, 0] = (np.arange(n, dtype=np.int32) * step % 180).astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = bgr[:, 0, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(rgb).to(device=device, dtype=dtype)


class TrajectoryGaussianAccumSharedState:
    """
    全物体下采样粒子：固定 MPM 下标；每个输出时刻只构建 **一批** 当前世界坐标微型 3D 高斯
    （可选与上一时刻中点插值，使折线更连贯）。`build_raster_batch` 每仿真输出帧只应调用 **一次**
    （多视角共用同一 batch，仅相机不同），以便 `prev_pos` 与上一帧正确对齐。
    """

    def __init__(
        self,
        num_tracks: int,
        rng_seed: int,
        midpoint_bridge: bool,
    ) -> None:
        self.num_tracks = max(1, int(num_tracks))
        self.rng = np.random.RandomState(int(rng_seed))
        self.midpoint_bridge = bool(midpoint_bridge)
        self.indices: Optional[np.ndarray] = None
        self.prev_pos: Optional[torch.Tensor] = None
        self._rgb_lut: Optional[torch.Tensor] = None

    def ensure_indices(self, gs_num: int) -> None:
        if self.indices is not None:
            return
        g = int(gs_num)
        if g < 1:
            self.indices = np.zeros((0,), dtype=np.int64)
            return
        nt = min(self.num_tracks, g)
        self.indices = self.rng.choice(g, size=nt, replace=False)
        self.indices.sort()

    def build_raster_batch(
        self,
        pos_mpm_world: torch.Tensor,
        sigma_world: float,
        device: torch.device,
        dtype: torch.dtype,
        splat_opacity: float = 1.0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        返回 (means3D, cov6, colors_precomp, opacities)；本帧调用后更新 prev_pos。
        splat_opacity：轨迹点不透明度，通常 1.0 配合较小 σ 更易得到清晰线感。
        """
        g = int(pos_mpm_world.shape[0])
        self.ensure_indices(g)
        if self.indices is None or self.indices.size == 0:
            self.prev_pos = None
            return None
        idx = torch.as_tensor(self.indices, device=device, dtype=torch.long)
        cur = pos_mpm_world[idx].to(dtype=dtype)
        n_cur = int(cur.shape[0])
        if self._rgb_lut is None or self._rgb_lut.shape[0] != n_cur:
            self._rgb_lut = _track_colors_rgb_distinct_cv2(n_cur, device, dtype)

        chunks_pos: List[torch.Tensor] = [cur]
        chunks_rgb: List[torch.Tensor] = [self._rgb_lut]
        if (
            self.midpoint_bridge
            and self.prev_pos is not None
            and self.prev_pos.shape == cur.shape
        ):
            mid = 0.5 * (cur + self.prev_pos.to(dtype=dtype))
            chunks_pos.append(mid)
            chunks_rgb.append(self._rgb_lut)

        self.prev_pos = cur.detach().clone()

        pos_t = torch.cat(chunks_pos, dim=0)
        rgb_t = torch.cat(chunks_rgb, dim=0)
        n_tot = int(pos_t.shape[0])
        opv = float(splat_opacity)
        opv = max(0.01, min(1.0, opv))
        op_t = torch.full((n_tot, 1), opv, device=device, dtype=dtype)
        cov_t = isotropic_cov6_batch(sigma_world, n_tot, device, dtype)
        return pos_t, cov_t, rgb_t, op_t


def write_run_parameters_json(
    path: str,
    payload: Dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_subsampled_world_tracks_ortho_pngs(
    xyz_world: np.ndarray,
    out_dir: str,
    width: int,
    height: int,
    axes: str = "xz",
) -> None:
    """
    不用相机/光栅化：将世界坐标轨迹 (T,N,3) 用正交投影两维画成逐帧 PNG（黑底彩色折线）。

    第 t 张图画每条轨迹从时刻 0 到 t 的折线；全局用全部 T、N 的 min/max 做 2D 归一化到画布。
    axes: 'xy' | 'xz' | 'yz'（在世界系下取哪两个分量作为屏幕轴）。
    """
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax = str(axes).strip().lower()
    if ax not in axis_map:
        raise ValueError(f"axes 应为 xy/xz/yz，得到: {axes!r}")
    a0, a1 = axis_map[ax]

    arr = np.asarray(xyz_world, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"xyz_world 期望 (T,N,3)，得到 {arr.shape}")
    T, N, _ = arr.shape
    w = max(32, int(width))
    h = max(32, int(height))
    margin = 16

    pair = arr[:, :, [a0, a1]].reshape(-1, 2)
    lo = pair.min(axis=0)
    hi = pair.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)

    os.makedirs(out_dir, exist_ok=True)

    def _track_color_bgr(n: int) -> Tuple[int, int, int]:
        hue = int(180.0 * float(n) / float(max(N, 1))) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    for t in range(T):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for n in range(N):
            seg = arr[: t + 1, n, :]
            u = (seg[:, a0] - lo[0]) / span[0] * float(w - 2 * margin) + float(margin)
            v = (seg[:, a1] - lo[1]) / span[1] * float(h - 2 * margin) + float(margin)
            uv = np.stack([u, v], axis=1).astype(np.int32)
            col = _track_color_bgr(n)
            if uv.shape[0] >= 2:
                cv2.polylines(
                    img,
                    [uv.reshape(-1, 1, 2)],
                    isClosed=False,
                    color=col,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            elif uv.shape[0] == 1:
                cv2.circle(img, (int(uv[0, 0]), int(uv[0, 1])), 2, col, -1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"{t:04d}.png"), img)
