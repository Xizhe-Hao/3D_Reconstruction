import os
import json
from typing import Tuple, Optional, List, Dict
from collections import deque

import numpy as np
import torch


def _extract_largest_cluster_points(
    pos: np.ndarray, voxel_size: Optional[float] = None
) -> np.ndarray:
    """
    用体素连通域近似提取最大簇，过滤漂浮散点。
    返回原数组的一个子集；若失败则回退原点集。
    """
    if pos.ndim != 2 or pos.shape[0] <= 1:
        return pos

    raw_min = np.min(pos, axis=0)
    raw_max = np.max(pos, axis=0)
    raw_extent = float(np.max(np.maximum(raw_max - raw_min, 1e-12)))
    if voxel_size is None or float(voxel_size) <= 0.0:
        voxel_size = max(1e-5, 0.03 * raw_extent)
    else:
        voxel_size = max(1e-5, float(voxel_size))

    coords = np.floor((pos - raw_min[None, :]) / voxel_size).astype(np.int32)
    voxel_to_indices: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, c in enumerate(coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        voxel_to_indices.setdefault(key, []).append(idx)

    if len(voxel_to_indices) <= 1:
        return pos

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    unvisited = set(voxel_to_indices.keys())
    best_component: List[Tuple[int, int, int]] = []
    best_points = 0

    while unvisited:
        start = unvisited.pop()
        q = deque([start])
        component = [start]
        point_count = len(voxel_to_indices[start])
        while q:
            vx, vy, vz = q.popleft()
            for dx, dy, dz in neighbor_offsets:
                nb = (vx + dx, vy + dy, vz + dz)
                if nb in unvisited:
                    unvisited.remove(nb)
                    q.append(nb)
                    component.append(nb)
                    point_count += len(voxel_to_indices[nb])
        if point_count > best_points:
            best_points = point_count
            best_component = component

    if best_points <= 0 or best_points >= pos.shape[0]:
        return pos

    keep_indices: List[int] = []
    for key in best_component:
        keep_indices.extend(voxel_to_indices[key])
    if not keep_indices:
        return pos
    return pos[np.asarray(keep_indices, dtype=np.int64)]


def _filter_points_for_bbox(
    pos: np.ndarray,
    material_params: Dict,
    *,
    prefix: str,
) -> np.ndarray:
    """按最大簇过滤后，再用于 bbox/center/edge 判定。"""
    use_cluster = bool(material_params.get(f"{prefix}_bbox_use_largest_cluster", True))
    if not use_cluster:
        return pos
    voxel_size = material_params.get(
        f"{prefix}_bbox_cluster_voxel_size",
        material_params.get("bbox_cluster_voxel_size", None),
    )
    return _extract_largest_cluster_points(pos, voxel_size=voxel_size)


def _compute_quantile_bbox(
    pos: np.ndarray, q_low: float = 0.01, q_high: float = 0.99
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用分位数包围盒抑制离群散点影响。"""
    q_low = max(0.0, min(0.49, float(q_low)))
    q_high = max(0.51, min(1.0, float(q_high)))
    if q_high <= q_low:
        q_low, q_high = 0.01, 0.99
    min_xyz = np.quantile(pos, q_low, axis=0)
    max_xyz = np.quantile(pos, q_high, axis=0)
    center_xyz = 0.5 * (min_xyz + max_xyz)
    extents = np.maximum(max_xyz - min_xyz, 1e-12)
    return min_xyz, max_xyz, center_xyz, extents


def _select_force_axis(extents: np.ndarray, axis_len: int) -> int:
    """在与主轴正交的两轴中，选正对面积更大的施力轴。"""
    cand_axes = [a for a in (0, 1, 2) if a != axis_len]
    assert len(cand_axes) == 2

    def _face_area_for_force_axis(a: int) -> float:
        other_axes = [k for k in (0, 1, 2) if k != a]
        return float(
            max(1e-12, extents[other_axes[0]]) * max(1e-12, extents[other_axes[1]])
        )

    return int(max(cand_axes, key=_face_area_for_force_axis))


def _estimate_local_surface_coordinate(
    pos: np.ndarray,
    *,
    sample_axis: int,
    sample_center: float,
    target_axis: int,
    side: str,
    local_band: float,
    fallback: float,
) -> float:
    """在 sample_axis 的局部带里，估计 target_axis 方向上的局部表面坐标。"""
    mask = np.abs(pos[:, sample_axis] - float(sample_center)) <= float(local_band)
    if not np.any(mask):
        return float(fallback)
    vals = pos[mask, target_axis]
    if str(side).strip().lower() == "max":
        return float(np.max(vals))
    return float(np.min(vals))


def build_press_boundary_conditions(
    mpm_init_pos: torch.Tensor,
    material_params: Dict,
) -> List[Dict]:
    """
    基于初始粒子位置自动构造 “press” 类型的边界条件:

    - 一个全场 bounding_box 约束
    - 物体底部的 surface_collider（近似地面）
    - 物体顶部上方的刚体薄板，通过 cuboid 以恒速向下压缩
    """
    pos = mpm_init_pos.detach().cpu().numpy()
    pos = _filter_points_for_bbox(pos, material_params, prefix="press")
    grid_lim = float(material_params.get("grid_lim", 2.0))

    min_xyz = pos.min(axis=0)
    max_xyz = pos.max(axis=0)
    center_xyz = pos.mean(axis=0)

    min_z = float(min_xyz[2])
    max_z = float(max_xyz[2])
    extent_z = max(1e-12, float(max_z - min_z))
    center_x = float(center_xyz[0])
    center_y = float(center_xyz[1])

    # 底部地面：放在物体最底部略微下方，法向指向 +z
    dz_floor = 0.01 * grid_lim
    floor_point = [center_x, center_y, min_z - dz_floor]
    floor_normal = [0.0, 0.0, 1.0]

    # 顶部刚体薄板：覆盖整个仿真区域 [0, grid_lim]^2
    dz_plate = 0.02 * grid_lim
    plate_z = max_z + dz_plate
    plate_point = [grid_lim * 0.5, grid_lim * 0.5, plate_z]
    plate_size = [grid_lim * 0.5, grid_lim * 0.5, 0.05 * grid_lim]
    # 压缩速度：3 秒内压缩 z 方向最大宽度的 70%，剩余 1 秒撤出力（在主流程中保证总时长为 4 秒）
    press_duration_s = 3.0
    press_ratio = 0.7
    plate_speed = press_ratio * extent_z / press_duration_s
    plate_velocity = [0.0, 0.0, -float(plate_speed)]
    plate_end_time = press_duration_s

    bc_list: List[Dict] = [
        {"type": "bounding_box"},
        {
            "type": "surface_collider",
            "point": floor_point,
            "normal": floor_normal,
            "surface": "slip",
            "friction": 0.5,
            "start_time": 0.0,
            "end_time": 1e3,
        },
        {
            "type": "cuboid",
            "point": plate_point,
            "size": plate_size,
            "velocity": plate_velocity,
            "start_time": 0.0,
            "end_time": plate_end_time,
            "reset": 0,
        },
    ]

    return bc_list


def build_drop_boundary_conditions(
    mpm_init_pos: torch.Tensor,
    material_params: Dict,
) -> List[Dict]:
    """
    自动构造 “drop” 类型边界条件:

    - 一个全场 bounding_box 约束
    - 底部 surface_collider：取物体粒子 z 最小值 min_z，地面在 min_z 下方固定 0.25 处（即下落净空 0.25）
    - 一个 very short 的 particle_impulse（用于初始化扰动/释放）
    """
    pos = mpm_init_pos.detach().cpu().numpy()
    pos = _filter_points_for_bbox(pos, material_params, prefix="drop")
    min_xyz = pos.min(axis=0)
    center_xyz = pos.mean(axis=0)
    min_z = float(min_xyz[2])
    drop_gap = 0.25
    floor_z = min_z - drop_gap
    floor_point = [float(center_xyz[0]), float(center_xyz[1]), float(floor_z)]
    floor_normal = [0.0, 0.0, 1.0]

    bc_list: List[Dict] = [
        {"type": "bounding_box"},
        {
            "type": "surface_collider",
            "point": floor_point,
            "normal": floor_normal,
            "surface": "slip",
            "friction": 0.5,
            "start_time": 0.0,
            "end_time": 1e3,
        },
        {
            "type": "particle_impulse",
            "force": [0.0, 0.0, 1.0],
            "num_dt": 2,
            "start_time": 0.0,
        },
    ]

    return bc_list


def build_shear_boundary_conditions(
    mpm_init_pos: torch.Tensor,
    material_params: Dict,
) -> List[Dict]:
    """
    自动构造 “shear” 类型边界条件（仿照 bend 的选轴逻辑，保留中心固定块）：

    - 一个全场 bounding_box
    - 先取最长轴 axis_len（与 bend 一致）
    - 在与 axis_len 正交的两轴中，选“正对面积更大”的轴作为 axis_force（与 bend 一致）
    - 在 axis_len 中部保留一个固定 cuboid，作为中间约束
    - 两块施力 cuboid 分布在 axis_len 两端，并分别位于 axis_force 的正/负两侧（贴近局部表面，避免推空）
    - 两块 cuboid 沿 axis_force 相反方向运动，形成剪切

    速度规则：参考 bend，3 秒内两端在 axis_force 上的相对推进量按该方向宽度的固定比例给出；
    若物体在该方向过薄，则单块位移至少为一个固定下限。
    """
    pos = mpm_init_pos.detach().cpu().numpy()
    pos = _filter_points_for_bbox(pos, material_params, prefix="shear")
    grid_lim = float(material_params.get("grid_lim", 2.0))
    q_low = float(material_params.get("shear_bbox_q_low", 0.01))
    q_high = float(material_params.get("shear_bbox_q_high", 0.99))
    min_xyz, max_xyz, center_xyz, extents = _compute_quantile_bbox(
        pos, q_low=q_low, q_high=q_high
    )

    # 1) bend 同款主轴：最长轴
    axis_len = int(np.argmax(extents))
    extent_len = float(extents[axis_len])

    # 2) bend 同款施力轴：在与 axis_len 正交的两轴中，选“正对面积更大”的轴
    axis_force = _select_force_axis(extents, axis_len)
    axis_other = [a for a in (0, 1, 2) if a not in (axis_len, axis_force)][0]
    extent_force = float(extents[axis_force])

    # 3) 中间固定块：仿照 bend，在 axis_len 中部保留一个薄片约束
    mid_plate_point = [float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])]
    mid_plate_size = [0.0, 0.0, 0.0]
    for a in range(3):
        if a == axis_len:
            mid_plate_size[a] = 0.05 * extent_len
        else:
            mid_plate_size[a] = 0.5 * grid_lim

    # 4) 施力块尺寸：仿照 bend
    # - force 方向很薄
    # - other 方向覆盖较大区域
    # - length 方向为端部薄片长度（不是整段覆盖）
    clamp_size = [0.0, 0.0, 0.0]
    # force 方向厚度做自适应：避免物体在该轴很薄时两块施力板重叠
    # 常规目标厚度沿用 bend；若物体太薄则按该轴宽度收缩
    force_thickness_target = 0.05 * grid_lim
    clamp_size[axis_force] = min(force_thickness_target, 0.8 * extent_force)
    clamp_size[axis_force] = max(clamp_size[axis_force], 1e-4 * grid_lim)
    end_len_size = 0.05 * extent_len
    clamp_size[axis_len] = end_len_size
    clamp_size[axis_other] = 0.5 * grid_lim

    # 5) 仿照 bend，在主轴两端布置两块施力板，避免两块都堆在中间
    gap = 0.02 * extent_len
    mid_thickness = mid_plate_size[axis_len]
    margin = 0.5 * (mid_thickness + end_len_size) + gap
    min_len = float(min_xyz[axis_len])
    max_len = float(max_xyz[axis_len])
    p_len_neg = min_len + margin
    p_len_pos = max_len - margin
    if p_len_neg >= p_len_pos:
        p_len_neg = min_len + 0.15 * extent_len
        p_len_pos = max_len - 0.15 * extent_len

    # 6) 在 axis_force 两侧贴近主簇 bbox 两侧表面
    # 注意：solver 中 cuboid.size 表示“半尺寸/半边长”，因此若把中心放到表面内侧，
    # 即使中心一正一负，体积仍可能大量重叠。这里统一改为“外侧贴面”放置。
    force_half_thickness = 0.5 * clamp_size[axis_force]
    contact_eps = 0.002 * grid_lim

    center_neg = center_xyz.copy()
    center_pos = center_xyz.copy()
    # 对角放置：length 负端配 force 正侧；length 正端配 force 负侧
    center_neg[axis_len] = p_len_neg
    center_pos[axis_len] = p_len_pos
    # length 负端放在 force 正侧外部；length 正端放在 force 负侧外部
    center_neg[axis_force] = float(max_xyz[axis_force]) + force_half_thickness - contact_eps
    center_pos[axis_force] = float(min_xyz[axis_force]) - force_half_thickness + contact_eps

    clamp_point_neg = [
        float(center_neg[0]),
        float(center_neg[1]),
        float(center_neg[2]),
    ]
    clamp_point_pos = [
        float(center_pos[0]),
        float(center_pos[1]),
        float(center_pos[2]),
    ]

    shear_duration_s = 3.0
    # 参考 bend：速度大小按施力轴宽度的固定比例 / 持续时间给出
    shear_ratio = float(material_params.get("shear_ratio", 0.5))
    shear_min_displacement = float(
        material_params.get("shear_min_displacement", 0.08)
    )
    shear_displacement = max(shear_ratio * extent_force, shear_min_displacement)
    shear_speed = shear_displacement / shear_duration_s

    # 7) 速度方向：沿 axis_force，一正一负（force 正侧向负推，force 负侧向正推）
    vel_pos = [0.0, 0.0, 0.0]
    vel_neg = [0.0, 0.0, 0.0]
    vel_pos[axis_force] = float(shear_speed)
    vel_neg[axis_force] = -float(shear_speed)

    bc_list: List[Dict] = [
        {"type": "bounding_box"},
        {
            "type": "cuboid",
            "point": mid_plate_point,
            "size": mid_plate_size,
            "velocity": [0.0, 0.0, 0.0],
            # 中间固定块全程保留，仅两侧施力块在 3 秒后撤销
            "start_time": 0.0,
            "end_time": 1e3,
            "reset": 0,
        },
        {
            "type": "cuboid",
            "point": clamp_point_pos,
            "size": clamp_size,
            "velocity": vel_pos,
            "start_time": 0.0,
            "end_time": float(shear_duration_s),
            "reset": 0,
        },
        {
            "type": "cuboid",
            "point": clamp_point_neg,
            "size": clamp_size,
            "velocity": vel_neg,
            "start_time": 0.0,
            "end_time": float(shear_duration_s),
            "reset": 0,
        },
    ]

    return bc_list


def build_stretch_boundary_conditions(
    mpm_init_pos: torch.Tensor,
    material_params: Dict,
) -> List[Dict]:
    """
    自动构造 “stretch” 类型边界条件（两端夹具拉伸）：

    - 一个全场 bounding_box
    - 在物体 y 轴两端各放一个薄片刚体夹具
      - y_min 端夹具向 -y 匀速移动
      - y_max 端夹具向 +y 匀速移动

    速度规则：3 秒内两夹具“总拉伸距离”（两端位移之和）为物体 y 向总宽度的 30%
    即每端位移为 15% * width_y
    """
    pos = mpm_init_pos.detach().cpu().numpy()
    pos = _filter_points_for_bbox(pos, material_params, prefix="stretch")
    grid_lim = float(material_params.get("grid_lim", 2.0))

    min_xyz = pos.min(axis=0)
    max_xyz = pos.max(axis=0)
    center_xyz = pos.mean(axis=0)

    min_y = float(min_xyz[1])
    max_y = float(max_xyz[1])
    extent_y = max(1e-12, float(max_y - min_y))

    center_x = float(center_xyz[0])
    center_z = float(center_xyz[2])

    # 两端夹具沿 y 方向略微“咬入”物体，使两侧都能真实夹住而不是完全在外侧滑动。
    # 注意：solver 中 cuboid.size 是半尺寸，这里默认直接控制写入 boundary_conditions.json
    # 的 y 向 size 值。默认 0.02，对应总厚度约 0.04。
    plate_thickness_y = float(material_params.get("stretch_plate_half_thickness_y", 0.02))
    plate_thickness_y = max(1e-4 * grid_lim, plate_thickness_y)
    # 让薄片中心落在物体内部一点（约半个板厚），保证与粒子有足够重叠
    inset = 0.5 * plate_thickness_y
    clamp_y_min = min_y + inset
    clamp_y_max = max_y - inset

    clamp_point_min = [center_x, float(clamp_y_min), center_z]
    clamp_point_max = [center_x, float(clamp_y_max), center_z]

    # 薄片：在 y 方向很薄
    clamp_size = [grid_lim * 0.5, plate_thickness_y, grid_lim * 0.5]

    stretch_duration_s = 3.0
    total_ratio = 0.3
    per_end_ratio = 0.5 * total_ratio  # 0.15
    stretch_speed = per_end_ratio * extent_y / stretch_duration_s

    bc_list: List[Dict] = [
        {"type": "bounding_box"},
        {
            "type": "cuboid",
            "point": clamp_point_min,
            "size": clamp_size,
            "velocity": [0.0, -float(stretch_speed), 0.0],
            "start_time": 0.0,
            "end_time": float(stretch_duration_s),
            "reset": 0,
        },
        {
            "type": "cuboid",
            "point": clamp_point_max,
            "size": clamp_size,
            "velocity": [0.0, float(stretch_speed), 0.0],
            "start_time": 0.0,
            "end_time": float(stretch_duration_s),
            "reset": 0,
        },
    ]

    return bc_list


def build_bend_boundary_conditions(
    mpm_init_pos: torch.Tensor,
    material_params: Dict,
) -> List[Dict]:
    """
    自动构造 “bend” 类型边界条件：

    - 先根据包围盒确定物体最长边所在轴（x/y/z 中 extent 最大的轴），记为 length 轴
    - 在 length 轴中部放置一个刚性薄片 cuboid，速度为 0（仅作为中段刚体支点），
      该 cuboid 是「垂直于 length 轴的平面区域」，在 length 方向上的厚度为该轴长度的 5%
    - 在 force 轴负侧（物体外侧）放置两块刚性 cuboid，沿 +force 方向同向推进产生碰撞
      - force 轴按 xyz 轮换对称规则选取，保证 force ⟂ length
        例如：length=z(2) -> force=y(1)；length=y(1) -> force=x(0)；length=x(0) -> force=z(2)
      - 两块 cuboid 在 length 轴上对称分布（“一正一负”）
      - cuboid 初始不贯穿物体：整体从 force 轴负侧外侧开始向正侧推进

    速度规则：参考当前 shear/bend 统一逻辑，3 秒内单块在施力轴方向上的位移
    按该方向宽度的固定比例给出；若物体在该方向过薄，则单块位移至少为固定下限。
    """
    pos = mpm_init_pos.detach().cpu().numpy()
    pos = _filter_points_for_bbox(pos, material_params, prefix="bend")
    grid_lim = float(material_params.get("grid_lim", 2.0))
    q_low = float(material_params.get("bend_bbox_q_low", 0.01))
    q_high = float(material_params.get("bend_bbox_q_high", 0.99))
    min_xyz, max_xyz, center_xyz, extents = _compute_quantile_bbox(
        pos, q_low=q_low, q_high=q_high
    )
    # 找到最长轴（弯折方向的“梁长”）：0=x, 1=y, 2=z
    axis_len = int(np.argmax(extents))

    # 中部刚性片：在 length 轴中部放一个 cuboid，速度为 0（固定“杠杆支点”）
    # 该 cuboid 的法向为 length 轴，其在 length 方向的厚度为 length_extent 的 5%
    center_x, center_y, center_z = center_xyz.tolist()
    mid_plate_point = [center_x, center_y, center_z]

    mid_plate_size = [0.0, 0.0, 0.0]
    length_extent = float(max(1e-12, extents[axis_len]))
    # 在 length 方向上厚度为 5% * length_extent，其余方向覆盖较大区域
    for i in range(3):
        if i == axis_len:
            mid_plate_size[i] = 0.05 * length_extent
        else:
            mid_plate_size[i] = 0.5 * grid_lim

    # 选择施力轴：在与 length 轴正交的两种方向中，选择“面向物体的面积区域更大”的一种。
    # 对于候选施力轴 a，其与物体接触的“面”近似为垂直于 a 的截面，其面积 ~ extents[other1] * extents[other2]，
    # 其中 other1/other2 为除 a 外的两个轴。我们选该面积更大的 a。
    axis_force = _select_force_axis(extents, axis_len)
    extent_force = float(max(1e-12, extents[axis_force]))

    bend_duration_s = 3.0
    bend_ratio = 0.5
    bend_min_displacement = float(
        material_params.get("bend_min_displacement", 0.08)
    )
    bend_displacement = max(bend_ratio * extent_force, bend_min_displacement)
    speed_force = bend_displacement / bend_duration_s

    # 两块施力 cuboid 在 length 轴上尽量靠近两端（远离中段固定薄片）
    # margin 同时考虑：中段薄片厚度、施力薄片自身 length 尺寸、以及一个小间隙
    mid_thickness = 0.05 * length_extent
    end_len_size = 0.05 * length_extent  # 施力部分也做成薄片：length 方向不做长条
    gap = 0.02 * length_extent
    margin = 0.5 * (mid_thickness + end_len_size) + gap
    min_len = float(min_xyz[axis_len])
    max_len = float(max_xyz[axis_len])
    p_len_neg = min_len + margin
    p_len_pos = max_len - margin
    # 防御：若物体太短导致 margin 过大，则退化为更保守的两端位置
    if p_len_neg >= p_len_pos:
        p_len_neg = min_len + 0.15 * length_extent
        p_len_pos = max_len - 0.15 * length_extent

    # 端部刚性片尺寸：
    # - force 方向很薄（避免贯穿）
    # - length 方向占一段长度（形成“沿 length 分布的两个力”）
    # - 剩余轴覆盖较大区域
    axis_other = [a for a in (0, 1, 2) if a not in (axis_len, axis_force)][0]
    end_plate_size = [0.0, 0.0, 0.0]
    end_plate_size[axis_force] = 0.05 * grid_lim
    end_plate_size[axis_other] = 0.5 * grid_lim
    end_plate_size[axis_len] = end_len_size

    # 施力 cuboid 的初始位置：沿 +force 推进方向，起始时贴近物体表面，避免“推空”。
    # 做法：在每个端部位置（axis_len=p_len_*）附近选一段粒子，估计该局部在 force 轴上的最小坐标；
    # 将施力板中心放到该表面外侧半个板厚处（允许极小咬入 epsilon 以确保首帧接触）。
    force_half_thickness = 0.5 * end_plate_size[axis_force]
    contact_eps = 0.002 * grid_lim
    global_force_min = float(min_xyz[axis_force])
    extent_len = float(max(1e-12, max_len - min_len))
    local_band = max(0.04 * extent_len, 0.01 * grid_lim)

    local_force_min_a = _estimate_local_surface_coordinate(
        pos,
        sample_axis=axis_len,
        sample_center=p_len_pos,
        target_axis=axis_force,
        side="min",
        local_band=local_band,
        fallback=global_force_min,
    )
    local_force_min_b = _estimate_local_surface_coordinate(
        pos,
        sample_axis=axis_len,
        sample_center=p_len_neg,
        target_axis=axis_force,
        side="min",
        local_band=local_band,
        fallback=global_force_min,
    )
    force_start_a = local_force_min_a - force_half_thickness + contact_eps
    force_start_b = local_force_min_b - force_half_thickness + contact_eps

    # 施力方向：沿 +force 方向推进（从负侧外侧推向物体）
    vel_force = [0.0, 0.0, 0.0]
    vel_force[axis_force] = float(speed_force)

    end_plate_a = center_xyz.copy()
    end_plate_b = center_xyz.copy()
    end_plate_a[axis_force] = force_start_a
    end_plate_b[axis_force] = force_start_b
    end_plate_a[axis_len] = p_len_pos
    end_plate_b[axis_len] = p_len_neg

    bc_list: List[Dict] = [
        {"type": "bounding_box"},
        {
            # 中部刚性片的实体体积，用于碰撞（速度为 0）
            "type": "cuboid",
            "point": mid_plate_point,
            "size": mid_plate_size,
            "velocity": [0.0, 0.0, 0.0],
            "start_time": 0.0,
            # 中间固定块保持全程有效；仅两侧施力块在 bend_duration_s 后撤销
            "end_time": 1e3,
            "reset": 0,
        },
        {
            # 两块施力 cuboid：同向推进，在 length 轴上对称分布
            "type": "cuboid",
            "point": end_plate_a.tolist(),
            "size": end_plate_size,
            "velocity": vel_force,
            "start_time": 0.0,
            "end_time": float(bend_duration_s),
            "reset": 0,
        },
        {
            "type": "cuboid",
            "point": end_plate_b.tolist(),
            "size": end_plate_size,
            "velocity": vel_force,
            "start_time": 0.0,
            "end_time": float(bend_duration_s),
            "reset": 0,
        },
    ]

    return bc_list


def save_boundary_condition_info(bc_params, material_params, save_dir: str) -> None:
    """
    将当前边界条件和重力信息导出为 JSON，便于分析与复现。
    """
    os.makedirs(save_dir, exist_ok=True)

    bc_summary = {
        "gravity_g": material_params.get("g", [0.0, 0.0, 0.0]),
        "boundary_conditions": [],
    }

    if bc_params is None:
        bc_params = []

    for i, bc in enumerate(bc_params):
        bc_item = {
            "index": i,
            "raw": bc,
        }

        if "surface_collider" in bc:
            sc = bc["surface_collider"]
            bc_item["surface_collider"] = {
                "point": sc.get("point", None),
                "normal": sc.get("normal", None),
                "surface": sc.get("surface", None),
                "friction": sc.get("friction", None),
                "start_time": sc.get("start_time", None),
                "end_time": sc.get("end_time", None),
                "size": sc.get("size", None),
            }

        if "set_velocity_on_cuboid" in bc:
            sv = bc["set_velocity_on_cuboid"]
            bc_item["set_velocity_on_cuboid"] = {
                "point": sv.get("point", None),
                "size": sv.get("size", None),
                "velocity": sv.get("velocity", None),
                "direction": sv.get("velocity", None),
                "start_time": sv.get("start_time", None),
                "end_time": sv.get("end_time", None),
                "reset": sv.get("reset", None),
            }

        if "enforce_particle_translation" in bc:
            ep = bc["enforce_particle_translation"]
            bc_item["enforce_particle_translation"] = {
                "point": ep.get("point", None),
                "size": ep.get("size", None),
                "velocity": ep.get("velocity", None),
                "direction": ep.get("velocity", None),
                "start_time": ep.get("start_time", None),
                "end_time": ep.get("end_time", None),
            }

        bc_summary["boundary_conditions"].append(bc_item)

    with open(
        os.path.join(save_dir, "boundary_conditions.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(bc_summary, f, indent=2, ensure_ascii=False)


def save_external_force_info(bc_params, material_params, save_dir: str) -> None:
    """
    导出外力 / 速度驱动边界条件信息（重力 + 速度边界的模长与方向）。
    """
    os.makedirs(save_dir, exist_ok=True)

    g = material_params.get("g", [0.0, 0.0, 0.0])
    g_arr = np.array(g, dtype=float)
    g_norm = float(np.linalg.norm(g_arr))
    g_dir = (g_arr / g_norm).tolist() if g_norm > 1e-12 else [0.0, 0.0, 0.0]

    info: Dict = {
        "gravity": {
            "g": g,
            "magnitude": g_norm,
            "direction": g_dir,
            "type": "body_force_acceleration",
        },
        "velocity_driven_boundary_conditions": [],
    }

    if bc_params is None:
        bc_params = []

    for i, bc in enumerate(bc_params):
        if str(bc.get("type", "")).strip().lower() == "cuboid":
            v = np.array(bc.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
            mag = float(np.linalg.norm(v))
            if mag > 1e-12:
                direction = (v / mag).tolist()
                info["velocity_driven_boundary_conditions"].append(
                    {
                        "index": i,
                        "type": "cuboid",
                        "point": bc.get("point", None),
                        "size": bc.get("size", None),
                        "velocity": v.tolist(),
                        "magnitude": mag,
                        "direction": direction,
                        "start_time": bc.get("start_time", None),
                        "end_time": bc.get("end_time", None),
                    }
                )
        if "set_velocity_on_cuboid" in bc:
            sv = bc["set_velocity_on_cuboid"]
            v = np.array(sv.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
            mag = float(np.linalg.norm(v))
            direction = (v / mag).tolist() if mag > 1e-12 else [0.0, 0.0, 0.0]
            info["velocity_driven_boundary_conditions"].append(
                {
                    "index": i,
                    "type": "set_velocity_on_cuboid",
                    "point": sv.get("point", None),
                    "size": sv.get("size", None),
                    "velocity": v.tolist(),
                    "magnitude": mag,
                    "direction": direction,
                    "start_time": sv.get("start_time", None),
                    "end_time": sv.get("end_time", None),
                }
            )

        if "enforce_particle_translation" in bc:
            ep = bc["enforce_particle_translation"]
            v = np.array(ep.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
            mag = float(np.linalg.norm(v))
            direction = (v / mag).tolist() if mag > 1e-12 else [0.0, 0.0, 0.0]
            info["velocity_driven_boundary_conditions"].append(
                {
                    "index": i,
                    "type": "enforce_particle_translation",
                    "point": ep.get("point", None),
                    "size": ep.get("size", None),
                    "velocity": v.tolist(),
                    "magnitude": mag,
                    "direction": direction,
                    "start_time": ep.get("start_time", None),
                    "end_time": ep.get("end_time", None),
                }
            )

    with open(
        os.path.join(save_dir, "external_actions.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def save_initial_force_mask_and_arrow_info(
    bc_params,
    mpm_init_pos: torch.Tensor,
    save_dir: str,
    *,
    contact_band_ratio: float = 0.05,
) -> Dict:
    """
    导出“初始受力区域 mask + 方向箭头”标签（基于速度驱动 cuboid）。

    输出：
    - meta/initial_force_mask_arrow.json：可读摘要（每个施力 cuboid 的方向、接触点等）
    - meta/initial_force_mask_arrow.npz：粒子级 mask（union + 每个 cuboid 的索引）

    说明：
    - 仅处理带 velocity 且速度模长 > 0 的 cuboid 边界（set_velocity_on_cuboid / enforce_particle_translation）
    - “初始接触区域”通过几何近似求得：在 cuboid 速度主轴正前方（或负前方）、
      同时落在 cuboid 其它两个轴 footprint 内，且距离接触面的点。
    """
    os.makedirs(save_dir, exist_ok=True)
    if bc_params is None:
        bc_params = []

    pos = mpm_init_pos.detach().cpu().numpy().astype(np.float64)
    n = int(pos.shape[0])
    if n <= 0:
        payload = {
            "num_particles": 0,
            "num_force_cuboids": 0,
            "force_cuboids": [],
            "union_mask_count": 0,
        }
        with open(os.path.join(save_dir, "initial_force_mask_arrow.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        np.savez_compressed(
            os.path.join(save_dir, "initial_force_mask_arrow.npz"),
            union_indices=np.array([], dtype=np.int64),
        )
        return {
            "union_indices": np.array([], dtype=np.int64),
            "force_cuboids": [],
            "num_particles": 0,
        }

    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    obj_extent = np.maximum(pos_max - pos_min, 1e-12)

    force_items: List[Dict] = []
    union_mask = np.zeros(n, dtype=bool)
    first_contact_mask = np.zeros(n, dtype=bool)
    per_item_indices: List[np.ndarray] = []

    def _extract_cuboid_velocity_bc(bc: Dict) -> Optional[Tuple[str, Dict]]:
        # 兼容当前自动 BC 生成格式：{"type":"cuboid","point":...,"size":...,"velocity":...}
        if str(bc.get("type", "")).strip().lower() == "cuboid":
            return ("cuboid", bc)
        if "set_velocity_on_cuboid" in bc:
            return ("set_velocity_on_cuboid", bc["set_velocity_on_cuboid"])
        if "enforce_particle_translation" in bc:
            return ("enforce_particle_translation", bc["enforce_particle_translation"])
        return None

    for bc_i, bc in enumerate(bc_params):
        parsed = _extract_cuboid_velocity_bc(bc)
        if parsed is None:
            continue
        bc_type, c = parsed
        p = np.array(c.get("point", [0.0, 0.0, 0.0]), dtype=np.float64)
        s = np.array(c.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)
        v = np.array(c.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
        v_norm = float(np.linalg.norm(v))
        if not np.isfinite(v_norm) or v_norm <= 1e-12:
            continue

        dir_vec = v / v_norm
        axis = int(np.argmax(np.abs(dir_vec)))
        sign = 1.0 if dir_vec[axis] >= 0.0 else -1.0
        other_axes = [a for a in (0, 1, 2) if a != axis]

        # footprint：只保留落在 cuboid 横向截面附近的粒子
        in_footprint = np.ones(n, dtype=bool)
        for a in other_axes:
            tol = max(1e-8, 0.05 * obj_extent[a])
            in_footprint &= np.abs(pos[:, a] - p[a]) <= (s[a] + tol)

        # 接触面（沿速度方向的“前表面”）
        if sign > 0:
            face = p[axis] + s[axis]
            dist = pos[:, axis] - face
            ahead = dist >= 0.0
        else:
            face = p[axis] - s[axis]
            dist = face - pos[:, axis]
            ahead = dist >= 0.0

        cand = in_footprint & ahead
        if not np.any(cand):
            # 退化：只看 footprint，取沿速度主轴最靠近前表面的点
            cand = in_footprint.copy()

        if np.any(cand):
            d = np.abs(dist[cand])
            band = max(1e-8, float(contact_band_ratio) * float(obj_extent[axis]))
            choose = d <= band
            cand_idx = np.where(cand)[0]
            if np.any(choose):
                sel_idx = cand_idx[choose]
            else:
                # 若 band 内为空，至少保留最邻近一小批点
                k = min(256, cand_idx.shape[0])
                ord_idx = np.argsort(d)[:k]
                sel_idx = cand_idx[ord_idx]
        else:
            sel_idx = np.array([], dtype=np.int64)

        # “第一批接触/最先被驱动”粒子：在候选集中取最小沿主轴距离的窄层
        first_idx = np.array([], dtype=np.int64)
        if np.any(cand):
            cand_idx = np.where(cand)[0]
            d = np.abs(dist[cand])
            d_min = float(np.min(d))
            first_band = max(1e-8, 0.01 * float(obj_extent[axis]))
            choose_first = d <= (d_min + first_band)
            if np.any(choose_first):
                first_idx = cand_idx[choose_first]
            else:
                kf = min(128, cand_idx.shape[0])
                first_idx = cand_idx[np.argsort(d)[:kf]]

        if sel_idx.size > 0:
            union_mask[sel_idx] = True
            contact_center = pos[sel_idx].mean(axis=0).tolist()
        else:
            contact_center = [float(p[0]), float(p[1]), float(p[2])]
        if first_idx.size > 0:
            first_contact_mask[first_idx] = True

        per_item_indices.append(sel_idx.astype(np.int64))
        force_items.append(
            {
                "bc_index": int(bc_i),
                "bc_type": bc_type,
                "point": [float(x) for x in p.tolist()],
                "size": [float(x) for x in s.tolist()],
                "velocity": [float(x) for x in v.tolist()],
                "arrow_direction": [float(x) for x in dir_vec.tolist()],
                "arrow_magnitude": float(v_norm),
                "contact_center_world": [float(x) for x in contact_center],
                "mask_count": int(sel_idx.size),
                "first_contact_count": int(first_idx.size),
            }
        )

    union_indices = np.where(union_mask)[0].astype(np.int64)
    first_contact_indices = np.where(first_contact_mask)[0].astype(np.int64)

    payload = {
        "num_particles": int(n),
        "num_force_cuboids": int(len(force_items)),
        "force_cuboids": force_items,
        "union_mask_count": int(union_indices.size),
        "first_contact_count": int(first_contact_indices.size),
        "note": "mask 为初始时刻近似接触区域；arrow_direction 来自 cuboid velocity 归一化向量",
    }
    with open(
        os.path.join(save_dir, "initial_force_mask_arrow.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    npz_payload: Dict[str, np.ndarray] = {
        "union_indices": union_indices,
        "first_contact_indices": first_contact_indices,
        "num_particles": np.array([n], dtype=np.int64),
    }
    for i, arr in enumerate(per_item_indices):
        npz_payload[f"cuboid_{i:02d}_indices"] = arr.astype(np.int64)
    np.savez_compressed(
        os.path.join(save_dir, "initial_force_mask_arrow.npz"),
        **npz_payload,
    )
    return {
        "union_indices": union_indices,
        "first_contact_indices": first_contact_indices,
        "force_cuboids": force_items,
        "num_particles": int(n),
    }


def setup_field_output_dirs(
    base_output_path: Optional[str],
    output_deformation: bool,
    output_stress: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """
    准备形变场 / 应力场输出目录。
    """
    deformation_dir: Optional[str] = None
    stress_dir: Optional[str] = None

    if base_output_path is None:
        return deformation_dir, stress_dir

    if output_deformation:
        deformation_dir = os.path.join(base_output_path, "deformation_field")
        os.makedirs(deformation_dir, exist_ok=True)

    if output_stress:
        stress_dir = os.path.join(base_output_path, "stress_field")
        os.makedirs(stress_dir, exist_ok=True)

    return deformation_dir, stress_dir


def save_fields_for_frame(
    mpm_solver,
    frame_id: int,
    deformation_dir: Optional[str],
    stress_dir: Optional[str],
    output_deformation: bool,
    output_stress: bool,
) -> None:
    """
    根据开关，在指定帧导出形变场 / 应力场。
    """
    if output_deformation and deformation_dir is not None:
        mpm_solver.save_deformation_field(
            frame_id=frame_id,
            save_dir=deformation_dir,
            include_F_trial=True,
        )

    if output_stress and stress_dir is not None:
        mpm_solver.save_stress_field(
            frame_id=frame_id,
            save_dir=stress_dir,
            use_F="trial",
            save_J=True,
        )


def write_stress_pcd_camera_meta_json(
    path: str,
    *,
    viewpoint_center_worldspace: np.ndarray,
    observant_coordinates: np.ndarray,
    num_views: int,
    init_azimuthm: float,
    init_elevation: float,
    init_radius: float,
    model_path: Optional[str] = None,
    default_camera_index: int = -1,
    move_camera: bool = False,
    delta_a: float = 0.0,
    delta_e: float = 0.0,
    delta_r: float = 0.0,
    field_output_interval: int = 1,
    fov_vertical_deg: float = 60.0,
    fov_horizontal_deg: float = 60.0,
    width_hint: int = 800,
    height_hint: int = 800,
    mpm_to_world: Optional[Dict] = None,
    mpm_space_viewpoint_center: Optional[List[float]] = None,
) -> None:
    """
    写入与 modified_simulation 中「方位角均匀分布 + synthetic 轨道相机」一致的参数，
    供 my_utils/visualize_fields.py 离线渲染多视角点云应力图（与 Gaussian 合成相机大致对齐）。

    camera_motion（delta_* / move_camera）供离线按仿真帧号重算 eye，与 get_camera_view(..., current_frame=frame) 一致。

    mpm_to_world：与 undo_all_transforms 一致，将 deformation_field 中 MPM 坐标变到与 3DGS 渲染相同的世界系。

    说明：若存在 cameras.json 且 default_camera_index > -1，Gaussian 侧可能固定使用 COLMAP 相机，
    此时本 meta 中的环绕视角与 RGB 渲染不一定一致，见写入的 warnings 字段。
    """
    from utils.camera_view_utils import get_camera_position_and_rotation

    vc = np.asarray(viewpoint_center_worldspace, dtype=np.float64).reshape(3)
    obs = np.asarray(observant_coordinates, dtype=np.float64).reshape(3, 3)
    up = obs[:, 2].astype(np.float64)
    nup = np.linalg.norm(up)
    if nup < 1e-12:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up = up / nup

    nv = max(1, int(num_views))
    views: List[Dict] = []
    az_base = float(init_azimuthm)
    el_base = float(init_elevation)
    rad = float(init_radius)

    for k in range(nv):
        az = (az_base + 360.0 * k / nv) % 360.0
        el = el_base
        name = f"az{int(round(az))}_el{int(round(el))}"
        eye, _R = get_camera_position_and_rotation(az, el, rad, vc, obs)
        eye = np.asarray(eye, dtype=np.float64).reshape(3)
        views.append(
            {
                "name": name,
                "azimuth": az,
                "elevation": el,
                "eye": eye.tolist(),
            }
        )

    warnings: List[str] = []
    if model_path:
        cj = os.path.join(model_path, "cameras.json")
        if os.path.isfile(cj) and int(default_camera_index) > -1:
            warnings.append(
                "检测到 cameras.json 且 default_camera_index>-1：Gaussian 可能始终使用单一 COLMAP 相机，"
                "与本文件中的多视角环绕轨迹不一致。"
            )

    payload = {
        "version": 3,
        "description": (
            "与 utils.camera_view_utils.get_camera_position_and_rotation 一致，"
            "对应 modified_simulation 中 num_views / num_render_views 的方位角分布；"
            "v3 增加 mpm_to_world，使 deformation_field 与 3DGS 使用同一世界坐标系。"
        ),
        "look_at": vc.tolist(),
        "world_up": up.tolist(),
        "observant_coordinates": obs.tolist(),
        "fov_vertical_deg": float(fov_vertical_deg),
        "fov_horizontal_deg": float(fov_horizontal_deg),
        "width_hint": int(width_hint),
        "height_hint": int(height_hint),
        "init_radius": rad,
        "camera_motion": {
            "move_camera": bool(move_camera),
            "delta_a": float(delta_a),
            "delta_e": float(delta_e),
            "delta_r": float(delta_r),
            "field_output_interval": int(field_output_interval),
        },
        "warnings": warnings,
        "views": views,
    }
    if mpm_space_viewpoint_center is not None:
        payload["mpm_space_viewpoint_center"] = [
            float(x) for x in mpm_space_viewpoint_center
        ]
    if mpm_to_world is not None:
        payload["mpm_to_world"] = mpm_to_world

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def mpm_positions_to_world_numpy(
    pos: np.ndarray, spec: Optional[Dict]
) -> np.ndarray:
    """
    与 modified_simulation 中 pos_world = undo_all_transforms(particle_x) 一致。
    deformation_field / stress_field 里存 MPM 空间坐标；3DGS / 可视化用世界系。
    spec 为 meta 中 mpm_to_world（rotation_matrices, scale_origin, original_mean_pos）。
    """
    if spec is None:
        return np.asarray(pos, dtype=np.float64)
    x = np.asarray(pos, dtype=np.float64).reshape(-1, 3).copy()
    x -= np.array([1.0, 1.0, 1.0], dtype=np.float64)
    scale = float(spec["scale_origin"])
    om = np.asarray(spec["original_mean_pos"], dtype=np.float64).reshape(3)
    x = om + x / scale
    mats = spec.get("rotation_matrices") or []
    for mi in range(len(mats) - 1, -1, -1):
        R = np.asarray(mats[mi], dtype=np.float64).reshape(3, 3)
        x = x @ R
    return x

