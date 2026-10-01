#!/usr/bin/env python3
"""Four-view web player for FoundationStereo depth predictions."""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    PROJECT_ROOT
    / "outputs/foundationstereo_smoke_full"
    / "seconds_12.4_13.5_target_7_foundationstereo"
    / "foundationstereo"
)


def resolve_result_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if (path / "foundationstereo").is_dir():
        path = path / "foundationstereo"
    required = (path / "manifest.json", path / "tracking_clip.npz")
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} under {path}")
    if not list(path.glob("frame_*.npz")):
        raise FileNotFoundError(f"No frame_*.npz files under {path}")
    return path


@lru_cache(maxsize=4)
def load_result_info(folder: str) -> tuple[dict, tuple[Path, ...], np.ndarray]:
    result_dir = resolve_result_dir(folder)
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    frame_files = tuple(sorted(result_dir.glob("frame_*.npz")))
    with np.load(result_dir / "tracking_clip.npz") as data:
        rgbs = np.asarray(data["rgbs"], dtype=np.uint8)
    if rgbs.ndim != 5 or rgbs.shape[0] != 4 or rgbs.shape[2] != 3:
        raise ValueError(f"Expected tracking RGB [4,T,3,H,W], got {rgbs.shape}")
    if rgbs.shape[1] != len(frame_files):
        raise ValueError(
            f"RGB timestamps ({rgbs.shape[1]}) do not match frame NPZ files ({len(frame_files)})"
        )
    return manifest, frame_files, rgbs


@lru_cache(maxsize=2)
def load_prediction(path_string: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path_string) as data:
        depths = np.asarray(data["depths_m"], dtype=np.float32)
        disparities = np.asarray(data["disparities_px"], dtype=np.float32)
        masks = np.asarray(data["valid_masks"], dtype=bool)
        intrinsics = np.asarray(data["rectified_intrinsics"], dtype=np.float32)
    expected = depths.shape
    if depths.ndim != 3 or depths.shape[0] != 4:
        raise ValueError(f"Expected full-resolution depth [4,H,W], got {depths.shape}")
    if disparities.shape != expected or masks.shape != expected or intrinsics.shape != (4, 3, 3):
        raise ValueError(
            f"Incompatible frame arrays: depth={depths.shape}, disparity={disparities.shape}, "
            f"mask={masks.shape}, intrinsics={intrinsics.shape}"
        )
    return depths, disparities, masks, intrinsics


def baseline_by_view(manifest: dict) -> np.ndarray:
    result = np.zeros(4, dtype=np.float32)
    pairs = manifest.get("pairs_rectified", [])
    baselines = manifest.get("baselines_m", [])
    if len(pairs) != len(baselines):
        raise ValueError("manifest pairs_rectified and baselines_m are inconsistent")
    for pair, baseline in zip(pairs, baselines):
        for view in pair:
            result[int(view)] = float(baseline)
    if np.any(result <= 0):
        raise ValueError(f"manifest does not provide a positive baseline for every view: {result}")
    return result


def raw_metric_depth(
    disparities: np.ndarray, intrinsics: np.ndarray, baselines: np.ndarray
) -> np.ndarray:
    depths = np.zeros_like(disparities, dtype=np.float32)
    for view in range(4):
        valid = np.isfinite(disparities[view]) & (disparities[view] > 1e-6)
        depths[view, valid] = (
            float(intrinsics[view, 0, 0] * baselines[view]) / disparities[view, valid]
        )
    return depths


def colorize(
    values: np.ndarray,
    valid: np.ndarray,
    low: float,
    high: float,
    output_size: tuple[int, int],
) -> np.ndarray:
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    if (colored.shape[1], colored.shape[0]) != output_size:
        colored = cv2.resize(colored, output_size, interpolation=cv2.INTER_AREA)
    return colored


def mask_image(mask: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    image = np.repeat(np.uint8(mask[..., None]) * 255, 3, axis=2)
    if (image.shape[1], image.shape[0]) != output_size:
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_NEAREST)
    return image


def render(
    folder: str,
    timestamp: float,
    mode: str,
    auto_range: bool,
    fixed_min: float,
    fixed_max: float,
):
    manifest, frame_files, rgbs = load_result_info(folder)
    index = int(np.clip(round(timestamp), 0, len(frame_files) - 1))
    depths, disparities, masks, intrinsics = load_prediction(str(frame_files[index]))
    baselines = baseline_by_view(manifest)
    raw_depths = raw_metric_depth(disparities, intrinsics, baselines)

    if mode == "Filtered depth":
        values = depths
        valid = masks & np.isfinite(depths) & (depths > 0)
        unit = "m"
    elif mode == "Raw depth":
        values = raw_depths
        valid = np.isfinite(raw_depths) & (raw_depths > 0)
        unit = "m"
    elif mode == "Disparity":
        values = disparities
        valid = np.isfinite(disparities) & (disparities > 0)
        unit = "px"
    elif mode == "LR valid mask":
        values = masks.astype(np.float32)
        valid = np.ones_like(masks, dtype=bool)
        unit = "binary"
    else:
        raise ValueError(f"Unknown display mode: {mode}")

    sample = values[valid]
    if mode == "LR valid mask":
        low, high = 0.0, 1.0
    elif auto_range:
        low, high = np.percentile(sample, [2, 98]) if sample.size else (0.0, 1.0)
    else:
        low, high = float(fixed_min), float(fixed_max)
        if high <= low:
            raise ValueError("Fixed maximum must be greater than fixed minimum")

    output_size = (int(rgbs.shape[-1]), int(rgbs.shape[-2]))
    visuals = []
    statistics = []
    for view in range(4):
        rgb = rgbs[view, index].transpose(1, 2, 0)
        prediction = (
            mask_image(masks[view], output_size)
            if mode == "LR valid mask"
            else colorize(values[view], valid[view], low, high, output_size)
        )
        visuals.extend((rgb, prediction))
        filtered = masks[view]
        raw = np.isfinite(raw_depths[view]) & (raw_depths[view] > 0)
        if filtered.any():
            depth_values = depths[view, filtered]
            depth_text = (
                f"depth median={np.median(depth_values):.4f}m, "
                f"P10-P90={np.percentile(depth_values, 10):.4f}-"
                f"{np.percentile(depth_values, 90):.4f}m"
            )
        else:
            depth_text = "no filtered depth"
        statistics.append(
            f"V{view}: raw={raw.mean():.1%}, LR-valid={filtered.mean():.2%}, {depth_text}"
        )

    frame_indices = manifest.get("frame_indices", list(range(len(frame_files))))
    source_frame = int(frame_indices[index]) if index < len(frame_indices) else index
    inference_size = manifest.get("inference_size", list(depths.shape[-1::-1]))
    header = (
        f"{frame_files[index].name}; timestamp={index + 1}/{len(frame_files)}; "
        f"source frame={source_frame}; inference={inference_size}; "
        f"mode={mode}; range={low:.5f}–{high:.5f} {unit}"
    )
    return (*visuals, header + "\n" + "\n".join(statistics))


def reload_result(
    folder: str,
    mode: str,
    auto_range: bool,
    fixed_min: float,
    fixed_max: float,
):
    load_result_info.cache_clear()
    load_prediction.cache_clear()
    _, frame_files, _ = load_result_info(folder)
    rendered = render(folder, 0, mode, auto_range, fixed_min, fixed_max)
    import gradio as gr

    return gr.update(minimum=0, maximum=len(frame_files) - 1, value=0), len(frame_files), *rendered


def next_timestamp(current: float, count: int) -> int:
    return (int(round(current)) + 1) % max(int(count), 1)


def main() -> None:
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"

    import gradio as gr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT))
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    args = parser.parse_args()

    initial_dir = resolve_result_dir(args.result_dir)
    _, initial_files, _ = load_result_info(str(initial_dir))
    initial_count = len(initial_files)

    with gr.Blocks(title="FoundationStereo four-view depth player") as app:
        gr.Markdown(
            "# FoundationStereo four-view depth player\n"
            "四路校正 RGB 与深度同步显示。Filtered depth 是左右一致性过滤后的输入，"
            "Raw depth 是由原始视差直接换算的米制深度。"
        )
        with gr.Row():
            folder = gr.Textbox(str(initial_dir), label="FoundationStereo result directory", scale=5)
            reload_button = gr.Button("Load / Reload", scale=1)
        with gr.Row():
            play = gr.Button("Play", variant="primary")
            timestamp = gr.Slider(
                0, initial_count - 1, value=0, step=1, label="Synchronized timestamp"
            )
            playback_fps = gr.Slider(0.5, 10, value=2, step=0.5, label="Playback FPS")
        with gr.Row():
            mode = gr.Dropdown(
                ["Filtered depth", "Raw depth", "LR valid mask", "Disparity"],
                value="Filtered depth",
                label="Display mode",
            )
            auto_range = gr.Checkbox(True, label="Shared auto range (P2–P98)")
            fixed_min = gr.Number(0.03, label="Fixed minimum")
            fixed_max = gr.Number(0.50, label="Fixed maximum")

        panels = []
        for view in range(4):
            with gr.Row():
                panels.append(gr.Image(label=f"Camera {view} rectified RGB"))
                panels.append(gr.Image(label=f"Camera {view} prediction"))
        information = gr.Textbox(label="Frame statistics", lines=6)

        count = gr.State(initial_count)
        playing = gr.State(False)
        timer = gr.Timer(value=0.5, active=False)
        inputs = [folder, timestamp, mode, auto_range, fixed_min, fixed_max]
        outputs = [*panels, information]

        app.load(render, inputs, outputs)
        timestamp.change(render, inputs, outputs, show_progress="hidden")
        for control in (mode, auto_range, fixed_min, fixed_max):
            control.change(render, inputs, outputs, show_progress="hidden")
        reload_button.click(
            reload_result,
            [folder, mode, auto_range, fixed_min, fixed_max],
            [timestamp, count, *outputs],
        )
        folder.submit(
            reload_result,
            [folder, mode, auto_range, fixed_min, fixed_max],
            [timestamp, count, *outputs],
        )
        play.click(
            lambda state: (
                not state,
                gr.update(active=not state),
                "Pause" if not state else "Play",
            ),
            playing,
            [playing, timer, play],
            show_progress="hidden",
        )
        timer.tick(next_timestamp, [timestamp, count], timestamp)
        playback_fps.change(
            lambda value: gr.update(value=1.0 / value),
            playback_fps,
            timer,
            show_progress="hidden",
        )

    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
