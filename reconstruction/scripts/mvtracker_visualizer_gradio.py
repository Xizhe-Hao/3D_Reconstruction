#!/usr/bin/env python3
"""Interactive Gradio viewer and MP4 renderer for MVTracker result folders."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from plyfile import PlyData
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MVTRACKER_ROOT = PROJECT_ROOT / "submodule" / "mvtracker"
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "outputs/data_test/frames_0_1414_target_96_duster"
)
_RERUN_PROCESS: subprocess.Popen | None = None
_RERUN_RESULT: Path | None = None
_RERUN_LOCK = threading.Lock()
_RENDER_LOCK = threading.Lock()


def configure_localhost_proxy_bypass() -> None:
    for variable in ("NO_PROXY", "no_proxy"):
        entries = [item for item in os.environ.get(variable, "").split(",") if item]
        for host in ("127.0.0.1", "localhost"):
            if host not in entries:
                entries.append(host)
        os.environ[variable] = ",".join(entries)


def resolve_result_dir(value: str | Path) -> Path:
    result_dir = Path(value).expanduser()
    if not result_dir.is_absolute():
        result_dir = PROJECT_ROOT / result_dir
    result_dir = result_dir.resolve()
    required = [result_dir / "tracks_4d.npz", result_dir / "tracks_4d.rrd"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing result files: " + ", ".join(missing))
    if not (result_dir / "ply_dense").is_dir():
        raise FileNotFoundError(
            f"Missing dense point-cloud directory: {result_dir / 'ply_dense'}"
        )
    return result_dir


def load_tracks(result_dir: Path) -> dict[str, np.ndarray]:
    with np.load(result_dir / "tracks_4d.npz") as data:
        required = {"trajectories_m", "visibility", "frame_indices", "video_times_s"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"tracks_4d.npz is missing keys: {sorted(missing)}")
        tracks = {
            "trajectories_m": np.asarray(data["trajectories_m"], dtype=np.float32),
            "visibility": np.asarray(data["visibility"], dtype=bool),
            "frame_indices": np.asarray(data["frame_indices"], dtype=np.int64),
            "video_times_s": np.asarray(data["video_times_s"], dtype=np.float64),
        }
    trajectories = tracks["trajectories_m"]
    visibility = tracks["visibility"]
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        raise ValueError(f"Unexpected trajectory shape: {trajectories.shape}")
    if visibility.shape != trajectories.shape[:2]:
        raise ValueError(
            f"Visibility shape {visibility.shape} does not match {trajectories.shape[:2]}"
        )
    if len(tracks["frame_indices"]) != trajectories.shape[0]:
        raise ValueError("Frame-index count does not match trajectory timestamps")
    return tracks


def dense_ply_paths(result_dir: Path, frame_indices: np.ndarray) -> list[Path]:
    paths = [
        result_dir / "ply_dense" / f"scene_{int(frame_index):06d}.ply"
        for frame_index in frame_indices
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path.name) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} dense PLY files, including: {preview}. "
            "Rerun reconstruction without --no-dense-ply."
        )
    return paths


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"].data
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
        np.float32
    )
    names = set(vertex.dtype.names or ())
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack(
            [vertex["red"], vertex["green"], vertex["blue"]]
        ).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 190, dtype=np.uint8)
    valid = np.isfinite(points).all(axis=1)
    return points[valid], colors[valid]


def track_colors(count: int) -> np.ndarray:
    if count == 0:
        return np.empty((0, 3), dtype=np.uint8)
    cmap = plt.get_cmap("turbo")
    return (cmap(np.linspace(0.0, 1.0, count, endpoint=False))[:, :3] * 255).astype(
        np.uint8
    )


def estimate_bounds(
    ply_paths: list[Path],
    trajectories: np.ndarray,
    visibility: np.ndarray,
    max_points_per_frame: int = 10000,
) -> tuple[np.ndarray, float]:
    samples = []
    time_step = max(1, len(ply_paths) // 12)
    rng = np.random.default_rng(72)
    for path in ply_paths[::time_step]:
        points, _ = load_ply(path)
        if len(points) > max_points_per_frame:
            points = points[rng.choice(len(points), max_points_per_frame, replace=False)]
        samples.append(points)
    visible_tracks = trajectories[visibility]
    if len(visible_tracks):
        samples.append(visible_tracks)
    points = np.concatenate(samples, axis=0)
    lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
    center = (lower + upper) / 2
    radius = float(np.max(upper - lower) * 0.58)
    return center.astype(np.float32), max(radius, 0.02)


def render_video(
    result_dir_value: str,
    fps: float,
    trail_length: int,
    azimuth: float,
    elevation: float,
    max_render_points: int,
    video_width: int,
    video_height: int,
) -> tuple[str, str]:
    with _RENDER_LOCK:
        result_dir = resolve_result_dir(result_dir_value)
        tracks = load_tracks(result_dir)
        trajectories = tracks["trajectories_m"]
        visibility = tracks["visibility"]
        frame_indices = tracks["frame_indices"]
        video_times = tracks["video_times_s"]
        ply_paths = dense_ply_paths(result_dir, frame_indices)
        if fps <= 0 or trail_length < 1 or max_render_points < 100:
            raise ValueError("FPS must be positive, trail length >=1, max points >=100")
        if video_width < 320 or video_height < 240:
            raise ValueError("Video dimensions must be at least 320x240")

        center, radius = estimate_bounds(ply_paths, trajectories, visibility)
        colors = track_colors(trajectories.shape[1])
        render_dir = result_dir / "visualization"
        render_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = render_dir / f"deformation_tracks_{stamp}.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is not installed or not in PATH")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{int(video_width)}x{int(video_height)}",
            "-framerate",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        figure = plt.figure(
            figsize=(video_width / 100, video_height / 100),
            dpi=100,
            facecolor="#080b12",
        )
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection="3d")
        rng = np.random.default_rng(72)
        try:
            for time_index, ply_path in tqdm(
                enumerate(ply_paths),
                total=len(ply_paths),
                desc="Render deformation video",
                unit="frame",
                dynamic_ncols=True,
            ):
                points, point_colors = load_ply(ply_path)
                if len(points) > max_render_points:
                    keep = rng.choice(len(points), max_render_points, replace=False)
                    points, point_colors = points[keep], point_colors[keep]

                axis.clear()
                axis.set_facecolor("#080b12")
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=point_colors.astype(np.float32) / 255.0,
                    s=0.35,
                    alpha=0.72,
                    linewidths=0,
                    depthshade=False,
                )
                current_visible = visibility[time_index]
                current = trajectories[time_index, current_visible]
                if len(current):
                    axis.scatter(
                        current[:, 0],
                        current[:, 1],
                        current[:, 2],
                        c=colors[current_visible].astype(np.float32) / 255.0,
                        s=8.0,
                        alpha=1.0,
                        linewidths=0,
                        depthshade=False,
                    )

                first = max(0, time_index - int(trail_length))
                segments, segment_colors = [], []
                for segment_time in range(first, time_index):
                    valid = visibility[segment_time] & visibility[segment_time + 1]
                    if valid.any():
                        segments.append(
                            np.stack(
                                [
                                    trajectories[segment_time, valid],
                                    trajectories[segment_time + 1, valid],
                                ],
                                axis=1,
                            )
                        )
                        age = (segment_time - first + 1) / max(1, time_index - first)
                        rgba = np.column_stack(
                            [
                                colors[valid].astype(np.float32) / 255.0,
                                np.full(valid.sum(), 0.15 + 0.75 * age),
                            ]
                        )
                        segment_colors.append(rgba)
                if segments:
                    axis.add_collection3d(
                        Line3DCollection(
                            np.concatenate(segments),
                            colors=np.concatenate(segment_colors),
                            linewidths=0.7,
                        )
                    )

                axis.set_xlim(center[0] - radius, center[0] + radius)
                axis.set_ylim(center[1] - radius, center[1] + radius)
                axis.set_zlim(center[2] - radius, center[2] + radius)
                axis.set_box_aspect((1, 1, 1))
                axis.view_init(elev=float(elevation), azim=float(azimuth))
                axis.set_axis_off()
                axis.set_title(
                    f"frame {int(frame_indices[time_index])}   "
                    f"t={float(video_times[time_index]):.3f}s   "
                    f"tracks={int(current_visible.sum())}",
                    color="white",
                    fontsize=11,
                    pad=1,
                )
                figure.subplots_adjust(left=0, right=1, bottom=0, top=0.96)
                canvas.draw()
                rgba = np.asarray(canvas.buffer_rgba())
                if encoder.stdin is None:
                    raise RuntimeError("ffmpeg stdin is unavailable")
                encoder.stdin.write(np.ascontiguousarray(rgba[:, :, :3]).tobytes())
        except Exception:
            if encoder.stdin is not None:
                encoder.stdin.close()
            encoder.kill()
            encoder.wait()
            output_path.unlink(missing_ok=True)
            raise
        finally:
            plt.close(figure)
        assert encoder.stdin is not None
        encoder.stdin.close()
        return_code = encoder.wait()
        if return_code != 0 or not output_path.is_file():
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
        status = (
            f"Rendered {len(ply_paths)} frames, {trajectories.shape[1]} tracks; "
            f"view azimuth={azimuth:g}, elevation={elevation:g}; output={output_path}"
        )
        return str(output_path), status


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def stop_rerun_server() -> None:
    global _RERUN_PROCESS, _RERUN_RESULT
    with _RERUN_LOCK:
        if _RERUN_PROCESS is not None:
            _terminate_process_group(_RERUN_PROCESS)
        _RERUN_PROCESS = None
        _RERUN_RESULT = None


def start_rerun_viewer(
    result_dir_value: str,
    viewer_url: str,
    bind_host: str,
    web_port: int,
    ws_port: int,
) -> tuple[str, str, str]:
    global _RERUN_PROCESS, _RERUN_RESULT
    result_dir = resolve_result_dir(result_dir_value)
    rrd_path = result_dir / "tracks_4d.rrd"
    rerun_cli = shutil.which("rerun")
    if rerun_cli is None:
        raise RuntimeError("rerun CLI is not installed in the active environment")
    with _RERUN_LOCK:
        same_server = (
            _RERUN_PROCESS is not None
            and _RERUN_PROCESS.poll() is None
            and _RERUN_RESULT == result_dir
        )
        if not same_server:
            if _RERUN_PROCESS is not None and _RERUN_PROCESS.poll() is None:
                _terminate_process_group(_RERUN_PROCESS)
            elif _port_is_open("127.0.0.1", int(web_port)):
                raise RuntimeError(
                    f"Port {web_port} is already occupied by another process. "
                    "Choose another --rerun-web-port."
                )
            command = [
                rerun_cli,
                str(rrd_path),
                "--serve-web",
                "--bind",
                bind_host,
                "--web-viewer-port",
                str(int(web_port)),
                "--ws-server-port",
                str(int(ws_port)),
                "--server-memory-limit",
                "4GB",
            ]
            _RERUN_PROCESS = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            _RERUN_RESULT = result_dir
            deadline = time.time() + 15
            while time.time() < deadline:
                if _RERUN_PROCESS.poll() is not None:
                    raise RuntimeError(
                        f"Rerun web server exited with code {_RERUN_PROCESS.returncode}"
                    )
                if _port_is_open("127.0.0.1", int(web_port)):
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("Timed out waiting for the Rerun web viewer")
    url = viewer_url.rstrip("/")
    html = (
        f'<div style="height:720px;width:100%;background:#080b12">'
        f'<iframe src="{url}" style="height:100%;width:100%;border:0" '
        f'allow="clipboard-read; clipboard-write; fullscreen"></iframe></div>'
        f'<p><a href="{url}" target="_blank">Open Rerun in a separate tab</a></p>'
    )
    tracks = load_tracks(result_dir)
    status = (
        f"Loaded {len(tracks['frame_indices'])} timestamps and "
        f"{tracks['trajectories_m'].shape[1]} tracks from {result_dir}. "
        f"Rerun web={url}, websocket port={ws_port}."
    )
    return html, str(rrd_path), status


def build_app(args: argparse.Namespace):
    import gradio as gr

    app = gr.Blocks()
    with app:
        gr.Markdown(
            "# MVTracker 4D reconstruction viewer\n"
            "Load an existing result folder for interactive Rerun inspection, or "
            "render its fused PLY sequence plus MVTracker trails to MP4."
        )
        result_dir = gr.Textbox(
            value=str(args.result_dir), label="MVTracker result directory"
        )
        with gr.Row():
            load_button = gr.Button("Load interactive Rerun")
            render_button = gr.Button("Render deformation MP4")
        with gr.Row():
            fps = gr.Slider(1, 30, step=1, value=args.fps, label="Video FPS")
            trail = gr.Slider(
                1, 100, step=1, value=args.trail_length, label="Track trail frames"
            )
            max_points = gr.Slider(
                1000,
                200000,
                step=1000,
                value=args.max_render_points,
                label="Max rendered PLY points/frame",
            )
        with gr.Row():
            azimuth = gr.Slider(
                -180, 180, step=1, value=args.azimuth, label="Camera azimuth"
            )
            elevation = gr.Slider(
                -90, 90, step=1, value=args.elevation, label="Camera elevation"
            )
            width = gr.Number(value=args.video_width, label="Video width")
            height = gr.Number(value=args.video_height, label="Video height")
        status = gr.Textbox(label="Status")
        viewer = gr.HTML(label="Interactive Rerun viewer")
        rrd_file = gr.File(label="Download RRD")
        video = gr.Video(label="Rendered deformation video")
        video_file = gr.File(label="Download MP4")

        load_button.click(
            fn=lambda path: start_rerun_viewer(
                path, args.viewer_url, args.rerun_bind, args.rerun_web_port, args.rerun_ws_port
            ),
            inputs=result_dir,
            outputs=[viewer, rrd_file, status],
        )

        def render_for_ui(path, out_fps, out_trail, out_azimuth, out_elevation, out_max, out_w, out_h):
            rendered, message = render_video(
                path,
                float(out_fps),
                int(out_trail),
                float(out_azimuth),
                float(out_elevation),
                int(out_max),
                int(out_w),
                int(out_h),
            )
            return rendered, rendered, message

        render_button.click(
            fn=render_for_ui,
            inputs=[
                result_dir,
                fps,
                trail,
                azimuth,
                elevation,
                max_points,
                width,
                height,
            ],
            outputs=[video, video_file, status],
        )
    return app.queue(default_concurrency_limit=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--rerun-web-port", type=int, default=9090)
    parser.add_argument("--rerun-ws-port", type=int, default=9877)
    parser.add_argument("--rerun-bind", default="127.0.0.1")
    parser.add_argument(
        "--viewer-url",
        default="http://127.0.0.1:9090",
        help="Browser-visible Rerun web URL used by the Gradio iframe",
    )
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--trail-length", type=int, default=20)
    parser.add_argument("--azimuth", type=float, default=-60)
    parser.add_argument("--elevation", type=float, default=20)
    parser.add_argument("--max-render-points", type=int, default=80000)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_localhost_proxy_bypass()
    atexit.register(stop_rerun_server)
    if args.render_only:
        video, status = render_video(
            str(args.result_dir),
            args.fps,
            args.trail_length,
            args.azimuth,
            args.elevation,
            args.max_render_points,
            args.video_width,
            args.video_height,
        )
        print(status, flush=True)
        print(video, flush=True)
        return
    app = build_app(args)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(args.result_dir.expanduser().resolve())],
    )


if __name__ == "__main__":
    main()
