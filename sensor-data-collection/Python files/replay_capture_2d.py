#!/usr/bin/env python3
"""Replay a capture session with a 2D top-down tactile image.

The verification view for data collection. It is the same replay as
replay_capture_multimodal.py -- same session loading, same camera decoding,
same force panel, same playback controls -- with the 3D bar panel replaced by
a top-down 2D image of the sensor.

Why 2D is the default for checking a capture:

* The 3D z-axis is voltage, not displacement. Showing it as a height implies a
  physical quantity that has not been calibrated yet; a flat image does not.
* No occlusion. Tall bars hide the cells behind them, and cells that read
  strongly negative form a wall across the near edge. All 256 cells are always
  visible here.
* A top-down image registers directly against the camera previews above it, so
  "the indenter is here, the tactile blob is there" is one glance, not a mental
  rotation.
* Reading a cell address off rotated 3D tick labels is hard; it is immediate on
  a 2D grid.

Use replay_capture_multimodal.py when bar height genuinely helps -- judging
relative depth, or a demo -- and this one for everything else.

Everything shared with the 3D replay is imported from it rather than copied,
so the two stay consistent. Keep both files in the same folder.

Usage:
    python replay_capture_2d.py "path\\to\\capture\\session"
    python replay_capture_2d.py "..." --labels          # per-cell values
    python replay_capture_2d.py "..." --range 1.0       # fixed colour scale
    python replay_capture_2d.py "..." --check-only      # decode check, no window

    # Render the whole view to a video instead of opening a window:
    python replay_capture_2d.py "..." --export clip.mp4 --export-range 42 49
    python replay_capture_2d.py "..." --export clip.gif --export-fps 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from replay_capture_multimodal import (
    COLS,
    ROWS,
    MultimodalReplay,
    VideoFrameReader,
    camera_grid_shape,
    load_session,
    prepare_sensor_data,
    print_summary,
    validate_video_decoding,
)

# Percentile of |signal| used when the colour scale is chosen automatically.
AUTO_RANGE_PERCENTILE = 99.5
AUTO_RANGE_FLOOR_V = 0.05

# replay_capture_multimodal selects TkAgg on import. Exporting needs no window,
# so switch to the offscreen renderer -- before pyplot creates any figure.
if any(item == "--export" or item.startswith("--export=") for item in sys.argv):
    matplotlib.use("Agg", force=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay the synchronized camera streams with a 2D tactile image "
            "and aligned IntelliMESUR force data."
        )
    )
    parser.add_argument("session", type=Path, help="Capture session directory")
    parser.add_argument(
        "--cameras",
        type=int,
        default=0,
        help=(
            "Expected camera count. 0 discovers it from the session "
            "(export metadata, session.json, or camera_* directories)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("change", "voltage", "relative-max"),
        default="change",
    )
    parser.add_argument(
        "--sensor-source",
        choices=("auto", "raw", "corrected", "corrected-v2", "filled"),
        default="auto",
    )
    parser.add_argument(
        "--missing-sensor",
        choices=("blank", "interpolate", "nearest"),
        default="blank",
        help=(
            "How to display camera frames whose sensor packet was lost. "
            "Interpolation and nearest-neighbor values are visualization only."
        ),
    )
    parser.add_argument("--fps", type=float)
    parser.add_argument("--baseline-frames", type=int, default=24)
    parser.add_argument("--adc-ref", type=float, default=5.0)
    parser.add_argument("--vdrive", type=float, default=5.0)
    parser.add_argument(
        "--range",
        dest="display_range",
        type=float,
        help=(
            "Colour-scale half-range in volts. Omit to size it from the "
            "session's own signal, which a flat image needs far more than a "
            "bar chart does."
        ),
    )
    parser.add_argument(
        "--no-auto-range",
        action="store_true",
        help="Use the fixed default scale instead of sizing it to the data.",
    )
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--filter-alpha", type=float, default=1.0)
    parser.add_argument("--preview-width", type=int, default=480)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bicubic upsampling for a continuous image (default: enabled).",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.05,
        help=(
            "Absolute floor in volts. A frame whose peak is below this is "
            "reported as no contact."
        ),
    )
    parser.add_argument(
        "--contact-fraction",
        type=float,
        default=0.4,
        help=(
            "Cells above this fraction of the frame's own peak count as "
            "contact. Being relative to the frame keeps the centroid on the "
            "real contact patch when the whole array has drifted off baseline."
        ),
    )
    parser.add_argument(
        "--no-contact-marker",
        action="store_true",
        help="Hide the contact centroid and peak-cell markers.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--export",
        type=Path,
        help=(
            "Render the whole view (cameras, tactile image, force curve) to a "
            "video file instead of opening a window. The extension picks the "
            "format: .mp4 or .gif."
        ),
    )
    parser.add_argument(
        "--export-fps",
        type=float,
        default=12.0,
        help=(
            "Output frame rate. Frames are sampled from the capture timeline "
            "at this rate, so the video plays at real speed."
        ),
    )
    parser.add_argument(
        "--export-range",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        help=(
            "Export only this span of capture time in seconds. Exporting a "
            "whole session is slow; a span around the event is usually enough."
        ),
    )
    parser.add_argument(
        "--export-dpi",
        type=int,
        default=80,
        help="Output resolution. The figure is 17 inches wide, so 80 -> 1360 px.",
    )
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.baseline_frames < 1:
        parser.error("--baseline-frames must be at least 1")
    if args.adc_ref <= 0 or args.vdrive <= 0:
        parser.error("--adc-ref and --vdrive must be positive")
    if args.display_range is not None and args.display_range <= 0:
        parser.error("--range must be positive")
    if args.gain <= 0:
        parser.error("--gain must be positive")
    if not 0 < args.filter_alpha <= 1:
        parser.error("--filter-alpha must satisfy 0 < alpha <= 1")
    if args.preview_width < 160:
        parser.error("--preview-width must be at least 160")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.cameras < 0:
        parser.error("--cameras must be non-negative")
    if args.contact_threshold < 0:
        parser.error("--contact-threshold must be non-negative")
    if not 0 < args.contact_fraction <= 1:
        parser.error("--contact-fraction must satisfy 0 < value <= 1")
    if args.export is not None:
        if args.export.suffix.lower() not in (".mp4", ".gif"):
            parser.error("--export must end in .mp4 or .gif")
        if args.export_fps <= 0:
            parser.error("--export-fps must be positive")
        if args.export_dpi < 20:
            parser.error("--export-dpi must be at least 20")
        if args.export_range and args.export_range[0] >= args.export_range[1]:
            parser.error("--export-range START_S must be less than END_S")
    return args


def autoscale_range(sensor_data, mode):
    """Half-range that keeps the bulk of the signal inside the colour scale.

    The 3D replay defaults to a fixed +/-2 V, which is fine when bar height
    carries the magnitude. On a flat image that scale washes a typical run out
    to near-white, so size it from the data instead.
    """
    if mode == "voltage":
        return None
    finite = sensor_data[np.isfinite(sensor_data)]
    if finite.size == 0:
        return None
    limit = float(np.percentile(np.abs(finite), AUTO_RANGE_PERCENTILE))
    return max(limit, AUTO_RANGE_FLOOR_V)


class Replay2D(MultimodalReplay):
    """The 3D replay with a top-down image in place of the bar panel.

    Playback, the frame/speed controls, the force panel, and the video readers
    are inherited unchanged; only the sensor panel is re-implemented. The
    parent __init__ is deliberately not called because it builds the 3D figure.
    """

    def __init__(
        self,
        session,
        sensor_data,
        description,
        z_min,
        z_max,
        cmap,
        norm,
        args,
        export=False,
    ):
        # Export mode renders offscreen: no animation timer, and the playback
        # buttons are hidden because they are meaningless in a video file.
        self.export = export
        self.session = session
        self.sensor_data = sensor_data
        self.description = description
        self.z_min = z_min
        self.z_max = z_max
        self.cmap = cmap
        self.norm = norm
        self.args = args
        self.camera_count = session["cameras"]["count"]
        self.frame_count = session["cameras"]["sequence"].size
        self.fps = args.fps or session["fps"]
        self.current_frame = min(args.start_frame, self.frame_count - 1)
        self.playing = not args.paused
        self.speed = 1.0
        self.last_tick = time.perf_counter()
        self.frame_accumulator = 0.0
        self.updating_slider = False
        self.show_markers = not args.no_contact_marker
        self.video_readers = [
            VideoFrameReader(
                path,
                session["cameras"]["video_fps"],
                int(info["width"]),
                int(info["height"]),
                args.preview_width,
            )
            for path, info in zip(
                session["cameras"]["videos"],
                session["cameras"]["video_info"],
            )
        ]
        self._build_figure()
        self.show_frame(self.current_frame, force=True)

    def _build_figure(self):
        preview_rows, preview_columns = camera_grid_shape(self.camera_count)
        figure_height = 10.0 + 2.6 * (preview_rows - 1)
        preview_share = 0.72 + 0.62 * (preview_rows - 1)
        self.fig = plt.figure(figsize=(17, figure_height))
        grid = self.fig.add_gridspec(
            2,
            1,
            height_ratios=(preview_share, 1.28),
            left=0.035,
            right=0.965,
            top=0.925,
            bottom=0.14,
            hspace=0.14,
        )

        preview_grid = grid[0].subgridspec(
            preview_rows,
            preview_columns,
            hspace=0.18,
            wspace=0.06,
        )
        self.image_artists = []
        for camera_index in range(self.camera_count):
            row, column = divmod(camera_index, preview_columns)
            axis = self.fig.add_subplot(preview_grid[row, column])
            preview = self.video_readers[camera_index].read(self.current_frame)
            artist = axis.imshow(preview, interpolation="nearest")
            axis.set_title(
                f"[{camera_index}] "
                f"{self.session['cameras']['names'][camera_index]}",
                fontsize=9,
            )
            axis.set_axis_off()
            self.image_artists.append(artist)

        # Four columns so the inherited _build_force_plot still finds the force
        # axes at [0, 3]. A flat sensor image needs far less width than the 3D
        # box did, so the ratios give the force curve roughly half the row
        # instead of the quarter it had before.
        # The image is square, so its on-screen size is capped by the row
        # height. Give its columns only about that much width, otherwise the
        # square floats in a wide empty cell and the force curve is pushed to
        # the far right edge.
        data_grid = grid[1].subgridspec(
            1, 4, width_ratios=(1.0, 1.0, 1.0, 7.0), wspace=0.30
        )
        self.sensor_axis = self.fig.add_subplot(data_grid[0, :3])
        self.sensor_image = self.sensor_axis.imshow(
            self.matrix_for_display(self.current_frame),
            cmap=self.cmap,
            norm=self.norm,
            interpolation="bicubic" if self.args.smooth else "nearest",
            origin="upper",
            extent=(-0.5, COLS - 0.5, ROWS - 0.5, -0.5),
            aspect="equal",
        )
        self.sensor_axis.set_xticks(range(0, COLS, 2))
        self.sensor_axis.set_yticks(range(0, ROWS, 2))
        self.sensor_axis.set_xticklabels(
            [f"C{index + 1}" for index in range(0, COLS, 2)], fontsize=8
        )
        self.sensor_axis.set_yticklabels(
            [f"R{index + 1}" for index in range(0, ROWS, 2)], fontsize=8
        )
        self.sensor_axis.set_xlabel("Column")
        self.sensor_axis.set_ylabel("Row")
        colorbar = self.fig.colorbar(
            self.sensor_image, ax=self.sensor_axis, shrink=0.82, pad=0.03
        )
        colorbar.set_label(f"{self.description} (V)")

        self.peak_marker, = self.sensor_axis.plot(
            [], [], "x", color="#101010", markersize=10, markeredgewidth=1.8
        )
        self.centroid_marker, = self.sensor_axis.plot(
            [],
            [],
            "o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor="#0b0b0b",
            markeredgewidth=1.6,
        )

        self.label_artists = []
        if self.args.labels:
            for row in range(ROWS):
                for column in range(COLS):
                    self.label_artists.append(
                        self.sensor_axis.text(
                            column,
                            row,
                            "",
                            ha="center",
                            va="center",
                            fontsize=5.2,
                            color="#111111",
                        )
                    )

        self._build_force_plot(data_grid)
        self._build_controls()
        if self.export:
            # Keep the frame slider: it reads as a progress bar in the video.
            for button in self.buttons:
                button.ax.set_visible(False)
            self.speed_slider.ax.set_visible(False)
            self.timer = None
            return
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def contact_metrics(self, matrix):
        """Peak cell and contact centroid, or None where not meaningful."""
        if not np.isfinite(matrix).any():
            return None, None, 0
        peak = np.unravel_index(int(np.nanargmax(matrix)), matrix.shape)
        if self.args.mode == "voltage":
            return peak, None, 0
        peak_value = float(matrix[peak])
        if peak_value <= self.args.contact_threshold:
            return peak, None, 0
        # Relative to this frame's own peak: an absolute cut-off flags most of
        # the array once the baseline has drifted, which puts the centroid in
        # the middle of the sensor instead of on the contact patch.
        threshold = max(
            self.args.contact_threshold,
            self.args.contact_fraction * peak_value,
        )
        contact = np.isfinite(matrix) & (matrix > threshold)
        cells = int(contact.sum())
        if cells == 0:
            return peak, None, 0
        weights = np.where(contact, matrix, 0.0)
        total = weights.sum()
        rows, columns = np.indices(matrix.shape)
        centroid = (
            float((weights * rows).sum() / total),
            float((weights * columns).sum() / total),
        )
        return peak, centroid, cells

    def show_frame(self, index, force=False):
        index = int(np.clip(index, 0, self.frame_count - 1))
        if index == self.current_frame and not force:
            return
        self.current_frame = index

        for camera_index, artist in enumerate(self.image_artists):
            artist.set_data(self.video_readers[camera_index].read(index))

        matrix = self.matrix_for_display(index)
        available = np.isfinite(self.sensor_data[index]).any()
        self.sensor_image.set_data(matrix)
        # A lost sensor packet must not look like a flat zero reading.
        self.sensor_image.set_alpha(1.0 if available else 0.25)

        peak, centroid, contact_cells = (
            self.contact_metrics(matrix) if available else (None, None, 0)
        )
        if self.show_markers and peak is not None:
            self.peak_marker.set_data([peak[1]], [peak[0]])
        else:
            self.peak_marker.set_data([], [])
        if self.show_markers and centroid is not None:
            self.centroid_marker.set_data([centroid[1]], [centroid[0]])
        else:
            self.centroid_marker.set_data([], [])

        self.update_labels(index, matrix)

        sequence = int(self.session["cameras"]["sequence"][index])
        elapsed = float(self.session["time_s"][index])
        state = self.sensor_state(index)
        value_range = (
            f"{np.nanmin(self.sensor_data[index]):+.3f} to "
            f"{np.nanmax(self.sensor_data[index]):+.3f} V"
            if available
            else "unavailable"
        )
        if centroid is not None:
            contact_text = (
                f"contact {contact_cells} cells @ "
                f"(R{centroid[0] + 1:.1f}, C{centroid[1] + 1:.1f})"
            )
            # A contact patch covering half the array is almost never a real
            # contact; it means the array has drifted away from the baseline
            # taken at the start of the run. Say so rather than reporting a
            # meaningless centroid.
            if contact_cells > matrix.size // 2:
                contact_text += " -- over half the array, check baseline drift"
        elif peak is not None:
            contact_text = f"peak @ (R{peak[0] + 1}, C{peak[1] + 1})"
        else:
            contact_text = "no sensor frame"
        # Two lines only; a third crowds the preview row above.
        self.sensor_axis.set_title(
            f"{self.description} | sequence {sequence} | t={elapsed:.3f}s\n"
            f"{state} | range {value_range} | {contact_text}",
            color="#b32020" if state == "MISSING" else "#202020",
            fontsize=9.5,
        )
        self.update_force(index, elapsed)

        self.updating_slider = True
        self.progress_slider.set_val(index)
        self.updating_slider = False
        self.fig.suptitle(
            f"Frame {index + 1}/{self.frame_count} | "
            f"capture_sequence_index={sequence} | "
            f"{self.camera_count} cameras",
            fontsize=11,
            y=0.985,
        )
        self.fig.canvas.draw_idle()

    def update_labels(self, index, matrix):
        if not self.label_artists:
            return
        available = np.isfinite(self.sensor_data[index]).any()
        for flat_index, artist in enumerate(self.label_artists):
            row, column = divmod(flat_index, COLS)
            value = float(matrix[row, column])
            artist.set_text(f"{value:+.2f}" if available else "--")

    def on_key_press(self, event):
        if event.key == "m":
            self.show_markers = not self.show_markers
            self.show_frame(self.current_frame, force=True)
            return
        if event.key == "i":
            self.args.smooth = not self.args.smooth
            self.sensor_image.set_interpolation(
                "bicubic" if self.args.smooth else "nearest"
            )
            self.fig.canvas.draw_idle()
            return
        super().on_key_press(event)


def export_frame_indices(session, args):
    """Frame indices sampled at --export-fps across the requested span."""
    time_s = session["time_s"]
    start, stop = float(time_s[0]), float(time_s[-1])
    if args.export_range:
        start = max(start, args.export_range[0])
        stop = min(stop, args.export_range[1])
        if start >= stop:
            raise RuntimeError(
                f"--export-range is outside the capture (0 to {time_s[-1]:.1f}s)"
            )
    wanted = np.arange(start, stop + 1e-9, 1.0 / args.export_fps)
    # Nearest real frame to each output instant, de-duplicated so a slow
    # capture never emits the same frame twice in a row.
    indices = np.unique(np.abs(time_s[None, :] - wanted[:, None]).argmin(axis=1))
    return indices, start, stop


def build_export_command(ffmpeg, path, width, height, fps):
    common = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.9g}",
        "-i",
        "pipe:0",
        "-an",
    ]
    if path.suffix.lower() == ".gif":
        # One pass: build a palette from the clip and apply it, otherwise the
        # default 216-colour web palette wrecks the colour map.
        return common + [
            "-filter_complex",
            "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse",
            str(path),
        ]
    return common + [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        # libx264 with yuv420p needs even dimensions.
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags",
        "+faststart",
        str(path),
    ]


def export_animation(session, sensor_data, description, z_min, z_max, cmap,
                     norm, args):
    import imageio_ffmpeg

    indices, start, stop = export_frame_indices(session, args)
    replay = Replay2D(
        session, sensor_data, description, z_min, z_max, cmap, norm, args,
        export=True,
    )
    figure = replay.fig
    figure.set_dpi(args.export_dpi)
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()

    output = args.export.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_export_command(
        imageio_ffmpeg.get_ffmpeg_exe(), output, width, height, args.export_fps
    )
    print(
        f"Exporting {len(indices)} frames covering {start:.2f}-{stop:.2f}s "
        f"at {args.export_fps:g} fps, {width}x{height} -> {output.name}",
        flush=True,
    )

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    started = time.perf_counter()
    try:
        assert process.stdin is not None
        for position, index in enumerate(indices):
            replay.show_frame(int(index), force=True)
            figure.canvas.draw()
            frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
            process.stdin.write(frame.tobytes())
            if position % 25 == 0 or position == len(indices) - 1:
                done = position + 1
                rate = done / max(time.perf_counter() - started, 1e-6)
                remaining = (len(indices) - done) / max(rate, 1e-6)
                print(
                    f"  {done}/{len(indices)} frames "
                    f"({rate:.1f} fps, ~{remaining:.0f}s left)",
                    flush=True,
                )
        process.stdin.close()
        code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        for reader in replay.video_readers:
            reader.close()
        plt.close(figure)

    if code:
        raise RuntimeError(f"FFmpeg failed with exit code {code}")
    size_mb = output.stat().st_size / 1024**2
    print(
        f"Wrote {output} ({size_mb:.1f} MiB, "
        f"{time.perf_counter() - started:.0f}s)"
    )
    return output


def main():
    args = parse_args()
    try:
        session = load_session(args.session, args)
        sensor_data, description, z_min, z_max, cmap, norm = (
            prepare_sensor_data(
                session["sensor"]["raw"],
                session["sensor"]["exact"],
                args,
            )
        )
        if args.display_range is None and not args.no_auto_range:
            limit = autoscale_range(sensor_data, args.mode)
            if limit is not None:
                z_min, z_max = -limit, limit
                norm = mcolors.TwoSlopeNorm(
                    vmin=z_min, vcenter=0.0, vmax=z_max
                )
                print(
                    f"Colour scale: auto +/-{limit:.3f} V "
                    f"(P{AUTO_RANGE_PERCENTILE:g} of |signal|; "
                    "use --range or --no-auto-range to override)"
                )
        fps = args.fps or session["fps"]
        if fps <= 0:
            raise RuntimeError("Playback fps must be positive")
    except Exception as exc:
        raise SystemExit(f"Replay setup failed: {exc}") from exc

    print_summary(session, args)
    print(f"Sensor display: {description} (2D top-down)")
    if args.check_only:
        validate_video_decoding(session["cameras"], args.preview_width)
        return 0

    if args.export is not None:
        try:
            export_animation(
                session, sensor_data, description, z_min, z_max, cmap, norm,
                args,
            )
        except Exception as exc:
            sys.stdout.flush()  # keep the error after the summary, not before
            print(f"Export failed: {exc}", file=sys.stderr)
            return 1
        return 0

    Replay2D(
        session,
        sensor_data,
        description,
        z_min,
        z_max,
        cmap,
        norm,
        args,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
