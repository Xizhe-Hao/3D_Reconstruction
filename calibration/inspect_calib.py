#!/usr/bin/env python3
"""Look at a calibration instead of reading its numbers.

    python inspect_calib.py <calibration.json> <detections.npz> --session <capture_dir>

RUN WITH THE **CALIBRATION** ENVIRONMENT (.venv-calib).

Why this exists
---------------
`validate.py` answers "is it good enough" with a table. It does not answer the
question you actually have while standing at the rig, which is *where* the error
is and whether the thing is usable at all. A number like "7.9 px" is impossible
to picture; two dots 8 px apart on a photograph is not.

The check this draws is the one a person can judge directly: **a point in the
world is one point, so every camera must agree on where it is.** Pick a corner,
triangulate it from the views that saw it, project it back into all four, and
look at how far the prediction lands from the feature it is supposed to hit. A
calibration that is right puts the cross on the dot in every view.

Three things are on screen:

* **the four views**, with the detected corners (green) and where the shared 3D
  point re-projects (red). The gap between them is the error, drawn at
  `--magnify` times life size so a sub-pixel error is still visible;
* **epipolar lines** -- click anywhere in any view. A point in one image must lie
  on a known line in every other image, and that line depends only on the
  calibration, not on any 3D reconstruction. Click a recognisable feature (a
  screw head, a corner of the sensor) and see whether the lines pass through the
  same feature elsewhere. This works on parts of the scene that are not the
  board, which is what makes it a real check rather than a restatement of the
  fit;
* **the 3D panel**: camera positions, optical axes, and the triangulated board.

Keys:  left/right = previous/next frame, m = cycle magnification,
       a = jump to the worst frame, r = reset the clicked point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2 as cv  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from geometry import triangulate, undistort_to_pixels  # noqa: E402
from session_io import load_session, read_gray  # noqa: E402

MAGNIFICATIONS = [1, 5, 20, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--session", type=Path, default=None,
                        help="capture directory; without it the views are drawn "
                             "on a blank background instead of the photographs")
    parser.add_argument("--frame", type=int, default=None,
                        help="frame to open on; default is the one the most "
                             "cameras see the board in")
    parser.add_argument("--magnify", type=int, default=20,
                        help="error vectors are drawn this many times life size")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here and exit instead of opening a window")
    return parser.parse_args()


class Inspector:
    def __init__(self, args):
        self.args = args
        cal = json.loads(args.calibration.read_text(encoding="utf-8"))
        self.cams = cal["cameras"]
        self.n = len(self.cams)
        self.K = [np.array(c["K"], float) for c in self.cams]
        self.D = [np.array(c["dist"], float) for c in self.cams]
        self.R = [np.array(c["R_world_to_cam"], float) for c in self.cams]
        self.t = [np.array(c["t_world_to_cam"], float) for c in self.cams]
        self.pos = [np.array(c["position_world"], float) for c in self.cams]
        self.axis = [np.array(c["optical_axis_world"], float) for c in self.cams]
        self.serials = [c["serial"] for c in self.cams]

        det = np.load(args.detections, allow_pickle=True)
        self.corners = det["corners"]
        self.counts = det["counts"]
        self.frame_indices = list(det["frame_indices"])
        self.size = det["image_sizes"][0]

        self.session = load_session(args.session) if args.session else None
        self.magnify = args.magnify
        self.click = None            # (camera, (u, v)) of the clicked pixel

        seen = (self.counts >= 12).sum(axis=0) * 1000 + self.counts.sum(axis=0)
        self.frame = int(np.argmax(seen)) if args.frame is None else \
            self.frame_indices.index(args.frame)

    # ------------------------------------------------------------------
    def reconstruct(self, fi):
        """Triangulate every corner two or more cameras measured in this frame."""
        out = {}
        for pid in range(self.corners.shape[2]):
            obs, cams = [], []
            for ci in range(self.n):
                if self.corners[ci, fi, pid, 0] >= 0:
                    uv = undistort_to_pixels(self.corners[ci, fi, pid], self.K[ci],
                                             self.D[ci])[0]
                    obs.append((self.K[ci], self.R[ci], self.t[ci], uv))
                    cams.append(ci)
            if len(obs) >= 2:
                X = triangulate(obs)
                if np.isfinite(X).all():
                    out[pid] = (X, cams)
        return out

    def project(self, ci, X):
        rvec, _ = cv.Rodrigues(self.R[ci])
        uv, _ = cv.projectPoints(X.reshape(-1, 1, 3), rvec, self.t[ci],
                                 self.K[ci], self.D[ci])
        return uv.reshape(-1, 2)

    def fundamental(self, i, j):
        Rij = self.R[j] @ self.R[i].T
        tij = self.t[j] - Rij @ self.t[i]
        tx = np.array([[0, -tij[2], tij[1]], [tij[2], 0, -tij[0]], [-tij[1], tij[0], 0]])
        return np.linalg.inv(self.K[j]).T @ tx @ Rij @ np.linalg.inv(self.K[i])

    def distort_pixels(self, ci, uv):
        """Undistorted pixels -> where they actually land on the photograph.

        The fundamental matrix relates *undistorted* coordinates, but the image
        on screen is the raw one. Drawing the straight line directly onto it
        would be off by tens of pixels near the edges at these lens parameters
        (k1 = -0.19), which is the same order as the error being judged -- so the
        line is sampled in undistorted space and pushed back through the lens.
        """
        uv = np.asarray(uv, float).reshape(-1, 2)
        K = self.K[ci]
        normalised = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0],
                                      (uv[:, 1] - K[1, 2]) / K[1, 1],
                                      np.ones(len(uv))])
        out, _ = cv.projectPoints(normalised, np.zeros(3), np.zeros(3), K, self.D[ci])
        return out.reshape(-1, 2)

    def epipolar_curve(self, src, uv, dst, samples=400):
        """The clicked point's epipolar line in camera `dst`, drawn on the raw image."""
        p = undistort_to_pixels(np.asarray(uv, float).reshape(1, 2),
                                self.K[src], self.D[src])[0]
        line = self.fundamental(src, dst) @ np.array([p[0], p[1], 1.0])
        w, h = float(self.size[0]), float(self.size[1])
        if abs(line[1]) >= abs(line[0]):
            xs = np.linspace(-0.25 * w, 1.25 * w, samples)
            ys = -(line[0] * xs + line[2]) / line[1]
        else:
            ys = np.linspace(-0.25 * h, 1.25 * h, samples)
            xs = -(line[1] * ys + line[2]) / line[0]
        pts = self.distort_pixels(dst, np.column_stack([xs, ys]))
        keep = ((pts[:, 0] > -w) & (pts[:, 0] < 2 * w)
                & (pts[:, 1] > -h) & (pts[:, 1] < 2 * h))
        return pts[keep]

    def image(self, ci, fi):
        if self.session is None:
            return None
        path = self.session.image_path(ci, self.frame_indices[fi])
        if path is None or not Path(path).exists():
            return None
        return read_gray(path, self.session.bayer)

    def worst_frame(self):
        best, worst = -1.0, self.frame
        for fi in range(self.corners.shape[1]):
            if (self.counts[:, fi] >= 12).sum() < 2:
                continue
            errs = self.errors(fi)
            if errs and np.mean([e for v in errs.values() for e in v]) > best:
                best, worst = np.mean([e for v in errs.values() for e in v]), fi
        return worst

    def errors(self, fi):
        """Re-projection error per camera, in pixels, for this frame."""
        rec = self.reconstruct(fi)
        per = {ci: [] for ci in range(self.n)}
        for pid, (X, cams) in rec.items():
            for ci in cams:
                uv = self.project(ci, X)[0]
                per[ci].append(float(np.linalg.norm(uv - self.corners[ci, fi, pid])))
        return {ci: v for ci, v in per.items() if v}

    # ------------------------------------------------------------------
    def draw(self):
        fi = self.frame
        rec = self.reconstruct(fi)
        self.fig.suptitle(
            f"frame {self.frame_indices[fi]}   "
            f"({len(rec)} corners seen by 2+ cameras)   "
            f"error vectors x{self.magnify}"
            + (f"   -- clicked in cam {self.click[0]}, epipolar lines elsewhere"
               if self.click else "   -- click any view to draw epipolar lines"),
            fontsize=11)

        for ci, ax in enumerate(self.image_axes):
            ax.clear()
            img = self.image(ci, fi)
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xlim(0, self.size[0]); ax.set_ylim(self.size[1], 0)
            ax.set_xticks([]); ax.set_yticks([])

            det, pred = [], []
            for pid, (X, cams) in rec.items():
                if ci not in cams:
                    continue
                det.append(self.corners[ci, fi, pid])
                pred.append(self.project(ci, X)[0])
            title = f"cam {ci}  {self.serials[ci]}"
            if det:
                det, pred = np.array(det), np.array(pred)
                err = np.linalg.norm(pred - det, axis=1)
                # Draw the error at `magnify` times life size, anchored on the
                # measurement, so the direction of the mistake stays readable.
                tip = det + (pred - det) * self.magnify
                ax.plot(det[:, 0], det[:, 1], ".", ms=3, color="#2ca02c",
                        label="detected")
                ax.plot([det[:, 0], tip[:, 0]], [det[:, 1], tip[:, 1]],
                        "-", lw=0.7, color="#d62728")
                ax.plot(tip[:, 0], tip[:, 1], "x", ms=3, mew=0.7, color="#d62728",
                        label="re-projected 3D point")
                title += f"   rms {np.sqrt((err ** 2).mean()):.2f} px"
            else:
                title += "   (board not seen)"
            ax.set_title(title, fontsize=9)

            if self.click is not None and self.click[0] != ci:
                src, uv = self.click
                curve = self.epipolar_curve(src, uv, ci)
                if len(curve) > 1:
                    ax.plot(curve[:, 0], curve[:, 1], "-", lw=1.4, color="#1f77b4")
            if self.click is not None and self.click[0] == ci:
                ax.plot(*self.click[1], "o", ms=9, mfc="none", mew=1.6, color="#1f77b4")
            if ci == 0 and det is not None and len(det):
                ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

        ax3 = self.ax3d
        ax3.clear()
        for ci in range(self.n):
            p, a = self.pos[ci], self.axis[ci]
            ax3.scatter(*p, s=40)
            ax3.quiver(*p, *(a * 60), color="0.4", lw=1)
            ax3.text(*p, f" {ci}", fontsize=8)
        if rec:
            P = np.array([X for X, _ in rec.values()])
            ax3.scatter(P[:, 0], P[:, 1], P[:, 2], s=3, color="#2ca02c")
        ax3.scatter([0], [0], [0], marker="x", s=60, color="red")
        ax3.set_xlabel("x (mm)", fontsize=8); ax3.set_ylabel("y (mm)", fontsize=8)
        ax3.set_zlabel("z (mm)", fontsize=8)
        ax3.set_title("cameras + triangulated board\n(red x = tactile array origin)",
                      fontsize=9)
        ax3.tick_params(labelsize=7)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def on_click(self, event):
        for ci, ax in enumerate(self.image_axes):
            if event.inaxes is ax and event.xdata is not None:
                self.click = (ci, (float(event.xdata), float(event.ydata)))
                self.draw()
                return

    def on_key(self, event):
        total = self.corners.shape[1]
        if event.key == "right":
            self.frame = (self.frame + 1) % total
        elif event.key == "left":
            self.frame = (self.frame - 1) % total
        elif event.key == "m":
            self.magnify = MAGNIFICATIONS[
                (MAGNIFICATIONS.index(self.magnify) + 1) % len(MAGNIFICATIONS)
                if self.magnify in MAGNIFICATIONS else 0]
        elif event.key == "a":
            self.frame = self.worst_frame()
        elif event.key == "r":
            self.click = None
        else:
            return
        self.draw()

    def run(self):
        self.fig = plt.figure(figsize=(16, 8))
        grid = GridSpec(2, 3, figure=self.fig, width_ratios=[1, 1, 1.1])
        self.image_axes = [self.fig.add_subplot(grid[r, c])
                           for r in (0, 1) for c in (0, 1)][:self.n]
        self.ax3d = self.fig.add_subplot(grid[:, 2], projection="3d")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()
        if self.args.save:
            self.fig.savefig(self.args.save, dpi=130, bbox_inches="tight")
            print(f"wrote {self.args.save}")
        else:
            plt.show()


def main() -> int:
    args = parse_args()
    ins = Inspector(args)
    errs = ins.errors(ins.frame)
    print(f"opening on frame {ins.frame_indices[ins.frame]}")
    for ci, v in errs.items():
        print(f"  cam {ci} ({ins.serials[ci]}): {len(v):3d} corners, "
              f"rms {np.sqrt((np.array(v) ** 2).mean()):6.2f} px, "
              f"max {max(v):6.2f} px")
    print("\nclick any view to draw epipolar lines in the others; "
          "left/right = frame, m = magnify, a = worst frame, r = reset")
    ins.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
