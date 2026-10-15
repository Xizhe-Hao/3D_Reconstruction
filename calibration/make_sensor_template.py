#!/usr/bin/env python3
"""Printable corner tags + a 1:1 layout sheet for the 16x16 tactile array.

    python calibration/make_sensor_template.py                       # -> calibration/sensor_template/
    python calibration/make_sensor_template.py --tag-mm 12 15 --paper a4

RUN WITH THE CALIBRATION ENVIRONMENT (.venv-calib).

Writes sensor_template.pdf (print at 100% / "actual size") and
sensor_template.json, which records everything a detector needs.

Page 1, tags: four AprilTags per size, one per corner of the active area,
two sets per size. Cut along the thin grey line (it sits just outside the
white quiet zone -- keep the white, the detector needs it).

    ID 0 -> R1C1   ID 1 -> R1C16   ID 2 -> R16C16   ID 3 -> R16C1

Placement: each tag sits diagonally OUTSIDE its corner, edges parallel to the
array edges, its inner corner (the one pointing at the array centre) on the
corner of the 120 x 120 mm active area. The tag's rotation does not matter: the
inner corner is the one nearest the centre of the four tags. If there is no
room to put them exactly on the corners, keep all four at the SAME diagonal
offset; the grid is placed from the tags' centre and edge directions plus the
known pitch, exactly as sensor_grid.py does with the tapes.

Page 2, layout (1:1): the active area, all 256 electrode pads, row/column
labels, the tag footprints and the bench-test press points P1-P9. Seen from
above with R1 at the top, C1 to C16 runs left to right and R1 to R16 top to
bottom, which is how the calibrated world frame sees the sensor (+z up).
Print it on transparency film to lay over the sensor and mark press points.

Simulated in the four calibrated views (calibration.json + sensor_grid.json),
tag36h11 at 12-15 mm is read in 3-4 cameras per corner with sharp focus; the
views are oblique enough that larger tags start leaving the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAPER_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
FAMILIES = {  # name -> (OpenCV dictionary, black-square width in bits)
    "36h11": ("DICT_APRILTAG_36h11", 8),
    "25h9": ("DICT_APRILTAG_25h9", 7),
    "16h5": ("DICT_APRILTAG_16h5", 6),
}
CORNERS = {0: "R1C1", 1: "R1C16", 2: "R16C16", 3: "R16C1"}

CELLS = 16
PITCH_MM = 7.5
PAD_MM = 5.0
SIDE_MM = CELLS * PITCH_MM

# Bench-test press points, (u, v) in mm from the outer corner of R1C1:
# u along C1 -> C16, v along R1 -> R16.  Cell (r, c) centre = ((c-1)*7.5 + 3.75, (r-1)*7.5 + 3.75).
TEST_POINTS = [
    ("P1", 56.25, 56.25, "pad centre R8C8 (single-cell tests T2-T4)"),
    ("P2", 60.00, 56.25, "gap between R8C8 and R8C9"),
    ("P3", 60.00, 60.00, "4-pad gap intersection, array centre"),
    ("P4", 26.25, 26.25, "pad centre R4C4"),
    ("P5", 93.75, 26.25, "pad centre R4C13"),
    ("P6", 26.25, 93.75, "pad centre R13C4"),
    ("P7", 93.75, 93.75, "pad centre R13C13"),
    ("P8", 56.25, 3.75, "edge pad R1C8"),
    ("P9", 3.75, 3.75, "corner pad R1C1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate corner AprilTags and a 1:1 layout sheet for the tactile array",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family", default="36h11", choices=sorted(FAMILIES))
    parser.add_argument("--tag-mm", type=float, nargs="+", default=[12.0, 15.0],
                        help="black-square side of each tag size to print")
    parser.add_argument("--layout-tag-mm", type=float, default=None,
                        help="tag size drawn as footprints on the layout page; default: largest --tag-mm")
    parser.add_argument("--paper", default="letter", choices=sorted(PAPER_MM))
    parser.add_argument("--dpi", type=float, default=600.0)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "sensor_template")
    return parser.parse_args()


class Page:
    """A white raster page addressed in millimetres."""

    def __init__(self, paper_mm: tuple[float, float], dpi: float) -> None:
        self.ppm = dpi / 25.4
        self.dpi = dpi
        w, h = (int(round(v * self.ppm)) for v in paper_mm)
        self.image = Image.new("L", (w, h), 255)
        self.draw = ImageDraw.Draw(self.image)

    def px(self, mm: float) -> int:
        return int(round(mm * self.ppm))

    def font(self, size_mm: float) -> ImageFont.FreeTypeFont:
        size = max(6, self.px(size_mm))
        for name in ("arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default(size=size)

    def text(self, x_mm: float, y_mm: float, text: str, size_mm: float = 3.0,
             fill: int = 0, anchor: str = "la") -> None:
        self.draw.text((self.px(x_mm), self.px(y_mm)), text, fill=fill,
                       font=self.font(size_mm), anchor=anchor)

    def rect(self, x_mm: float, y_mm: float, w_mm: float, h_mm: float,
             outline: int | None = 0, fill: int | None = None, width_mm: float = 0.2) -> None:
        self.draw.rectangle(
            (self.px(x_mm), self.px(y_mm), self.px(x_mm + w_mm) - 1, self.px(y_mm + h_mm) - 1),
            outline=outline, fill=fill, width=max(1, self.px(width_mm)))

    def dashed_rect(self, x_mm: float, y_mm: float, w_mm: float, h_mm: float,
                    fill: int = 0, width_mm: float = 0.2, dash_mm: float = 1.0) -> None:
        corners = [(x_mm, y_mm), (x_mm + w_mm, y_mm), (x_mm + w_mm, y_mm + h_mm), (x_mm, y_mm + h_mm)]
        for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
            length = float(np.hypot(x1 - x0, y1 - y0))
            steps = max(1, int(length / dash_mm))
            for i in range(0, steps, 2):
                a, b = i / steps, min(1.0, (i + 1) / steps)
                self.draw.line((self.px(x0 + (x1 - x0) * a), self.px(y0 + (y1 - y0) * a),
                                self.px(x0 + (x1 - x0) * b), self.px(y0 + (y1 - y0) * b)),
                               fill=fill, width=max(1, self.px(width_mm)))

    def paste(self, tile: np.ndarray, x_mm: float, y_mm: float) -> None:
        self.image.paste(Image.fromarray(tile), (self.px(x_mm), self.px(y_mm)))

    def scale_bar(self, x_mm: float, y_mm: float, length_mm: float = 100.0) -> None:
        width = max(1, self.px(0.25))
        self.draw.line((self.px(x_mm), self.px(y_mm), self.px(x_mm + length_mm), self.px(y_mm)),
                       fill=0, width=width)
        for mm in range(int(length_mm) + 1):
            tick = 3.0 if mm % 10 == 0 else 1.5 if mm % 5 == 0 else 0.8
            self.draw.line((self.px(x_mm + mm), self.px(y_mm), self.px(x_mm + mm), self.px(y_mm - tick)),
                           fill=0, width=width)
        self.text(x_mm + length_mm + 2, y_mm, f"{length_mm:g} mm -- measure to check the print scale",
                  2.5, anchor="lm")


def tag_tile(dictionary: str, black_bits: int, tag_id: int, black_mm: float, ppm: float) -> np.ndarray:
    """Black-bordered tag plus a one-bit white quiet zone, at exactly black_mm."""
    aruco = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dictionary))
    total_bits = black_bits + 2
    side_px = int(round(black_mm * total_bits / black_bits * ppm))
    small = cv.aruco.generateImageMarker(aruco, tag_id, black_bits * 10, borderBits=1)
    small = cv.copyMakeBorder(small, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=255)
    return cv.resize(small, (side_px, side_px), interpolation=cv.INTER_NEAREST)


def tags_page(args: argparse.Namespace, dictionary: str, bits: int) -> Image.Image:
    page = Page(PAPER_MM[args.paper], args.dpi)
    margin = 15.0
    page.text(margin, margin, f"Tactile array corner tags -- tag{args.family}", 5.0)
    page.text(margin, margin + 7, "Print at 100% / actual size.  Cut on the grey line; keep the white border.", 3.0)
    page.text(margin, margin + 11.5,
              "Each tag goes diagonally outside its corner, inner corner on the corner of the 120 mm active area.", 3.0)
    page.text(margin, margin + 16, "All four at the same offset.  Rotation does not matter.  Glue flat, no gloss.", 3.0)
    y = margin + 26
    for black_mm in args.tag_mm:
        total = black_mm * (bits + 2) / bits
        page.text(margin, y, f"black square {black_mm:g} mm  (with white border {total:.1f} mm)", 3.5)
        y += 6
        for copy in range(2):
            x = margin
            for tag_id, corner in CORNERS.items():
                page.paste(tag_tile(dictionary, bits, tag_id, black_mm, page.ppm), x, y)
                page.rect(x - 0.4, y - 0.4, total + 0.8, total + 0.8, outline=170, width_mm=0.1)
                page.text(x + total / 2, y + total + 1.5, f"ID {tag_id} -> {corner}", 2.6, anchor="ma")
                x += max(total, 22.0) + 12.0
            y += total + 9.0
        y += 4
    page.scale_bar(margin, PAPER_MM[args.paper][1] - margin)
    return page.image


def layout_page(args: argparse.Namespace, bits: int) -> Image.Image:
    page = Page(PAPER_MM[args.paper], args.dpi)
    paper_w, paper_h = PAPER_MM[args.paper]
    tag_total = args.layout_tag_mm * (bits + 2) / bits
    ox = (paper_w - SIDE_MM) / 2
    oy = 15.0 + 22.0 + tag_total + 6.0
    page.text(15, 15, "Tactile array layout, 1:1  (seen from above, R1 at the top)", 5.0)
    page.text(15, 22, f"Active area {SIDE_MM:g} x {SIDE_MM:g} mm, 16 x 16 pads of {PAD_MM:g} x {PAD_MM:g} mm "
              f"at {PITCH_MM:g} mm pitch.  Print on transparency to lay over the sensor.", 3.0)
    page.text(15, 26.5, f"Dashed squares: tag footprints ({args.layout_tag_mm:g} mm tags).  "
              "+ : bench-test press points P1-P9.", 3.0)

    for r in range(CELLS):
        for c in range(CELLS):
            x = ox + c * PITCH_MM + (PITCH_MM - PAD_MM) / 2
            y = oy + r * PITCH_MM + (PITCH_MM - PAD_MM) / 2
            page.rect(x, y, PAD_MM, PAD_MM, outline=150, fill=232, width_mm=0.1)
    page.rect(ox, oy, SIDE_MM, SIDE_MM, outline=0, width_mm=0.3)
    for i in range(CELLS):
        centre = (i + 0.5) * PITCH_MM
        page.text(ox + centre, oy - 1.5, f"C{i + 1}", 2.0, anchor="md")
        page.text(ox - 1.5, oy + centre, f"R{i + 1}", 2.0, anchor="rm")

    offsets = {0: (-tag_total, -tag_total), 1: (SIDE_MM, -tag_total),
               2: (SIDE_MM, SIDE_MM), 3: (-tag_total, SIDE_MM)}
    for tag_id, (du, dv) in offsets.items():
        page.dashed_rect(ox + du, oy + dv, tag_total, tag_total, width_mm=0.2)
        page.text(ox + du + tag_total / 2, oy + dv + tag_total / 2, f"ID {tag_id}", 3.0, anchor="mm")

    arm = 2.5
    for name, u, v, _ in TEST_POINTS:
        x, y = ox + u, oy + v
        page.draw.line((page.px(x - arm), page.px(y), page.px(x + arm), page.px(y)), fill=0, width=page.px(0.2))
        page.draw.line((page.px(x), page.px(y - arm), page.px(x), page.px(y + arm)), fill=0, width=page.px(0.2))
        page.text(x + 1.0, y - 1.0, name, 2.4, anchor="ld")

    y = oy + SIDE_MM + tag_total + 8
    page.text(15, y, "Press points (u, v) in mm from the outer corner of R1C1; u along C1->C16, v along R1->R16:", 2.8)
    for i, (name, u, v, note) in enumerate(TEST_POINTS):
        page.text(18 + (i % 2) * 95, y + 5 + (i // 2) * 4.2, f"{name}  ({u:.2f}, {v:.2f})  {note}", 2.5)
    page.scale_bar(15, paper_h - 12)
    return page.image


def main() -> None:
    args = parse_args()
    dictionary, bits = FAMILIES[args.family]
    if args.layout_tag_mm is None:
        args.layout_tag_mm = max(args.tag_mm)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pages = [tags_page(args, dictionary, bits), layout_page(args, bits)]
    pdf = args.out_dir / "sensor_template.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=args.dpi)

    spec = {
        "schema": "sensor-template/1",
        "units": "millimetre",
        "family": args.family,
        "opencv_dictionary": dictionary,
        "black_bits": bits,
        "tag_black_mm": args.tag_mm,
        "tag_total_mm": [round(m * (bits + 2) / bits, 3) for m in args.tag_mm],
        "tag_ids": {str(k): v for k, v in CORNERS.items()},
        "placement": ("each tag diagonally outside its corner of the active area, edges parallel to the "
                      "array, inner corner (nearest the centre of the four tags) on the active-area corner; "
                      "all four at the same offset if not exactly on the corners"),
        "array": {"cells": CELLS, "pitch_mm": PITCH_MM, "pad_mm": PAD_MM, "active_side_mm": SIDE_MM,
                  "frame": "u along C1->C16, v along R1->R16, origin at the outer corner of R1C1; "
                           "u x v points into the sensor (-z of the calibrated world)"},
        "test_points": [{"name": n, "u_mm": u, "v_mm": v, "note": note} for n, u, v, note in TEST_POINTS],
        "print_note": "print at 100%; check the 100 mm scale bar and the tag black squares with callipers",
    }
    (args.out_dir / "sensor_template.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"wrote {pdf}")
    print(f"wrote {args.out_dir / 'sensor_template.json'}")


if __name__ == "__main__":
    main()
