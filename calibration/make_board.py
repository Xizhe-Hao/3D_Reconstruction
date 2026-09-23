#!/usr/bin/env python3
"""Generate a printable ChArUco target plus the board.json that defines it.

    python make_board.py --out-dir board

Print the PDF at 100% / "actual size" (no "fit to page"), glue it to a rigid
flat substrate (glass or aluminium -- a curled sheet of paper is the single
largest error source in a calibration like this), then MEASURE a square with
callipers and, if the print scaled, correct `square_mm` in board.json. Every
later script reads that file, so the measured value propagates everywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

from charuco_board import ARUCO_DICTS, BoardSpec

PAPER_MM = {"a4": (210.0, 297.0), "a3": (297.0, 420.0), "letter": (215.9, 279.4)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a ChArUco calibration target and its board.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--squares-x", type=int, default=8, help="squares across")
    parser.add_argument("--squares-y", type=int, default=11, help="squares down")
    parser.add_argument("--square-mm", type=float, default=12.0)
    parser.add_argument(
        "--marker-mm",
        type=float,
        default=None,
        help="marker side; default is 0.75 x square-mm",
    )
    parser.add_argument("--dict", dest="dictionary", default="DICT_4X4_50",
                        choices=sorted(ARUCO_DICTS))
    parser.add_argument("--dpi", type=float, default=600.0,
                        help="target print resolution; 600 keeps marker edges crisp")
    parser.add_argument("--margin-mm", type=float, default=10.0,
                        help="white quiet zone around the board (needed for detection)")
    parser.add_argument("--paper", default="a4", choices=sorted(PAPER_MM) + ["none"])
    parser.add_argument("--out-dir", type=Path, default=Path("board"))
    parser.add_argument("--note", default="", help="free text stored in board.json")
    return parser.parse_args()


def draw_scale_bar(page: np.ndarray, px_per_mm: float, origin_px: tuple[int, int],
                   length_mm: float = 50.0) -> None:
    """A ruler printed next to the board so print scaling can be checked."""
    x0, y0 = origin_px
    length_px = int(round(length_mm * px_per_mm))
    thickness = max(1, int(round(0.3 * px_per_mm)))
    cv.line(page, (x0, y0), (x0 + length_px, y0), 0, thickness)
    for millimetre in range(0, int(length_mm) + 1):
        if millimetre % 10 == 0:
            tick = int(round(3.0 * px_per_mm))
        elif millimetre % 5 == 0:
            tick = int(round(2.0 * px_per_mm))
        else:
            tick = int(round(1.0 * px_per_mm))
        x = x0 + int(round(millimetre * px_per_mm))
        cv.line(page, (x, y0), (x, y0 - tick), 0, thickness)
    cv.putText(page, f"{length_mm:g} mm -- measure me", (x0, y0 + int(4.5 * px_per_mm)),
               cv.FONT_HERSHEY_SIMPLEX, 0.045 * px_per_mm, 0, thickness)


def main() -> int:
    args = parse_args()
    marker_mm = args.marker_mm if args.marker_mm is not None else 0.75 * args.square_mm

    spec = BoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_mm=args.square_mm,
        marker_mm=round(marker_mm, 4),
        dictionary=args.dictionary,
        note=args.note,
    )
    spec.validate()
    board = spec.build()

    # Snap one square to a whole number of pixels and then report the DPI that
    # makes that exact: this removes the sub-pixel rounding that would
    # otherwise make the printed square differ from board.json.
    px_per_square = int(round(spec.square_mm * args.dpi / 25.4))
    if px_per_square < 40:
        raise SystemExit(
            f"only {px_per_square} px per square at {args.dpi} dpi -- raise --dpi"
        )
    effective_dpi = px_per_square * 25.4 / spec.square_mm
    px_per_mm = px_per_square / spec.square_mm

    board_px = (spec.squares_x * px_per_square, spec.squares_y * px_per_square)
    board_img = board.generateImage(board_px, marginSize=0, borderBits=1)

    margin_px = int(round(args.margin_mm * px_per_mm))
    caption_px = int(round(22.0 * px_per_mm))
    page = np.full(
        (board_px[1] + 2 * margin_px + caption_px, board_px[0] + 2 * margin_px),
        255, np.uint8,
    )
    page[margin_px:margin_px + board_px[1], margin_px:margin_px + board_px[0]] = board_img

    text_y = margin_px + board_px[1] + int(round(6.0 * px_per_mm))
    caption = (
        f"ChArUco {spec.squares_x}x{spec.squares_y}  square={spec.square_mm:g}mm  "
        f"marker={spec.marker_mm:g}mm  {spec.dictionary}  --  PRINT AT 100%"
    )
    cv.putText(page, caption, (margin_px, text_y), cv.FONT_HERSHEY_SIMPLEX,
               0.05 * px_per_mm, 0, max(1, int(round(0.3 * px_per_mm))))
    draw_scale_bar(page, px_per_mm, (margin_px, text_y + int(round(9.0 * px_per_mm))))

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_{spec.squares_x}x{spec.squares_y}_{spec.square_mm:g}mm"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    json_path = out_dir / "board.json"

    pil = Image.fromarray(page)
    pil.save(png_path, dpi=(effective_dpi, effective_dpi))
    pil.convert("L").save(pdf_path, resolution=effective_dpi)
    spec.save(json_path)

    page_w_mm = page.shape[1] / px_per_mm
    page_h_mm = page.shape[0] / px_per_mm
    print(f"board      : {spec.describe()}")
    print(f"print at   : {effective_dpi:.3f} dpi  ({px_per_square} px per square)")
    print(f"page size  : {page_w_mm:.1f} x {page_h_mm:.1f} mm")
    if args.paper != "none":
        paper_w, paper_h = PAPER_MM[args.paper]
        fits = (page_w_mm <= paper_w and page_h_mm <= paper_h) or (
            page_w_mm <= paper_h and page_h_mm <= paper_w)
        print(f"fits {args.paper.upper():6}: {'yes' if fits else 'NO -- shrink the board or use bigger paper'}")
    print(f"wrote      : {pdf_path}")
    print(f"             {png_path}")
    print(f"             {json_path}")
    print()
    print("After printing: measure one square with callipers. If it is not")
    print(f"{spec.square_mm:g} mm, put the MEASURED value into {json_path} -- the")
    print("whole calibration is scaled by that number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
