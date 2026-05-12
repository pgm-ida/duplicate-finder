#!/usr/bin/env python3
"""Side-by-side comparison of rembg and OpenCV magazine-quad detectors.

For each JPG in the input folder, this script:

  1. Runs both `_find_magazine_quad_rembg` and `_find_magazine_quad` directly
     on the image (no warp/tighten/deskew — only raw quad detection).
  2. Writes a downscaled overlay (`<stem>_rembg_quad.jpg`,
     `<stem>_opencv_quad.jpg`) showing the detected polygon and corners.
  3. Runs the full `crop_magazine` pipeline twice — once with only the
     rembg detector active, once with only the OpenCV detector active — and
     writes the warped output (`<stem>_rembg_crop.jpg`, `<stem>_opencv_crop.jpg`).

Outputs land in `<input_folder>/_test_results/`.

Usage:
    python test_cropping.py                       # uses ./_demo_test_photos
    python test_cropping.py path/to/folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

import find_duplicates_local as fdl


IMAGE_EXTS = {".jpg", ".jpeg"}
MAX_OVERLAY_WIDTH = 1200
POLYLINE_THICKNESS = 12
CORNER_RADIUS = 25
GREEN = (0, 255, 0)        # BGR
RED = (0, 0, 255)          # BGR


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _draw_quad_overlay(img_bgr: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    """Return a downscaled copy of `img_bgr` with the quad drawn on top
    (green polyline + red corner circles). If `quad` is None, just downscale."""
    out = img_bgr.copy()
    if quad is not None:
        ordered = fdl._order_quad(quad)
        pts = ordered.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=GREEN,
                      thickness=POLYLINE_THICKNESS, lineType=cv2.LINE_AA)
        for (x, y) in ordered.astype(int):
            cv2.circle(out, (int(x), int(y)), CORNER_RADIUS, RED,
                       thickness=-1, lineType=cv2.LINE_AA)

    h, w = out.shape[:2]
    if w > MAX_OVERLAY_WIDTH:
        scale = MAX_OVERLAY_WIDTH / w
        out = cv2.resize(out, (MAX_OVERLAY_WIDTH, int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return out


def _quad_area(quad: np.ndarray | None) -> float:
    if quad is None:
        return 0.0
    return float(cv2.contourArea(quad.astype(np.float32)))


# Pixel threshold for the per-corner verdict. Each corner is judged against
# its counterpart in the other detector's quad. If, averaged across the four
# corners, the rembg quad lies more than this many pixels INWARD of the
# opencv quad (or vice versa), that detector is judged to have cropped
# tighter. Calibrated empirically: identical-quad noise sits around 1-2 px,
# a single-side bleed (e.g. opencv including the bound-book binding on the
# left only) averages 20-30 px across all four corners. Threshold of 20 picks
# up that case while staying well above the noise floor.
CORNER_INWARD_THRESHOLD_PX = 20


def _avg_inward_displacement(rembg_quad: np.ndarray, opencv_quad: np.ndarray) -> float:
    """Signed average pixel distance that rembg corners lie INWARD of opencv
    corners. Positive = rembg cropped tighter; negative = opencv cropped tighter.

    Inward direction is corner-specific: TL inward = down + right, TR inward
    = down + left, BR inward = up + left, BL inward = up + right.
    """
    r = fdl._order_quad(rembg_quad)
    o = fdl._order_quad(opencv_quad)
    tl = (r[0, 0] - o[0, 0]) + (r[0, 1] - o[0, 1])
    tr = (o[1, 0] - r[1, 0]) + (r[1, 1] - o[1, 1])
    br = (o[2, 0] - r[2, 0]) + (o[2, 1] - r[2, 1])
    bl = (r[3, 0] - o[3, 0]) + (o[3, 1] - r[3, 1])
    # Each term sums an x and a y delta, so divide by 2 to get per-axis pixels,
    # then average across the four corners.
    return float((tl + tr + br + bl) / 8)


def _verdict(rembg_quad: np.ndarray | None,
             opencv_quad: np.ndarray | None) -> tuple[str, float]:
    """Returns (verdict, score). Score is signed inward displacement in pixels
    (positive = rembg tighter, negative = opencv tighter; 0 when one failed)."""
    if rembg_quad is None and opencv_quad is None:
        return ("both failed", 0.0)
    if opencv_quad is None:
        return ("rembg only (opencv failed)", 0.0)
    if rembg_quad is None:
        return ("opencv only (rembg failed)", 0.0)
    score = _avg_inward_displacement(rembg_quad, opencv_quad)
    if score > CORNER_INWARD_THRESHOLD_PX:
        return ("rembg tighter", score)
    if score < -CORNER_INWARD_THRESHOLD_PX:
        return ("opencv tighter", score)
    return ("tie", score)


def _crop_with_forced_method(src: Path, dst: Path, method: str) -> tuple[int, int] | None:
    """Run `crop_magazine` forcing one detector. `method` is 'rembg' or 'opencv'.

    Uses monkeypatching of `fdl` module globals to disable the unwanted
    detector for the duration of the call.
    """
    if method == "opencv":
        original_flag = fdl._REMBG_AVAILABLE
        fdl._REMBG_AVAILABLE = False
        try:
            return fdl.crop_magazine(src, dst)
        finally:
            fdl._REMBG_AVAILABLE = original_flag

    if method == "rembg":
        # Disable the OpenCV fallback by replacing the function with one that
        # always returns None. crop_magazine's two-tier logic will then only
        # succeed if rembg succeeds.
        original_fn = fdl._find_magazine_quad
        fdl._find_magazine_quad = lambda _img: None
        try:
            return fdl.crop_magazine(src, dst)
        finally:
            fdl._find_magazine_quad = original_fn

    raise ValueError(f"unknown method: {method}")


def process_one(img_path: Path, out_dir: Path) -> dict:
    """Returns a per-photo result dict for the summary."""
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        return {"path": img_path, "error": "cv2.imread returned None"}

    quad_rembg = fdl._find_magazine_quad_rembg(img) if fdl.rembg_available() else None
    quad_opencv = fdl._find_magazine_quad(img) if fdl.cropping_available() else None

    cv2.imwrite(str(out_dir / f"{stem}_rembg_quad.jpg"),
                _draw_quad_overlay(img, quad_rembg))
    cv2.imwrite(str(out_dir / f"{stem}_opencv_quad.jpg"),
                _draw_quad_overlay(img, quad_opencv))

    rembg_crop = _crop_with_forced_method(
        img_path, out_dir / f"{stem}_rembg_crop.jpg", "rembg"
    )
    opencv_crop = _crop_with_forced_method(
        img_path, out_dir / f"{stem}_opencv_crop.jpg", "opencv"
    )

    return {
        "path": img_path,
        "rembg_quad": quad_rembg,
        "opencv_quad": quad_opencv,
        "rembg_crop": rembg_crop,
        "opencv_crop": opencv_crop,
        "rembg_area": _quad_area(quad_rembg),
        "opencv_area": _quad_area(quad_opencv),
    }


def _print_summary(results: list[dict]) -> None:
    total = len(results)
    rembg_ok = sum(1 for r in results if r.get("rembg_quad") is not None)
    opencv_ok = sum(1 for r in results if r.get("opencv_quad") is not None)
    rembg_crop_ok = sum(1 for r in results if r.get("rembg_crop") is not None)
    opencv_crop_ok = sum(1 for r in results if r.get("opencv_crop") is not None)

    print()
    print("=" * 60)
    print(f"Photos processed       : {total}")
    print(f"rembg quad detected    : {rembg_ok}/{total}")
    print(f"opencv quad detected   : {opencv_ok}/{total}")
    print(f"rembg crop_magazine OK : {rembg_crop_ok}/{total}")
    print(f"opencv crop_magazine OK: {opencv_crop_ok}/{total}")
    print("=" * 60)

    # Per-photo verdict by signed inward corner displacement (see
    # _avg_inward_displacement). Catches the "opencv latches onto the bound
    # book's leather binding, rembg cuts along the page edge" case even when
    # the area difference is only a few percent.
    verdict_counts = {
        "rembg tighter": 0,
        "opencv tighter": 0,
        "tie": 0,
        "rembg only (opencv failed)": 0,
        "opencv only (rembg failed)": 0,
        "both failed": 0,
    }
    print()
    print("Per-photo verdict (rembg vs opencv quad alignment):")
    for r in results:
        verdict, score = _verdict(r.get("rembg_quad"), r.get("opencv_quad"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        score_str = f" ({score:+.0f}px)" if score else ""
        print(f"  {r['path'].name:36s}  {verdict}{score_str}")

    print()
    print("Verdict summary:")
    for label, count in verdict_counts.items():
        if count:
            print(f"  {label:32s}  {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "folder", nargs="?", default="./_demo_test_photos",
        help="Input folder of JPGs (default: ./_demo_test_photos)",
    )
    args = parser.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        return 2

    if not fdl.cropping_available():
        print("ERROR: opencv (cv2) not available — cannot run comparison.",
              file=sys.stderr)
        return 2
    if not fdl.rembg_available():
        print("WARNING: rembg not available — only opencv outputs will be useful.")

    images = _list_images(folder)
    if not images:
        print(f"No JPGs found in {folder}")
        return 0

    out_dir = folder / "_test_results"
    out_dir.mkdir(exist_ok=True)
    print(f"Writing outputs to {out_dir}")

    results = []
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            results.append(process_one(img_path, out_dir))
        except Exception as exc:
            print(f"  failed: {exc}")
            results.append({"path": img_path, "error": str(exc)})

    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
