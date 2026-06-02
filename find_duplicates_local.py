#!/usr/bin/env python3
"""
Magazine duplicate finder — 100% local, no API needed.

Uses perceptual image hashing (pHash) to find photos that show the same
magazine page. When several consecutive photos in one stack match several
consecutive photos in another part of the stack, that's a duplicate copy
of the same magazine issue.

Usage:
    python find_duplicates_local.py <images_folder>

File naming expected: YYYY_MM_DD_PublicationName_NNNN.JPG
    e.g. 2026_05_08_NME_0611.JPG
         2026_05_08_Recordmirror_0050.JPG

Walks subfolders, so images organized by collection
(e.g. images/NME/, images/Recordmirror/) are picked up automatically.
"""

import json
import logging
import os
import re
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

_LOGGER = logging.getLogger(__name__)

# Force unbuffered output so progress prints visibly during long runs
# Also force UTF-8 so unicode characters in the report don't crash on Windows
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

import imagehash
from PIL import Image

# Optional OCR support. If pytesseract or the tesseract binary is missing,
# we silently fall back to "no OCR" — items will lack extracted dates but
# the rest of the pipeline (hash + spread detection) keeps working.
_OCR_AVAILABLE = None  # tri-state: None=untested, True=working, False=unavailable
try:
    import pytesseract
    _PYTESSERACT_IMPORTED = True
except ImportError:
    _PYTESSERACT_IMPORTED = False


def ocr_available() -> bool:
    """Returns True if pytesseract+tesseract are usable (caches result)."""
    global _OCR_AVAILABLE
    if _OCR_AVAILABLE is not None:
        return _OCR_AVAILABLE
    if not _PYTESSERACT_IMPORTED:
        _OCR_AVAILABLE = False
        return False
    try:
        # A cheap call that actually invokes the binary.
        pytesseract.get_tesseract_version()
        _OCR_AVAILABLE = True
    except Exception:
        _OCR_AVAILABLE = False
    return _OCR_AVAILABLE


# ─── OpenCV cropping (optional) ──────────────────────────────────────────────
# We use opencv-python-headless for contour-based perspective unwarp. If cv2 is
# missing, we silently fall back to "no cropping" — the pipeline still works on
# the original images.
try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None
    _CV2_AVAILABLE = False


def cropping_available() -> bool:
    return _CV2_AVAILABLE


# ─── rembg foreground segmentation (optional) ───────────────────────────────
# U^2-Net based foreground segmentation. Used as the primary magazine-detector
# in crop_magazine, with the OpenCV contour-based detector as fallback. The
# actual `from rembg import ...` is deferred to first use — importing rembg
# pulls in onnxruntime + scipy + numba and adds ~60 seconds to .exe startup
# on cold disk. With lazy import, startup is fast and the first crop pays
# the init cost on the tether worker thread (where the user doesn't notice).
#
# When running from a PyInstaller bundle (`sys.frozen`), point U2NET_HOME at
# the embedded model directory so rembg doesn't try to download weights on
# first launch. build.ps1 stages `u2net.onnx` into `<bundle>/u2net/`.
if getattr(sys, "frozen", False):
    _bundled_u2net = Path(getattr(sys, "_MEIPASS", "")) / "u2net"
    if _bundled_u2net.is_dir():
        os.environ.setdefault("U2NET_HOME", str(_bundled_u2net))

_REMBG_AVAILABLE: "bool | None" = None  # None = not yet probed
_rembg_new_session = None  # populated on first successful probe
_rembg_remove = None
_REMBG_SESSION = None


def rembg_available() -> bool:
    """Returns True if rembg can be imported. Caches the result of the first
    probe — the actual import only happens on the first call."""
    global _REMBG_AVAILABLE, _rembg_new_session, _rembg_remove
    if _REMBG_AVAILABLE is not None:
        return _REMBG_AVAILABLE
    try:
        from rembg import new_session as _ns
        from rembg import remove as _rm
        _rembg_new_session = _ns
        _rembg_remove = _rm
        _REMBG_AVAILABLE = True
    except Exception as exc:
        # Broader than ImportError on purpose: rembg's onnxruntime backend can
        # raise OSError/RuntimeError or metadata-lookup errors in a frozen
        # .exe. Treat any failure as "not available" so the app falls back
        # to the OpenCV cropper instead of crashing.
        _LOGGER.warning("rembg import failed: %s", exc)
        _REMBG_AVAILABLE = False
    return _REMBG_AVAILABLE


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None and rembg_available():
        _REMBG_SESSION = _rembg_new_session("u2net")
    return _REMBG_SESSION


def _order_quad(pts):
    """Order 4 corner points as TL, TR, BR, BL."""
    import numpy as _np
    rect = _np.zeros((4, 2), dtype=_np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[_np.argmin(s)]
    rect[2] = pts[_np.argmax(s)]
    d = _np.diff(pts, axis=1)
    rect[1] = pts[_np.argmin(d)]
    rect[3] = pts[_np.argmax(d)]
    return rect


def _find_magazine_quad(img_bgr):
    """Return 4 corner points of the largest contiguous magazine region,
    or None if no plausible quad was found.

    Strategy: threshold against corner-median background → morphological close
    → largest external contour → approximate to 4 points (try several
    epsilons) → fall back to rotated bounding rectangle.
    """
    import numpy as _np
    h, w = img_bgr.shape[:2]
    gray = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2GRAY)

    # Background reference = median of corner regions
    cs = max(20, min(h, w) // 50)
    corner_pixels = _np.concatenate([
        gray[:cs, :cs].flatten(),
        gray[:cs, -cs:].flatten(),
        gray[-cs:, :cs].flatten(),
        gray[-cs:, -cs:].flatten(),
    ])
    bg = int(_np.median(corner_pixels))

    # Magazine = anything significantly darker than the background
    # (works for both white-paper and yellowed-paper magazines: both are darker
    # than the light-grey/white photo background)
    threshold = max(0, bg - 25)
    _, mask = _cv2.threshold(gray, threshold, 255, _cv2.THRESH_BINARY_INV)

    # Morphological close to fill internal text gaps so the magazine becomes
    # one solid blob; size proportional to image resolution
    k = max(15, min(h, w) // 100)
    kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (k, k))
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=_cv2.contourArea)
    # Reject if it's a tiny speck (< 10% of frame)
    if _cv2.contourArea(largest) < 0.1 * h * w:
        return None

    peri = _cv2.arcLength(largest, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = _cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(_np.float32)
    # Fallback: rotated bounding rectangle
    rect = _cv2.minAreaRect(largest)
    return _cv2.boxPoints(rect).astype(_np.float32)


def _find_magazine_quad_rembg(img_bgr):
    """Return 4 corner points of the magazine using U^2-Net foreground
    segmentation, or None if no plausible quad was found.

    Used as the primary detector in `crop_magazine` because rembg handles
    complex/textured backgrounds and bound-book scenes that the corner-median
    threshold in `_find_magazine_quad` can't separate. Returns None on any
    rejection so the caller can fall back to `_find_magazine_quad`.
    """
    if not _CV2_AVAILABLE or not rembg_available():
        return None
    import numpy as _np
    h, w = img_bgr.shape[:2]

    # rembg expects RGB PIL input.
    rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    try:
        mask_pil = _rembg_remove(
            pil_img, session=_get_rembg_session(), only_mask=True
        )
    except Exception as exc:
        # Corrupt input, OOM, missing model on first-run with no network, etc.
        _LOGGER.warning("rembg.remove failed: %s", exc)
        return None

    mask = _np.asarray(mask_pil, dtype=_np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    _, mask = _cv2.threshold(mask, 128, 255, _cv2.THRESH_BINARY)

    # Same close-kernel size as the OpenCV detector for consistency: bridges
    # small holes (text gutters, magazine fold) so the page becomes one blob.
    k = max(15, min(h, w) // 100)
    kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (k, k))
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=_cv2.contourArea)
    frame_area = float(h * w)
    area = float(_cv2.contourArea(largest))

    # Too small → rembg found nothing plausible. Typically means the input
    # is already cropped (no clear foreground/background contrast) or the
    # subject is genuinely tiny — either way fall back to the OpenCV detector.
    if area < 0.15 * frame_area:
        return None
    # Too large → whole-frame mask. rembg sometimes does this on
    # already-cropped inputs or when foreground/background are inseparable.
    # An accepted near-full quad would defeat the purpose of cropping.
    if area > 0.97 * frame_area:
        return None

    peri = _cv2.arcLength(largest, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = _cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(_np.float32)
            break
    if quad is None:
        rect = _cv2.minAreaRect(largest)
        quad = _cv2.boxPoints(rect).astype(_np.float32)

    # Validate quad shape: reject if opposite sides differ by more than 40%
    # in length. Extreme skew usually means rembg latched onto a weird blob
    # (eg. a dark photographer-hand fragment) rather than the page.
    ordered = _order_quad(quad)
    tl, tr, br, bl = ordered
    top = float(_np.linalg.norm(tr - tl))
    bottom = float(_np.linalg.norm(br - bl))
    left = float(_np.linalg.norm(bl - tl))
    right = float(_np.linalg.norm(br - tr))

    def _too_uneven(a: float, b: float) -> bool:
        m = max(a, b)
        if m <= 0:
            return True
        return abs(a - b) / m > 0.40

    if _too_uneven(top, bottom) or _too_uneven(left, right):
        return None

    return quad


def _find_magazine_bbox_bgsubtr(img_bgr):
    """Border-sampling background subtraction.

    Learns the background colour from the image border, builds a foreground
    mask, and returns an axis-aligned bounding box as
    np.float32 [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]
    (TL, TR, BR, BL order), or None if the foreground is indistinguishable
    from the background (e.g. white magazine on white background).
    """
    if not _CV2_AVAILABLE:
        return None
    import numpy as _np
    h, w = img_bgr.shape[:2]

    # Sample border strip (≈5% of shorter dimension, min 10 px)
    bw = max(10, min(h, w) // 20)
    border = _np.concatenate([
        img_bgr[:bw, :].reshape(-1, 3),
        img_bgr[-bw:, :].reshape(-1, 3),
        img_bgr[:, :bw].reshape(-1, 3),
        img_bgr[:, -bw:].reshape(-1, 3),
    ])
    bg_color = _np.median(border, axis=0).astype(_np.float32)

    # Per-pixel max-channel distance from background colour
    diff = _np.abs(img_bgr.astype(_np.float32) - bg_color).max(axis=2)

    # Light backgrounds (nearly white) need a stricter threshold to avoid
    # paper-texture noise registering as foreground.
    bg_brightness = float(bg_color.max())
    threshold = 30 if bg_brightness > 200 else 20

    fg_mask = (diff > threshold).astype(_np.uint8) * 255

    # White-on-white / very-low-contrast: nothing to detect
    fg_ratio = float(fg_mask.sum()) / (255.0 * h * w)
    if fg_ratio < 0.05:
        return None

    # Morphological cleanup: close bridges text/fold gaps; open removes noise
    k = max(15, min(h, w) // 100)
    kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (k, k))
    fg_mask = _cv2.morphologyEx(fg_mask, _cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = _cv2.morphologyEx(fg_mask, _cv2.MORPH_OPEN,  kernel, iterations=1)

    ys, xs = _np.where(fg_mask > 0)
    if not xs.size:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    # Near-full-frame result means the background sample is unreliable
    if (x_max - x_min) * (y_max - y_min) > 0.97 * h * w:
        return None

    return _np.array([
        [x_min, y_min],  # TL
        [x_max, y_min],  # TR
        [x_max, y_max],  # BR
        [x_min, y_max],  # BL
    ], dtype=_np.float32)


def crop_magazine_with_meta(src_path, dst_path):
    """Detect + warp the magazine and return (dims, meta).

    dims is (width, height) if a crop was written, else None.
    meta = {
      method:        'rembg'|'opencv'|'bgsubtr'|None,
      confidence:    'high'|'low',
      proposed_quad: [[x,y],…] | None,   # raw detector output (4 pts)
      bgsubtr_bbox:  [[x,y],…] | None,   # axis-aligned TL/TR/BR/BL
      frame_size:    [w, h],
      reasons:       [str, …],           # why confidence is 'low'
    }

    Resolution order:
      1. Validated quad → perspective warp           → confidence='high'
      2. No quad at all + bgsubtr bbox available
         → axis-aligned crop                         → confidence='high'
      3. Quad present but rejected, OR both absent   → no crop written
                                                     → confidence='low'
    """
    _err = lambda r: (None, {"method": None, "confidence": "low",
                              "proposed_quad": None, "bgsubtr_bbox": None,
                              "frame_size": [0, 0], "reasons": [r]})
    if not _CV2_AVAILABLE:
        return _err("cv2_unavailable")
    import numpy as _np
    img = _cv2.imread(str(src_path))
    if img is None:
        return _err("image_unreadable")

    h, w = img.shape[:2]
    frame_area = float(h * w)

    bgsubtr_bbox = _find_magazine_bbox_bgsubtr(img)
    bgsubtr_list = bgsubtr_bbox.tolist() if bgsubtr_bbox is not None else None

    # Primary: rembg; fallback: opencv
    quad = None
    method_used = None
    if rembg_available():
        quad = _find_magazine_quad_rembg(img)
        if quad is not None:
            method_used = "rembg"
    if quad is None:
        quad = _find_magazine_quad(img)
        if quad is not None:
            method_used = "opencv"

    proposed_quad = quad.tolist() if quad is not None else None

    # ── Confidence assessment ────────────────────────────────────────────
    reasons: list = []
    if quad is None:
        reasons.append("no_quad")
    else:
        quad_area = float(_cv2.contourArea(quad.astype(_np.float32)))

        # Envelope: quad must cover ≥70% of bgsubtr bbox area
        if bgsubtr_bbox is not None:
            bbox_area = float(_cv2.contourArea(bgsubtr_bbox.astype(_np.float32)))
            if bbox_area > 0 and quad_area < 0.70 * bbox_area:
                reasons.append("quad_inside_bbox_envelope")

        # Size: quad < 15% of frame is implausible
        if quad_area < 0.15 * frame_area:
            reasons.append("quad_too_small")

        # Orientation: portrait source should not yield landscape quad
        ordered = _order_quad(quad)
        tl, tr, br, bl = ordered
        q_w = float(_np.linalg.norm(tr - tl))
        q_h = float(_np.linalg.norm(bl - tl))
        if h > w and q_w > 0 and q_h > 0 and q_w > q_h:
            reasons.append("orientation_flip")

        # Aspect ratio
        if q_w > 0 and q_h > 0:
            asp = q_w / q_h
            if asp < 0.3 or asp > 2.0:
                reasons.append("aspect_out_of_range")

    quad_valid = quad is not None and not reasons

    # ── Resolution order ─────────────────────────────────────────────────
    dims = None
    final_method = method_used
    confidence = "high"

    if quad_valid:
        # 1. Validated quad → perspective warp
        ordered = _order_quad(quad)
        tl, tr, br, bl = ordered
        out_w = int(max(_np.linalg.norm(br - bl), _np.linalg.norm(tr - tl)))
        out_h = int(max(_np.linalg.norm(tr - br), _np.linalg.norm(tl - bl)))
        if out_w >= 100 and out_h >= 100:
            dst_pts = _np.array(
                [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                dtype=_np.float32,
            )
            M = _cv2.getPerspectiveTransform(ordered, dst_pts)
            warped = _cv2.warpPerspective(img, M, (out_w, out_h))
            _cv2.imwrite(str(dst_path), warped,
                         [_cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
            dims = (out_w, out_h)
        else:
            reasons.append("quad_too_small_after_warp")
            confidence = "low"

    elif quad is None and bgsubtr_bbox is not None:
        # 2. No quad detected at all, but bgsubtr found the cover extent
        x_min = int(bgsubtr_bbox[:, 0].min())
        x_max = int(bgsubtr_bbox[:, 0].max())
        y_min = int(bgsubtr_bbox[:, 1].min())
        y_max = int(bgsubtr_bbox[:, 1].max())
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        out_w = x_max - x_min
        out_h = y_max - y_min
        if out_w >= 100 and out_h >= 100:
            cropped = img[y_min:y_max, x_min:x_max]
            _cv2.imwrite(str(dst_path), cropped,
                         [_cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
            dims = (out_w, out_h)
            final_method = "bgsubtr"
        else:
            reasons.append("bgsubtr_bbox_too_small")
            confidence = "low"

    else:
        # 3. Quad present but rejected, or both detectors failed → low
        confidence = "low"
        if not reasons:
            reasons.append("no_bbox_detected")

    _LOGGER.info("crop %s: method=%s confidence=%s reasons=%s",
                 Path(src_path).name, final_method or "none", confidence, reasons)

    return dims, {
        "method": final_method,
        "confidence": confidence,
        "proposed_quad": proposed_quad,
        "bgsubtr_bbox": bgsubtr_list,
        "frame_size": [w, h],
        "reasons": reasons,
    }


def crop_magazine(src_path, dst_path) -> tuple[int, int] | None:
    """Detect the magazine in `src_path`, perspective-warp it onto a clean
    rectangle, and save to `dst_path` as JPEG. Returns (width, height) of the
    output, or None on failure (in which case dst_path is not written).

    Backward-compatible thin wrapper around crop_magazine_with_meta.
    """
    dims, _ = crop_magazine_with_meta(src_path, dst_path)
    return dims


def apply_crop_from_meta(src_path, dst_path, meta: dict) -> "tuple[int,int] | None":
    """Apply the bgsubtr_bbox from a previous crop_magazine_with_meta call.

    Used when the user accepts a low-confidence auto crop: the bgsubtr bbox
    is the reliable fallback (the proposed_quad is the suspect detection).
    Returns (out_w, out_h) or None.
    """
    bgsubtr_bbox = meta.get("bgsubtr_bbox")
    if bgsubtr_bbox is None:
        return None
    import numpy as _np
    bbox = _np.array(bgsubtr_bbox, dtype=_np.float32)
    x_min = int(bbox[:, 0].min()); x_max = int(bbox[:, 0].max())
    y_min = int(bbox[:, 1].min()); y_max = int(bbox[:, 1].max())
    return apply_rect_crop(src_path, dst_path,
                           {"x": x_min, "y": y_min,
                            "w": x_max - x_min, "h": y_max - y_min})


def apply_rect_crop(src_path, dst_path, rect: dict) -> "tuple[int,int] | None":
    """Crop src_path to the axis-aligned rect {x, y, w, h} (pixels in the
    original image) and save to dst_path as JPEG. Returns (out_w, out_h) or None.
    """
    if not _CV2_AVAILABLE:
        return None
    import numpy as _np
    img = _cv2.imread(str(src_path))
    if img is None:
        return None
    ih, iw = img.shape[:2]
    x  = max(0, int(rect.get("x", 0)))
    y  = max(0, int(rect.get("y", 0)))
    rw = int(rect.get("w", iw))
    rh = int(rect.get("h", ih))
    x2 = min(iw, x + rw)
    y2 = min(ih, y + rh)
    out_w = x2 - x
    out_h = y2 - y
    if out_w < 10 or out_h < 10:
        return None
    cropped = img[y:y2, x:x2]
    _cv2.imwrite(str(dst_path), cropped, [_cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
    return (out_w, out_h)


# ─── Config ──────────────────────────────────────────────────────────────────

CACHE_FILE = "phash_cache.json"
REPORT_FILE = "duplicates_report.txt"
HTML_REPORT_FILE = "duplicates_report.html"
THUMB_SIZE = 300        # max longest edge of embedded thumbnails (base64 in HTML)
THUMB_QUALITY = 72      # JPEG quality for embedded thumbnails (smaller HTML)
# Cache versions:
#   v1 = filename → phash_str
#   v2 = filename → phash_str (algorithm tweaks)
#   v3 = filename → {hash, edge_std}
#   v4 = filename → {phash, dhash, edge_std, ocr_text, dates}
#   v5 = v4 + cropping: features computed from auto-cropped versions in <folder>/cropped/
CACHE_VERSION = 5

# Cropping config
CROPPED_DIR = "cropped"     # subfolder under each analysed folder
CROP_JPEG_QUALITY = 92      # high quality for archival/server upload

HASH_SIZE = 16          # 16 → 256-bit hash (more discriminating than default 64-bit)
MATCH_THRESHOLD = 24    # Hamming distance below this = same page
                        # (out of 256 bits; ~10% tolerance handles lighting/angle differences)

# Magazine-mode constants. The user photographs each magazine as:
#   front (single-page)  →  back (single-page)  →  colophon/index (two-page spread, optional)
# Or sometimes just (front, back) with no spread, or (front, spread). We detect
# magazine boundaries by looking for the two-page spread photo: it's a single
# JPEG showing two pages side-by-side, so the magazine fills the camera frame.
# Single-page photos have whitespace at the left/right because a portrait page
# doesn't fill the landscape frame.
#
# Detection: edge-strip standard-deviation. Single-page → background → low std.
# Spread → magazine content extends to edges → high std. Bimodal distribution.
SPREAD_EDGE_STD_THRESHOLD = 10  # avg(left_std, right_std) above this = spread
MAX_PAGES_PER_MAGAZINE = 6      # safety cap when no spread is found in a run
PAGES_PER_MAGAZINE = 4          # legacy default; not used when edge_std is available
WARN_COPIES_PER_GROUP = 6       # Warn if one duplicate group has this many copies or more.


# ─── File parsing ─────────────────────────────────────────────────────────────

FILENAME_RE = re.compile(
    r"^(\d{4}_\d{2}_\d{2})_(.+?)_(\d{4,5})\.(jpe?g|JPE?G)$"
)

def parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return {
        "date_taken": m.group(1),
        "publication": m.group(2),
        "seq": int(m.group(3)),
        "filename": name,
    }


def scan_images(folder: Path) -> list[dict]:
    """Recursively find all parseable image files, sorted by (publication, seq).

    Excludes images inside our own output subdirectories (`cropped/`, `_thumbs/`,
    `_backup_*`) so they aren't double-counted when re-running on a folder that
    already has cached cropped output.
    """
    images = []
    excluded_dirs = {CROPPED_DIR, "_thumbs"}
    for f in folder.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        # Skip anything under our own output directories (relative to source)
        rel = f.relative_to(folder)
        if any(part in excluded_dirs or part.startswith("_backup_") for part in rel.parts):
            continue
        info = parse_filename(f.name)
        if info:
            info["path"] = str(f)
            images.append(info)
    images.sort(key=lambda x: (x["publication"].lower(), x["seq"]))
    return images


# ─── Hashing ──────────────────────────────────────────────────────────────────

# ─── Date extraction (OCR support) ──────────────────────────────────────────

# Recognise common date formats found on magazine and book pages. Multiple
# captures are normalized to (year, month, day) tuples. A "two-digit year"
# is interpreted as 19YY for YY >= 50, otherwise 20YY (covers the 1970s-1990s
# print era + recent issues).
_MONTH_LOOKUP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_DATE_PATTERNS = [
    # "November 27, 1976"  /  "Nov. 27 1976"  /  "November 27 76"
    re.compile(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+'
        r'(\d{1,2})(?:st|nd|rd|th)?[\s,.\-]+(\d{2,4})\b',
        re.IGNORECASE,
    ),
    # "27 November 1976"  /  "27th Nov 76"
    re.compile(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+'
        r'(January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?[\s,.\-]+(\d{2,4})\b',
        re.IGNORECASE,
    ),
    # "27/11/76", "27-11-76"  (day/month/year — UK format common in NME)
    re.compile(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b'),
]


def _normalize_year(y_str: str) -> int | None:
    try:
        y = int(y_str)
    except ValueError:
        return None
    if y < 100:
        return 2000 + y if y < 50 else 1900 + y
    if 1900 <= y <= 2099:
        return y
    return None


def extract_dates(text: str) -> list[tuple[int, int, int]]:
    """Return distinct (year, month, day) tuples found in `text`. May be empty."""
    found: list[tuple[int, int, int]] = []
    seen: set = set()

    def add(y, m, d):
        if y is None or m is None or d is None:
            return
        if not (1 <= m <= 12 and 1 <= d <= 31):
            return
        t = (y, m, d)
        if t not in seen:
            seen.add(t)
            found.append(t)

    for m_match in _DATE_PATTERNS[0].finditer(text):
        month = _MONTH_LOOKUP.get(m_match.group(1).lower().rstrip("."))
        try:
            day = int(m_match.group(2))
        except ValueError:
            continue
        year = _normalize_year(m_match.group(3))
        add(year, month, day)
    for m_match in _DATE_PATTERNS[1].finditer(text):
        try:
            day = int(m_match.group(1))
        except ValueError:
            continue
        month = _MONTH_LOOKUP.get(m_match.group(2).lower().rstrip("."))
        year = _normalize_year(m_match.group(3))
        add(year, month, day)
    for m_match in _DATE_PATTERNS[2].finditer(text):
        try:
            day = int(m_match.group(1))
            month = int(m_match.group(2))
        except ValueError:
            continue
        year = _normalize_year(m_match.group(3))
        # Reject if it doesn't look like a sensible date (two-digit ambiguous):
        # the d/m/y pattern accepts day > 12 to disambiguate from m/d/y, but we
        # accept all and rely on consensus across an item to filter noise.
        add(year, month, day)
    return found


# ─── Image features ──────────────────────────────────────────────────────────

OCR_TIMEOUT_SEC = 20  # tesseract occasionally hangs on certain images; cap it.


def _run_ocr(image: Image.Image) -> str:
    """Run OCR on a PIL image. Returns extracted text or empty string.

    OCR is run at full resolution because dates in magazine headers/footers
    are small and aggressive downscaling loses them — empirically, 2500px
    catches some dates but misses others; 4000px catches all in our test set.
    Cost: ~1.5–4s per photo. Cached after first run.

    A hard timeout (OCR_TIMEOUT_SEC) is enforced via pytesseract's `timeout=`
    param: tesseract occasionally hangs indefinitely on particular images
    (we've seen 3+ stalls on real data). On timeout, this returns empty
    string and the rest of the pipeline keeps working.
    """
    if not ocr_available():
        return ""
    try:
        # pytesseract rejects lazy/unloaded PIL images even when mode == "RGB".
        # `convert("RGB")` forces a real decode and returns a fresh image that
        # pytesseract can handle. Always copy/convert; calling on an already-RGB
        # image is cheap.
        ocr_img = image.convert("RGB")
        # Default PSM (3 = fully automatic page segmentation) reliably picks up
        # small date prints in headers and footers.
        return pytesseract.image_to_string(ocr_img, timeout=OCR_TIMEOUT_SEC)
    except (pytesseract.pytesseract.TesseractError, RuntimeError):
        # Includes the "Tesseract process timeout" RuntimeError raised by
        # pytesseract's `timeout=` enforcement.
        return ""
    except Exception:
        return ""


def compute_features(path: str, *, run_ocr: bool = True) -> dict:
    """Compute all per-photo features in one image open.

    Returns a dict with:
      phash      : hex string of perceptual hash (HASH_SIZE-bit grid)
      dhash      : hex string of difference hash (gradient-based, robust to
                   brightness; uncorrelated with pHash failure modes — used for
                   ensemble matching to boost recall)
      edge_std   : avg std-dev of left+right edge strips (spread vs single page)
      ocr_text   : raw OCR text, truncated to 4000 chars (for debugging)
      dates      : list of (Y,M,D) tuples extracted from ocr_text

    If OCR is unavailable or `run_ocr=False`, ocr_text=""/dates=[].
    """
    img = Image.open(path)
    img.draft("L", (1500, 1500))
    if img.mode != "L":
        img_gray = img.convert("L")
    else:
        img_gray = img

    # Aspect ratio of the source image. After cropping, this reliably classifies
    # spread vs single-page (spread = landscape, single = portrait). For the
    # original uncropped images (which are all 3984x2656 landscape regardless of
    # content), this is uninformative — we fall back to edge_std there.
    src_w, src_h = img.size
    aspect_ratio = src_w / src_h if src_h else 1.0

    # edge_std on a small thumbnail. Useful for uncropped images (background
    # whitespace → low std for portrait pages, high std for spread pages); for
    # cropped images both kinds fill the frame so edge_std is meaningless.
    import numpy as _np
    es_thumb = img_gray.copy()
    es_thumb.thumbnail((400, 400), Image.BILINEAR)
    arr = _np.asarray(es_thumb, dtype=_np.uint8)
    h, w = arr.shape
    strip = max(1, w // 10)
    edge_std = float((arr[:, :strip].std() + arr[:, -strip:].std()) / 2)

    # is_spread: prefer aspect_ratio when it's informative (i.e., source image
    # is not the camera-native 3984x2656≈1.50). On cropped output, aspect ratio
    # of the magazine itself decides: > 1.1 = landscape spread.
    # For uncropped camera-native images (aspect very close to 1.50), fall back
    # to edge_std.
    if abs(aspect_ratio - 1.50) < 0.05:
        is_spread = edge_std > SPREAD_EDGE_STD_THRESHOLD
    else:
        is_spread = aspect_ratio > 1.1

    # Hashes. Both run on the same downsampled image to keep this fast.
    hash_img = img_gray.copy()
    if max(hash_img.size) > 256:
        scale = 256 / max(hash_img.size)
        hash_img = hash_img.resize(
            (int(hash_img.width * scale), int(hash_img.height * scale)),
            Image.BILINEAR,
        )
    phash = str(imagehash.phash(hash_img, hash_size=HASH_SIZE))
    dhash = str(imagehash.dhash(hash_img, hash_size=HASH_SIZE))

    # OCR (slow). Skipped on 2-page spreads: spread photos rarely contain
    # readable date prints (the magazine is open at internal pages, not the
    # masthead) AND OCR is unusually slow on them (5-12s vs 1-2s for single
    # pages). Spreads are already detected as item boundaries, so missing OCR
    # on them costs nothing.
    ocr_text = ""
    dates: list = []
    if run_ocr and ocr_available() and not is_spread:
        # Re-open with original colour for OCR — gray conversion may hurt readability.
        ocr_img = Image.open(path)
        ocr_text = _run_ocr(ocr_img)
        if ocr_text:
            dates = extract_dates(ocr_text)
            ocr_text = ocr_text[:4000]  # truncate for cache size

    return {
        "phash": phash,
        "dhash": dhash,
        "edge_std": edge_std,
        "aspect_ratio": aspect_ratio,
        "is_spread": is_spread,
        "ocr_text": ocr_text,
        "dates": dates,
    }


def load_cache(folder: Path) -> dict:
    """Returns {filename: feature_dict} or {} on version mismatch."""
    p = folder / CACHE_FILE
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("_version") == CACHE_VERSION:
            return {k: v for k, v in data.items() if not k.startswith("_")}
        else:
            print(f"  (cache v{data.get('_version', '?')} → v{CACHE_VERSION}, will recompute)")
    return {}


def save_cache(folder: Path, cache: dict):
    out = {"_version": CACHE_VERSION, **cache}
    with open(folder / CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def cache_get_phash(entry) -> str:
    """Cache entries are either v4 dicts or legacy strings/dicts."""
    if isinstance(entry, dict):
        return entry.get("phash") or entry.get("hash") or ""
    return entry or ""


def cache_get_dhash(entry) -> str:
    if isinstance(entry, dict):
        return entry.get("dhash", "")
    return ""


def cache_get_edge_std(entry) -> float:
    if isinstance(entry, dict):
        return float(entry.get("edge_std", 0.0))
    return 0.0


def cache_get_is_spread(entry, fallback_threshold: float = 10.0) -> bool:
    """Returns whether this photo is a 2-page spread (magazine boundary marker).

    Prefers the explicit `is_spread` flag (set during cropping-aware feature
    extraction). Falls back to edge_std comparison for legacy cache entries
    that don't have the flag.
    """
    if isinstance(entry, dict):
        if "is_spread" in entry:
            return bool(entry["is_spread"])
        return entry.get("edge_std", 0.0) > fallback_threshold
    return False


def cache_get_dates(entry) -> list:
    if isinstance(entry, dict):
        return [tuple(d) for d in entry.get("dates", [])]
    return []


# Legacy alias used elsewhere in the file.
def cache_get_hash(entry) -> str:
    return cache_get_phash(entry)


# ─── Magazine-mode duplicate detection ──────────────────────────────────────

def _build_items_by_spread(images: list[dict], cache: dict,
                            spread_threshold: float = SPREAD_EDGE_STD_THRESHOLD,
                            max_pages: int = MAX_PAGES_PER_MAGAZINE) -> list[dict]:
    """Phase 3a: chunk photos into items using 2-page-spread photos as boundaries.

    Convention: each item = run of consecutive (by seq) photos ending at the
    first 2-page spread photo (front, back, [optional inside pages,] spread).
    Spread detection: edge_std (avg std-dev of left+right 10% strips) > threshold.

    Falls back to a max-page cap if no spread is found in a run.
    """
    by_pub = defaultdict(list)
    for img in images:
        by_pub[img["publication"]].append(img)

    items = []
    for pub, imgs in by_pub.items():
        imgs.sort(key=lambda x: x["seq"])
        current: list[dict] = []
        for img in imgs:
            current.append(img)
            entry = cache.get(img["filename"])
            # Prefer the explicit is_spread flag (works on cropped images where
            # edge_std is meaningless); fall back to edge_std for legacy entries.
            is_spread = cache_get_is_spread(entry, fallback_threshold=spread_threshold)
            if is_spread or len(current) >= max_pages:
                items.append(_make_item(pub, current, is_spread, cache))
                current = []
        if current:
            items.append(_make_item(pub, current, False, cache))
    return items


def _make_item(pub: str, photo_imgs: list[dict], ended_with_spread: bool,
               cache: dict) -> dict:
    """Build an item dict from a list of photo dicts (sorted by seq)."""
    front = photo_imgs[0]
    photos = [p["filename"] for p in photo_imgs]
    # Collect all OCR'd dates across the item's photos.
    all_dates = []
    for fn in photos:
        all_dates.extend(cache_get_dates(cache.get(fn)))
    # Consensus date = the date that appears in the most distinct photos.
    # Counting per-photo (not per-occurrence) avoids one OCR-noisy page
    # outweighing a date that legitimately appears once on each of three pages.
    date_per_photo: list[set] = []
    for fn in photos:
        date_per_photo.append(set(cache_get_dates(cache.get(fn))))
    date_votes: dict = {}
    for ds in date_per_photo:
        for d in ds:
            date_votes[d] = date_votes.get(d, 0) + 1
    consensus_date = None
    if date_votes:
        # Tie-break: prefer date that appears in the front+back pages.
        front_back_dates = set()
        for ds in date_per_photo[:2]:
            front_back_dates.update(ds)
        consensus_date = max(
            date_votes,
            key=lambda d: (date_votes[d], 1 if d in front_back_dates else 0),
        )
    return {
        "publication": pub,
        "front_filename": front["filename"],
        "front_seq": front["seq"],
        "seq_range": [photo_imgs[0]["seq"], photo_imgs[-1]["seq"]],
        "photos": photos,
        "page_count": len(photos),
        "ended_with_spread": ended_with_spread,
        "all_dates": sorted({d for ds in date_per_photo for d in ds}),
        "consensus_date": consensus_date,
    }


def _refine_items_by_dates(items: list[dict], images: list[dict], cache: dict
                            ) -> tuple[list[dict], list[str]]:
    """Phase 3b: split/merge items using OCR'd date evidence.

    Two refinements:
    1. SPLIT: if an item contains photos with multiple distinct consensus dates,
       it's likely two items the spread-detector incorrectly merged. Find the
       boundary (where dates change) and split.
    2. MERGE: if two adjacent items in the same publication share a date, they
       were likely one item the spread-detector incorrectly split. Merge them.

    Both refinements only fire when *positive* date evidence supports them —
    items with no extracted dates pass through unchanged.

    Returns (refined_items, warnings).
    """
    warnings: list[str] = []
    img_lookup = {img["filename"]: img for img in images}
    if not items:
        return items, warnings

    # ── Pass 1: split items whose photos have inconsistent dates ──
    refined: list[dict] = []
    for item in items:
        photos = item["photos"]
        if len(photos) <= 1:
            refined.append(item)
            continue
        # For each photo, what dates does its OCR say?
        per_photo_dates = [set(cache_get_dates(cache.get(fn))) for fn in photos]
        # Find a "split point" — a position where photos before it consistently
        # show date A and photos after consistently show date B (different).
        split_at = None
        for k in range(1, len(photos)):
            before = set().union(*per_photo_dates[:k])
            after = set().union(*per_photo_dates[k:])
            if before and after and before.isdisjoint(after):
                # Disjoint date sets on each side → real boundary.
                split_at = k
                break
        if split_at is None:
            refined.append(item)
            continue
        # Split into two items at the boundary. Inherit ended_with_spread for
        # the second half only (it has the original spread, if any).
        before_imgs = [img_lookup[fn] for fn in photos[:split_at]]
        after_imgs = [img_lookup[fn] for fn in photos[split_at:]]
        refined.append(_make_item(item["publication"], before_imgs, False, cache))
        refined.append(_make_item(item["publication"], after_imgs, item.get("ended_with_spread", False), cache))
        warnings.append(
            f"Split item at seq {photos[split_at-1].split('_')[-1].replace('.JPG','')}/"
            f"{photos[split_at].split('_')[-1].replace('.JPG','')} based on date OCR"
        )

    # ── Pass 2: merge adjacent same-publication items with matching dates ──
    if not refined:
        return refined, warnings
    merged: list[dict] = []
    i = 0
    while i < len(refined):
        cur = refined[i]
        # Greedily merge into cur as long as next item is same publication AND
        # both have a consensus_date AND those dates match. Cap at MAX_PAGES_PER_MAGAZINE
        # to avoid runaway merges from OCR noise.
        while (i + 1 < len(refined)
               and refined[i + 1]["publication"] == cur["publication"]
               and cur.get("consensus_date") is not None
               and cur["consensus_date"] == refined[i + 1].get("consensus_date")
               and cur["page_count"] + refined[i + 1]["page_count"] <= MAX_PAGES_PER_MAGAZINE * 2):
            nxt = refined[i + 1]
            combined_imgs = [img_lookup[fn] for fn in cur["photos"] + nxt["photos"]]
            cur = _make_item(cur["publication"], combined_imgs,
                             nxt.get("ended_with_spread", False), cache)
            warnings.append(
                f"Merged adjacent items at seq {cur['seq_range'][0]:04d}-{cur['seq_range'][1]:04d} "
                f"(shared OCR date {cur['consensus_date']})"
            )
            i += 1
        merged.append(cur)
        i += 1

    return merged, warnings




def _match_items(items: list[dict], cache: dict, threshold: int,
                  date_boost_threshold: int = 90) -> dict[tuple[int, int], dict]:
    """Phase 4: pairwise comparison of items within each publication.

    Multi-strategy matching for high recall WITHOUT false-positive cascades:
      • POSITION-ALIGNED matching: position 0 (front) vs position 0; position 1
        (back) vs position 1. Inside pages are NOT compared because random
        inside-page similarities chain unrelated items via union-find.
      • Use BOTH pHash and dHash on each aligned position; the smaller distance
        wins per algorithm. Pair flagged if either pHash ≤ threshold OR dHash
        ≤ threshold (ensemble OR — recall priority on cover photo variations).
      • If both items have an OCR'd consensus_date AND they MATCH, accept the
        pair up to `date_boost_threshold` (catches duplicates with very
        different cover photographs but the same printed publication date).
      • If both items have consensus_dates AND they DIFFER, reject the pair
        (confidently not the same issue), unless the hash distance is ≤ 12
        (≈ identical pages → probably an OCR misread).

    Returns dict {(i, j): {"phash": int, "dhash": int, "best_pair": (fa, fb),
                            "best_algo": "phash"|"dhash"|"date",
                            "date_match": bool, "date_conflict": bool}}.
    """
    by_pub_idx: dict[str, list[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        by_pub_idx[it["publication"]].append(idx)

    # Cache parsed hashes and dates per item.
    item_phashes: dict[int, list] = {}
    item_dhashes: dict[int, list] = {}
    item_dates: dict[int, set] = {}
    for idx, it in enumerate(items):
        phs, dhs = [], []
        for fn in it["photos"]:
            entry = cache.get(fn)
            if entry is None:
                continue
            ph_hex = cache_get_phash(entry)
            dh_hex = cache_get_dhash(entry)
            if ph_hex:
                phs.append((fn, imagehash.hex_to_hash(ph_hex)))
            if dh_hex:
                dhs.append((fn, imagehash.hex_to_hash(dh_hex)))
        item_phashes[idx] = phs
        item_dhashes[idx] = dhs
        item_dates[idx] = {it["consensus_date"]} if it.get("consensus_date") else set()
        # Also include all_dates as candidates for date-match (any overlap is enough)
        item_dates[idx].update(it.get("all_dates", []))
        item_dates[idx].discard(None)

    pair_info: dict[tuple[int, int], dict] = {}
    for pub, idxs in by_pub_idx.items():
        for ii in range(len(idxs)):
            i = idxs[ii]
            phs_i = item_phashes[i]
            dhs_i = item_dhashes[i]
            dates_i = item_dates[i]
            if not phs_i and not dhs_i:
                continue
            for jj in range(ii + 1, len(idxs)):
                j = idxs[jj]
                phs_j = item_phashes[j]
                dhs_j = item_dhashes[j]
                dates_j = item_dates[j]

                # Date analysis (cheap, do first):
                # - date_match: any overlap between item dates → strong positive
                # - date_conflict: both have *consensus* dates AND they differ → strong negative
                cons_i = items[i].get("consensus_date")
                cons_j = items[j].get("consensus_date")
                date_match = bool(dates_i and dates_j and (dates_i & dates_j))
                date_conflict = (
                    cons_i is not None and cons_j is not None
                    and cons_i != cons_j
                    # Allow off-by-a-couple-days tolerance for OCR misreads
                    and abs((cons_i[0] - cons_j[0])*365 + (cons_i[1] - cons_j[1])*30 + (cons_i[2] - cons_j[2])) > 4
                )
                # Reject conflicting-date pairs UNLESS the hash similarity is
                # very strong (could indicate OCR misread on one side).
                # Use an inner threshold check below.

                # Position-aligned matching at EACH position k. We compute the
                # per-position best (pHash and dHash) and ALSO count how many
                # positions had at least one hash ≤ threshold ("good positions").
                # The cascade-prevention rule below requires ≥ 2 good positions
                # for items with ≥ 2 pages — unrelated items rarely have 2
                # coincidentally-matching positions, but real duplicates almost
                # always have multiple matching pages (front, back, spread, etc.).
                # dHash on real magazine data is much looser than pHash —
                # unrelated items often have dHash distances of 16-50 while
                # pHash distances stay 80-130. So we use a tighter threshold
                # for dHash (half of pHash threshold). Real-data verification
                # on NME confirms this separates duplicates from non-duplicates.
                dhash_threshold = max(20, threshold // 2)

                best_ph = None; best_ph_pair = None
                best_dh = None; best_dh_pair = None
                good_positions = 0
                near_identical_position = False
                positions_to_check = min(len(phs_i), len(phs_j), len(dhs_i), len(dhs_j))
                for k in range(positions_to_check):
                    fa, ha_p = phs_i[k]
                    fb, hb_p = phs_j[k]
                    ha_d = dhs_i[k][1]
                    hb_d = dhs_j[k][1]
                    d_p = ha_p - hb_p
                    d_d = ha_d - hb_d
                    if best_ph is None or d_p < best_ph:
                        best_ph = d_p; best_ph_pair = (fa, fb)
                    if best_dh is None or d_d < best_dh:
                        best_dh = d_d; best_dh_pair = (fa, fb)
                    # A position is "good" if pHash ≤ threshold OR dHash ≤ tight threshold.
                    if d_p <= threshold or d_d <= dhash_threshold:
                        good_positions += 1
                    # Near-identical override: a single position with very low
                    # distance (≤ 12) is strong enough on its own.
                    if d_p <= 12 or d_d <= 12:
                        near_identical_position = True

                min_pages = min(items[i]["page_count"], items[j]["page_count"])
                # Require ≥ 2 good positions when both items have ≥ 2 pages.
                # Single-page items (or where one item has only 1 page) accept
                # 1 good position because there's nothing more to align.
                required_good = 2 if min_pages >= 2 else 1

                # Determine if this pair is a match.
                # Hash override: if any single position is near-identical
                # (≤ 12 bits ≈ visually identical), accept the pair regardless
                # of date conflict or low good-position count.
                hash_override = near_identical_position

                # Reject pairs with *conflicting* OCR dates unless we have a
                # near-identical hash (most likely OCR misread on one side).
                if date_conflict and not hash_override:
                    continue

                # Decision: enough "good positions" to call this a match?
                #   Multi-page items (≥ 2 pages each) need ≥ 2 good positions.
                #   Single-page items need 1.
                #   The hash_override (single near-identical page) bypasses this.
                #   The date_match (matching OCR'd publication dates) also
                #   bypasses, accepting up to date_boost_threshold.
                multi_page_match = good_positions >= required_good
                date_ok = date_match and (
                    (best_ph if best_ph is not None else 999) <= date_boost_threshold
                    or (best_dh if best_dh is not None else 999) <= date_boost_threshold
                )

                is_match = multi_page_match or hash_override or date_ok
                if not is_match:
                    continue

                # For reporting / "best_algo": pick the strongest signal.
                ph_ok = best_ph is not None and best_ph <= threshold
                dh_ok = best_dh is not None and best_dh <= threshold

                # Pick which algorithm "won" (smallest normalized distance).
                if ph_ok and (not dh_ok or (best_ph or 999) <= (best_dh or 999)):
                    best_algo = "phash"
                    best_pair = best_ph_pair
                elif dh_ok:
                    best_algo = "dhash"
                    best_pair = best_dh_pair
                else:
                    best_algo = "date"
                    best_pair = best_ph_pair or best_dh_pair

                pair_info[(i, j)] = {
                    "phash": best_ph if best_ph is not None else 999,
                    "dhash": best_dh if best_dh is not None else 999,
                    "best_pair": best_pair,
                    "best_algo": best_algo,
                    "date_match": date_match,
                    "date_conflict": date_conflict,
                }

    return pair_info


def find_duplicate_groups(images: list[dict], cache: dict, threshold: int,
                          pages_per_magazine: int = PAGES_PER_MAGAZINE
                          ) -> tuple[list[dict], list[str]]:
    """Find duplicate items via multi-strategy matching with date confirmation.

    Pipeline (phases):
      1. Build candidate items by spread-boundary detection.
      2. Refine boundaries using OCR'd dates (split inconsistent items, merge
         adjacent same-date items).
      3. Match items pairwise: every photo of A vs every photo of B, using
         pHash + dHash ensemble; date overlap confirms or rejects.
      4. Cluster matched pairs via union-find into duplicate groups.
      5. Annotate each group with confidence, KEEP recommendation, etc.

    Returns (groups, warnings).
    """
    warnings: list[str] = []

    # Phase 1: candidate items by spread detection.
    items = _build_items_by_spread(images, cache,
                                    max_pages=max(2, pages_per_magazine + 2))
    if not items:
        return [], warnings

    # Phase 2: date-based refinement (split/merge using OCR evidence).
    items, refine_warnings = _refine_items_by_dates(items, images, cache)
    # Surface a few diagnostic refinement notes (cap at 5 to avoid spam).
    for w in refine_warnings[:5]:
        warnings.append("Date OCR refinement: " + w)
    if len(refine_warnings) > 5:
        warnings.append(f"Date OCR refinement: ...and {len(refine_warnings) - 5} more refinements")

    if not items:
        return [], warnings

    # Phase 3: pairwise matching.
    pair_info = _match_items(items, cache, threshold)

    # Phase 4: union-find clustering.
    parent = list(range(len(items)))
    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j) in pair_info:
        _union(i, j)

    by_root: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(items)):
        by_root[_find(idx)].append(idx)

    # Phase 5: build group dicts.
    groups = []
    for root, item_idxs in by_root.items():
        if len(item_idxs) < 2:
            continue
        item_idxs.sort(key=lambda i: items[i]["front_seq"])

        ph_distances = []
        dh_distances = []
        date_match_count = 0
        for ii in range(len(item_idxs)):
            for jj in range(ii + 1, len(item_idxs)):
                a, b = item_idxs[ii], item_idxs[jj]
                key = (a, b) if a < b else (b, a)
                info = pair_info.get(key)
                if info:
                    ph_distances.append(info["phash"])
                    dh_distances.append(info["dhash"])
                    if info["date_match"]:
                        date_match_count += 1

        copies = []
        for i in item_idxs:
            it = items[i]
            copies.append({
                "seq_range": list(it["seq_range"]),
                "photos": list(it["photos"]),
                "photo_count": it["page_count"],
                "front_seq": it["front_seq"],
                "ended_with_spread": it.get("ended_with_spread", False),
                "consensus_date": it.get("consensus_date"),
                "all_dates": list(it.get("all_dates", [])),
            })

        # KEEP = most pages photographed. Tie-break by lowest front_seq.
        keep_idx = max(
            range(len(copies)),
            key=lambda k: (copies[k]["photo_count"], -copies[k]["front_seq"])
        )

        # Best distance is the lowest of either algorithm.
        best_ph = min(ph_distances) if ph_distances else None
        best_dh = min(dh_distances) if dh_distances else None
        best_dist = min([d for d in (best_ph, best_dh) if d is not None and d != 999], default=None)
        worst_ph = max((d for d in ph_distances if d != 999), default=None)
        worst_dh = max((d for d in dh_distances if d != 999), default=None)
        worst_dist = max([d for d in (worst_ph, worst_dh) if d is not None], default=None)

        # Confidence ladder. Date-match boosts confidence by one tier.
        if best_dist is None:
            conf_base = "LOW"
        elif best_dist <= 10:
            conf_base = "VERY HIGH"
        elif best_dist <= 25:
            conf_base = "HIGH"
        elif best_dist <= 40:
            conf_base = "MEDIUM"
        else:
            conf_base = "LOW"
        if date_match_count > 0 and conf_base in ("MEDIUM", "LOW"):
            confidence = "HIGH" if conf_base == "MEDIUM" else "MEDIUM"
        else:
            confidence = conf_base

        groups.append({
            "publication": items[item_idxs[0]]["publication"],
            "copy_count": len(copies),
            "total_photos": sum(c["photo_count"] for c in copies),
            "keep_index": keep_idx,
            "copies": copies,
            "best_distance": best_dist,
            "worst_distance": worst_dist,
            "best_phash": best_ph,
            "best_dhash": best_dh,
            "match_pair_count": len(ph_distances),
            "date_match_count": date_match_count,
            "confidence": confidence,
        })

    # Warn about suspiciously large groups (likely a recurring-cover situation —
    # e.g., an artist who appeared on 8 NME covers in a year).
    for g in groups:
        if g["copy_count"] >= WARN_COPIES_PER_GROUP:
            warnings.append(
                f"Unusually large group in {g['publication']}: {g['copy_count']} copies "
                f"(starts at seq {g['copies'][0]['seq_range'][0]:04d}). "
                f"Verify visually — may be a recurring cover or coincidental similarity."
            )

    groups.sort(key=lambda g: (g["publication"].lower(), g["copies"][0]["seq_range"][0]))
    return groups, warnings


# ─── Report ───────────────────────────────────────────────────────────────────

def _fmt_range(lo: int, hi: int) -> str:
    return f"{lo:04d}" if lo == hi else f"{lo:04d}-{hi:04d}"


def generate_report(images, groups, out_path: Path, threshold: int,
                    warnings: list[str] | None = None,
                    pages_per_magazine: int = PAGES_PER_MAGAZINE):
    by_pub = defaultdict(list)
    for img in images:
        by_pub[img["publication"]].append(img)

    lines = []
    lines.append("=" * 72)
    lines.append("MAGAZINE DUPLICATE REPORT  (front-page matching, local)")
    lines.append("=" * 72)
    lines.append(f"\nTotal photos        : {len(images)}")
    lines.append(f"Publications        : {', '.join(sorted(by_pub))}")
    for pub, ims in sorted(by_pub.items()):
        seqs = sorted(i["seq"] for i in ims)
        lines.append(f"  {pub:<20} {len(ims):>4} photos  (seq {seqs[0]:04d}-{seqs[-1]:04d})")
    lines.append(f"\nPages per magazine    : {pages_per_magazine}")
    lines.append(f"Match threshold       : {threshold} / 256 bits  (lower = stricter)")
    lines.append(f"Duplicate magazines   : {len(groups)}")
    triple_plus = sum(1 for g in groups if g["copy_count"] >= 3)
    lines.append(f"  with 3+ copies      : {triple_plus}")

    if warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in warnings:
            lines.append(f"  ! {w}")
    lines.append("")

    if not groups:
        lines.append("No duplicates detected.")
    else:
        lines.append("─" * 72)
        lines.append("DUPLICATE MAGAZINE ISSUES")
        lines.append("─" * 72)
        lines.append("Each entry is one magazine issue that exists in multiple physical copies.")
        lines.append("Recommendation: keep ONE copy, physically remove the others from the stack.")
        lines.append("\nConfidence guide: best_distance is hash distance of the strongest match;")
        lines.append("  0–10 = nearly identical photos | 10–25 = high confidence")
        lines.append("  25–40 = medium (verify visually) | 40+ = low (could be coincidence)\n")

        for n, g in enumerate(groups, 1):
            best = g["best_distance"]
            # Use the algorithm-provided confidence (which considers date match
            # boost) if available; otherwise fall back to distance-based.
            confidence = g.get("confidence") or (
                "VERY HIGH" if best is not None and best <= 10 else
                "HIGH" if best is not None and best <= 25 else
                "MEDIUM" if best is not None and best <= 40 else
                "LOW"
            )
            ph = g.get("best_phash")
            dh = g.get("best_dhash")
            date_matches = g.get("date_match_count", 0)
            extras = []
            if ph is not None and ph != 999:
                extras.append(f"pHash {ph}")
            if dh is not None and dh != 999:
                extras.append(f"dHash {dh}")
            if date_matches:
                extras.append(f"{date_matches} date-match{'es' if date_matches > 1 else ''}")
            extras_str = " | ".join(extras) if extras else f"dist {best}/256"
            lines.append(
                f"[{n}] {g['publication']}  -  {g['copy_count']} copies  "
                f"({confidence})  [{extras_str}]"
            )
            keep_idx = g.get("keep_index", 0)
            for ci, copy in enumerate(g["copies"], 1):
                pos = _fmt_range(copy["seq_range"][0], copy["seq_range"][1])
                is_keep = (ci - 1) == keep_idx
                tag = "  <- KEEP (most photos)" if is_keep else "  <- remove from stack"
                date_str = ""
                if copy.get("consensus_date"):
                    y, m, d = copy["consensus_date"]
                    date_str = f", date: {y:04d}-{m:02d}-{d:02d}"
                lines.append(f"    Copy {ci}: positions {pos}  ({copy['photo_count']} photos{date_str}){tag}")
                for p in copy["photos"]:
                    lines.append(f"        - {p}")
            lines.append("")

    report = "\n".join(lines)
    # utf-8-sig adds a BOM so Windows tools (Notepad, type, Get-Content) detect UTF-8.
    out_path.write_text(report, encoding="utf-8-sig")
    print(report)


# ─── HTML report with embedded thumbnails (base64) ───────────────────────────

def make_thumbnail_b64(src_path: str) -> str:
    """Return a base64-encoded JPEG thumbnail (data-URI body) for embedding in HTML."""
    import base64
    import io
    img = Image.open(src_path)
    img.draft("RGB", (THUMB_SIZE * 2, THUMB_SIZE * 2))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=THUMB_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_html_report(images, groups, folder: Path, threshold: int,
                         warnings: list[str] | None = None,
                         pages_per_magazine: int = PAGES_PER_MAGAZINE,
                         feature_source: dict[str, str] | None = None):
    """Build a self-contained HTML report (base64-embedded thumbnails).

    When `feature_source` is provided, thumbnails are generated from the
    cropped versions (cleaner-looking report). Falls back to originals when
    a cropped version is missing.
    """
    img_lookup = {img["filename"]: img for img in images}

    # Build base64 thumbnails for every photo referenced in any group
    print("Building embedded thumbnails ...")
    needed = set()
    for g in groups:
        for copy in g["copies"]:
            for fname in copy["photos"]:
                needed.add(fname)
    thumbs_b64: dict[str, str] = {}
    for n, fname in enumerate(needed, 1):
        if fname in img_lookup:
            try:
                # Prefer cropped version (cleaner thumbnail, smaller file size)
                src = (feature_source or {}).get(fname) or img_lookup[fname]["path"]
                thumbs_b64[fname] = make_thumbnail_b64(src)
            except Exception as e:
                print(f"  WARN failed thumbnail for {fname}: {e}")
        if n % 50 == 0:
            print(f"  [{n}/{len(needed)}] thumbs encoded")
    print(f"  Done ({len(thumbs_b64)} thumbnails embedded inline).")

    # Build HTML
    by_pub = defaultdict(list)
    for img in images:
        by_pub[img["publication"]].append(img)
    triple_plus = sum(1 for g in groups if g["copy_count"] >= 3)

    pub_summary = " · ".join(
        f'<span class="pub-tag">{pub}</span> {len(ims)} photos'
        for pub, ims in sorted(by_pub.items())
    )

    # Pre-compute per-confidence counts so filter buttons can show them.
    def _conf_for(best):
        if best is None: return "LOW"
        if best <= 10: return "VERY HIGH"
        if best <= 25: return "HIGH"
        if best <= 40: return "MEDIUM"
        return "LOW"
    conf_count = defaultdict(int)
    for g in groups:
        conf_count[_conf_for(g["best_distance"])] += 1
    high_count = conf_count["VERY HIGH"] + conf_count["HIGH"]
    medium_count = conf_count["MEDIUM"]
    strong_count = high_count + medium_count  # everything except LOW

    if warnings:
        from html import escape as _esc
        warnings_html = (
            '<div class="warnings"><strong>Heads up:</strong>'
            'Some clusters look suspicious — verify these visually.<ul>'
            + "".join(f"<li>{_esc(w)}</li>" for w in warnings)
            + '</ul></div>'
        )
    else:
        warnings_html = ""

    html = []
    html.append(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Magazine Duplicates</title>
<style>
:root {{
  --bg: #0d0d0f;
  --bg-card: #16171a;
  --bg-card-hover: #1c1d21;
  --bg-elevated: #1f2025;
  --border: #25272d;
  --border-strong: #34373f;
  --text: #e8e8ea;
  --text-dim: #8a8d96;
  --text-muted: #5c5f68;
  --accent: #5b8def;
  --keep: #4ea866;
  --keep-bg: rgba(78, 168, 102, 0.08);
  --remove: #c95555;
  --remove-bg: rgba(201, 85, 85, 0.06);
  --conf-very-high: #4ea866;
  --conf-high: #6db981;
  --conf-medium: #d4a047;
  --conf-low: #c95555;
}}

* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif;
  padding: 0 0 80px 0;
}}

/* ─── Header ─────────────────────────────────────────────── */
header {{
  background: linear-gradient(180deg, #1a1b1f 0%, #131418 100%);
  border-bottom: 1px solid var(--border);
  padding: 28px 40px 20px;
}}
header h1 {{
  margin: 0 0 4px 0;
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.3px;
}}
header .subtitle {{
  color: var(--text-dim);
  font-size: 13px;
  margin-bottom: 14px;
}}
.stats {{
  display: flex;
  gap: 32px;
  margin-top: 16px;
  flex-wrap: wrap;
}}
.stat {{
  display: flex;
  flex-direction: column;
  gap: 2px;
}}
.stat .num {{
  font-size: 24px;
  font-weight: 600;
  color: var(--text);
}}
.stat .lbl {{
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.6px;
}}
.pub-tag {{
  display: inline-block;
  padding: 2px 8px;
  background: var(--bg-elevated);
  border-radius: 4px;
  font-size: 12px;
  color: var(--text-dim);
  margin-right: 4px;
}}

/* ─── Warnings banner ───────────────────────────────────── */
.warnings {{
  background: rgba(212, 160, 71, 0.08);
  border-bottom: 1px solid rgba(212, 160, 71, 0.3);
  padding: 12px 40px;
  font-size: 12px;
  color: #e8d5a8;
}}
.warnings strong {{ color: #d4a047; margin-right: 6px; }}
.warnings ul {{ margin: 6px 0 0 0; padding-left: 22px; }}
.warnings li {{ margin: 2px 0; }}

/* ─── Filter bar ─────────────────────────────────────────── */
.filter-bar {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(13, 13, 15, 0.92);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 14px 40px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}}
.filter-bar .label {{
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-right: 4px;
}}
.btn {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 6px 12px;
  font-size: 12px;
  border-radius: 6px;
  cursor: pointer;
  font-family: inherit;
  transition: all 0.15s;
}}
.btn:hover {{
  background: var(--bg-card-hover);
  color: var(--text);
  border-color: var(--border-strong);
}}
.btn.active {{
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}}
.btn .count {{
  display: inline-block;
  margin-left: 4px;
  padding: 1px 5px;
  background: rgba(255,255,255,0.08);
  border-radius: 8px;
  font-size: 10px;
  font-weight: 600;
}}
.progress {{
  margin-left: auto;
  font-size: 12px;
  color: var(--text-dim);
}}
.progress strong {{ color: var(--text); }}

/* ─── Groups ─────────────────────────────────────────────── */
main {{ padding: 24px 40px; max-width: 1400px; margin: 0 auto; }}

.group {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 16px;
  transition: border-color 0.15s;
}}
.group:hover {{ border-color: var(--border-strong); }}
.group.verified {{ opacity: 0.45; }}
.group.dismissed {{ opacity: 0.25; }}
.group.dismissed .copies {{ display: none; }}

.group-head {{
  padding: 16px 20px 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}}
.group-head .num {{
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  font-size: 13px;
  font-weight: 500;
}}
.group-head .pub {{
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
}}
.group-head .copies-pill {{
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  padding: 3px 9px;
  border-radius: 12px;
  font-size: 11px;
  color: var(--text-dim);
  font-weight: 500;
}}
.group-head .copies-pill.three {{ background: rgba(212, 160, 71, 0.12); color: #d4a047; border-color: rgba(212, 160, 71, 0.3); }}
.group-head .copies-pill.four-plus {{ background: rgba(201, 85, 85, 0.12); color: #c95555; border-color: rgba(201, 85, 85, 0.3); }}

.conf {{
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 4px;
  background: var(--bg-elevated);
}}
.conf-VERY-HIGH, .conf-HIGH {{ color: var(--conf-high); background: rgba(109, 185, 129, 0.1); }}
.conf-MEDIUM {{ color: var(--conf-medium); background: rgba(212, 160, 71, 0.1); }}
.conf-LOW {{ color: var(--conf-low); background: rgba(201, 85, 85, 0.1); }}

.dist {{
  font-size: 11px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}}
.actions {{ margin-left: auto; display: flex; gap: 6px; }}
.action-btn {{
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 5px 10px;
  font-size: 11px;
  border-radius: 5px;
  cursor: pointer;
  font-family: inherit;
  transition: all 0.15s;
}}
.action-btn:hover {{ background: var(--bg-elevated); color: var(--text); }}
.action-btn.confirmed {{ background: rgba(78, 168, 102, 0.15); border-color: var(--keep); color: var(--keep); }}
.action-btn.dismissed {{ background: rgba(201, 85, 85, 0.12); border-color: var(--remove); color: var(--remove); }}

.copies {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 12px;
  padding: 4px 20px 20px;
}}
.copy {{
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  position: relative;
}}
.copy.keep {{ border-left: 3px solid var(--keep); background: linear-gradient(90deg, var(--keep-bg) 0%, var(--bg-elevated) 30%); }}
.copy.remove {{ border-left: 3px solid var(--remove); background: linear-gradient(90deg, var(--remove-bg) 0%, var(--bg-elevated) 30%); }}

.copy-head {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}}
.copy-label {{
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}}
.copy.keep .copy-label {{ color: var(--keep); }}
.copy.remove .copy-label {{ color: var(--remove); }}
.copy-pos {{
  font-size: 13px;
  color: var(--text);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}}
.copy-meta {{
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 2px;
}}
.photos {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 6px;
}}
.photo {{
  position: relative;
  cursor: zoom-in;
  border-radius: 4px;
  overflow: hidden;
  background: #000;
  border: 1px solid var(--border);
  transition: border-color 0.15s, transform 0.15s;
}}
.photo:hover {{ border-color: var(--accent); transform: translateY(-1px); }}
.photo img {{
  width: 100%;
  height: auto;
  display: block;
}}
.photo-cap {{
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 3px 5px;
  font-size: 9px;
  color: var(--text-dim);
  background: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.85) 100%);
  font-variant-numeric: tabular-nums;
  text-align: center;
  letter-spacing: 0.3px;
}}

/* ─── Lightbox ───────────────────────────────────────────── */
.lightbox {{
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.95);
  z-index: 100;
  justify-content: center;
  align-items: center;
  padding: 40px;
  cursor: zoom-out;
}}
.lightbox.open {{ display: flex; }}
.lightbox img {{
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}}
.lightbox-cap {{
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  color: var(--text);
  font-size: 12px;
  background: var(--bg-card);
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid var(--border);
}}

/* ─── Empty state when filter shows nothing ────────────── */
.empty {{
  text-align: center;
  padding: 60px 20px;
  color: var(--text-muted);
}}
.empty.hidden {{ display: none; }}

/* ─── Covers-only mode: hide non-cover thumbs per copy ──── */
body.covers-only .photo:not(.cover) {{ display: none; }}
body.covers-only .copy .more-pages {{ display: inline-block; }}
.copy .more-pages {{
  display: none;
  margin-top: 6px;
  font-size: 11px;
  color: var(--text-muted);
  font-style: italic;
}}

/* ─── Segmented control (replaces multi-button filter bar) ─ */
.seg {{
  display: inline-flex;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}}
.seg .btn {{
  border: 0;
  border-radius: 0;
  border-right: 1px solid var(--border);
  margin: 0;
}}
.seg .btn:last-child {{ border-right: 0; }}
.filter-axis {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.filter-axis-label {{
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-right: 2px;
}}

/* ─── Per-copy date pill ───────────────────────────────── */
.copy-date {{
  display: inline-block;
  margin-left: 8px;
  padding: 2px 7px;
  background: rgba(91, 141, 239, 0.12);
  border: 1px solid rgba(91, 141, 239, 0.3);
  border-radius: 4px;
  font-size: 10px;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.3px;
}}
.date-match-badge {{
  display: inline-block;
  margin-left: 6px;
  padding: 2px 7px;
  background: rgba(78, 168, 102, 0.12);
  border: 1px solid rgba(78, 168, 102, 0.3);
  border-radius: 4px;
  font-size: 10px;
  color: var(--keep);
  font-weight: 600;
}}

/* ─── Compare modal ─────────────────────────────────────── */
.compare-modal {{
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.92);
  z-index: 200;
  flex-direction: column;
  padding: 24px;
}}
.compare-modal.open {{ display: flex; }}
.compare-modal-head {{
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  color: var(--text);
}}
.compare-modal-head h3 {{ margin: 0; font-size: 16px; font-weight: 600; }}
.compare-modal-close {{
  margin-left: auto;
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
}}
.compare-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  flex: 1;
  overflow: auto;
}}
.compare-side {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  overflow: auto;
}}
.compare-side h4 {{
  margin: 0 0 10px 0;
  font-size: 13px;
  color: var(--text-dim);
  font-weight: 500;
}}
.compare-side .compare-photo {{
  margin-bottom: 12px;
}}
.compare-side .compare-photo img {{
  width: 100%;
  border-radius: 4px;
  border: 1px solid var(--border);
}}
.compare-side .compare-photo .label {{
  font-size: 10px;
  color: var(--text-muted);
  text-align: center;
  margin-top: 3px;
  font-variant-numeric: tabular-nums;
}}

@media (max-width: 768px) {{
  header, .filter-bar, main {{ padding-left: 20px; padding-right: 20px; }}
  .copies {{ grid-template-columns: 1fr; }}
}}
</style>
</head><body>

<header>
  <h1>Magazine Duplicates</h1>
  <div class="subtitle">{pub_summary}</div>
  <div class="stats">
    <div class="stat"><span class="num">{len(images)}</span><span class="lbl">Photos</span></div>
    <div class="stat"><span class="num">{len(groups)}</span><span class="lbl">Duplicate magazines</span></div>
    <div class="stat"><span class="num">{triple_plus}</span><span class="lbl">3+ copies</span></div>
    <div class="stat"><span class="num">{threshold}<span style="font-size:14px;color:var(--text-muted)">/256</span></span><span class="lbl">Match threshold</span></div>
    <div class="stat"><span class="num">{pages_per_magazine}</span><span class="lbl">Pages/magazine</span></div>
  </div>
  <div style="margin-top:14px;font-size:12px;color:var(--text-muted)">
    Detection compares front + back page hashes only (the first 2 photos of each magazine).
    Inside pages are listed but not used for matching.
  </div>
</header>
{warnings_html}
<div class="filter-bar">
  <div class="filter-axis">
    <span class="filter-axis-label">Confidence</span>
    <div class="seg">
      <button class="btn active" data-conf="strong" title="Hide LOW-confidence groups (likely false positives)">Strong <span class="count">{strong_count}</span></button>
      <button class="btn" data-conf="all">All <span class="count">{len(groups)}</span></button>
    </div>
  </div>
  <div class="filter-axis">
    <span class="filter-axis-label">Copies</span>
    <div class="seg">
      <button class="btn active" data-copies="any">Any</button>
      <button class="btn" data-copies="3plus">3+ <span class="count">{triple_plus}</span></button>
    </div>
  </div>
  <div class="filter-axis">
    <span class="filter-axis-label">Status</span>
    <div class="seg">
      <button class="btn active" data-status="all">All</button>
      <button class="btn" data-status="unverified">Unverified</button>
      <button class="btn" data-status="confirmed">Confirmed</button>
    </div>
  </div>
  <button class="btn" id="covers-toggle" title="Show only the cover photo per copy">Covers only</button>
  <span class="progress">Verified: <strong id="progress-num">0</strong> / {len(groups)}</span>
</div>

<main id="groups">""")

    for n, g in enumerate(groups, 1):
        best = g["best_distance"]
        ph = g.get("best_phash")
        dh = g.get("best_dhash")
        date_match_count = g.get("date_match_count", 0)
        confidence = g.get("confidence") or (
            "VERY HIGH" if best is not None and best <= 10 else
            "HIGH" if best is not None and best <= 25 else
            "MEDIUM" if best is not None and best <= 40 else
            "LOW"
        )
        copies_pill_cls = ""
        if g["copy_count"] == 3:
            copies_pill_cls = " three"
        elif g["copy_count"] >= 4:
            copies_pill_cls = " four-plus"

        html.append(
            f'<div class="group" id="g-{n}" '
            f'data-copies="{g["copy_count"]}" data-confidence="{confidence}" '
            f'data-date-match="{1 if date_match_count else 0}">'
        )
        html.append('<div class="group-head">')
        html.append(f'<span class="num">#{n:02d}</span>')
        html.append(f'<span class="pub">{g["publication"]}</span>')
        html.append(f'<span class="copies-pill{copies_pill_cls}">{g["copy_count"]} copies</span>')
        html.append(f'<span class="conf conf-{confidence.replace(" ", "-")}">{confidence}</span>')
        # Distance details: pHash + dHash + date-match
        dist_parts = []
        if ph is not None and ph != 999:
            dist_parts.append(f"pHash {ph}/256")
        if dh is not None and dh != 999:
            dist_parts.append(f"dHash {dh}/256")
        dist_parts.append(f'{g["match_pair_count"]} pair-match{"es" if g["match_pair_count"] != 1 else ""}')
        html.append(f'<span class="dist">{" · ".join(dist_parts)}</span>')
        if date_match_count:
            html.append(f'<span class="date-match-badge">📅 date match</span>')
        html.append('<div class="actions">')
        if g["copy_count"] >= 2:
            html.append(f'<button class="action-btn" data-compare="{n}" title="Open side-by-side compare">⇄ Compare</button>')
        html.append(f'<button class="action-btn" data-action="confirm" data-id="{n}">✓ Confirmed</button>')
        html.append(f'<button class="action-btn" data-action="dismiss" data-id="{n}">✕ Not a dup</button>')
        html.append('</div>')
        html.append('</div>')

        html.append('<div class="copies">')
        keep_idx = g.get("keep_index", 0)
        for ci, copy in enumerate(g["copies"], 1):
            is_keep = (ci - 1) == keep_idx
            cls = "keep" if is_keep else "remove"
            label = "KEEP" if is_keep else "REMOVE"
            pos = _fmt_range(copy["seq_range"][0], copy["seq_range"][1])
            html.append(f'<div class="copy {cls}">')
            html.append('<div class="copy-head">')
            date_pill = ""
            if copy.get("consensus_date"):
                y, mo, d = copy["consensus_date"]
                date_pill = f'<span class="copy-date">{y:04d}-{mo:02d}-{d:02d}</span>'
            html.append(f'<div><div class="copy-label">{label}</div>'
                        f'<div class="copy-meta">Copy {ci} · {copy["photo_count"]} photo{"s" if copy["photo_count"] > 1 else ""}{date_pill}</div></div>')
            html.append(f'<div class="copy-pos">pos {pos}</div>')
            html.append('</div>')

            html.append('<div class="photos">')
            # First photo per copy is treated as the "cover" (lowest seq).
            for pi, fname in enumerate(copy["photos"]):
                short = fname.split("_")[-1].replace(".JPG", "").replace(".jpg", "")
                b64 = thumbs_b64.get(fname, "")
                src = f"data:image/jpeg;base64,{b64}" if b64 else ""
                photo_cls = "photo cover" if pi == 0 else "photo"
                html.append(
                    f'<div class="{photo_cls}" data-full="{src}" data-name="{fname}">'
                    f'<img src="{src}" alt="{fname}" loading="lazy">'
                    f'<div class="photo-cap">{short}</div>'
                    f'</div>'
                )
            html.append('</div>')
            extra_pages = copy["photo_count"] - 1
            if extra_pages > 0:
                html.append(
                    f'<div class="more-pages">+ {extra_pages} more page'
                    f'{"s" if extra_pages > 1 else ""} hidden — click "Covers only" to toggle</div>'
                )
            html.append('</div>')
        html.append('</div></div>')

    html.append("""</main>

<div class="empty hidden" id="empty">No duplicates match this filter.</div>

<div class="lightbox" id="lightbox">
  <img id="lightbox-img" alt="">
  <div class="lightbox-cap" id="lightbox-cap"></div>
</div>

<div class="compare-modal" id="compare-modal">
  <div class="compare-modal-head">
    <h3 id="compare-title">Compare</h3>
    <span id="compare-info" style="font-size:12px;color:#8a8d96"></span>
    <button class="compare-modal-close" onclick="closeCompare()">Close (Esc)</button>
  </div>
  <div class="compare-grid" id="compare-grid"></div>
</div>

<script>
// ── Multi-axis filter logic (orthogonal: confidence × copies × verification) ──
const groups = document.querySelectorAll('.group');
const emptyState = document.getElementById('empty');
const filterState = { conf: 'strong', copies: 'any', status: 'all' };

function applyFilters() {
  let visible = 0;
  groups.forEach(g => {
    const conf = g.dataset.confidence;
    const copies = parseInt(g.dataset.copies);
    const verified = g.classList.contains('verified');
    const dismissed = g.classList.contains('dismissed');

    let pass = true;
    if (filterState.conf === 'strong' && conf === 'LOW') pass = false;
    if (filterState.copies === '3plus' && copies < 3) pass = false;
    if (filterState.status === 'unverified' && (verified || dismissed)) pass = false;
    if (filterState.status === 'confirmed' && !verified) pass = false;
    g.style.display = pass ? '' : 'none';
    if (pass) visible++;
  });
  emptyState.classList.toggle('hidden', visible > 0);
}

document.querySelectorAll('.filter-bar [data-conf]').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.filter-bar [data-conf]').forEach(x => x.classList.toggle('active', x === b));
    filterState.conf = b.dataset.conf;
    applyFilters();
  });
});
document.querySelectorAll('.filter-bar [data-copies]').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.filter-bar [data-copies]').forEach(x => x.classList.toggle('active', x === b));
    filterState.copies = b.dataset.copies;
    applyFilters();
  });
});
document.querySelectorAll('.filter-bar [data-status]').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.filter-bar [data-status]').forEach(x => x.classList.toggle('active', x === b));
    filterState.status = b.dataset.status;
    applyFilters();
  });
});
applyFilters();

// ── Verification state (persisted in localStorage) ──
const STORAGE_KEY = 'magazine-dup-verification';
const state = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');

function applyState() {
  let verifiedCount = 0;
  groups.forEach(g => {
    const id = g.id.replace('g-', '');
    const s = state[id];
    g.classList.toggle('verified', s === 'confirm');
    g.classList.toggle('dismissed', s === 'dismiss');
    if (s) verifiedCount++;
    g.querySelectorAll('.action-btn').forEach(btn => {
      btn.classList.toggle('confirmed', s === 'confirm' && btn.dataset.action === 'confirm');
      btn.classList.toggle('dismissed', s === 'dismiss' && btn.dataset.action === 'dismiss');
    });
  });
  document.getElementById('progress-num').textContent = verifiedCount;
}

document.querySelectorAll('.action-btn').forEach(btn => {
  btn.addEventListener('click', e => {
    const id = btn.dataset.id;
    const action = btn.dataset.action;
    if (state[id] === action) delete state[id]; // toggle off
    else state[id] = action;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    applyState();
  });
});

applyState();

// ── Covers-only toggle (persisted) ──
const COVERS_KEY = 'magazine-dup-covers-only';
const coversBtn = document.getElementById('covers-toggle');
function applyCoversMode() {
  const on = localStorage.getItem(COVERS_KEY) !== 'false';  // default ON
  document.body.classList.toggle('covers-only', on);
  coversBtn.classList.toggle('active', on);
  coversBtn.textContent = on ? '✓ Covers only' : 'Covers only';
}
coversBtn.addEventListener('click', () => {
  const on = localStorage.getItem(COVERS_KEY) !== 'false';
  localStorage.setItem(COVERS_KEY, on ? 'false' : 'true');
  applyCoversMode();
});
applyCoversMode();

// ── Lightbox ──
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxCap = document.getElementById('lightbox-cap');

document.querySelectorAll('.photo').forEach(p => {
  p.addEventListener('click', () => {
    lightboxImg.src = p.dataset.full;
    lightboxCap.textContent = p.dataset.name;
    lightbox.classList.add('open');
  });
});

lightbox.addEventListener('click', () => lightbox.classList.remove('open'));

// ── Side-by-side compare modal ──
const compareModal = document.getElementById('compare-modal');
const compareGrid = document.getElementById('compare-grid');
const compareTitle = document.getElementById('compare-title');
const compareInfo = document.getElementById('compare-info');

function openCompare(groupNum) {
  const group = document.getElementById('g-' + groupNum);
  if (!group) return;
  const pub = group.querySelector('.pub').textContent;
  const conf = group.dataset.confidence;
  const copies = group.querySelectorAll('.copy');
  compareTitle.textContent = pub + ' — Group #' + groupNum;
  compareInfo.textContent = `${copies.length} copies · ${conf}`;
  compareGrid.innerHTML = '';
  // Stack all copies side-by-side. CSS grid limits to 2 columns naturally;
  // additional copies wrap to a second row.
  copies.forEach((cp, idx) => {
    const side = document.createElement('div');
    side.className = 'compare-side';
    const isKeep = cp.classList.contains('keep');
    const pos = cp.querySelector('.copy-pos').textContent;
    side.innerHTML = `<h4>${isKeep ? '✓ KEEP' : 'REMOVE'} — Copy ${idx+1} (${pos})</h4>`;
    cp.querySelectorAll('.photo').forEach(ph => {
      const div = document.createElement('div');
      div.className = 'compare-photo';
      const img = ph.querySelector('img');
      div.innerHTML = `<img src="${img.src}" alt=""><div class="label">${ph.dataset.name}</div>`;
      side.appendChild(div);
    });
    compareGrid.appendChild(side);
  });
  // If many copies, switch to N-column grid.
  compareGrid.style.gridTemplateColumns = `repeat(${Math.min(copies.length, 4)}, 1fr)`;
  compareModal.classList.add('open');
}
function closeCompare() { compareModal.classList.remove('open'); }
document.querySelectorAll('[data-compare]').forEach(b => {
  b.addEventListener('click', () => openCompare(b.dataset.compare));
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if (compareModal.classList.contains('open')) closeCompare();
    else if (lightbox.classList.contains('open')) lightbox.classList.remove('open');
  }
});

// ── Keyboard shortcuts (single-axis quick toggles) ──
document.addEventListener('keydown', e => {
  if (lightbox.classList.contains('open')) return;
  if (compareModal.classList.contains('open')) return;
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 's') document.querySelector('.filter-bar [data-conf="strong"]').click();
  else if (e.key === 'a') document.querySelector('.filter-bar [data-conf="all"]').click();
  else if (e.key === '3') document.querySelector('.filter-bar [data-copies="3plus"]').click();
  else if (e.key === 'u') document.querySelector('.filter-bar [data-status="unverified"]').click();
  else if (e.key === 'c') coversBtn.click();
});
</script>
</body></html>""")

    out = folder / HTML_REPORT_FILE
    out.write_text("\n".join(html), encoding="utf-8")
    print(f"HTML report saved to: {out}")
    print(f"  Open it in your browser to visually verify each duplicate.")


# ─── Library entry point (for desktop app) ───────────────────────────────────

def _crop_path_for(folder: Path, filename: str) -> Path:
    return folder / CROPPED_DIR / filename


def ensure_cropped(folder: Path, images: list[dict], progress_cb=None) -> dict[str, str]:
    """Ensure each image has a cropped version in <folder>/cropped/.

    Returns a dict mapping filename → path of the image to use for feature
    extraction (cropped if it exists or got generated successfully, else
    falls back to the original).
    """
    crop_dir = folder / CROPPED_DIR
    crop_dir.mkdir(exist_ok=True)
    paths: dict[str, str] = {}
    todo = []
    for img in images:
        cropped = _crop_path_for(folder, img["filename"])
        if cropped.exists() and cropped.stat().st_size > 0:
            paths[img["filename"]] = str(cropped)
        else:
            todo.append(img)
    if not todo or not cropping_available():
        # Fall back to originals for anything not yet cropped (e.g. cv2 missing)
        for img in todo:
            paths[img["filename"]] = img["path"]
        return paths
    if progress_cb:
        progress_cb("crop", 0, len(todo))
    for n, img in enumerate(todo, 1):
        dst = _crop_path_for(folder, img["filename"])
        try:
            dims = crop_magazine(img["path"], dst)
            paths[img["filename"]] = str(dst) if dims else img["path"]
        except Exception:
            paths[img["filename"]] = img["path"]
        if progress_cb and (n % 10 == 0 or n == len(todo)):
            progress_cb("crop", n, len(todo))
    return paths


def analyze_folder(folder_path, threshold: int = MATCH_THRESHOLD,
                   pages_per_magazine: int = PAGES_PER_MAGAZINE,
                   rehash: bool = False, progress_cb=None):
    """
    Run the full pipeline on a folder. Returns (images, groups, stats) or
    (None, [], {"error": ...}) if the folder has no magazine images.

    progress_cb(phase: str, n: int, total: int) is called periodically:
      - phase="scan", n=files_found, total=files_found
      - phase="crop", n=cropped_so_far, total=images_to_crop
      - phase="hash", n=images_hashed_so_far, total=images_to_hash
      - phase="match", n=0, total=0
      - phase="render", n=0, total=0
    """
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        return None, [], {"error": f"Not a directory: {folder}"}

    if progress_cb: progress_cb("scan", 0, 0)
    images = scan_images(folder)
    if not images:
        return None, [], {"error": "No magazine images found (expected names like 2026_05_08_NME_0611.JPG)"}
    if progress_cb: progress_cb("scan", len(images), len(images))

    # Crop step: produce cleaned versions in <folder>/cropped/ for both server
    # upload and downstream feature extraction.
    feature_source = ensure_cropped(folder, images, progress_cb)

    cache = {} if rehash else load_cache(folder)
    # An entry is "complete" only if it has hash + edge_std (v4 schema).
    todo = [img for img in images
            if img["filename"] not in cache
            or not isinstance(cache[img["filename"]], dict)
            or "edge_std" not in cache[img["filename"]]]

    if todo:
        if progress_cb: progress_cb("hash", 0, len(todo))
        for n, img in enumerate(todo, 1):
            src = feature_source.get(img["filename"], img["path"])
            try:
                cache[img["filename"]] = compute_features(src)
            except Exception:
                continue
            if n % 25 == 0 or n == len(todo):
                save_cache(folder, cache)
                if progress_cb: progress_cb("hash", n, len(todo))
        save_cache(folder, cache)

    if progress_cb: progress_cb("match", 0, 0)
    groups, warnings = find_duplicate_groups(images, cache, threshold, pages_per_magazine)

    if progress_cb: progress_cb("render", 0, 0)

    triple_plus = sum(1 for g in groups if g["copy_count"] >= 3)
    stats = {
        "total_images": len(images),
        "total_groups": len(groups),
        "triple_plus": triple_plus,
        "publications": sorted({i["publication"] for i in images}),
        "threshold": threshold,
        "pages_per_magazine": pages_per_magazine,
        "folder": str(folder),
        "cropped_folder": str(folder / CROPPED_DIR),
        "cropping_enabled": cropping_available(),
        "warnings": warnings,
    }

    # Always write the standalone report files so they can be opened separately.
    # The report renderer uses the cropped images for thumbnails when available
    # (cleaner-looking report, smaller HTML file).
    generate_report(images, groups, folder / REPORT_FILE, threshold, warnings, pages_per_magazine)
    generate_html_report(images, groups, folder, threshold, warnings, pages_per_magazine,
                         feature_source=feature_source)

    return images, groups, stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Find duplicate magazine issues using perceptual hashing")
    parser.add_argument("folder", help="Folder containing magazine JPGs")
    parser.add_argument("--rehash", action="store_true", help="Recompute all hashes (ignore cache)")
    parser.add_argument("--threshold", type=int, default=MATCH_THRESHOLD,
                        help=f"Match threshold (lower = stricter, default {MATCH_THRESHOLD})")
    parser.add_argument("--pages-per-magazine", type=int, default=PAGES_PER_MAGAZINE,
                        help=f"Photos per magazine in your photographing convention "
                             f"(default {PAGES_PER_MAGAZINE} = front, back, colophon, index)")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        return

    threshold = args.threshold
    pages_per_magazine = args.pages_per_magazine

    print(f"Scanning {folder} ...")
    images = scan_images(folder)
    if not images:
        print("No magazine images found (expected names like 2026_05_08_NME_0611.JPG)")
        return

    pubs = sorted(set(i["publication"] for i in images))
    print(f"Found {len(images)} images across publications: {', '.join(pubs)}")

    cache = {} if args.rehash else load_cache(folder)
    todo = [img for img in images
            if img["filename"] not in cache
            or not isinstance(cache[img["filename"]], dict)
            or "edge_std" not in cache[img["filename"]]]
    print(f"{len(cache)} cached, {len(todo)} new features to compute.")

    if todo:
        start = time.time()
        for n, img in enumerate(todo, 1):
            try:
                cache[img["filename"]] = compute_features(img["path"])
            except Exception as e:
                print(f"  ERROR computing features for {img['filename']}: {e}")
                continue
            if n % 25 == 0 or n == len(todo):
                elapsed = time.time() - start
                rate = n / elapsed if elapsed > 0 else 0
                eta = (len(todo) - n) / rate if rate > 0 else 0
                print(f"  [{n}/{len(todo)}]  {rate:.1f} img/s  ETA {eta:.0f}s")
                save_cache(folder, cache)
        save_cache(folder, cache)
        print(f"Hashing done in {time.time() - start:.1f}s.\n")

    print(f"Searching for duplicate magazines "
          f"(threshold={threshold}/256, {pages_per_magazine} pages/magazine) ...")
    groups, warnings = find_duplicate_groups(images, cache, threshold, pages_per_magazine)
    n_pairs = sum(g["match_pair_count"] for g in groups)
    triple_plus = sum(1 for g in groups if g["copy_count"] >= 3)
    print(f"Found {len(groups)} duplicate magazine(s) "
          f"({triple_plus} with 3+ copies, {n_pairs} cover-to-cover matches).")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")
    print()

    generate_report(images, groups, folder / REPORT_FILE, threshold, warnings, pages_per_magazine)
    generate_html_report(images, groups, folder, threshold, warnings, pages_per_magazine)
    print(f"\nFull report saved to: {folder / REPORT_FILE}")
    print(f"Hash cache saved to:  {folder / CACHE_FILE}")


if __name__ == "__main__":
    main()
