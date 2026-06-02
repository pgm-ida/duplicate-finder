#!/usr/bin/env python3
"""Fixture-based tests for crop_magazine_with_meta and _find_magazine_bbox_bgsubtr.

Tests use synthetic images and monkeypatching so they run without real cameras
or the full rembg neural-net model.
"""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Make sure we import the worktree's copy of fdl, not any installed version.
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
import find_duplicates_local as fdl


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _dark_cover_on_white(h=1200, w=900, margin=60):
    """Synthetic: dark magazine cover (near-black) on white table top."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    img[margin:h-margin, margin:w-margin] = 25  # dark cover
    return img


def _holographic_on_white(h=1200, w=900, margin=60):
    """Synthetic: mid-grey holographic cover on white background."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    img[margin:h-margin, margin:w-margin] = 120  # mid-grey cover
    return img


def _grey_background(h=1200, w=900, margin=60):
    """Synthetic: dark cover on a grey scan surface."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 160  # grey bg
    img[margin:h-margin, margin:w-margin] = 30        # dark cover
    return img


def _write_jpg(tmp_path: Path, img: np.ndarray, name="test.jpg") -> Path:
    import cv2
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


def _small_quad(img: np.ndarray, scale=0.25) -> np.ndarray:
    """Return a quad occupying ~scale fraction of the image area (centred)."""
    h, w = img.shape[:2]
    sw = int(w * scale); sh = int(h * scale)
    cx, cy = w // 2, h // 2
    return np.array([
        [cx - sw//2, cy - sh//2],
        [cx + sw//2, cy - sh//2],
        [cx + sw//2, cy + sh//2],
        [cx - sw//2, cy + sh//2],
    ], dtype=np.float32)


def _full_cover_quad(img: np.ndarray, margin=60) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array([
        [margin, margin],
        [w - margin, margin],
        [w - margin, h - margin],
        [margin, h - margin],
    ], dtype=np.float32)


# ─── Tests: _find_magazine_bbox_bgsubtr ──────────────────────────────────────

class TestFindBgsubtr:
    def test_dark_cover_on_white_returns_bbox(self, tmp_path):
        """Dark magazine on white background — bgsubtr must find the cover."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")
        img = _dark_cover_on_white()
        bbox = fdl._find_magazine_bbox_bgsubtr(img)
        assert bbox is not None, "expected bbox for dark cover on white"
        xs = bbox[:, 0]; ys = bbox[:, 1]
        # Cover spans most of the image except the 60px margin
        assert xs.min() < 100, f"x_min too large: {xs.min()}"
        assert xs.max() > 800, f"x_max too small: {xs.max()}"
        assert ys.min() < 100, f"y_min too large: {ys.min()}"
        assert ys.max() > 1100, f"y_max too small: {ys.max()}"

    def test_holographic_returns_bbox(self, tmp_path):
        """Mid-grey cover on white — bgsubtr must still find it."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")
        img = _holographic_on_white()
        bbox = fdl._find_magazine_bbox_bgsubtr(img)
        assert bbox is not None, "expected bbox for holographic/grey cover"

    def test_grey_background_returns_bbox(self, tmp_path):
        """Dark cover on grey surface — bgsubtr learns grey as background."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")
        img = _grey_background()
        bbox = fdl._find_magazine_bbox_bgsubtr(img)
        assert bbox is not None, "expected bbox on grey background"

    def test_white_on_white_returns_none(self):
        """All-white image (white magazine on white table) → None."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")
        img = np.ones((1000, 800, 3), dtype=np.uint8) * 245
        bbox = fdl._find_magazine_bbox_bgsubtr(img)
        assert bbox is None, f"expected None for white-on-white, got {bbox}"

    def test_near_uniform_returns_none(self):
        """Near-uniform low-contrast image → None."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")
        img = np.ones((1000, 800, 3), dtype=np.uint8) * 200
        img[200:800, 150:650] = 210  # only 10-unit contrast
        bbox = fdl._find_magazine_bbox_bgsubtr(img)
        # Low contrast — should return None (below 20-px threshold)
        assert bbox is None, f"expected None for near-uniform image, got {bbox}"


# ─── Tests: crop_magazine_with_meta — confidence gate ────────────────────────

class TestCropMagazineWithMeta:

    # ── envelope check ────────────────────────────────────────────────────────

    def test_envelope_check_marks_nested_quad_as_low(self, tmp_path):
        """Detector returns a small nested quad (silver circle / photo inset).
        The quad area is well below 70% of the bgsubtr bbox → LOW confidence,
        no crop written.
        """
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")

        img = _dark_cover_on_white()
        src = _write_jpg(tmp_path, img)
        dst = tmp_path / "out.jpg"

        # Small nested quad (≈6% of frame area)
        small = _small_quad(img, scale=0.25)

        orig_rembg = fdl._find_magazine_quad_rembg
        orig_cv = fdl._find_magazine_quad
        try:
            fdl._find_magazine_quad_rembg = lambda _img: small
            fdl._find_magazine_quad       = lambda _img: small

            dims, meta = fdl.crop_magazine_with_meta(str(src), str(dst))

            assert meta["confidence"] == "low", (
                f"expected confidence=low for nested quad, got {meta['confidence']!r}. "
                f"reasons={meta['reasons']}"
            )
            assert "quad_inside_bbox_envelope" in meta["reasons"]
            assert not dst.exists(), "should not write a crop for low confidence"
            assert dims is None
        finally:
            fdl._find_magazine_quad_rembg = orig_rembg
            fdl._find_magazine_quad       = orig_cv

    def test_valid_quad_high_confidence(self, tmp_path):
        """Full-cover quad passes all checks → HIGH confidence, crop written."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")

        img = _dark_cover_on_white()
        src = _write_jpg(tmp_path, img)
        dst = tmp_path / "out.jpg"

        good = _full_cover_quad(img)

        orig_rembg = fdl._find_magazine_quad_rembg
        orig_cv = fdl._find_magazine_quad
        try:
            fdl._find_magazine_quad_rembg = lambda _img: good
            fdl._find_magazine_quad       = lambda _img: good

            dims, meta = fdl.crop_magazine_with_meta(str(src), str(dst))

            assert meta["confidence"] == "high", (
                f"expected high, got {meta['confidence']!r}. reasons={meta['reasons']}"
            )
            assert not meta["reasons"]
            assert dst.exists(), "should write a crop for high confidence"
            assert dims is not None
        finally:
            fdl._find_magazine_quad_rembg = orig_rembg
            fdl._find_magazine_quad       = orig_cv

    def test_no_quad_bgsubtr_rescue(self, tmp_path):
        """Both detectors return None but bgsubtr finds the cover →
        axis-aligned crop written, confidence HIGH.
        """
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")

        img = _dark_cover_on_white()
        src = _write_jpg(tmp_path, img)
        dst = tmp_path / "out.jpg"

        orig_rembg = fdl._find_magazine_quad_rembg
        orig_cv = fdl._find_magazine_quad
        try:
            fdl._find_magazine_quad_rembg = lambda _img: None
            fdl._find_magazine_quad       = lambda _img: None

            dims, meta = fdl.crop_magazine_with_meta(str(src), str(dst))

            assert meta["confidence"] == "high", (
                f"expected high (bgsubtr rescue), got {meta['confidence']!r}. "
                f"reasons={meta['reasons']}"
            )
            assert meta["method"] == "bgsubtr"
            assert dst.exists()
            assert dims is not None
        finally:
            fdl._find_magazine_quad_rembg = orig_rembg
            fdl._find_magazine_quad       = orig_cv

    def test_no_quad_no_bgsubtr_low(self, tmp_path):
        """All detectors fail (white-on-white) → no crop, confidence LOW."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")

        img = np.ones((1000, 800, 3), dtype=np.uint8) * 245  # white-on-white
        src = _write_jpg(tmp_path, img)
        dst = tmp_path / "out.jpg"

        orig_rembg = fdl._find_magazine_quad_rembg
        orig_cv = fdl._find_magazine_quad
        try:
            fdl._find_magazine_quad_rembg = lambda _img: None
            fdl._find_magazine_quad       = lambda _img: None

            dims, meta = fdl.crop_magazine_with_meta(str(src), str(dst))

            assert meta["confidence"] == "low"
            assert dims is None
            assert not dst.exists()
        finally:
            fdl._find_magazine_quad_rembg = orig_rembg
            fdl._find_magazine_quad       = orig_cv

    def test_backward_compat_crop_magazine(self, tmp_path):
        """crop_magazine() still returns (w, h) or None — no meta leakage."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")

        img = _dark_cover_on_white()
        src = _write_jpg(tmp_path, img)
        dst = tmp_path / "compat.jpg"

        good = _full_cover_quad(img)
        orig_rembg = fdl._find_magazine_quad_rembg
        orig_cv = fdl._find_magazine_quad
        try:
            fdl._find_magazine_quad_rembg = lambda _img: good
            fdl._find_magazine_quad       = lambda _img: good

            result = fdl.crop_magazine(str(src), str(dst))
            assert isinstance(result, tuple) and len(result) == 2
            assert dst.exists()
        finally:
            fdl._find_magazine_quad_rembg = orig_rembg
            fdl._find_magazine_quad       = orig_cv

    # ── aspect / orientation checks ───────────────────────────────────────────

    def test_aspect_out_of_range_low(self, tmp_path):
        """Very wide quad (aspect > 2.0) → LOW."""
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")

        img = _dark_cover_on_white(h=900, w=1200)
        src = _write_jpg(tmp_path, img)
        dst = tmp_path / "out.jpg"

        h, w = img.shape[:2]
        # Very wide quad: aspect ≈ 5:1
        wide_quad = np.array([
            [100, 400], [1100, 400], [1100, 600], [100, 600]
        ], dtype=np.float32)

        orig_rembg = fdl._find_magazine_quad_rembg
        orig_cv = fdl._find_magazine_quad
        try:
            fdl._find_magazine_quad_rembg = lambda _img: wide_quad
            fdl._find_magazine_quad       = lambda _img: wide_quad
            _, meta = fdl.crop_magazine_with_meta(str(src), str(dst))
            assert meta["confidence"] == "low"
            assert "aspect_out_of_range" in meta["reasons"]
        finally:
            fdl._find_magazine_quad_rembg = orig_rembg
            fdl._find_magazine_quad       = orig_cv


# ─── Tests: apply_rect_crop ──────────────────────────────────────────────────

class TestApplyRectCrop:
    def test_basic_crop(self, tmp_path):
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        img[100:900, 50:750] = 200
        src = _write_jpg(tmp_path, img)
        dst = tmp_path / "out.jpg"
        dims = fdl.apply_rect_crop(str(src), str(dst), {"x": 50, "y": 100, "w": 700, "h": 800})
        assert dims == (700, 800)
        assert dst.exists()

    def test_clamps_to_image_bounds(self, tmp_path):
        if not fdl.cropping_available():
            pytest.skip("cv2 not available")
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        src = _write_jpg(tmp_path, img)
        dst = tmp_path / "out.jpg"
        dims = fdl.apply_rect_crop(str(src), str(dst), {"x": 300, "y": 400, "w": 500, "h": 500})
        assert dims is not None
        assert dims[0] <= 400 and dims[1] <= 500


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
