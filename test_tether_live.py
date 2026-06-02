#!/usr/bin/env python3
"""End-to-end test of the LIVE tether path (the confidence gate in shooting mode).

Unlike test_cropping_meta.py (which tests the crop function in isolation), this
drives the real `TetherWatcher._process_one` — the exact method the live worker
thread runs when a photo lands in the watch folder — and the review-resolution
methods the GUI calls (accept_auto_crop / apply_manual_crop / discard_review).

It uses REAL photos from the deewee input set so the detector behaves exactly as
it will in production:
  - a HIGH-confidence cover  → auto-cropped, original deleted, verdict emitted
  - a LOW-confidence cover   → original STAGED in _review/ (never deleted),
                               crop_review event emitted, cover verdict deferred
                               to PENDING_REVIEW, worker free to continue
  - Accept / Adjust / Reshoot resolve the staged photo correctly

If the real input photos aren't present, the whole module is skipped.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import tether

INPUT = Path(r"C:\Users\Ida\Desktop\deewee\input")
HIGH_IMG = INPUT / "2026_06_01_input_2327.JPG"   # eval: HIGH, rembg
LOW_IMGS = [INPUT / f"2026_06_01_input_{n}.JPG"   # eval: LOW, quad_inside_bbox_envelope
            for n in (2331, 2333, 2334)]

pytestmark = pytest.mark.skipif(
    not HIGH_IMG.exists() or not all(p.exists() for p in LOW_IMGS),
    reason="real deewee input photos not available",
)


def _make_watcher(tmp_path) -> tuple[tether.TetherWatcher, list]:
    """Build a watcher wired to temp dirs, WITHOUT starting the observer/hotkey
    threads (we drive _process_one directly, exactly as the worker would)."""
    events: list = []
    w = tether.TetherWatcher(
        watch_dir=str(tmp_path / "watch"),
        library_dir=str(tmp_path / "library"),
        publication="TEST",
        cover_check=True,
        match_threshold=60,
        on_event=events.append,
    )
    Path(w.watch_dir).mkdir(parents=True, exist_ok=True)
    Path(w.library_dir).mkdir(parents=True, exist_ok=True)
    w._cache = {}            # start() normally loads this; we skip start()
    return w, events


def _drop(w: tether.TetherWatcher, real_img: Path) -> Path:
    """Copy a real photo into the watch dir and return its path (simulates the
    camera writing a file that the watchdog would enqueue)."""
    dst = Path(w.watch_dir) / real_img.name
    shutil.copy2(real_img, dst)
    return dst


def _types(events) -> list:
    return [e["type"] for e in events]


def _of_type(events, t) -> list:
    return [e for e in events if e["type"] == t]


# ─── HIGH-confidence path ────────────────────────────────────────────────────

def test_high_confidence_crops_and_deletes_original(tmp_path):
    w, events = _make_watcher(tmp_path)
    src = _drop(w, HIGH_IMG)

    w._process_one(src)

    mag = w._current_magazine
    assert mag is not None, "a magazine should have been auto-opened"
    cropped = mag / HIGH_IMG.name
    assert cropped.exists(), "HIGH-confidence crop must be written to the magazine"
    assert not src.exists(), "original must be deleted on a HIGH-confidence crop"
    assert not (w._review_dir / HIGH_IMG.name).exists(), "must NOT be staged for review"
    # First photo of a new magazine → a real (non-pending) cover verdict fired.
    verdicts = _of_type(events, "verdict")
    assert verdicts, "a cover verdict should be emitted for photo #1"
    assert verdicts[-1]["kind"] in ("UNIQUE", "DUPLICATE", "UNCERTAIN")
    assert not w._pending_reviews


# ─── LOW-confidence path: the core of the fix ────────────────────────────────

def test_low_confidence_stages_original_and_defers(tmp_path):
    w, events = _make_watcher(tmp_path)
    src = _drop(w, LOW_IMGS[0])

    w._process_one(src)

    mag = w._current_magazine
    assert mag is not None
    # The original must be preserved in _review/, NOT cropped, NOT deleted.
    staged = w._review_dir / LOW_IMGS[0].name
    assert staged.exists(), "LOW-confidence original MUST be staged in _review/"
    assert not src.exists(), "original is moved out of the watch dir (into _review/)"
    assert not (mag / LOW_IMGS[0].name).exists(), \
        "no crop should be written for a LOW-confidence photo"
    # A crop_review event must be emitted, carrying the data the UI needs.
    reviews = _of_type(events, "crop_review")
    assert reviews, "a crop_review event must be emitted"
    rv = reviews[-1]
    assert rv["name"] == LOW_IMGS[0].name
    assert rv["frame_size"] and rv["reasons"]
    # Cover verdict for photo #1 must be DEFERRED, not a real UNIQUE/DUPLICATE.
    verdicts = _of_type(events, "verdict")
    assert verdicts and verdicts[-1]["kind"] == "PENDING_REVIEW", \
        "first-photo verdict must be PENDING_REVIEW while crop is unresolved"
    # And the worker recorded the pending state (so it can keep shooting).
    assert LOW_IMGS[0].name in w._pending_reviews


def test_accept_auto_crop_resolves_review(tmp_path):
    w, events = _make_watcher(tmp_path)
    name = LOW_IMGS[0].name
    w._process_one(_drop(w, LOW_IMGS[0]))
    assert name in w._pending_reviews
    mag = w._current_magazine

    res = w.accept_auto_crop(name)

    assert res["success"], res
    assert (mag / name).exists(), "accepting must write the crop into the magazine"
    assert not (w._review_dir / name).exists(), "staged original must be removed"
    assert name not in w._pending_reviews
    assert name in w._cache, "features must be computed after resolution"
    # The deferred cover verdict now fires for real.
    kinds = [e["kind"] for e in _of_type(events, "verdict")]
    assert any(k in ("UNIQUE", "DUPLICATE", "UNCERTAIN") for k in kinds), \
        "a real cover verdict should fire once the review is resolved"


def test_apply_manual_crop_resolves_review(tmp_path):
    w, events = _make_watcher(tmp_path)
    name = LOW_IMGS[1].name
    w._process_one(_drop(w, LOW_IMGS[1]))
    mag = w._current_magazine
    fw, fh = w._pending_reviews[name]["meta"]["frame_size"]
    # User drags a rectangle (here: a generous centred crop).
    rect = {"x": int(fw * 0.1), "y": int(fh * 0.1),
            "w": int(fw * 0.8), "h": int(fh * 0.8)}

    res = w.apply_manual_crop(name, rect)

    assert res["success"], res
    assert (mag / name).exists(), "manual crop must be written into the magazine"
    assert not (w._review_dir / name).exists()
    assert name not in w._pending_reviews


def test_discard_review_reshoot(tmp_path):
    w, events = _make_watcher(tmp_path)
    name = LOW_IMGS[2].name
    w._process_one(_drop(w, LOW_IMGS[2]))
    mag = w._current_magazine

    res = w.discard_review(name)

    assert res["success"], res
    assert not (w._review_dir / name).exists(), "reshoot must delete the staged file"
    assert not (mag / name).exists(), "reshoot must not produce a crop"
    assert name not in w._pending_reviews


def test_worker_not_blocked_two_lows_accumulate(tmp_path):
    """Two LOW photos in a row must both stage and accumulate as pending — the
    worker never blocks waiting for the user to resolve the first."""
    w, events = _make_watcher(tmp_path)
    w._process_one(_drop(w, LOW_IMGS[0]))
    w._process_one(_drop(w, LOW_IMGS[1]))
    assert LOW_IMGS[0].name in w._pending_reviews
    assert LOW_IMGS[1].name in w._pending_reviews
    assert (w._review_dir / LOW_IMGS[0].name).exists()
    assert (w._review_dir / LOW_IMGS[1].name).exists()
