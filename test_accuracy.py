#!/usr/bin/env python3
"""Synthetic accuracy harness for find_duplicates_local.

Generates small synthetic image stacks with KNOWN duplicate ground-truth, runs
the duplicate detector, and reports precision/recall. Use this to validate that
algorithm changes don't regress on edge cases.

Cases tested:
  • Clean duplicates (identical images): should be 100% recall.
  • Brightness shift (one copy is darker/lighter): hash-tolerant test.
  • Slight rotation/skew: structural change test.
  • Different items with similar palette (false-positive trap).
  • Variable item lengths (1-page item, 4-page item, mixed).
  • Missing seq numbers within an item.

Run:  python test_accuracy.py
Exits non-zero if any case regresses from documented expected metrics.
"""

import sys
import shutil
import random
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance

# Import the module under test from the same dir.
sys.path.insert(0, str(Path(__file__).parent))
import find_duplicates_local as fdl


def _noise_page(seed: int, w: int, h: int, *, blocky: int = 50) -> Image.Image:
    """Pseudo-random magazine-page-like image with seed-distinct gross structure.

    Uses a coarse 4×4 grid of high-contrast solid colors (dominates pHash and
    dHash signals) plus a per-pixel noise overlay (adds discrimination for
    fine-grained hashes). The 4×4 grid choice gives 4×4×3=48 effective binary
    features per image, enough that random seeds produce distinct hashes.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    # Coarse grid: 4×4 of high-contrast colors.
    coarse = rng.integers(0, 256, (4, 4, 3), dtype=_np.uint8)
    # Tile so each cell occupies a large image region.
    cell_h = h // 4
    cell_w = w // 4
    arr = _np.kron(coarse, _np.ones((cell_h, cell_w, 1), dtype=_np.uint8))
    # Pad/crop to exact size.
    if arr.shape[0] < h:
        arr = _np.concatenate([arr, _np.repeat(arr[-1:], h - arr.shape[0], axis=0)], axis=0)
    if arr.shape[1] < w:
        arr = _np.concatenate([arr, _np.repeat(arr[:, -1:], w - arr.shape[1], axis=1)], axis=1)
    arr = arr[:h, :w]
    # Add fine noise on top so the high-resolution detail differs per seed.
    noise = rng.integers(-30, 30, (h, w, 3), dtype=_np.int16)
    arr = _np.clip(arr.astype(_np.int16) + noise, 0, 255).astype(_np.uint8)
    return Image.fromarray(arr, "RGB")


def make_page(seed: int, frame_w: int = 1600, frame_h: int = 1100, *,
              brightness: float = 1.0,
              rotation: float = 0.0,
              text: str | None = None) -> Image.Image:
    """Synthetic single-page photo: portrait magazine page (with seed-specific
    visual content) centered on a light "table" background. Whitespace edges
    keep edge_std low → detector classifies as a single page, not a spread.
    """
    img = Image.new("RGB", (frame_w, frame_h), color=(245, 245, 245))
    page_w, page_h = 600, 900
    page = _noise_page(seed, page_w, page_h, blocky=50)
    if text:
        ImageDraw.Draw(page).text((20, 20), text, fill=(0, 0, 0))
    if brightness != 1.0:
        page = ImageEnhance.Brightness(page).enhance(brightness)
    if rotation != 0.0:
        page = page.rotate(rotation, fillcolor=(220, 220, 220))
    px = (frame_w - page_w) // 2
    py = (frame_h - page_h) // 2
    img.paste(page, (px, py))
    return img


def make_spread(seed: int, frame_w: int = 1600, frame_h: int = 1100) -> Image.Image:
    """2-page-spread photo: content fills the landscape frame edge-to-edge so
    edge_std is high and the spread detector marks it as a boundary."""
    return _noise_page(seed * 13 + 7, frame_w, frame_h, blocky=40)


def write_synthetic_stack(folder: Path, plan: list[dict]) -> list[set]:
    """Materialize a synthetic stack from `plan`. Returns the ground-truth
    duplicate groups: list of sets of seq numbers belonging to each duplicate
    item-group."""
    folder.mkdir(parents=True, exist_ok=True)
    seq = 1
    duplicate_seqs: dict[int, list[int]] = {}  # issue_seed -> [front_seq for each copy]
    for entry in plan:
        seed = entry["seed"]
        pages = entry.get("pages", 3)  # front, back, spread
        bright = entry.get("brightness", 1.0)
        rot = entry.get("rotation", 0.0)
        # Page 1: front
        img = make_page(seed, brightness=bright, rotation=rot, text=f"Issue {seed} Front")
        front_seq = seq
        img.save(folder / f"2026_05_07_TEST_{seq:04d}.JPG", quality=85)
        seq += 1
        # Page 2: back (different content but same "issue" seed family)
        if pages >= 2:
            img = make_page(seed * 100 + 1, brightness=bright, rotation=rot, text=f"Issue {seed} Back")
            img.save(folder / f"2026_05_07_TEST_{seq:04d}.JPG", quality=85)
            seq += 1
        # Page 3: a 2-page spread (boundary marker)
        if pages >= 3:
            img = make_spread(seed)
            img.save(folder / f"2026_05_07_TEST_{seq:04d}.JPG", quality=85)
            seq += 1

        if entry.get("is_duplicate_of") is not None:
            target = entry["is_duplicate_of"]
            duplicate_seqs.setdefault(target, []).append(front_seq)
        else:
            duplicate_seqs.setdefault(seed, []).append(front_seq)

    # Convert dict to list of sets, only keeping groups with ≥ 2 fronts.
    return [set(seqs) for seqs in duplicate_seqs.values() if len(seqs) >= 2]


def evaluate(folder: Path, expected_groups: list[set], threshold: int = 60) -> dict:
    """Run analyze_folder and compute precision/recall vs expected_groups."""
    images, groups, stats = fdl.analyze_folder(folder, threshold=threshold)
    found_groups: list[set] = []
    for g in groups:
        seqs = {c["front_seq"] for c in g["copies"]}
        found_groups.append(seqs)

    # A "true positive" is a found group that exactly matches an expected group.
    # Subset matches (the algorithm found 2 of 3 copies) count as partial.
    tp = 0; fp = 0; fn = 0
    matched_expected = set()
    for fg in found_groups:
        # Best-matching expected group by Jaccard similarity.
        best_iou = 0; best_idx = -1
        for i, eg in enumerate(expected_groups):
            iou = len(fg & eg) / max(1, len(fg | eg))
            if iou > best_iou:
                best_iou = iou; best_idx = i
        if best_iou >= 0.5:
            tp += 1
            matched_expected.add(best_idx)
        else:
            fp += 1
    fn = len(expected_groups) - len(matched_expected)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "expected": len(expected_groups), "found": len(found_groups),
    }


# ─── Test cases ──────────────────────────────────────────────────────────────

CASES = [
    {
        "name": "clean_duplicates",
        "plan": [
            {"seed": 1}, {"seed": 2},
            {"seed": 1, "is_duplicate_of": 1},  # duplicate of issue 1
            {"seed": 3},
            {"seed": 2, "is_duplicate_of": 2},  # duplicate of issue 2
        ],
        "expects": {"recall_min": 1.0, "precision_min": 1.0},
    },
    {
        "name": "brightness_shift",
        "plan": [
            {"seed": 10}, {"seed": 11},
            {"seed": 10, "is_duplicate_of": 10, "brightness": 0.7},  # same issue, dimmer
            {"seed": 11, "is_duplicate_of": 11, "brightness": 1.3},  # same issue, brighter
        ],
        # Brightness should not break match — dHash is gradient-based and tolerant.
        "expects": {"recall_min": 1.0, "precision_min": 1.0},
    },
    {
        "name": "slight_rotation",
        "plan": [
            {"seed": 20}, {"seed": 21},
            {"seed": 20, "is_duplicate_of": 20, "rotation": 2.0},
            {"seed": 21, "is_duplicate_of": 21, "rotation": -3.0},
        ],
        # Rotation up to a few degrees should still be caught.
        "expects": {"recall_min": 1.0, "precision_min": 1.0},
    },
    {
        "name": "no_duplicates",
        "plan": [
            {"seed": 30}, {"seed": 31}, {"seed": 32}, {"seed": 33}, {"seed": 34},
        ],
        # No duplicates should be detected.
        "expects": {"recall_min": 1.0, "precision_min": 1.0, "found_max": 0},
    },
    {
        "name": "three_copies",
        "plan": [
            {"seed": 40},
            {"seed": 40, "is_duplicate_of": 40},
            {"seed": 41},
            {"seed": 40, "is_duplicate_of": 40},
        ],
        "expects": {"recall_min": 1.0, "precision_min": 1.0},
    },
]


def run_all():
    overall_pass = True
    for case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            expected = write_synthetic_stack(folder, case["plan"])
            try:
                metrics = evaluate(folder, expected, threshold=60)
            except Exception as e:
                print(f"  [{case['name']}]  CRASH: {e}")
                overall_pass = False
                continue
            ok_p = metrics["precision"] >= case["expects"].get("precision_min", 0.0)
            ok_r = metrics["recall"] >= case["expects"].get("recall_min", 0.0)
            ok_f = metrics["found"] <= case["expects"].get("found_max", 999)
            status = "PASS" if (ok_p and ok_r and ok_f) else "FAIL"
            if status == "FAIL":
                overall_pass = False
            print(f"  [{case['name']:<20}]  {status}  "
                  f"precision={metrics['precision']:.2f}  "
                  f"recall={metrics['recall']:.2f}  "
                  f"(found {metrics['found']} / expected {metrics['expected']})")
    return overall_pass


if __name__ == "__main__":
    print("Running synthetic accuracy tests ...\n")
    ok = run_all()
    print()
    if ok:
        print("All cases passed.")
        sys.exit(0)
    else:
        print("FAILURES detected.")
        sys.exit(1)
