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
import re
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

# Force unbuffered output so progress prints visibly during long runs
# Also force UTF-8 so unicode characters in the report don't crash on Windows
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

import imagehash
from PIL import Image


# ─── Config ──────────────────────────────────────────────────────────────────

CACHE_FILE = "phash_cache.json"
REPORT_FILE = "duplicates_report.txt"
HTML_REPORT_FILE = "duplicates_report.html"
THUMB_SIZE = 300        # max longest edge of embedded thumbnails (base64 in HTML)
THUMB_QUALITY = 72      # JPEG quality for embedded thumbnails (smaller HTML)
CACHE_VERSION = 2       # bump to invalidate cache when algorithm changes

HASH_SIZE = 16          # 16 → 256-bit hash (more discriminating than default 64-bit)
MATCH_THRESHOLD = 24    # Hamming distance below this = same page
                        # (out of 256 bits; ~10% tolerance handles lighting/angle differences)
MIN_GAP = 3             # Two photos must be at least this far apart in sequence
                        # to be considered "different magazines" (not same magazine)
SAME_MAG_GAP = 2        # Two matched photos within this seq distance (same publication)
                        # are treated as the same physical magazine — but ONLY if their
                        # respective duplicates are also within this distance (parallel
                        # evidence). Gap of 2 covers typical front+back / front+colophon
                        # combinations without over-merging adjacent papers.


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
    """Recursively find all parseable image files, sorted by (publication, seq)."""
    images = []
    for f in folder.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        info = parse_filename(f.name)
        if info:
            info["path"] = str(f)
            images.append(info)
    images.sort(key=lambda x: (x["publication"].lower(), x["seq"]))
    return images


# ─── Hashing ──────────────────────────────────────────────────────────────────

def compute_phash(path: str) -> str:
    img = Image.open(path)
    # Use Pillow's built-in fast resize during decode (3-5x faster than full decode + resize)
    img.draft("L", (512, 512))
    if img.mode != "L":
        img = img.convert("L")
    # Final downscale to 256px — pHash internally goes to 4*hash_size anyway
    if max(img.size) > 256:
        scale = 256 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    return str(imagehash.phash(img, hash_size=HASH_SIZE))


def load_cache(folder: Path) -> dict:
    p = folder / CACHE_FILE
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        # Cache versioning: old caches without version are v1, ignore them
        if isinstance(data, dict) and data.get("_version") == CACHE_VERSION:
            return {k: v for k, v in data.items() if not k.startswith("_")}
        else:
            print(f"  (cache version mismatch — will recompute hashes)")
    return {}


def save_cache(folder: Path, cache: dict):
    out = {"_version": CACHE_VERSION, **cache}
    with open(folder / CACHE_FILE, "w") as f:
        json.dump(out, f, indent=2)


# ─── Duplicate detection ──────────────────────────────────────────────────────

def find_match_pairs(images: list[dict], hashes: dict, threshold: int) -> list[tuple]:
    """All pairs of photos within same publication whose hashes match."""
    pairs = []
    by_pub = defaultdict(list)
    for img in images:
        by_pub[img["publication"]].append(img)

    for pub, imgs in by_pub.items():
        imgs.sort(key=lambda x: x["seq"])
        h_objs = [imagehash.hex_to_hash(hashes[i["filename"]]) for i in imgs]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                if imgs[j]["seq"] - imgs[i]["seq"] < MIN_GAP:
                    continue
                dist = h_objs[i] - h_objs[j]
                if dist <= threshold:
                    pairs.append((imgs[i], imgs[j], dist))
    return pairs


def find_duplicate_groups(images: list[dict], hashes: dict, threshold: int) -> list[dict]:
    """
    Build a graph of matching photos, find connected components, then within
    each component identify the N distinct physical copies by clustering on
    sequence proximity.

    Each result represents a magazine issue that exists in multiple copies in
    the stack (2, 3, 4, ... copies). Photos from one physical copy will be
    consecutive in sequence; the gap between copies will be much larger.
    """
    pairs = find_match_pairs(images, hashes, threshold)

    # Build adjacency by filename
    adj = defaultdict(set)
    distance_lookup = {}
    img_lookup = {}
    for img in images:
        img_lookup[img["filename"]] = img
    for a, b, dist in pairs:
        adj[a["filename"]].add(b["filename"])
        adj[b["filename"]].add(a["filename"])
        key = tuple(sorted([a["filename"], b["filename"]]))
        distance_lookup[key] = dist

    # Add "same physical magazine" edges, but only with parallel-match evidence.
    #
    # Two matched photos a, b in same publication within SAME_MAG_GAP seq positions
    # are merged into the same physical magazine ONLY IF:
    #     a has a match X, b has a match Y, where X and Y are also within
    #     SAME_MAG_GAP seq positions in the same publication.
    #
    # This prevents over-merging: just being adjacent in the stack isn't enough,
    # we need evidence that their *duplicates* are also adjacent (i.e. they
    # appear together as a paper in another copy too).
    match_map = defaultdict(set)
    for a, b, _ in pairs:
        match_map[a["filename"]].add(b["filename"])
        match_map[b["filename"]].add(a["filename"])

    matched_imgs_by_pub = defaultdict(list)
    for fname in adj:
        img = img_lookup[fname]
        matched_imgs_by_pub[img["publication"]].append(img)
    for pub, imgs in matched_imgs_by_pub.items():
        imgs.sort(key=lambda x: x["seq"])
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                if imgs[j]["seq"] - imgs[i]["seq"] > SAME_MAG_GAP:
                    break
                # Parallel-evidence check
                i_matches = match_map[imgs[i]["filename"]]
                j_matches = match_map[imgs[j]["filename"]]
                has_parallel = False
                for im_name in i_matches:
                    im_img = img_lookup[im_name]
                    for jm_name in j_matches:
                        jm_img = img_lookup[jm_name]
                        if (im_img["publication"] == jm_img["publication"]
                                and abs(im_img["seq"] - jm_img["seq"]) <= SAME_MAG_GAP
                                and im_name != jm_name):
                            has_parallel = True
                            break
                    if has_parallel:
                        break
                if has_parallel:
                    adj[imgs[i]["filename"]].add(imgs[j]["filename"])
                    adj[imgs[j]["filename"]].add(imgs[i]["filename"])

    # Find connected components (per publication, since adjacency only crosses
    # within a publication anyway)
    seen = set()
    components = []
    for fname in adj:
        if fname in seen:
            continue
        comp = set()
        queue = [fname]
        while queue:
            v = queue.pop()
            if v in comp:
                continue
            comp.add(v)
            seen.add(v)
            queue.extend(adj[v] - comp)
        components.append(comp)

    # For each component, cluster photos into physical copies by sequence proximity
    SEQ_GAP_THRESHOLD = 8  # photos with seq gap > this are different physical copies
    groups = []
    for comp in components:
        if len(comp) < 2:
            continue
        comp_imgs = sorted([img_lookup[f] for f in comp], key=lambda x: x["seq"])
        pub = comp_imgs[0]["publication"]

        copies = [[comp_imgs[0]]]
        for img in comp_imgs[1:]:
            if img["seq"] - copies[-1][-1]["seq"] <= SEQ_GAP_THRESHOLD:
                copies[-1].append(img)
            else:
                copies.append([img])

        if len(copies) < 2:
            continue  # all matches were within one tight cluster, not a real duplicate

        # Compute summary stats on match distances within the component
        comp_distances = []
        comp_files = list(comp)
        for i in range(len(comp_files)):
            for j in range(i + 1, len(comp_files)):
                key = tuple(sorted([comp_files[i], comp_files[j]]))
                if key in distance_lookup:
                    comp_distances.append(distance_lookup[key])

        groups.append({
            "publication": pub,
            "copy_count": len(copies),
            "total_photos": len(comp_imgs),
            "copies": [
                {
                    "seq_range": [c[0]["seq"], c[-1]["seq"]],
                    "photos": [x["filename"] for x in c],
                    "photo_count": len(c),
                }
                for c in copies
            ],
            "best_distance": min(comp_distances) if comp_distances else None,
            "worst_distance": max(comp_distances) if comp_distances else None,
            "match_pair_count": len(comp_distances),
        })

    groups.sort(key=lambda g: (g["publication"].lower(), g["copies"][0]["seq_range"][0]))
    return groups


# ─── Cluster nearby duplicate matches ────────────────────────────────────────

# ─── Report ───────────────────────────────────────────────────────────────────

def _fmt_range(lo: int, hi: int) -> str:
    return f"{lo:04d}" if lo == hi else f"{lo:04d}–{hi:04d}"


def generate_report(images, groups, out_path: Path, threshold: int):
    by_pub = defaultdict(list)
    for img in images:
        by_pub[img["publication"]].append(img)

    lines = []
    lines.append("=" * 72)
    lines.append("MAGAZINE DUPLICATE REPORT  (perceptual hash, local)")
    lines.append("=" * 72)
    lines.append(f"\nTotal photos     : {len(images)}")
    lines.append(f"Publications     : {', '.join(sorted(by_pub))}")
    for pub, ims in sorted(by_pub.items()):
        seqs = sorted(i["seq"] for i in ims)
        lines.append(f"  {pub:<20} {len(ims):>4} photos  (seq {seqs[0]:04d}–{seqs[-1]:04d})")
    lines.append(f"\nMatch threshold       : {threshold} / 256 bits  (lower = stricter)")
    lines.append(f"Duplicate magazines   : {len(groups)}")
    triple_plus = sum(1 for g in groups if g["copy_count"] >= 3)
    lines.append(f"  with 3+ copies      : {triple_plus}\n")

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
            confidence = (
                "VERY HIGH" if best is not None and best <= 10 else
                "HIGH" if best is not None and best <= 25 else
                "MEDIUM" if best is not None and best <= 40 else
                "LOW"
            )
            lines.append(
                f"[{n}] {g['publication']}  —  {g['copy_count']} copies  "
                f"(confidence: {confidence}, best dist {best}/256)"
            )
            for ci, copy in enumerate(g["copies"], 1):
                pos = _fmt_range(copy["seq_range"][0], copy["seq_range"][1])
                tag = "  ← KEEP" if ci == 1 else "  ← remove from stack"
                lines.append(f"    Copy {ci}: positions {pos}  ({copy['photo_count']} photos){tag}")
                for p in copy["photos"]:
                    lines.append(f"        - {p}")
            lines.append("")

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
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


def generate_html_report(images, groups, folder: Path, threshold: int):
    """Build a self-contained HTML report (base64-embedded thumbnails)."""
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
                thumbs_b64[fname] = make_thumbnail_b64(img_lookup[fname]["path"])
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
  </div>
</header>

<div class="filter-bar">
  <span class="label">Filter</span>
  <button class="btn active" data-filter="all">All <span class="count">{len(groups)}</span></button>
  <button class="btn" data-filter="3plus">3+ copies <span class="count">{triple_plus}</span></button>
  <button class="btn" data-filter="HIGH">High conf</button>
  <button class="btn" data-filter="MEDIUM">Medium conf</button>
  <button class="btn" data-filter="LOW">Low conf</button>
  <button class="btn" data-filter="uncertain">Uncertain (Med + Low)</button>
  <button class="btn" data-filter="unverified">Unverified only</button>
  <span class="progress">Verified: <strong id="progress-num">0</strong> / {len(groups)}</span>
</div>

<main id="groups">""")

    for n, g in enumerate(groups, 1):
        best = g["best_distance"]
        worst = g["worst_distance"]
        confidence = (
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
            f'data-copies="{g["copy_count"]}" data-confidence="{confidence}">'
        )
        html.append('<div class="group-head">')
        html.append(f'<span class="num">#{n:02d}</span>')
        html.append(f'<span class="pub">{g["publication"]}</span>')
        html.append(f'<span class="copies-pill{copies_pill_cls}">{g["copy_count"]} copies</span>')
        html.append(f'<span class="conf conf-{confidence.replace(" ", "-")}">{confidence}</span>')
        html.append(f'<span class="dist">best {best}/256 · worst {worst}/256 · {g["match_pair_count"]} pages</span>')
        html.append('<div class="actions">')
        html.append(f'<button class="action-btn" data-action="confirm" data-id="{n}">✓ Confirmed</button>')
        html.append(f'<button class="action-btn" data-action="dismiss" data-id="{n}">✕ Not a dup</button>')
        html.append('</div>')
        html.append('</div>')

        html.append('<div class="copies">')
        for ci, copy in enumerate(g["copies"], 1):
            cls = "keep" if ci == 1 else "remove"
            label = "KEEP" if ci == 1 else "REMOVE"
            pos = _fmt_range(copy["seq_range"][0], copy["seq_range"][1])
            html.append(f'<div class="copy {cls}">')
            html.append('<div class="copy-head">')
            html.append(f'<div><div class="copy-label">{label}</div>'
                        f'<div class="copy-meta">Copy {ci} · {copy["photo_count"]} photo{"s" if copy["photo_count"] > 1 else ""}</div></div>')
            html.append(f'<div class="copy-pos">pos {pos}</div>')
            html.append('</div>')

            html.append('<div class="photos">')
            for fname in copy["photos"]:
                short = fname.split("_")[-1].replace(".JPG", "").replace(".jpg", "")
                b64 = thumbs_b64.get(fname, "")
                src = f"data:image/jpeg;base64,{b64}" if b64 else ""
                html.append(
                    f'<div class="photo" data-full="{src}" data-name="{fname}">'
                    f'<img src="{src}" alt="{fname}" loading="lazy">'
                    f'<div class="photo-cap">{short}</div>'
                    f'</div>'
                )
            html.append('</div></div>')
        html.append('</div></div>')

    html.append("""</main>

<div class="empty hidden" id="empty">No duplicates match this filter.</div>

<div class="lightbox" id="lightbox">
  <img id="lightbox-img" alt="">
  <div class="lightbox-cap" id="lightbox-cap"></div>
</div>

<script>
// ── Filter logic ──
const filterBtns = document.querySelectorAll('.filter-bar .btn');
const groups = document.querySelectorAll('.group');
const emptyState = document.getElementById('empty');

function applyFilter(filter) {
  filterBtns.forEach(b => b.classList.toggle('active', b.dataset.filter === filter));
  let visible = 0;
  groups.forEach(g => {
    let show = false;
    const conf = g.dataset.confidence;
    const copies = parseInt(g.dataset.copies);
    const verified = g.classList.contains('verified') || g.classList.contains('dismissed');
    if (filter === 'all') show = true;
    else if (filter === '3plus') show = copies >= 3;
    else if (filter === 'HIGH') show = (conf === 'HIGH' || conf === 'VERY HIGH');
    else if (filter === 'MEDIUM') show = conf === 'MEDIUM';
    else if (filter === 'LOW') show = conf === 'LOW';
    else if (filter === 'uncertain') show = (conf === 'MEDIUM' || conf === 'LOW');
    else if (filter === 'unverified') show = !verified;
    g.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  emptyState.classList.toggle('hidden', visible > 0);
}

filterBtns.forEach(b => b.addEventListener('click', () => applyFilter(b.dataset.filter)));

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
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') lightbox.classList.remove('open');
});

// ── Keyboard shortcuts ──
document.addEventListener('keydown', e => {
  if (lightbox.classList.contains('open')) return;
  if (e.target.tagName === 'INPUT') return;
  const keys = { '1': 'all', '2': '3plus', '3': 'HIGH', '4': 'MEDIUM', '5': 'LOW', '6': 'uncertain', '7': 'unverified' };
  if (keys[e.key]) applyFilter(keys[e.key]);
});
</script>
</body></html>""")

    out = folder / HTML_REPORT_FILE
    out.write_text("\n".join(html), encoding="utf-8")
    print(f"HTML report saved to: {out}")
    print(f"  Open it in your browser to visually verify each duplicate.")


# ─── Library entry point (for desktop app) ───────────────────────────────────

def analyze_folder(folder_path, threshold: int = MATCH_THRESHOLD,
                   rehash: bool = False, progress_cb=None):
    """
    Run the full pipeline on a folder. Returns (images, groups, stats) or
    (None, [], {"error": ...}) if the folder has no magazine images.

    progress_cb(phase: str, n: int, total: int) is called periodically:
      - phase="scan", n=files_found, total=files_found
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

    cache = {} if rehash else load_cache(folder)
    todo = [img for img in images if img["filename"] not in cache]

    if todo:
        if progress_cb: progress_cb("hash", 0, len(todo))
        for n, img in enumerate(todo, 1):
            try:
                cache[img["filename"]] = compute_phash(img["path"])
            except Exception:
                continue
            if n % 25 == 0 or n == len(todo):
                save_cache(folder, cache)
                if progress_cb: progress_cb("hash", n, len(todo))
        save_cache(folder, cache)

    if progress_cb: progress_cb("match", 0, 0)
    groups = find_duplicate_groups(images, cache, threshold)

    if progress_cb: progress_cb("render", 0, 0)

    triple_plus = sum(1 for g in groups if g["copy_count"] >= 3)
    stats = {
        "total_images": len(images),
        "total_groups": len(groups),
        "triple_plus": triple_plus,
        "publications": sorted({i["publication"] for i in images}),
        "threshold": threshold,
        "folder": str(folder),
    }

    # Always write the standalone report files so they can be opened separately
    generate_report(images, groups, folder / REPORT_FILE, threshold)
    generate_html_report(images, groups, folder, threshold)

    return images, groups, stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Find duplicate magazine issues using perceptual hashing")
    parser.add_argument("folder", help="Folder containing magazine JPGs")
    parser.add_argument("--rehash", action="store_true", help="Recompute all hashes (ignore cache)")
    parser.add_argument("--threshold", type=int, default=MATCH_THRESHOLD,
                        help=f"Match threshold (lower = stricter, default {MATCH_THRESHOLD})")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        return

    threshold = args.threshold

    print(f"Scanning {folder} ...")
    images = scan_images(folder)
    if not images:
        print("No magazine images found (expected names like 2026_05_08_NME_0611.JPG)")
        return

    pubs = sorted(set(i["publication"] for i in images))
    print(f"Found {len(images)} images across publications: {', '.join(pubs)}")

    cache = {} if args.rehash else load_cache(folder)
    todo = [img for img in images if img["filename"] not in cache]
    print(f"{len(cache)} hashes cached, {len(todo)} new to compute.")

    if todo:
        start = time.time()
        for n, img in enumerate(todo, 1):
            try:
                cache[img["filename"]] = compute_phash(img["path"])
            except Exception as e:
                print(f"  ERROR hashing {img['filename']}: {e}")
                continue
            if n % 25 == 0 or n == len(todo):
                elapsed = time.time() - start
                rate = n / elapsed if elapsed > 0 else 0
                eta = (len(todo) - n) / rate if rate > 0 else 0
                print(f"  [{n}/{len(todo)}]  {rate:.1f} img/s  ETA {eta:.0f}s")
                save_cache(folder, cache)
        save_cache(folder, cache)
        print(f"Hashing done in {time.time() - start:.1f}s.\n")

    print(f"Searching for duplicate magazine groups (threshold={threshold}/256) ...")
    groups = find_duplicate_groups(images, cache, threshold)
    n_pairs = sum(g["match_pair_count"] for g in groups)
    triple_plus = sum(1 for g in groups if g["copy_count"] >= 3)
    print(f"Found {len(groups)} duplicate magazine(s) "
          f"({triple_plus} with 3+ copies, {n_pairs} underlying page-matches).\n")

    generate_report(images, groups, folder / REPORT_FILE, threshold)
    generate_html_report(images, groups, folder, threshold)
    print(f"\nFull report saved to: {folder / REPORT_FILE}")
    print(f"Hash cache saved to:  {folder / CACHE_FILE}")


if __name__ == "__main__":
    main()
