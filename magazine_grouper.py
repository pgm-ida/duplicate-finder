#!/usr/bin/env python3
"""
Magazine photo grouper and duplicate detector.

Groups magazine photos into issues using Claude AI vision, then finds
duplicate copies in your stack and tells you where to find them.

Usage:
    python magazine_grouper.py <images_folder> [--api-key KEY]

File naming expected: YYYY_MM_DD_PublicationName_NNNN.JPG
    e.g. 2026_05_08_NME_0611.JPG
         2026_05_08_Recordmirror_0050.JPG
"""

import os
import sys
import json
import base64
import re
import argparse
from pathlib import Path
from collections import defaultdict
import io

import anthropic
from PIL import Image


# ─── Config ──────────────────────────────────────────────────────────────────

CACHE_FILE = "analysis_cache.json"
RESULTS_FILE = "grouping_results.json"
REPORT_FILE = "duplicates_report.txt"
MODEL = "claude-haiku-4-5-20251001"   # cheapest; change to claude-sonnet-4-6 if dates are missed
MAX_IMAGE_PX = 1024                   # longest edge when sending to API
IMAGE_QUALITY = 80                    # JPEG quality for API submission


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
    """Return all parseable image files, sorted by (publication, seq).
    Recursively walks subfolders, so images can be flat or in collection
    subfolders like images/NME/, images/Recordmirror/, etc."""
    images = []
    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        info = parse_filename(f.name)
        if info:
            info["path"] = str(f)
            images.append(info)
    images.sort(key=lambda x: (x["publication"].lower(), x["seq"]))
    return images


# ─── Image encoding ───────────────────────────────────────────────────────────

def encode_image(path: str) -> tuple[str, str]:
    """Resize and base64-encode an image for the Claude API."""
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Downscale to MAX_IMAGE_PX on longest side
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=IMAGE_QUALITY)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), "image/jpeg"


# ─── Claude analysis ─────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """\
This is a photo of a page from a physical music magazine (like NME, Record Mirror, Sounds, Melody Maker, etc.).

Extract the following and return ONLY a JSON object — no markdown, no explanation:

{
  "publication": "<exact publication name as printed, e.g. NME, Record Mirror>",
  "issue_date": "<date as printed on the magazine, e.g. '25 February 1984', '5 April 1980'>",
  "page_type": "<one of: front_cover, back_cover, index, colophon, interior_pages, unknown>",
  "is_front_cover": <true or false>,
  "confidence": "<high, medium, or low — how sure you are about the date>"
}

If you cannot read the date, set issue_date to "unknown" and confidence to "low".
Focus on the masthead/title area and the date line near the top of the page."""


def analyze_image(client: anthropic.Anthropic, path: str, model: str = MODEL) -> dict:
    img_b64, media_type = encode_image(path)
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": ANALYSIS_PROMPT},
            ],
        }],
    )
    raw = response.content[0].text.strip()
    # Strip code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "publication": "unknown",
            "issue_date": "unknown",
            "page_type": "unknown",
            "is_front_cover": False,
            "confidence": "low",
            "_raw": raw,
        }


# ─── Caching ──────────────────────────────────────────────────────────────────

def load_cache(folder: Path) -> dict:
    cache_path = folder / CACHE_FILE
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(folder: Path, cache: dict):
    with open(folder / CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ─── Grouping logic ───────────────────────────────────────────────────────────

def normalize_date(date_str: str) -> str:
    """Light normalization so '25 Feb 1984' == '25 February 1984'."""
    if not date_str or date_str.lower() == "unknown":
        return "unknown"
    month_map = {
        "jan": "january", "feb": "february", "mar": "march", "apr": "april",
        "may": "may", "jun": "june", "jul": "july", "aug": "august",
        "sep": "september", "oct": "october", "nov": "november", "dec": "december",
    }
    s = date_str.lower().strip()
    for short, full in month_map.items():
        s = s.replace(short + " ", full + " ").replace(short + ".", full)
    return s


def group_into_issues(images: list[dict], analyses: dict) -> list[dict]:
    """
    Group images into magazine issues.

    Strategy:
    - Within each publication, images are sorted by sequence number.
    - A new issue begins when we encounter a front cover OR when the
      detected issue_date changes from the previous image.
    - Fallback: if no front covers are detected at all, treat every
      run of images with the same date as one issue.
    """
    groups = []
    current_group: list[dict] = []
    current_date = None

    for img in images:
        key = img["filename"]
        analysis = analyses.get(key, {})
        date = normalize_date(analysis.get("issue_date", "unknown"))
        is_cover = analysis.get("is_front_cover", False)
        pub = analysis.get("publication") or img["publication"]

        if not current_group:
            # Start first group
            current_group = [img]
            current_date = date
        elif is_cover or (date != "unknown" and date != current_date):
            # Flush current group, start new one
            groups.append(_make_group(current_group, analyses))
            current_group = [img]
            current_date = date
        else:
            current_group.append(img)

    if current_group:
        groups.append(_make_group(current_group, analyses))

    return groups


def _make_group(images: list[dict], analyses: dict) -> dict:
    seqs = [img["seq"] for img in images]
    pub = images[0]["publication"]

    # Best date = from the front cover if present, else first image with a known date
    date = "unknown"
    for img in images:
        a = analyses.get(img["filename"], {})
        if a.get("is_front_cover") and normalize_date(a.get("issue_date", "")) != "unknown":
            date = normalize_date(a["issue_date"])
            break
    if date == "unknown":
        for img in images:
            a = analyses.get(img["filename"], {})
            d = normalize_date(a.get("issue_date", ""))
            if d != "unknown":
                date = d
                break

    front_cover_file = next(
        (img["filename"] for img in images if analyses.get(img["filename"], {}).get("is_front_cover")),
        images[0]["filename"],
    )

    return {
        "publication": pub,
        "issue_date": date,
        "seq_range": [min(seqs), max(seqs)],
        "photo_count": len(images),
        "photos": [img["filename"] for img in images],
        "front_cover_photo": front_cover_file,
    }


# ─── Duplicate detection ──────────────────────────────────────────────────────

def find_duplicates(groups: list[dict]) -> list[list[dict]]:
    """Return lists of groups that share the same publication + issue_date."""
    key_to_groups = defaultdict(list)
    for g in groups:
        if g["issue_date"] == "unknown":
            continue
        key = (g["publication"].lower(), g["issue_date"])
        key_to_groups[key].append(g)
    return [gs for gs in key_to_groups.values() if len(gs) > 1]


# ─── Report ───────────────────────────────────────────────────────────────────

def generate_report(groups: list[dict], duplicates: list[list[dict]], out_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("MAGAZINE DUPLICATE REPORT")
    lines.append("=" * 70)
    lines.append(f"\nTotal magazine issues detected: {len(groups)}")
    lines.append(f"Duplicate sets found: {len(duplicates)}\n")

    if not duplicates:
        lines.append("No duplicates found!")
    else:
        lines.append("─" * 70)
        lines.append("DUPLICATES TO RESOLVE")
        lines.append("─" * 70)
        for i, dup_set in enumerate(duplicates, 1):
            pub = dup_set[0]["publication"]
            date = dup_set[0]["issue_date"].title()
            lines.append(f"\n[{i}] {pub} — {date}  ({len(dup_set)} copies)")
            for copy in sorted(dup_set, key=lambda g: g["seq_range"][0]):
                seq_start, seq_end = copy["seq_range"]
                photos = copy["photo_count"]
                cover = copy["front_cover_photo"]
                if seq_start == seq_end:
                    pos_str = f"position {seq_start:04d}"
                else:
                    pos_str = f"positions {seq_start:04d}–{seq_end:04d}"
                lines.append(f"    Copy at {pos_str}  ({photos} photo{'s' if photos > 1 else ''})  [cover: {cover}]")

            # Recommendation: keep the one with most photos (best documented)
            best = max(dup_set, key=lambda g: g["photo_count"])
            remove = [g for g in dup_set if g is not best]
            remove_positions = ", ".join(
                f"{g['seq_range'][0]:04d}–{g['seq_range'][1]:04d}" if g["seq_range"][0] != g["seq_range"][1]
                else f"{g['seq_range'][0]:04d}"
                for g in remove
            )
            lines.append(f"    → KEEP position {best['seq_range'][0]:04d} ({best['photo_count']} photos)")
            lines.append(f"    → REMOVE position(s): {remove_positions}")

    lines.append("\n" + "─" * 70)
    lines.append("ALL DETECTED ISSUES (sorted by publication + sequence)")
    lines.append("─" * 70)
    for g in groups:
        seq_start, seq_end = g["seq_range"]
        pos = f"{seq_start:04d}–{seq_end:04d}" if seq_start != seq_end else f"{seq_start:04d}"
        date_str = g["issue_date"].title() if g["issue_date"] != "unknown" else "(date unknown)"
        dup_flag = ""
        for dup_set in duplicates:
            if g in dup_set:
                dup_flag = " *** DUPLICATE ***"
                break
        lines.append(f"  {g['publication']:<20} {date_str:<30} pos {pos}  {g['photo_count']}px{dup_flag}")

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    print(report)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Group magazine photos and find duplicates")
    parser.add_argument("folder", help="Folder containing magazine JPGs")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default=MODEL, help=f"Claude model to use (default: {MODEL})")
    parser.add_argument("--reanalyze", action="store_true", help="Ignore cache and re-analyze all images")
    parser.add_argument("--estimate", action="store_true", help="Show cost estimate and exit (no API calls)")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    if args.estimate:
        images = scan_images(folder)
        cache = load_cache(folder)
        todo = len([i for i in images if i["filename"] not in cache])
        # Haiku vision: ~1500 input tokens per image (image + prompt), $0.80/M
        est_cost = todo * 1500 / 1_000_000 * 0.80
        print(f"Images found      : {len(images)}")
        print(f"Already cached    : {len(images) - todo}")
        print(f"Need to analyze   : {todo}")
        print(f"Estimated cost    : ${est_cost:.2f}  (using {MODEL}, ~$0.80/M tokens)")
        print(f"  (Switch to claude-sonnet-4-6 for higher accuracy, ~4x cost)")
        return

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: provide --api-key or set ANTHROPIC_API_KEY environment variable")
        print("  Option 1: python magazine_grouper.py images --api-key sk-ant-...")
        print("  Option 2: $env:ANTHROPIC_API_KEY='sk-ant-...' then run script")
        sys.exit(1)

    model = args.model
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Scanning {folder} ...")
    images = scan_images(folder)
    if not images:
        print("No magazine images found (expected names like 2026_05_08_NME_0611.JPG)")
        sys.exit(1)

    print(f"Found {len(images)} images across publications: "
          f"{', '.join(sorted(set(i['publication'] for i in images)))}")

    # Load cache
    cache = {} if args.reanalyze else load_cache(folder)
    todo = [img for img in images if img["filename"] not in cache]
    print(f"{len(cache)} already analyzed, {len(todo)} to analyze now.")

    # Analyze with Claude
    for n, img in enumerate(todo, 1):
        print(f"  [{n}/{len(todo)}] {img['filename']} ...", end=" ", flush=True)
        try:
            result = analyze_image(client, img["path"], model)
            cache[img["filename"]] = result
            date = result.get("issue_date", "?")
            ptype = result.get("page_type", "?")
            print(f"{ptype} | {date} | confidence={result.get('confidence','?')}")
        except Exception as e:
            print(f"ERROR: {e}")
            cache[img["filename"]] = {"error": str(e), "issue_date": "unknown", "is_front_cover": False}
        # Save after every image so we can resume
        save_cache(folder, cache)

    # Group and detect duplicates
    print("\nGrouping into issues ...")
    groups = group_into_issues(images, cache)
    print(f"Detected {len(groups)} magazine issues.")

    duplicates = find_duplicates(groups)
    print(f"Found {len(duplicates)} duplicate sets.\n")

    # Save structured results
    results = {"groups": groups, "duplicates": [d for d in duplicates]}
    with open(folder / RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Write and print report
    generate_report(groups, duplicates, folder / REPORT_FILE)
    print(f"\nFull report saved to: {folder / REPORT_FILE}")
    print(f"Structured data saved to: {folder / RESULTS_FILE}")
    print(f"Analysis cache saved to: {folder / CACHE_FILE}")


if __name__ == "__main__":
    main()
