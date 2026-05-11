#!/usr/bin/env python3
"""One-shot migration: flat folder of magazine photos → per-magazine subfolders.

Before: `images/NME/2026_05_07_NME_0001.JPG` …670 photos flat in one folder.
After:  `images/NME/2026_05_07_NME_0001/2026_05_07_NME_0001.JPG` …grouped
        into one subfolder per detected magazine (named `<date>_<PUB>_<NNNN>`).

This matches the layout the live tether produces, so the live verdict algorithm
can compare new shoots against the existing library.

Usage:
    python migrate.py <folder> [<folder> ...]
    python migrate.py --dry-run <folder>     # show planned moves, don't touch anything

Algorithm:
    1. Reuse find_duplicates_local's item-detection pipeline:
       spread-detection → OCR-date refinement → list of items.
    2. For each item, name a folder using the photographer's date prefix
       (parsed from filenames) + publication + a sequence number.
    3. Move the CROPPED versions of the item's photos into the new folder.
    4. Move the originals to `<folder>/_originals_legacy/` (preserved, not
       deleted — user can purge later).
    5. Leave the existing phash_cache.json intact: cache keys are filenames
       (not paths), so the entries remain valid after the physical move.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import find_duplicates_local as fdl


ORIGINALS_BACKUP = "_originals_legacy"


def plan_folder(folder: Path) -> dict:
    """Build the rename/move plan for one source folder.

    Returns:
        {
          "folder": <Path>,
          "publication": <str>,         # e.g. "NME"
          "items": [
              {
                  "name": "2026_05_07_NME_0001",
                  "photo_filenames": [...],
                  "ended_with_spread": bool,
                  "consensus_date": (Y,M,D) or None,
              },
              ...
          ],
          "issues": [str, ...],         # e.g. "no photos detected"
        }
    """
    issues = []
    images = fdl.scan_images(folder)
    if not images:
        return {"folder": folder, "publication": "?", "items": [],
                "issues": ["no parseable images found"]}

    cache = fdl.load_cache(folder)
    if not cache:
        issues.append("no feature cache present — boundary detection may be weaker. "
                      "Run a full analysis first, then re-run migrate.")
        return {"folder": folder, "publication": images[0]["publication"],
                "items": [], "issues": issues}

    # Reuse the same item-detection used by the duplicate finder, so the
    # boundary decisions are consistent with the analysis pipeline.
    items_internal = fdl._build_items_by_spread(images, cache)
    items_internal, refine_msgs = fdl._refine_items_by_dates(items_internal, images, cache)

    publications = sorted({img["publication"] for img in images})

    # Group items by publication so each pub gets its own sequence numbers.
    by_pub: dict[str, list[dict]] = {}
    for it in items_internal:
        by_pub.setdefault(it["publication"], []).append(it)

    planned = []
    for pub, group in by_pub.items():
        group.sort(key=lambda it: it["front_seq"])
        for seq_idx, it in enumerate(group, 1):
            # Determine the "date prefix" of the folder name. Use the front
            # photo's date_taken (photographer's date stamp, parsed from the
            # filename), since that's what the live tether uses too.
            front_filename = it["photos"][0]
            front_img = next((im for im in images if im["filename"] == front_filename), None)
            date_prefix = front_img["date_taken"] if front_img else "0000_00_00"

            # Try to pick a consensus OCR date for diagnostics.
            consensus_date = None
            date_counts: Counter = Counter()
            for fname in it["photos"]:
                e = cache.get(fname)
                if isinstance(e, dict):
                    for d in (e.get("dates") or []):
                        date_counts[tuple(d)] += 1
            if date_counts:
                consensus_date = date_counts.most_common(1)[0][0]

            folder_name = f"{date_prefix}_{pub}_{seq_idx:04d}"
            planned.append({
                "name": folder_name,
                "photo_filenames": list(it["photos"]),
                "ended_with_spread": bool(it.get("ended_with_spread")),
                "consensus_date": consensus_date,
            })

    return {"folder": folder, "publication": ", ".join(publications),
            "items": planned, "issues": issues + refine_msgs[:10]}


def execute_plan(plan: dict, *, dry_run: bool = False) -> dict:
    """Execute (or print) the rename plan. Returns {"moved_photos": N,
    "moved_originals": N, "created_folders": N, "errors": [...]}.

    Source for the cropped files: `<folder>/cropped/<filename>.JPG`
    Source for the originals:     `<folder>/<filename>.JPG`
    """
    folder: Path = plan["folder"]
    cropped_src = folder / fdl.CROPPED_DIR
    backup_dir = folder / ORIGINALS_BACKUP

    stats = {"moved_photos": 0, "moved_originals": 0,
             "created_folders": 0, "errors": []}

    if not plan["items"]:
        stats["errors"].append("no items to migrate")
        return stats

    if not dry_run:
        backup_dir.mkdir(exist_ok=True)

    for item in plan["items"]:
        target = folder / item["name"]
        if dry_run:
            print(f"  → {item['name']}/  ({len(item['photo_filenames'])} photos)")
            for fname in item["photo_filenames"]:
                print(f"        {fname}")
            continue

        target.mkdir(exist_ok=True)
        stats["created_folders"] += 1

        for fname in item["photo_filenames"]:
            cropped = cropped_src / fname
            original = folder / fname
            # Move cropped version into the magazine folder.
            if cropped.exists():
                dst = target / fname
                try:
                    shutil.move(str(cropped), str(dst))
                    stats["moved_photos"] += 1
                except Exception as e:
                    stats["errors"].append(f"move cropped {fname}: {e}")
            else:
                stats["errors"].append(f"missing cropped: {fname}")
            # Move original to backup (preserve for safety; user can purge later).
            if original.exists():
                try:
                    shutil.move(str(original), str(backup_dir / fname))
                    stats["moved_originals"] += 1
                except Exception as e:
                    stats["errors"].append(f"move original {fname}: {e}")

    # If the cropped/ folder is now empty, remove it for tidiness.
    if not dry_run and cropped_src.exists() and not any(cropped_src.iterdir()):
        try:
            cropped_src.rmdir()
        except OSError:
            pass

    return stats


def migrate(folder_path: str, *, dry_run: bool = False) -> dict:
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        return {"error": f"not a directory: {folder}"}

    plan = plan_folder(folder)
    if not plan["items"]:
        return {"error": "; ".join(plan["issues"]) or "nothing to migrate"}

    print(f"\n=== {folder.name} ({plan['publication']}) — {len(plan['items'])} items planned ===")
    if plan["issues"]:
        for msg in plan["issues"][:5]:
            print(f"  ! {msg}")
        if len(plan["issues"]) > 5:
            print(f"  ! ...and {len(plan['issues']) - 5} more notes")

    if dry_run:
        print("\n(dry run — no files will be moved)")
    stats = execute_plan(plan, dry_run=dry_run)
    print(f"\n  created_folders : {stats['created_folders']}")
    print(f"  moved_cropped   : {stats['moved_photos']}")
    print(f"  moved_originals : {stats['moved_originals']}")
    if stats["errors"]:
        print(f"  errors          : {len(stats['errors'])}")
        for e in stats["errors"][:5]:
            print(f"      {e}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate flat magazine folder(s) to per-magazine subfolders.")
    parser.add_argument("folders", nargs="+", help="Folder(s) to migrate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan; don't move any files.")
    args = parser.parse_args()
    for f in args.folders:
        migrate(f, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
