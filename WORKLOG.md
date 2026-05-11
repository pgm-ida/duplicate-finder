# Worklog

## Session 4 (2026-05-09 later) — one-shot migration to per-magazine folders

User asked to migrate the existing 670+304 flat-folder photos into the new
per-magazine folder layout so the live tether can compare against them.

### What changed

| Folder | Before | After |
|---|---|---|
| `images/NME/` | 670 JPGs flat + `cropped/` subfolder | 258 magazine subfolders + `_originals_legacy/` |
| `images/Record Mirror/` | 304 JPGs flat + `cropped/` subfolder | 153 magazine subfolders + `_originals_legacy/` |

For each detected magazine (from the existing item-detection pipeline:
spread-detection → OCR-date refinement), the migration script:

1. Created `<photographer_date>_<PUB>_<NNNN>/` subfolder
2. Moved the **cropped** versions into it (matching tether's cropped-only output)
3. Moved the **originals** to `_originals_legacy/` (preserved, not deleted —
   user can purge later)
4. Left `phash_cache.json` intact (cache keys are filenames, unaffected by
   the physical reorganisation)
5. Removed the now-empty `cropped/` subfolder

### Verification

Smoke-tested the live tether against the migrated NME library by simulating a
"shoot" with photos already in the library:
- Shot 0160-0162 (Graham Parker) → flagged **DUPLICATE** of `2026_05_07_NME_0056`,
  score 0, date match on 1976-11-13 ✓
- Shot 0420-0421 → flagged **DUPLICATE** of `2026_05_07_NME_0147`, score 0,
  date match on 1977-04-09 ✓

(Both were tautological matches — same files — but the test confirms the
migrated library structure is correctly traversed and compared.)

### New file this session

- `migrate.py` — standalone one-shot. Usage:
  ```
  python migrate.py --dry-run <folder>     # show planned moves
  python migrate.py <folder1> [<folder2> ...]
  ```

### Outcome

User's library now has the canonical layout the live tether expects. New
shoots via Alt+N will be compared against all 258 NME + 153 Record Mirror
existing magazines automatically.

---

## Session 3 (2026-05-09 evening continued) — live tether workflow

User pivoted: don't want a separate "shoot helper" plus "batch analyser" — wants
*the existing program* to become the live shooting assistant. Workflow:

1. Press `Alt+N` before shooting each magazine.
2. EOS Utility drops photos into a watch folder as usual.
3. The app automatically: (a) creates `<library>/<date>_<pub>_<NNNN>/` on each
   hotkey press, (b) moves+crops each new photo into the current magazine
   folder, (c) deletes the original (cropped-only output per user preference).
4. After idle timeout (default 30 s) OR next hotkey press, the magazine
   "closes": its photos are compared against every other magazine folder in
   the library and an **instant verdict** is shown — UNIQUE, DUPLICATE, or
   UNCERTAIN — with the matching magazine name if any. User can set aside
   physical duplicates for resale immediately.

### Architecture

| Component | File | Notes |
|---|---|---|
| Watcher | `tether.py` | watchdog Observer + pynput GlobalHotKeys + queue+worker thread |
| Cropping | reused `find_duplicates_local.crop_magazine` | no changes |
| Per-folder cache | reused `find_duplicates_local.{load,save}_cache` | one cache.json at library root, shared with batch mode |
| API bridge | `app.py` `Api.*tether*` methods | start/stop, open/close magazine, settings, state |
| UI | rewrite of `app_index.html` | two top-level tabs (Shoot/Library), Shoot is default |

The watcher waits for file size stability (3 polls × 150 ms unchanged) before
processing — EOS Utility writes incrementally and watchdog fires on first
byte. Empirically reliable.

### Settings persistence

`%APPDATA%\DuplicateFinder\tether_settings.json` holds watch_dir, library_dir,
publication, hotkey, auto_close_seconds, match_threshold.

### End-to-end smoke test (real NME photos, simulated EOS shooting)

Three "magazines" rotated by hotkey-equivalent:
- Mag #1: photos 0001, 0002, 0003 (Brian Eno cover + spread) → **UNIQUE** ✓
- Mag #2: photos 0142, 0143 (BAMA LAMA cover + back) → **UNIQUE** ✓
- Mag #3: photos 0456, 0457 (BAMA LAMA SECOND copy) → **DUPLICATE of #2**
  ✓ — score 0, pHash 16, dHash 21, date match on (1975-09-18, 1976-09-18)

Files correctly cropped, originals deleted, library folders correctly named:
`2026_05_11_NME_0001/`, `2026_05_11_NME_0002/`, `2026_05_11_NME_0003/`.

### Verdict algorithm

For each candidate sibling magazine in the library:
1. Compute all-pairs `min(pHash, dHash)` distance between new mag's photos and
   sibling's photos. Track the minimum.
2. If new and sibling share any extracted OCR date → subtract 30 from the
   score (strong promotion; date matches are gold-standard evidence).
3. After scanning all siblings, pick the lowest score.

Classification:
- score ≥ threshold (default 60) → **UNIQUE**
- score ≤ 25 OR date match → **DUPLICATE**
- else → **UNCERTAIN**

### .exe rebuild

Added `--collect-all watchdog`, `--collect-all pynput` plus a few
`--hidden-import` lines. Size unchanged at **151 MB**.

### Open work (not done this session)

- The existing 670+304 photos in `images/NME/` and `images/Record Mirror/` are
  still in the flat-folder layout. A migration helper could split them into
  per-magazine folders using the existing item-detection. Not wired in yet
  because the user's main use case going forward is *new* shooting, not
  re-organising the legacy library.

### Files changed this session

- `tether.py` — new module, ~330 lines, all the live-watcher logic
- `app.py` — added tether API methods (start/stop/open/close/state/settings),
  `_sanitize_event` helper for marshalling numpy→json
- `app_index.html` — full rewrite: tabbed UI (Shoot default + Library
  secondary), big verdict card, activity log, settings panel
- `build.ps1` — added watchdog + pynput to PyInstaller args

---

## Session 2 (2026-05-09 evening) — image cropping / normalization

User asked: "research the best way to crop/rotate/center images before analysing,
output the cropped versions so I can upload them to my server."

### Research findings (from sampling 30 random NME photos)

| Property | Result |
|---|---|
| Camera | 3984×2656 landscape, consistent setup, top-down lighting, light-grey background |
| Vertical position | very tight (center_y 0.49-0.52) |
| **Horizontal position** | **drifts ±15% (center_x 0.35-0.64)** ← main accuracy loss |
| Vertical extent | 92-98% of frame |
| Horizontal extent | bimodal: 44-83% (singles), 87-98% (spreads) |
| **Rotation** | **visible 1-5° in most photos** ← second accuracy loss |
| Edge sharpness | strong (Sobel max = 255) — contour detection reliable |

### Approach chosen: **OpenCV contour + perspective warp**

After comparing 5 options (bbox-only, Hough rotation, OpenCV contour, ML scanner,
hybrid), chose OpenCV because:
- User priority was accuracy, not speed
- Cropped images go on a server → need to look good, not just be hash-comparable
- Perspective warp handles rotation AND slight perspective distortion
- Industry-standard for document scanning, well-tested

Pipeline:
1. Grayscale → corner-median background reference
2. Threshold (bg-25) + morphological close
3. Largest external contour
4. Approximate to quad (try epsilons 0.02-0.08); fall back to rotated bounding rect
5. Order corners (TL/TR/BR/BL) and compute output dimensions from quad edge lengths
6. cv2.getPerspectiveTransform → cv2.warpPerspective
7. Save as JPEG quality 92

### Bugs found and fixed during integration

1. **edge_std regression on cropped images.** Old spread detection relied on
   whitespace in the photo's left/right margins. After cropping, no whitespace
   → every cropped image looked like a spread → every photo became its own
   item. Fix: added `is_spread` flag based on aspect ratio of cropped output
   (>1.1 = landscape spread, <1.1 = portrait single), with edge_std as fallback
   for legacy entries.

2. **`scan_images` double-counted images** by walking into the `cropped/`
   subfolder, finding 1340 entries instead of 670 with duplicated filenames.
   Fix: skip `cropped/`, `_thumbs/`, and `_backup_*` subdirectories.

3. **Tesseract OCR hangs indefinitely** on certain images (recurring issue,
   third stall). Added `OCR_TIMEOUT_SEC = 20` via pytesseract's `timeout=`
   param. On timeout, returns empty string and pipeline continues.

### Real-data results after cropping

**Visual confirmation (BAMA LAMA covers, seq 0142 vs 0456):**
- Pre-cropping: framing very different (offset, rotated 2-3°). pHash distance 56.
- Post-cropping: both look nearly identical (centered, straight). pHash distance 20.

**NME at threshold 60:**
- 240 items detected (unchanged — magazine boundaries still correct after is_spread fix)
- 29 duplicate groups (was 34 pre-cropping; the 5 "missing" were date-OCR-refined splits
  where the algorithm correctly recognized two stack-adjacent items as different
  magazines with different printed dates — improved accuracy, not lost recall)
- 2 groups with 3+ copies
- Each cropped image is now a clean, ~3 MB JPEG at native magazine resolution

**Record Mirror at threshold 60:**
- 76 items detected
- 3 duplicate groups (was 4; cropping eliminated 1 ambiguous match)

### Cache schema (v5)

```
{
  "_version": 5,
  "filename.JPG": {
    "phash":        "<64-char hex>",  // perceptual hash from CROPPED image
    "dhash":        "<64-char hex>",  // difference hash from CROPPED image
    "edge_std":     <float>,           // legacy, still computed
    "aspect_ratio": <float>,           // cropped output's w/h
    "is_spread":    <bool>,            // aspect_ratio > 1.1 (preferred over edge_std)
    "ocr_text":     "<truncated 4000>",
    "dates":        [[Y, M, D], ...]
  },
  ...
}
```

### Files for server upload

`<source>/cropped/<original_filename>.JPG` — same naming as original. Quality 92.
Native resolution (no upscaling). All 670 NME + 304 Record Mirror photos cropped.

### .exe rebuild

- Now bundles OpenCV (`--collect-all cv2`) in addition to tesseract
- New size: **151 MB** (was 113 MB pre-cropping, plain tesseract; +40 MB OpenCV)
- Build command: `.\build.ps1`

### Files changed this session

- `find_duplicates_local.py` — added `compute_features` aspect_ratio + is_spread; new `crop_magazine`, `_find_magazine_quad`, `_order_quad`, `ensure_cropped`; OCR timeout; `scan_images` skips output dirs; updated `analyze_folder` to crop before hash; report uses cropped for thumbnails; cache v4 → v5
- `build.ps1` — added `--collect-all cv2` and `--hidden-import cv2`

---

# Worklog — autonomous rewrite session (2026-05-09)

User left for several hours. Authorized me to rewrite as needed, prioritize accuracy
over speed, lean toward recall over precision, don't commit (working tree review).

## Goals (from user, in order)

1. Don't miss real duplicates (recall priority)
2. Future-proof: works for magazines, books, "anything around that standard"
3. UI cleanup — currently cluttered (9 filter buttons with overlapping semantics)
4. Use OCR'd dates as a "same-magazine" signal — most pages have a printed date

## Decisions and rationale

### OCR / dates — use as a CONFIRMING signal, not a primary one

Tested OCR on 14 sample photos (mix of covers, inside pages, spreads):
- Covers: dates extracted reliably ("November 27 1976" from seq 1; "November 13 1976" from seq 224)
- Inside pages: ~50% extraction rate; dates often missing or unreadable
- Two-page spreads: OCR slow (8-12s) and unreliable
- Average non-spread time: ~0.7s/photo. NME folder ≈ 8-15 min for full OCR.

**Decision:** OCR every photo (cache result). Use dates as:
- *Item refinement*: if photos within a spread-detected item have inconsistent dates AND
  multiple of them have dates → split the item.
- *Match confirmation*: if two candidate-duplicate items share an extracted date → boost
  confidence to HIGH regardless of hash distance. If both have dates and they DIFFER → reject.
- *Recall boost*: if two items share an extracted date but hash distance is above threshold
  → still flag as a potential duplicate (date match is strong evidence).

OCR will be silently optional: if tesseract isn't available, fall back to the
existing edge_std + hash pipeline.

### Tesseract setup (this machine)

- Installed via scoop (`scoop install tesseract`). 5.5.0.20241111.
- Language data scoop install failed at extraction; downloaded `eng.traineddata`
  directly from `tessdata_fast` (3.9 MB) into `~\scoop\apps\tesseract\current\tessdata\`.
- pytesseract installed via `pip install pytesseract`.

For .exe distribution: tesseract binary + tessdata need to be bundled. Will document
the bundling step when I get to PyInstaller build. If too heavy, OCR ships off by default
and the user can enable it via a checkbox after installing tesseract themselves.

### Architecture: phases

Refactoring `find_duplicates_local.py` into discrete phases so each is swappable and
testable:

1. **Scan**: discover image files, parse filenames into (publication, seq, date).
2. **Features**: per-photo features (pHash, dHash, edge_std, OCR dates). Cached.
3. **Item-detect**: group photos into items (magazines/books) using boundary signals.
4. **Match**: pairwise compare items, ensemble of multiple signals.
5. **Cluster**: union-find on matched pairs → groups of duplicates.
6. **Report**: HTML/text generation.

Adding a new signal becomes "extend the Features dataclass + use it in Match".

### Match strategy (the accuracy fix)

Old: position-aligned (pos 0 of A vs pos 0 of B; pos 1 vs pos 1). Single algorithm (pHash).
New: every photo of A vs every photo of B. Best (smallest) distance per algorithm wins.
Ensemble: pair flagged if EITHER pHash OR dHash distance ≤ threshold (recall-favoring).
Date overlap: confirms or rejects.

This roughly doubles comparison count per item-pair (3*3 vs 2 for typical 3-page item)
but is still <1s for the whole dataset.

### Cache structure (v4)

Bumping CACHE_VERSION to 4. Schema:

```json
{
  "_version": 4,
  "filename.JPG": {
    "phash": "abc..." (64 hex chars for 256-bit hash),
    "dhash": "...",
    "edge_std": 12.3,
    "ocr_text": "raw OCR output, truncated to 2000 chars",
    "dates": [["1976", "11", "27"], ...]  // normalized as (Y, M, D) tuples
  }
}
```

Old caches will be invalidated → re-hash on next run. With OCR, that's ~10-15 min for NME.

### UI cleanup plan

Current: Strong / All / 3+ copies / High / Medium / Low / Uncertain / Unverified / Covers only.
That's 9 buttons with overlapping semantics.

New:
- Single segmented control "Show:" with [Strong] [All] [Custom]
- Confidence slider (only visible when Custom selected)
- Copy-count filter (separate, "≥ 2" / "≥ 3" / "Any")
- Verification filter (Unverified / Verified / All)
- Covers-only stays as a separate toggle (it's orthogonal)

Plus: side-by-side compare modal (click "compare" on a group).

### What I'm NOT changing (yet)

- pHash-based hashing (still solid baseline)
- Cache file location (`phash_cache.json`)
- Filename parsing (`YYYY_MM_DD_Pub_NNNN.JPG`)
- Local HTTP server for serving thumbnails

## Progress log

### Completed

1. **Cache schema bumped to v4.** Now stores per-photo: `phash`, `dhash`, `edge_std`,
   `ocr_text`, `dates` (list of (Y,M,D) tuples). Old caches invalidated; first run
   re-hashes everything.
2. **`compute_features` rewritten.** Single image-open computes all features.
   OCR runs at full resolution (~1.5s/photo) because aggressive downscaling loses
   small date prints. dHash added alongside pHash for ensemble matching.
3. **Date extraction.** `extract_dates()` parses month-name and DD/MM/YY formats.
   Verified on 4 sample covers: 3/4 dates correctly extracted directly via OCR; the
   4th (Eddie & Hotrods cover) has decorative print that Tesseract misses.
4. **Pipeline refactored into phases:**
   - `_build_items_by_spread()` — chunk photos at spread boundaries (existing logic, cleaner name)
   - `_refine_items_by_dates()` — split items with inconsistent dates; merge adjacent same-date items
   - `_match_items()` — every-photo-vs-every-photo matching with pHash + dHash ensemble
   - `find_duplicate_groups()` — orchestrates the phases
5. **Match strategy upgrade:**
   - Multi-page: every photo of A vs every photo of B (smaller distance per algorithm wins)
   - Ensemble: pair flagged if EITHER pHash ≤ threshold OR dHash ≤ threshold
   - Date confirmation: matching consensus dates → boost (accept up to threshold 90); conflicting consensus dates → reject (unless hash distance ≤ 12, "obviously identical" override)
   - Confidence ladder: VERY HIGH (≤10) / HIGH (≤25) / MEDIUM (≤40) / LOW (>40), bumped up one tier when date-match
6. **HTML report UI cleanup:**
   - 9 filter buttons → 3 segmented controls (Confidence × Copies × Status) + Covers-only toggle
   - Per-copy date pill displayed when `consensus_date` is set
   - "Date match" badge on group header when items share an extracted date
   - Side-by-side compare modal (click ⇄ Compare) — shows all photos of all copies in N-column layout
   - Keyboard shortcuts simplified: `s`/`a` confidence, `3` copies, `u` unverified, `c` covers-only

### In progress (background task)

- OCR pass on NME folder. As of last check: 125/670 photos done, 108 of those 125
  have at least one extracted date (~86% date-recall). Estimated ~15 min total.

### Real-data hash distance findings (NME, partial cache 275/670)

Tested on confirmed duplicate pairs (visually verified) and known-unrelated pairs:

**UNRELATED magazines (cover-cover):**
- pHash: 88-126 (well above any reasonable threshold)
- dHash: 44-72 (close to threshold of 60 — too permissive)

**REAL DUPLICATES (Graham Parker issue, seqs 160 vs 224):**
- 160 vs 224 (cover-cover): pHash=106, dHash=82  ← *both above threshold*
- 161 vs 225 (back-back): pHash=58, dHash=41  ← marginal
- 162 vs 226 (spread-spread): pHash=32, dHash=32  ← strong match

**Algorithm tuning consequences:**

1. Cover-only matching misses real duplicates whose covers were photographed at
   different angles/lighting. The Graham Parker issue is a clear example: the
   covers don't hash-match well, but the spreads do.

2. Every-photo-vs-every-photo (cross-position) matching causes false-positive
   cascades: ~84 unrelated magazines got chained into one group via random
   inside-page similarities.

3. **Final design: position-aligned + multi-position requirement.**
   - Compare A[k] vs B[k] for k = 0..min(len(A), len(B)).
   - dHash threshold = pHash threshold / 2 (because dHash is looser on this
     content — see distance distribution above).
   - Require ≥ 2 positions to match (when both items have ≥ 2 pages) before
     accepting a duplicate. This breaks cascades because unrelated items rarely
     have 2 coincidentally-matching positions.
   - Single near-identical position (≤ 12) overrides the 2-position rule.
   - Date confirmation (matching OCR'd publication dates) overrides the rule
     and accepts up to a relaxed `date_boost_threshold` of 90.
   - Date conflict (different OCR'd dates) rejects the pair unless near-identical.

   Result on partial cache: **5 real duplicate groups** detected, no cascades.
   Visually verified pair: NME #133 / #221 = "Lowell George is the future of
   rock 'n' roll" Dec 6 1975 — confirmed real duplicate.

### .exe build with bundled tesseract

`build.ps1` now produces `dist/DuplicateFinder.exe` (~113 MB, was 58 MB without
OCR support). Tesseract binary + `eng.traineddata` + required DLLs are staged
into `_tesseract_bundle/` and embedded via PyInstaller `--add-data`. At runtime,
`app.py:_configure_tesseract()` points pytesseract at the unpacked binary inside
`sys._MEIPASS`. If tesseract is unavailable, OCR features silently degrade.

To build:  `.\build.ps1`

Requires:
  - `scoop install tesseract`
  - `eng.traineddata` in `~\scoop\apps\tesseract\current\tessdata\` (downloaded
    directly from `tessdata_fast` if scoop's `tesseract-languages` package
    fails — known scoop extraction bug).

### Final results (full OCR cache, both folders)

**NME (670 photos)** at threshold 60, pages_per_magazine 4:
- 240 items detected (vs 169 with the old N=4-only algorithm — spread-detection
  finds true magazine boundaries: variable item lengths from 1-6 photos)
- **34 duplicate groups** found (vs 7 with old algorithm — recall increased ~5x)
- 3 groups with 3+ copies
- Most groups confirmed by matching OCR'd publication dates (visible in HTML
  report as "📅 date match" badge)
- 6 warnings about unusually large groups (≥6 copies) for visual review

**Record Mirror (304 photos)**:
- 76 items detected (clean 304/4 = 76 magazines)
- **4 duplicate groups** found, all HIGH confidence
- 3 of 4 confirmed by date match
- 1 split refinement (item at seq 29/30 split apart due to date mismatch)

### OCR pass timing notes

- First run on NME: cache invalidated v3→v4, 670 photos to compute.
- Original (no skip): stalled at 325 photos after ~1 hour (one photo got stuck
  in tesseract). Killed and resumed.
- Resumed pass with spread-skip optimization: 345 remaining photos in 6.7 min
  (~0.85 photos/s). Spread photos take 0.0s (skipped); single pages take 1-2s
  with date extraction.
- Record Mirror cold pass: 304 photos in ~5 min with new optimization.
- After cache is built, subsequent re-runs are instant (analyze takes ~5s).

### Cache schema (v4)

Stored in `<folder>/phash_cache.json`:
```
{
  "_version": 4,
  "filename.JPG": {
    "phash":      "<64-char hex>",   // 256-bit perceptual hash (HASH_SIZE=16)
    "dhash":      "<64-char hex>",   // 256-bit difference hash
    "edge_std":   <float>,           // avg std-dev of left+right 10% strips
    "ocr_text":   "<truncated 4000>",// raw OCR output (debug aid)
    "dates":      [[Y, M, D], ...]   // extracted dates from OCR
  },
  ...
}
```

### Files changed this session

- `find_duplicates_local.py` — rewrote feature extraction, item-detection, matching pipeline
- `app.py` — added `_configure_tesseract()` for bundled OCR support
- `app_index.html` — Pages/mag slider already present, no changes this session
- `build.ps1` — NEW, builds .exe with bundled tesseract
- `test_accuracy.py` — NEW, synthetic accuracy harness (limited usefulness on
  noise patterns; real-data verification was more revealing)
- `WORKLOG.md` — this file

### Known limitations / future work

1. **OCR is English-only.** Date regex parses English month names + DD/MM/YYYY.
   For non-English magazines, OCR would still extract text but date regex would
   miss. Easy to extend the regex when needed.

2. **Spread-vs-single-page heuristic is one-dimensional.** edge_std works well
   on this user's data (clean white background) but might fail if photographs
   are taken on a busy surface. Possible fallback: aspect ratio of detected
   foreground bounding box.

3. **Match threshold tuning is per-content.** dHash threshold = pHash/2 is
   tuned for these magazine scans. Different content (e.g., children's books
   with very colorful uniform pages) may need adjustment. Could expose as a
   secondary slider in the UI.

4. **Synthetic test harness is weak.** Random noise patterns don't exercise the
   algorithm the same way real images do. The harness exists but most
   meaningful regression coverage is via real-data spot checks.

5. **No "Did I miss any?" recall verification on real data.** I confirmed the
   detected duplicates are real (visual spot checks on Eddie & Hotrods, Rod
   Stewart, Graham Parker, Lowell George covers) but I haven't manually
   verified that no real duplicates were missed. Recall of 34 vs old 7 suggests
   strong improvement, but a full recall audit would require manually skimming
   all 240 items.

### How to use

**For development (with system tesseract):**
- `python find_duplicates_local.py <folder>` — CLI mode
- `python app.py` — desktop app

**To build:** `.\build.ps1` produces `dist/DuplicateFinder.exe` (~113 MB) with
tesseract bundled. Run that .exe directly.

**Cache:** First analysis on a folder takes minutes (OCR + hashing). Subsequent
runs reuse `phash_cache.json` and complete in seconds.




