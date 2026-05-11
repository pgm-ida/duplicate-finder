# Duplicate

A desktop app that flags duplicate magazines, books, or any other physical
artifacts as you photograph them — runs entirely locally, no API keys, no
cloud services. Built for sellers and archivists working through stacks
where you can't easily tell at a glance which titles you already have.

## Download

Get the latest `DuplicateFinder.exe` from the
[Releases page](../../releases/latest), double-click to run.

> First launch on Windows: dismiss SmartScreen with "More info → Run anyway".
> The binary isn't code-signed (no certificate yet), but you can also build
> it yourself — see below.

## Two modes

Tabs in the top bar switch between them.

### Shoot — live tethered duplicate check

The primary workflow. You photograph magazines one at a time and the app
tells you instantly whether each one is a duplicate.

1. **Setup** (one-time):
   - Watch folder — where your camera software (e.g. EOS Utility) drops
     new JPGs
   - Library folder — where the app organizes per-magazine subfolders
   - Duplicates folder — where confirmed duplicates get moved (defaults to
     `<library>/_duplicates`)
   - Hotkey — click the field and press the combo you want (e.g. `Alt+N`)
2. **Click Start watching**.
3. Press the hotkey **before each magazine** to open a fresh folder.
4. Shoot the **front cover first**. The moment it lands, the app compares
   it against every cover in the library and shows a verdict:
   - ✓ **UNIQUE** — keep shooting (back, colofon, index)
   - ⚠ **DUPLICATE** — set the physical magazine aside for resale, click
     **Move to duplicates**, and rotate to the next magazine
   - ? **UNCERTAIN** — verify visually, then either accept or move on
5. Continue with remaining photos for unique magazines (back cover, colofon,
   etc.); they auto-crop and stack into the magazine folder.

The thumbnail strip below the status bar shows everything in the current
magazine, and any photos that arrived late (after a rotation, routed by
EXIF time into the previous magazine) get a yellow border.

**Why cover-only**: interior pages — masthead, colofon, page index — are
nearly identical across unrelated issues and cause false-positive matches
in full-magazine comparison. The cover is the only page that reliably
identifies one issue.

### Library / batch — analyze an existing folder

For collections you've already photographed in bulk. Pick a folder of
sequentially-named JPGs (`YYYY_MM_DD_PublicationName_NNNN.JPG`), set a
match threshold, click **Analyze**. Generates an interactive HTML report
of every duplicate group with a side-by-side lightbox for visual
verification.

Filenames must follow the convention so the grouper can split consecutive
photos into per-magazine units:

```
2026_05_08_NME_0001.JPG   # mag 1, front
2026_05_08_NME_0002.JPG   # mag 1, back
2026_05_08_NME_0003.JPG   # mag 2, front
…
```

**Threshold guide**:
- **20–30** — strict, only near-identical matches
- **40–50** — balanced
- **60** — recommended, catches most copies including 3+ duplicate sets
- **70+** — loose, catches degraded duplicates but produces more false positives

## How the detection works

Each photo passes through a local pipeline:

1. **Auto-crop** — `crop_magazine` finds the magazine rectangle in the
   frame and cuts away background.
2. **Perceptual hashing** — pHash + dHash, both 64-bit. Hamming distance
   between hashes is the similarity score.
3. **OCR date extraction** — Tesseract pulls cover dates from the cropped
   image. A shared date between two covers boosts the match confidence
   significantly.

The bundled `.exe` ships with Tesseract and the English language model
inside (~70 MB of the binary's size).

## Run from source

```powershell
git clone https://github.com/pgm-ida/duplicate-finder.git
cd duplicate-finder
pip install -r requirements.txt
# Windows: scoop install tesseract  (or install Tesseract any way you like
# and make sure tesseract.exe is on PATH)
python app.py
```

Requires Python 3.10+. Tesseract must be reachable on `PATH` (or override
via `pytesseract.pytesseract.tesseract_cmd`).

## Build the binary

```powershell
.\build.ps1
```

The script:
1. Stages a Tesseract bundle from your scoop install into
   `_tesseract_bundle/`
2. Regenerates `icon.ico` from `make_icon.py` (the two-pill Ditto mark)
3. Runs PyInstaller with `--onefile --windowed`, embedding the icon,
   HTML, and Tesseract bundle

Output: `dist\DuplicateFinder.exe` (~150 MB — most of it is OpenCV,
NumPy/SciPy, and Tesseract).

If you don't have Tesseract via scoop, edit the `$tessSrc` path at the
top of `build.ps1` to wherever your `tesseract.exe` lives.

## CLI

The batch pipeline is also a standalone CLI:

```powershell
python find_duplicates_local.py <folder> --threshold 60
```

Outputs `duplicates_report.html` and `duplicates_report.txt` in the
folder.

## Project layout

| File | Role |
|---|---|
| `app.py` | pywebview entry point, JS↔Python bridge |
| `app_index.html` | Single-page UI (Shoot tab + Library tab) |
| `tether.py` | Live watcher: hotkey, file events, cover-only verdict |
| `find_duplicates_local.py` | Batch analysis pipeline + HTML report |
| `make_icon.py` | Renders `icon.ico` from the Ditto mark |
| `build.ps1` | One-command build (Tesseract staging + PyInstaller) |
| `migrate.py` | One-shot migration from flat folder → per-magazine subfolders |
| `test_accuracy.py` | Synthetic precision/recall harness |

## License

MIT — see [LICENSE](LICENSE).
