# Duplicate Finder

A desktop app that finds duplicate magazines, records, posters, books, or any other physical artifacts you've photographed — by analyzing the photos themselves. Runs entirely locally, no API keys or cloud services needed.

## Download

Get the latest pre-built app from the [Releases page](../../releases/latest):

- **Windows**: `DuplicateFinder.exe` — double-click to run
- **macOS**: `DuplicateFinder-macos.zip` — unzip, then double-click `DuplicateFinder.app`

> First time on macOS: right-click → Open (instead of double-click) to bypass the "unidentified developer" warning.

## How it works

1. Photograph each item in your collection sequentially. Use a consistent naming convention: `YYYY_MM_DD_CollectionName_NNNN.JPG` (e.g. `2026_05_08_NME_0001.JPG`)
2. Open the app, point it at the folder containing the photos
3. Click **Analyze** — it will:
   - Compute a perceptual hash of every photo (locally, ~25 images/sec)
   - Find pairs of photos that show the same physical item
   - Group consecutive photos into "physical copies" (using sequence numbers + parallel-evidence merging)
   - Report which items appear multiple times in the stack
4. Visually verify each duplicate using the built-in lightbox, mark them confirmed/dismissed (state persists in browser localStorage)
5. Walk to your physical stack and remove the duplicate copies

## File naming convention

Files must be named `YYYY_MM_DD_PublicationName_NNNN.JPG` where:
- `YYYY_MM_DD` — date the photos were taken
- `PublicationName` — collection name (e.g. `NME`, `Recordmirror`, `Books`)
- `NNNN` — 4-digit sequence number representing the position in the physical stack

Example layout:

```
my_archive/
├── 2026_05_08_NME_0001.JPG       # First magazine, front
├── 2026_05_08_NME_0002.JPG       # First magazine, back
├── 2026_05_08_NME_0003.JPG       # Second magazine, front
├── 2026_05_08_NME_0004.JPG       # Second magazine, back
└── 2026_05_08_NME_0005.JPG       # Third magazine, front
```

The sequence number is what lets the app tell you exactly which physical paper to remove from your stack.

## Tuning

The **Threshold** slider controls how lenient the duplicate detection is:

- **20–30** (strict): only nearly-identical photos match. Low false positives, but misses duplicates with significant lighting/angle differences.
- **40–50** (balanced): catches most real duplicates with few false positives.
- **60** (recommended): catches most copies including 3+ duplicate sets. Some false positives, but the verification UI lets you dismiss them quickly.
- **70+** (loose): catches even degraded duplicates. Expect to dismiss a third of matches as coincidental layout overlaps.

## Run from source

If you don't want to use the prebuilt binaries, or you want to modify the app:

```bash
git clone https://github.com/YOUR_USERNAME/duplicate-finder.git
cd duplicate-finder
pip install -r requirements.txt
python app.py
```

Works on Windows, macOS, and Linux. Requires Python 3.10+.

## Build a binary yourself

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name DuplicateFinder \
  --add-data "app_index.html:." --collect-all webview app.py
```

Output: `dist/DuplicateFinder.exe` (Windows) or `dist/DuplicateFinder.app` (macOS).

> Note: the separator between the source and destination paths in `--add-data` is `;` on Windows and `:` on macOS/Linux.

## CLI usage

If you'd rather not use the GUI, the analysis pipeline is also available as a command-line tool:

```bash
python find_duplicates_local.py /path/to/folder --threshold 60
```

This generates an `duplicates_report.html` and a plain-text `duplicates_report.txt` in the folder.

## License

MIT — see [LICENSE](LICENSE).
