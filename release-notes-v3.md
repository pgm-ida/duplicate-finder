# v3 — rembg cropping + compact UI

This release replaces the magazine-detection step with a foreground-segmentation
model (rembg / U²-Net) and tightens the shoot-mode UI for use next to
EOS Utility's fixed window.

## What's new

**Smarter cropping**
- **rembg (U²-Net)** is now the primary magazine-quad detector. The old
  corner-median OpenCV detector stays as fallback when rembg can't produce
  a plausible quad.
- Fixes the bound-book case where the OpenCV detector pulled the dark
  leather binding into the crop. Rembg cuts along the actual page edge,
  giving consistent crops whether the magazine is loose or in a book.
- Also handles complex/textured backgrounds where the page brightness is
  close to the surface — the old threshold-based detector silently failed
  on those.
- New rejections in the rembg detector: too-small mask (<15% of frame,
  likely an already-cropped input), too-large mask (>97%, whole-frame
  fallback that defeats cropping), and skewed quads (>40% mismatch between
  opposite sides).

**Compact shoot-mode UI**
- Sidebar is now collapsible (320 px when open, 0 when collapsed). It
  auto-collapses when watching starts and auto-expands when stopped.
  Manual toggle via the gear icon in the status strip.
- Status strip slimmed: single line with status dot, magazine name
  (truncated with ellipsis on overflow), and photo count.
- Verdict card halved in size (40 px → 26 px title, 32 px → 18 px padding)
  and the message text shortened ("No match in library." / "Matches X").
- "How it works" help panel removed from the sidebar.

**.exe build**
- `build.ps1` now stages `u2net.onnx` (~176 MB) into `_rembg_bundle/` and
  bundles it via PyInstaller `--add-data` alongside the existing Tesseract
  bundle. The exe sets `U2NET_HOME` at runtime so it never tries to
  download the model from the internet.
- Bundle includes `--copy-metadata rembg pymatting pooch tqdm onnxruntime`
  so rembg's runtime metadata lookups succeed in the frozen binary.
- Final size: ~370 MB (was ~150 MB in v2).
- rembg's heavy dependencies (onnxruntime, scipy, numba) are now
  lazy-imported — they only load on the first crop, not at app launch.
  This drops time-to-window from ~80 s to ~15 s on cold disk.

## Notes & caveats

- First crop after launch takes a few extra seconds (one-time rembg
  session init). Subsequent crops run at the steady-state ~2–3 s per
  photo. The tether worker runs cropping off the UI thread, so the user
  doesn't see it block.
- Existing libraries cropped with v2 are still valid; the cache schema
  (v5) hasn't changed. Newly-shot photos cropped with rembg compare
  against older opencv-cropped library photos via pHash, which is robust
  to the small differences between the two cropping outputs (typically
  under the 24/256-bit threshold).
- See [`WORKLOG.md`](WORKLOG.md) for the full session log and the
  validation run.
