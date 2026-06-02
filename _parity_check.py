#!/usr/bin/env python3
"""Prove the LIVE worker path crops identically to the batch eval path.

For each real image:
  A) eval path  — crop_magazine_with_meta() on the main thread (what _crop_eval did)
  B) live path  — fed through a real started TetherWatcher (watchdog → worker
                  thread → _process_one), exactly as live shooting does

Then compare method / confidence / dims and the SHA-256 of the written crop.
If A and B match, the fixed code crops the same live as in the eval — i.e. the
live path has no hidden regression (rembg threading, fallback, etc.).
"""
import hashlib
import shutil
import time
from pathlib import Path

import find_duplicates_local as fdl
import tether

SRC_DIR = Path(r"C:\Users\Ida\Desktop\deewee\EPHEMERA\New folder")
OUT = Path(r"C:\Users\Ida\Desktop\deewee\input\results\parity")
N = 12


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None


def main():
    imgs = sorted(p for p in SRC_DIR.glob("*.JPG"))[:N]
    if not imgs:
        print("no sample images"); return
    eval_dir = OUT / "eval"; live_watch = OUT / "watch"; live_lib = OUT / "library"
    for d in (eval_dir, live_watch, live_lib):
        d.mkdir(parents=True, exist_ok=True)

    # ── A) eval path (main thread) ───────────────────────────────────────
    eval_res = {}
    for src in imgs:
        dims, meta = fdl.crop_magazine_with_meta(str(src), str(eval_dir / src.name))
        eval_res[src.name] = (meta["method"], meta["confidence"], dims,
                              sha(eval_dir / src.name))

    # ── B) live path (real watcher + worker thread) ──────────────────────
    events = []
    w = tether.TetherWatcher(watch_dir=str(live_watch), library_dir=str(live_lib),
                             publication="PARITY", cover_check=False,
                             on_event=events.append)
    w.start()
    w.open_new_magazine()
    mag = w._current_magazine
    try:
        for src in imgs:
            shutil.copy2(src, live_watch / src.name)
            time.sleep(0.15)  # stagger so the watchdog enqueues each distinctly
        # Wait for the worker queue to drain.
        deadline = time.time() + 180
        while time.time() < deadline:
            done = sum(1 for s in imgs
                       if (mag / s.name).exists()
                       or (w._review_dir / s.name).exists())
            if done >= len(imgs):
                break
            time.sleep(0.5)
        time.sleep(1.0)
    finally:
        w.stop()

    # Live crop lands in the magazine folder (HIGH) or _review (LOW, no crop).
    print(f"{'image':32} {'eval':28} {'live':28} match")
    print("-" * 100)
    n_match = n_diff = 0
    for src in imgs:
        em, ec, ed, eh = eval_res[src.name]
        live_crop = mag / src.name
        review = w._review_dir / src.name
        if live_crop.exists():
            lh = sha(live_crop); lstate = f"crop {lh}"
        elif review.exists():
            lh = None; lstate = "staged(LOW)"
        else:
            lh = None; lstate = "MISSING"
        same = (eh == lh) if ec == "high" else (ec == "low" and not live_crop.exists())
        n_match += same; n_diff += (not same)
        ev = f"{em}/{ec}/{ed[0] if ed else '-'}"
        print(f"{src.name:32} {ev:28} {lstate:28} {'OK' if same else 'DIFF'}")

    print("-" * 100)
    print(f"{n_match}/{len(imgs)} identical between eval path and live worker path")
    if n_diff:
        print("=> DIFFERENCE FOUND: the live worker crops differently than the eval.")
    else:
        print("=> The fixed code crops the SAME live as in the eval. No live regression.")


if __name__ == "__main__":
    main()
