"""Live-tether watcher: organise EOS Utility output into per-magazine folders
and instantly detect duplicates as photos arrive.

Workflow
--------

1. User configures a `watch_dir` (where EOS Utility drops new JPGs) and a
   `library_dir` (where per-magazine subfolders should live).
2. User presses the global hotkey (default Alt+N) to start a new magazine.
   This creates `<library_dir>/<date>_<pub>_<NNNN>/` and points subsequent
   incoming photos there.
3. As each new JPG appears in watch_dir, the watcher:
     a) waits for it to finish being written
     b) moves it into the current magazine folder
     c) runs `find_duplicates_local.crop_magazine` on it
     d) deletes the original (per user preference: cropped-only output)
     e) computes features for the cropped version (phash, dhash, OCR, ...)
4. When the magazine "closes" (next hotkey press OR `auto_close_seconds`
   timeout after the last new photo), the watcher compares the new magazine
   against every other magazine in `library_dir` and emits a verdict via the
   `on_verdict` callback: UNIQUE, DUPLICATE_OF <other>, or UNCERTAIN.

Threading
---------

A single `TetherWatcher` instance owns:
  - one `watchdog.Observer` thread (file events)
  - one `pynput.GlobalHotKeys` listener thread
  - one worker thread that drains a queue of pending photos (so file events
    and hotkeys never block on slow operations like OCR)

Public API
----------

Configure via constructor or `update_config(...)`. State changes are pushed
through `on_event(event_dict)`. Caller (app.py or CLI) renders the UI.

Event dict shapes:
  {"type": "started"}
  {"type": "magazine_opened", "name": "...", "path": "..."}
  {"type": "photo_added", "name": "...", "magazine": "...", "count": N}
  {"type": "magazine_closed", "name": "...", "photos": N}
  {"type": "verdict", "name": "...", "kind": "UNIQUE|DUPLICATE|UNCERTAIN",
   "match": {...} or None}
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    from pynput import keyboard as _pynput_keyboard
except ImportError:
    _pynput_keyboard = None

from PIL import Image

import find_duplicates_local as fdl


SETTINGS_FILENAME = "tether_settings.json"


def _read_shot_time(path: Path) -> Optional[float]:
    """Returns EXIF DateTimeOriginal as a POSIX timestamp, or None.

    Used to route late-arriving photos to the magazine that was open when the
    shutter actually fired, rather than the one that's open right now."""
    try:
        with Image.open(path) as img:
            exif = img.getexif() if hasattr(img, "getexif") else None
            if exif is None:
                return None
            # 36867 = DateTimeOriginal, 36868 = DateTimeDigitized
            raw = exif.get(36867) or exif.get(36868) or exif.get(306)
            if not raw:
                return None
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        return None


def _today_str() -> str:
    return date.today().strftime("%Y_%m_%d")


def _next_magazine_seq(library_dir: Path, prefix: str) -> int:
    """Returns the next available magazine sequence number for today's prefix.

    Looks at existing folders matching `<prefix>_NNNN`; returns 1 if none."""
    if not library_dir.exists():
        return 1
    highest = 0
    for child in library_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(prefix + "_"):
            continue
        suffix = name[len(prefix) + 1:]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def _wait_for_stable_file(path: Path, poll_interval: float = 0.15,
                          stable_reads: int = 3, max_wait: float = 30.0) -> bool:
    """Block until a file's size has been stable for `stable_reads` polls,
    indicating that the writer is done. Returns True on success, False if
    the file vanished or the wait timed out."""
    deadline = time.monotonic() + max_wait
    last_size = -1
    stable_count = 0
    while time.monotonic() < deadline:
        if not path.exists():
            time.sleep(poll_interval)
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            time.sleep(poll_interval)
            continue
        if sz == last_size and sz > 0:
            stable_count += 1
            if stable_count >= stable_reads:
                return True
        else:
            stable_count = 0
            last_size = sz
        time.sleep(poll_interval)
    return False


class _WatchdogHandler(FileSystemEventHandler):
    """Pushes new JPG paths onto the supplied queue. Filtering is done here so
    the worker thread doesn't have to consider non-image events."""

    def __init__(self, queue_in: queue.Queue, accepted_exts: tuple):
        self._q = queue_in
        self._exts = accepted_exts

    def _maybe_enqueue(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() in self._exts and p.is_file():
            self._q.put(str(p))

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_moved(self, event):
        # EOS Utility sometimes writes to a .tmp then renames — catch the
        # rename target.
        if not event.is_directory:
            self._maybe_enqueue(event.dest_path)


class TetherWatcher:
    """Live-tether orchestrator. See module docstring for the workflow."""

    ACCEPTED_EXTS = (".jpg", ".jpeg")

    def __init__(self, *,
                 watch_dir: str,
                 library_dir: str,
                 publication: str = "NME",
                 hotkey: str = "<alt>+n",
                 auto_close_seconds: int = 30,
                 match_threshold: int = 60,
                 cover_check: bool = True,
                 rotation_grace_seconds: int = 3,
                 duplicates_dir: str = "",
                 on_event: Optional[Callable[[dict], None]] = None):
        self.watch_dir = Path(watch_dir).resolve()
        self.library_dir = Path(library_dir).resolve()
        self.publication = publication
        self.hotkey = hotkey
        self.auto_close_seconds = auto_close_seconds
        self.match_threshold = match_threshold
        self.cover_check = cover_check
        self.rotation_grace_seconds = max(0, int(rotation_grace_seconds))
        self.duplicates_dir = duplicates_dir or ""
        self.on_event = on_event or (lambda evt: None)

        self._photo_queue: queue.Queue = queue.Queue()
        self._observer: Optional[Observer] = None
        self._hotkey_listener = None
        self._worker_thread: Optional[threading.Thread] = None
        self._closer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._state_lock = threading.RLock()
        self._current_magazine: Optional[Path] = None
        self._current_photo_count = 0
        self._magazine_opened_at_wall: Optional[float] = None
        self._last_photo_at: Optional[float] = None
        # Previous magazine kept "alive" for a brief drain window so that
        # photos shot just before a rotation still land in the right folder.
        self._previous_magazine: Optional[Path] = None
        self._previous_opened_at_wall: Optional[float] = None
        self._previous_count = 0
        self._previous_drain_until: Optional[float] = None
        self._cache: dict = {}  # filename -> features (shared library cache)
        # Low-confidence crops staged in _review/ pending human decision.
        # Key = filename; value = {review_path, target, route_to_previous,
        #                          is_first_of_current, meta}
        self._pending_reviews: dict = {}

    # ── Public control ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start watching. Idempotent."""
        if self._observer is not None:
            return
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.library_dir.mkdir(parents=True, exist_ok=True)

        # Load library cache (one cache.json at the library_dir root)
        self._cache = fdl.load_cache(self.library_dir)

        # File watcher
        handler = _WatchdogHandler(self._photo_queue, self.ACCEPTED_EXTS)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_dir), recursive=True)
        self._observer.start()

        # Worker drains the queue. We use a sentinel (None) to signal stop.
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="tether-worker", daemon=True)
        self._worker_thread.start()

        # Auto-close watchdog (fires when no photo arrives for N seconds)
        self._closer_thread = threading.Thread(
            target=self._auto_close_loop, name="tether-closer", daemon=True)
        self._closer_thread.start()

        # Global hotkey
        if _pynput_keyboard is not None:
            try:
                self._hotkey_listener = _pynput_keyboard.GlobalHotKeys({
                    self.hotkey: self._on_hotkey,
                })
                self._hotkey_listener.start()
            except Exception as e:
                self._emit({"type": "error",
                            "message": f"hotkey listener failed: {e}"})
        else:
            self._emit({"type": "error",
                        "message": "pynput not installed — hotkey disabled"})

        self._emit({"type": "started"})

    def stop(self) -> None:
        """Stop watching. Finalizes any draining previous magazine and closes
        the current one first."""
        with self._state_lock:
            if self._previous_magazine is not None:
                self._finalize_previous_locked()
        self._close_current_magazine(reason="stop")
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
            self._hotkey_listener = None
        # Push sentinel so worker exits
        self._photo_queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None
        # Save cache one last time
        try:
            fdl.save_cache(self.library_dir, self._cache)
        except Exception:
            pass

    def open_new_magazine(self) -> Optional[str]:
        """Manually open a new magazine (equivalent to pressing the hotkey).
        Returns the new folder name, or None if rotation was rejected (e.g.
        the current magazine is still empty)."""
        return self._open_magazine_internal()

    def close_current_magazine(self) -> None:
        """Manually close the current magazine."""
        self._close_current_magazine(reason="manual")

    def list_current_photos(self) -> list[dict]:
        """Return [{name, magazine}] for photos in the current magazine,
        oldest first. Used by the UI to rebuild the thumbnail strip."""
        with self._state_lock:
            mag = self._current_magazine
        if mag is None:
            return []
        try:
            files = sorted([f for f in mag.iterdir()
                            if f.is_file() and f.suffix.lower() in self.ACCEPTED_EXTS])
        except OSError:
            return []
        return [{"name": f.name, "magazine": mag.name} for f in files]

    def move_magazine_to_duplicates(self, name: str) -> dict:
        """Move a magazine folder out of the library into the duplicates dir.
        Used after the user confirms a DUPLICATE verdict — the photos stay on
        disk so they can be referenced when reselling the magazine, but they
        no longer pollute the library used for future cover comparisons."""
        # Resolve source folder. Accept either bare name or full path.
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = self.library_dir / name
        if not candidate.exists() or not candidate.is_dir():
            return {"success": False, "error": f"Magazine not found: {name}"}
        ddir = Path(self.duplicates_dir).resolve() if self.duplicates_dir \
            else (self.library_dir / "_duplicates")
        ddir.mkdir(parents=True, exist_ok=True)
        dst = ddir / candidate.name
        if dst.exists():
            i = 2
            while (ddir / f"{candidate.name}_{i}").exists():
                i += 1
            dst = ddir / f"{candidate.name}_{i}"
        try:
            shutil.move(str(candidate), str(dst))
        except Exception as e:
            return {"success": False, "error": f"move failed: {e}"}
        # Drop cached features for files that no longer live in the library,
        # so future cover checks don't match against moved-out photos.
        try:
            for f in dst.iterdir():
                if f.is_file():
                    self._cache.pop(f.name, None)
            fdl.save_cache(self.library_dir, self._cache)
        except Exception:
            pass
        return {"success": True, "moved_to": str(dst)}

    def get_state(self) -> dict:
        with self._state_lock:
            return {
                "watching": self._observer is not None,
                "watch_dir": str(self.watch_dir),
                "library_dir": str(self.library_dir),
                "duplicates_dir": self.duplicates_dir,
                "publication": self.publication,
                "hotkey": self.hotkey,
                "current_magazine": (
                    self._current_magazine.name if self._current_magazine else None
                ),
                "current_count": self._current_photo_count,
                "auto_close_seconds": self.auto_close_seconds,
                "cover_check": self.cover_check,
                "rotation_grace_seconds": self.rotation_grace_seconds,
            }

    # ── Review staging ─────────────────────────────────────────────────────

    @property
    def _review_dir(self) -> Path:
        return self.library_dir / "_review"

    def accept_auto_crop(self, name: str) -> dict:
        """Accept the previously-proposed auto crop for a review-staged photo.

        Applies the bgsubtr_bbox from the original detection (not the suspect
        quad), writes the crop to the magazine folder, computes features, and
        emits the deferred cover verdict if this was photo #1.
        """
        with self._state_lock:
            review = self._pending_reviews.pop(name, None)
        if review is None:
            return {"success": False, "error": f"No pending review for {name!r}"}
        meta = review.get("meta", {})
        bgsubtr = meta.get("bgsubtr_bbox")
        if bgsubtr is None:
            return {"success": False, "error": "No bgsubtr_bbox in meta — cannot accept"}
        import numpy as _np
        bbox = _np.array(bgsubtr, dtype=_np.float32)
        rect = {
            "x": int(bbox[:, 0].min()), "y": int(bbox[:, 1].min()),
            "w": int(bbox[:, 0].max() - bbox[:, 0].min()),
            "h": int(bbox[:, 1].max() - bbox[:, 1].min()),
        }
        return self._resolve_review(name, review, rect=rect)

    def apply_manual_crop(self, name: str, rect: dict) -> dict:
        """Apply a user-specified axis-aligned crop (rect = {x, y, w, h} in
        pixels of the original staged image) and complete processing."""
        with self._state_lock:
            review = self._pending_reviews.pop(name, None)
        if review is None:
            return {"success": False, "error": f"No pending review for {name!r}"}
        return self._resolve_review(name, review, rect=rect)

    def discard_review(self, name: str) -> dict:
        """Discard a staged review photo (user chose Reshoot). Removes the
        staged file and drops the pending state."""
        with self._state_lock:
            review = self._pending_reviews.pop(name, None)
        if review is None:
            return {"success": False, "error": f"No pending review for {name!r}"}
        try:
            Path(review["review_path"]).unlink()
        except OSError:
            pass
        return {"success": True}

    def _resolve_review(self, name: str, review: dict, *, rect: dict) -> dict:
        """Internal: write a rect crop from the staged file, then finalize."""
        review_path = Path(review["review_path"])
        target = Path(review["target"])
        is_first_of_current = review.get("is_first_of_current", False)

        dst = target / name
        try:
            dims = fdl.apply_rect_crop(str(review_path), str(dst), rect)
        except Exception as e:
            return {"success": False, "error": f"crop failed: {e}"}

        if dims is None:
            # Fallback: just move the staged file to the target
            try:
                shutil.move(str(review_path), str(dst))
            except Exception as e:
                return {"success": False, "error": f"move failed: {e}"}
        else:
            try:
                review_path.unlink()
            except OSError:
                pass

        try:
            self._cache[name] = fdl.compute_features(str(dst))
            fdl.save_cache(self.library_dir, self._cache)
        except Exception as e:
            return {"success": False, "error": f"feature compute failed: {e}"}

        if is_first_of_current and self.cover_check:
            try:
                verdict = self._compare_cover_to_library(target, dst)
                self._emit({"type": "verdict",
                            "name": target.name,
                            "kind": verdict["kind"],
                            "match": verdict.get("match"),
                            "details": verdict.get("details"),
                            "cover_only": True})
            except Exception as e:
                self._emit({"type": "error",
                            "message": f"cover check failed for {target.name}: {e}"})

        return {"success": True}

    # ── Hotkey ─────────────────────────────────────────────────────────────

    def _on_hotkey(self) -> None:
        self._open_magazine_internal()

    def _open_magazine_internal(self) -> Optional[str]:
        """Rotate to a new magazine. Refuses if the current one is empty
        (avoids the 'I tapped the hotkey twice and now I have two empty
        folders' problem). The old magazine moves into a short drain window
        so in-flight photos can still be routed there by EXIF time."""
        now_mono = time.monotonic()
        with self._state_lock:
            if self._current_magazine is not None and self._current_photo_count == 0:
                self._emit({"type": "info",
                            "message": f"Take at least one photo before opening a new magazine "
                                       f"({self._current_magazine.name} is still empty)."})
                return None

            # Move current → previous (drain window). Only happens if current
            # had photos; the empty-guard above already handled the other case.
            if self._current_magazine is not None and self._current_photo_count > 0:
                # If something is already draining, finalize it now to avoid
                # piling up multiple drain magazines.
                if self._previous_magazine is not None:
                    self._finalize_previous_locked()
                self._previous_magazine = self._current_magazine
                self._previous_opened_at_wall = self._magazine_opened_at_wall
                self._previous_count = self._current_photo_count
                self._previous_drain_until = now_mono + self.rotation_grace_seconds

            prefix = f"{_today_str()}_{self.publication}"
            seq = _next_magazine_seq(self.library_dir, prefix)
            name = f"{prefix}_{seq:04d}"
            folder = self.library_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            self._current_magazine = folder
            self._current_photo_count = 0
            self._magazine_opened_at_wall = time.time()
            self._last_photo_at = now_mono

        self._emit({"type": "magazine_opened",
                    "name": name, "path": str(folder)})
        return name

    def _close_current_magazine(self, *, reason: str) -> None:
        """Close the current magazine right now (no drain window). Used by
        stop() and the manual 'Close' button."""
        with self._state_lock:
            mag = self._current_magazine
            count = self._current_photo_count
            self._current_magazine = None
            self._current_photo_count = 0
            self._magazine_opened_at_wall = None
            self._last_photo_at = None
        if mag is None or count == 0:
            return
        self._emit({"type": "magazine_closed", "name": mag.name, "photos": count})
        # When cover_check is on, the verdict already fired on photo #1.
        # Otherwise fall back to the full-magazine comparison at close time.
        if not self.cover_check:
            try:
                verdict = self._compare_magazine_to_library(mag)
                self._emit({"type": "verdict",
                            "name": mag.name,
                            "kind": verdict["kind"],
                            "match": verdict.get("match"),
                            "details": verdict.get("details")})
            except Exception as e:
                self._emit({"type": "error",
                            "message": f"verdict failed for {mag.name}: {e}"})

    def _finalize_previous_locked(self) -> None:
        """Finish the drained 'previous' magazine. Caller holds _state_lock.
        Emits the close event (verdict already fired on its first photo)."""
        mag = self._previous_magazine
        count = self._previous_count
        self._previous_magazine = None
        self._previous_opened_at_wall = None
        self._previous_count = 0
        self._previous_drain_until = None
        if mag is None or count == 0:
            return
        # Release the lock for the emit + (optional) full comparison so we
        # don't hold _state_lock through a slow verdict computation.
        try:
            self._state_lock.release()
            self._emit({"type": "magazine_closed",
                        "name": mag.name, "photos": count})
            if not self.cover_check:
                try:
                    verdict = self._compare_magazine_to_library(mag)
                    self._emit({"type": "verdict",
                                "name": mag.name,
                                "kind": verdict["kind"],
                                "match": verdict.get("match"),
                                "details": verdict.get("details")})
                except Exception as e:
                    self._emit({"type": "error",
                                "message": f"verdict failed for {mag.name}: {e}"})
        finally:
            self._state_lock.acquire()

    # ── Worker (cropping + feature computation) ────────────────────────────

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._photo_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                return  # sentinel
            try:
                self._process_one(Path(item))
            except Exception as e:
                self._emit({"type": "error", "message": f"process failed for {item}: {e}"})

    def _process_one(self, src: Path) -> None:
        """Wait for the file to be fully written, route it to the right
        magazine (using EXIF shutter time to catch in-flight photos that
        arrive after a rotation), then crop and compute features. Fires the
        cover-only verdict when this is the first photo of a new magazine."""
        if not _wait_for_stable_file(src):
            self._emit({"type": "error",
                        "message": f"file never stabilised: {src.name}"})
            return

        # EXIF time is read from the source file before crop (crop_magazine
        # writes a fresh JPEG without EXIF in its output).
        shot_time = _read_shot_time(src)

        # Pick target. Cases:
        #   - No magazine is open at all → auto-open one
        #   - The photo was shot before the current magazine was opened AND a
        #     drained previous magazine is still in its grace window → route
        #     to that previous magazine
        #   - Otherwise → current magazine
        if self._current_magazine is None:
            self._open_magazine_internal()

        with self._state_lock:
            cur = self._current_magazine
            cur_opened = self._magazine_opened_at_wall
            prev = self._previous_magazine
            prev_drain = self._previous_drain_until
            now_mono = time.monotonic()
            route_to_previous = (
                prev is not None
                and prev_drain is not None
                and now_mono < prev_drain
                and shot_time is not None
                and cur_opened is not None
                and shot_time + 0.5 < cur_opened
            )
            target = prev if route_to_previous else cur

        if target is None:
            return  # extremely unlikely

        # Crop into the magazine folder. High-confidence crops delete the
        # original; low-confidence crops stage it in _review/ for human review.
        dst = target / src.name
        crop_ok = False
        meta = None
        try:
            dims, meta = fdl.crop_magazine_with_meta(str(src), str(dst))
            crop_ok = dims is not None
        except Exception as e:
            self._emit({"type": "error",
                        "message": f"crop failed for {src.name}: {e}"})

        confidence = (meta or {}).get("confidence", "low")

        if confidence == "low":
            # Stage original for manual review (non-blocking — worker continues)
            review_dir = self._review_dir
            try:
                review_dir.mkdir(parents=True, exist_ok=True)
                review_dst = review_dir / src.name
                shutil.move(str(src), str(review_dst))
            except Exception as e:
                self._emit({"type": "error",
                            "message": f"review stage failed for {src.name}: {e}"})
                return

            is_first_of_current = False
            with self._state_lock:
                if route_to_previous:
                    self._previous_count += 1
                    count = self._previous_count
                else:
                    self._current_photo_count += 1
                    count = self._current_photo_count
                    self._last_photo_at = time.monotonic()
                    is_first_of_current = (count == 1)
                self._pending_reviews[src.name] = {
                    "review_path": str(review_dst),
                    "target": str(target),
                    "route_to_previous": route_to_previous,
                    "is_first_of_current": is_first_of_current,
                    "meta": meta or {},
                }

            self._emit({"type": "photo_added",
                        "name": src.name,
                        "magazine": target.name,
                        "count": count,
                        "routed_late": route_to_previous})
            self._emit({
                "type": "crop_review",
                "name": src.name,
                "magazine": target.name,
                "proposed_quad": (meta or {}).get("proposed_quad"),
                "bgsubtr_bbox": (meta or {}).get("bgsubtr_bbox"),
                "frame_size": (meta or {}).get("frame_size"),
                "reasons": (meta or {}).get("reasons", []),
            })
            # If this was photo #1 of a new magazine, emit a pending verdict
            if is_first_of_current and self.cover_check:
                self._emit({"type": "verdict",
                            "name": target.name,
                            "kind": "PENDING_REVIEW",
                            "match": None,
                            "details": None,
                            "cover_only": True,
                            "pending_crop": src.name})
            return

        if not crop_ok:
            try:
                shutil.move(str(src), str(dst))
            except Exception as e:
                self._emit({"type": "error",
                            "message": f"move failed for {src.name}: {e}"})
                return
        else:
            try:
                src.unlink()
            except OSError:
                pass

        # Compute features and stash in cache (keyed by filename so the
        # existing duplicate-finder cache lookup keeps working).
        try:
            self._cache[src.name] = fdl.compute_features(str(dst))
        except Exception as e:
            self._emit({"type": "error",
                        "message": f"feature compute failed for {src.name}: {e}"})

        is_first_of_current = False
        with self._state_lock:
            if route_to_previous:
                self._previous_count += 1
                count = self._previous_count
            else:
                self._current_photo_count += 1
                count = self._current_photo_count
                self._last_photo_at = time.monotonic()
                is_first_of_current = (count == 1)
        self._emit({"type": "photo_added",
                    "name": src.name,
                    "magazine": target.name,
                    "count": count,
                    "routed_late": route_to_previous})

        # Persist cache periodically (cheap; small JSON file)
        try:
            fdl.save_cache(self.library_dir, self._cache)
        except Exception:
            pass

        # Cover-only verdict: fires the moment the front cover lands.
        if is_first_of_current and self.cover_check:
            try:
                verdict = self._compare_cover_to_library(target, dst)
                self._emit({"type": "verdict",
                            "name": target.name,
                            "kind": verdict["kind"],
                            "match": verdict.get("match"),
                            "details": verdict.get("details"),
                            "cover_only": True})
            except Exception as e:
                self._emit({"type": "error",
                            "message": f"cover check failed for {target.name}: {e}"})

    # ── Auto-close after idle ──────────────────────────────────────────────

    def _auto_close_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.5)
            now = time.monotonic()
            # Finalize the drained previous magazine once its grace window expires.
            with self._state_lock:
                drain_expired = (
                    self._previous_magazine is not None
                    and self._previous_drain_until is not None
                    and now >= self._previous_drain_until
                )
                if drain_expired:
                    self._finalize_previous_locked()
            # Idle auto-close of the current magazine.
            with self._state_lock:
                last = self._last_photo_at
                open_mag = self._current_magazine is not None
                count = self._current_photo_count
            if open_mag and last is not None and count > 0:
                if now - last >= self.auto_close_seconds:
                    self._close_current_magazine(reason="auto-timeout")

    # ── Duplicate verdict ──────────────────────────────────────────────────

    def _compare_cover_to_library(self, new_mag: Path, cover_path: Path) -> dict:
        """Compare a single cover photo against the first photo of every other
        magazine in the library. The 'first photo' is the first JPG in the
        folder by name sort — EOS Utility numbers photos in shutter order, so
        this is the front cover.

        Cover-only matching deliberately ignores interior pages (colofon,
        masthead, page index, etc.) because those frequently repeat across
        unrelated magazines and cause false-positive duplicates."""
        import imagehash

        new_feat = self._cache.get(cover_path.name)
        if new_feat is None:
            new_feat = fdl.compute_features(str(cover_path))
            self._cache[cover_path.name] = new_feat

        def parse_h(s):
            return imagehash.hex_to_hash(s) if s else None

        n_ph = parse_h(new_feat.get("phash"))
        n_dh = parse_h(new_feat.get("dhash"))
        n_dates = {tuple(d) for d in (new_feat.get("dates") or [])}

        siblings = [
            p for p in self.library_dir.iterdir()
            if p.is_dir() and p != new_mag and not p.name.startswith(".")
            and not p.name.startswith("_")
        ]

        best_score: Optional[int] = None
        best_sib: Optional[Path] = None
        best_detail: Optional[dict] = None
        for sib in siblings:
            try:
                files = sorted([f for f in sib.iterdir()
                                if f.is_file() and f.suffix.lower() in self.ACCEPTED_EXTS])
            except OSError:
                continue
            if not files:
                continue
            cover = files[0]
            feat = self._cache.get(cover.name)
            if feat is None:
                try:
                    feat = fdl.compute_features(str(cover))
                    self._cache[cover.name] = feat
                except Exception:
                    continue
            s_ph = parse_h(feat.get("phash"))
            s_dh = parse_h(feat.get("dhash"))
            p_d = (n_ph - s_ph) if (n_ph is not None and s_ph is not None) else 999
            d_d = (n_dh - s_dh) if (n_dh is not None and s_dh is not None) else 999
            score = min(p_d, d_d)
            s_dates = {tuple(d) for d in (feat.get("dates") or [])}
            date_match = bool(n_dates & s_dates)
            if date_match and score < 999:
                # Shared cover date is a strong corroborating signal — promote.
                score = max(0, score - 30)
            if best_score is None or score < best_score:
                best_score = score
                best_sib = sib
                best_detail = {"phash": int(p_d), "dhash": int(d_d),
                               "date_match": date_match,
                               "shared_dates": sorted(list(n_dates & s_dates))}

        if best_score is None or best_score >= self.match_threshold:
            return {"kind": "UNIQUE",
                    "details": f"closest cover distance "
                               f"{best_score if best_score is not None else 'n/a'}"
                               f"; threshold {self.match_threshold}"}
        if best_score <= 25 or (best_detail and best_detail.get("date_match")):
            kind = "DUPLICATE"
        else:
            kind = "UNCERTAIN"
        return {
            "kind": kind,
            "match": {
                "name": best_sib.name if best_sib else None,
                "score": int(best_score),
                "detail": best_detail,
            },
            "details": (
                f"closest cover: {best_sib.name if best_sib else 'n/a'} "
                f"@ score {best_score}"
            ),
        }

    def _compare_magazine_to_library(self, new_mag: Path) -> dict:
        """Compute distances between `new_mag` and every other magazine folder
        in the library. Returns a verdict dict."""
        # Build temporary "images" list for analysis. Each subfolder of
        # library_dir is one item; the new magazine is one item too.
        siblings = [
            p for p in self.library_dir.iterdir()
            if p.is_dir() and p != new_mag and not p.name.startswith(".")
            and not p.name.startswith("_")
        ]

        def features_for(folder: Path) -> list:
            """Returns sorted list of feature dicts for photos in this folder."""
            files = sorted([f for f in folder.iterdir()
                            if f.is_file() and f.suffix.lower() in self.ACCEPTED_EXTS])
            out = []
            for f in files:
                feat = self._cache.get(f.name)
                if feat is None:
                    try:
                        feat = fdl.compute_features(str(f))
                        self._cache[f.name] = feat
                    except Exception:
                        continue
                out.append(feat)
            return out

        new_feats = features_for(new_mag)
        if not new_feats:
            return {"kind": "UNCERTAIN",
                    "details": "no photos detected in the new magazine"}

        import imagehash
        def parse_h(s):
            return imagehash.hex_to_hash(s) if s else None

        best_score = None  # smaller is better
        best_sibling = None
        best_detail = None
        for sib in siblings:
            sib_feats = features_for(sib)
            if not sib_feats:
                continue
            # Compare every photo of new vs every photo of sib; take the best
            # (smallest) min of (pHash distance, dHash distance).
            local_best = None
            local_detail = None
            for nf in new_feats:
                nph = parse_h(nf.get("phash"))
                ndh = parse_h(nf.get("dhash"))
                if nph is None and ndh is None:
                    continue
                for sf in sib_feats:
                    sph = parse_h(sf.get("phash"))
                    sdh = parse_h(sf.get("dhash"))
                    p_d = (nph - sph) if (nph is not None and sph is not None) else 999
                    d_d = (ndh - sdh) if (ndh is not None and sdh is not None) else 999
                    pair_best = min(p_d, d_d)
                    if local_best is None or pair_best < local_best:
                        local_best = pair_best
                        local_detail = {"phash": p_d, "dhash": d_d}
            # Date confirmation: if the new and sib share any extracted date,
            # boost confidence.
            new_dates = set()
            sib_dates = set()
            for f in new_feats:
                for d in f.get("dates") or []:
                    new_dates.add(tuple(d))
            for f in sib_feats:
                for d in f.get("dates") or []:
                    sib_dates.add(tuple(d))
            date_match = bool(new_dates & sib_dates)
            score = local_best if local_best is not None else 999
            if date_match and score < 999:
                # Halve effective score — strong promotion
                score = max(0, score - 30)
            if best_score is None or score < best_score:
                best_score = score
                best_sibling = sib
                best_detail = {**(local_detail or {}),
                               "date_match": date_match,
                               "shared_dates": sorted(
                                   list(new_dates & sib_dates))}

        if best_score is None or best_score >= self.match_threshold:
            return {"kind": "UNIQUE",
                    "details": f"closest neighbour distance "
                               f"{best_score if best_score is not None else 'n/a'}"
                               f"; threshold {self.match_threshold}"}
        if best_score <= 25 or (best_detail and best_detail.get("date_match")):
            kind = "DUPLICATE"
        else:
            kind = "UNCERTAIN"
        return {
            "kind": kind,
            "match": {
                "name": best_sibling.name if best_sibling else None,
                "score": best_score,
                "detail": best_detail,
            },
            "details": (
                f"closest match: {best_sibling.name if best_sibling else 'n/a'} "
                f"@ score {best_score}"
            ),
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _emit(self, evt: dict) -> None:
        try:
            self.on_event(evt)
        except Exception:
            pass


# ─── Persistence of tether settings ─────────────────────────────────────────

def load_settings(path: Optional[Path] = None) -> dict:
    """Returns persisted tether settings (or sensible defaults)."""
    p = path or _default_settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "watch_dir": "",
        "library_dir": "",
        "duplicates_dir": "",
        "publication": "NME",
        "hotkey": "<alt>+n",
        "auto_close_seconds": 30,
        "match_threshold": 60,
        "cover_check": True,
        "rotation_grace_seconds": 3,
    }


def save_settings(settings: dict, path: Optional[Path] = None) -> None:
    p = path or _default_settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _default_settings_path() -> Path:
    """Where to persist tether settings — user-local app data dir."""
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return Path(base) / "DuplicateFinder" / SETTINGS_FILENAME
