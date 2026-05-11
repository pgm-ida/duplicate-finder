#!/usr/bin/env python3
"""
Magazine Duplicates — desktop app.

Wraps the existing find_duplicates_local pipeline in a native window using
pywebview. Lets the user pick a folder, set the match threshold, run the
analysis, and view the same dark-themed report inside an app window.

Run with:   python app.py
Package:    pyinstaller --onefile --windowed --add-data "app_index.html;." app.py
"""

import http.server
import os
import socket
import socketserver
import sys
import threading
from pathlib import Path

import webview


def _configure_tesseract():
    """Point pytesseract at a bundled (PyInstaller) tesseract.exe if present.

    PyInstaller extracts data files to `sys._MEIPASS` at runtime. We bundle the
    tesseract binary + tessdata into the .exe; this function finds it before
    `find_duplicates_local` (which imports pytesseract) is loaded.
    """
    if not getattr(sys, "frozen", False):
        return  # Running from source — use whatever tesseract is on PATH.
    base = Path(getattr(sys, "_MEIPASS", "."))
    bundled = base / "tesseract" / "tesseract.exe"
    if bundled.exists():
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = str(bundled)
            tessdata = base / "tesseract" / "tessdata"
            if tessdata.exists():
                os.environ["TESSDATA_PREFIX"] = str(tessdata)
        except ImportError:
            pass


_configure_tesseract()
import find_duplicates_local as fdl
import tether as _tether


# ─── Local HTTP server for serving the report ────────────────────────────────
#
# Edge WebView2 (and most modern webviews) block iframe loads of file:// URLs
# from another file:// page for security. To get around this, we run a tiny
# local HTTP server and point the iframe at http://localhost:PORT/... instead.
# The server's served-directory is updated each time the user picks a folder.

class _DynamicHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with two configurable roots:
      - `served_directory` (the analyze report folder) for everything
      - `library_directory` for requests under the `/lib/` prefix
        (used by Shoot mode to show thumbnails of the current magazine)

    Both roots can be updated at runtime via class attributes."""

    served_directory = "."
    library_directory: "str | None" = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.__class__.served_directory, **kwargs)

    def translate_path(self, path):
        from urllib.parse import unquote
        cls = self.__class__
        raw = path.split("?", 1)[0].split("#", 1)[0]
        if cls.library_directory and raw.startswith("/lib/"):
            rel = unquote(raw[len("/lib/"):]).lstrip("/\\")
            # Disallow absolute paths or "..": normpath collapses ".." segments
            # which is enough for our local app context.
            norm = os.path.normpath(rel)
            if norm.startswith("..") or os.path.isabs(norm):
                return cls.library_directory
            return os.path.join(cls.library_directory, norm)
        # Refresh directory on every request so set_directory() takes effect.
        self.directory = cls.served_directory
        return super().translate_path(path)

    def log_message(self, format, *args):
        # Suppress per-request console spam
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ReportServer:
    def __init__(self):
        self.port = _free_port()
        self.server = socketserver.ThreadingTCPServer(
            ("127.0.0.1", self.port), _DynamicHandler
        )
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def set_directory(self, path: str):
        _DynamicHandler.served_directory = str(path)

    def set_library_directory(self, path: str):
        _DynamicHandler.library_directory = str(path) if path else None

    def url_for(self, relative_path: str) -> str:
        return f"http://127.0.0.1:{self.port}/{relative_path}"

    def lib_url_for(self, relative_path: str) -> str:
        # Forward slashes only in URLs, even on Windows.
        rel = str(relative_path).replace("\\", "/").lstrip("/")
        return f"http://127.0.0.1:{self.port}/lib/{rel}"


class Api:
    """JavaScript ↔ Python bridge exposed to the webview."""

    def __init__(self, server: ReportServer):
        self.window = None
        self.server = server
        self.last_folder = None
        self.last_progress = {"phase": "idle", "n": 0, "total": 0}
        self._tether_watcher: "_tether.TetherWatcher | None" = None
        self._tether_settings = _tether.load_settings()

    # ── Folder picker ─────────────────────────────────────────────────────
    def pick_folder(self):
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            folder = result[0] if isinstance(result, (list, tuple)) else result
            self.last_folder = folder
            return folder
        return None

    # ── Analysis ──────────────────────────────────────────────────────────
    def analyze(self, folder, threshold, pages_per_magazine=4):
        """Run the full analysis. Returns dict with file:// url to the report."""
        try:
            self.last_folder = folder

            def progress(phase, n, total):
                self.last_progress = {"phase": phase, "n": n, "total": total}
                # Push update to JS
                self._push_progress(phase, n, total)

            images, groups, stats = fdl.analyze_folder(
                folder, threshold=int(threshold),
                pages_per_magazine=int(pages_per_magazine),
                progress_cb=progress,
            )
            if "error" in stats:
                return {"success": False, "error": stats["error"]}

            # Point the local HTTP server at this folder so the iframe can load
            # the report (and its thumbnails) via http://localhost:PORT/...
            folder_resolved = Path(folder).resolve()
            self.server.set_directory(folder_resolved)
            url = self.server.url_for(fdl.HTML_REPORT_FILE)

            return {
                "success": True,
                "report_url": url,
                "stats": stats,
            }
        except Exception as e:
            import traceback
            return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}

    # ── Open report folder in OS file explorer ──────────────────────────
    def open_in_explorer(self, folder):
        try:
            import subprocess
            if sys.platform == "win32":
                subprocess.Popen(["explorer", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
            return True
        except Exception:
            return False

    # ── Internal: push progress to JS ───────────────────────────────────
    def _push_progress(self, phase, n, total):
        if self.window is None:
            return
        try:
            self.window.evaluate_js(
                f"window.onProgress && window.onProgress({n!r}, {total!r}, {phase!r})"
            )
        except Exception:
            # Window might not be ready, ignore
            pass

    # ── Tether (live shoot mode) ───────────────────────────────────────
    def get_tether_settings(self):
        return dict(self._tether_settings)

    def save_tether_settings(self, settings):
        try:
            merged = dict(self._tether_settings)
            merged.update(settings or {})
            _tether.save_settings(merged)
            self._tether_settings = merged
            return {"success": True, "settings": merged}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def pick_tether_dir(self, kind):
        """kind = 'watch' | 'library' | 'duplicates'. Opens a folder picker."""
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        folder = result[0] if isinstance(result, (list, tuple)) else result
        key_map = {
            "watch": "watch_dir",
            "library": "library_dir",
            "duplicates": "duplicates_dir",
        }
        key = key_map.get(kind)
        if key is None:
            return folder
        self._tether_settings[key] = folder
        try:
            _tether.save_settings(self._tether_settings)
        except Exception:
            pass
        return folder

    def start_tether(self, overrides=None):
        """Start the watcher with the persisted settings (optionally overridden)."""
        if self._tether_watcher is not None:
            return {"success": False, "error": "Tether is already running"}
        cfg = dict(self._tether_settings)
        if overrides:
            cfg.update(overrides)
        if not cfg.get("watch_dir") or not cfg.get("library_dir"):
            return {"success": False, "error": "Pick a watch folder and a library folder first"}
        try:
            self._tether_watcher = _tether.TetherWatcher(
                watch_dir=cfg["watch_dir"],
                library_dir=cfg["library_dir"],
                publication=cfg.get("publication") or "NME",
                hotkey=cfg.get("hotkey") or "<alt>+n",
                auto_close_seconds=int(cfg.get("auto_close_seconds") or 30),
                match_threshold=int(cfg.get("match_threshold") or 60),
                cover_check=bool(cfg.get("cover_check", True)),
                rotation_grace_seconds=int(cfg.get("rotation_grace_seconds") or 3),
                duplicates_dir=cfg.get("duplicates_dir") or "",
                on_event=self._push_tether_event,
            )
            self._tether_watcher.start()
            # Point the HTTP server's /lib/ prefix at the library so the UI
            # can render thumbnails of the current magazine's photos.
            self.server.set_library_directory(cfg["library_dir"])
            return {"success": True, "state": self._tether_watcher.get_state(),
                    "lib_url_base": self.server.lib_url_for("")}
        except Exception as e:
            import traceback
            self._tether_watcher = None
            return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}

    def stop_tether(self):
        if self._tether_watcher is None:
            return {"success": True}
        try:
            self._tether_watcher.stop()
        finally:
            self._tether_watcher = None
        return {"success": True}

    def tether_open_new_magazine(self):
        if self._tether_watcher is None:
            return {"success": False, "error": "Not running"}
        name = self._tether_watcher.open_new_magazine()
        return {"success": True, "name": name}

    def tether_close_magazine(self):
        if self._tether_watcher is None:
            return {"success": False, "error": "Not running"}
        self._tether_watcher.close_current_magazine()
        return {"success": True}

    def tether_state(self):
        if self._tether_watcher is None:
            return {"running": False, "settings": self._tether_settings}
        return {"running": True, "state": self._tether_watcher.get_state(),
                "settings": self._tether_settings,
                "lib_url_base": self.server.lib_url_for("")}

    def tether_list_current_photos(self):
        """Returns [{name, magazine, url}] for the photos in the open magazine,
        so the UI can rebuild the thumbnail strip on reload."""
        if self._tether_watcher is None:
            return []
        photos = self._tether_watcher.list_current_photos()
        return [{
            "name": p["name"],
            "magazine": p["magazine"],
            "url": self.server.lib_url_for(f"{p['magazine']}/{p['name']}"),
        } for p in photos]

    def tether_move_to_duplicates(self, magazine_name):
        """Move a magazine folder out of the library into the duplicates dir.
        Called after the user confirms a DUPLICATE verdict."""
        if self._tether_watcher is None:
            return {"success": False, "error": "Tether not running"}
        return self._tether_watcher.move_magazine_to_duplicates(magazine_name)

    def _push_tether_event(self, evt):
        """Called from a tether worker thread → marshal to JS."""
        if self.window is None:
            return
        try:
            # Convert numpy types (from imagehash distance) to plain Python ints
            # so they're JSON-serialisable inside evaluate_js.
            import json as _json
            payload = _json.dumps(_sanitize_event(evt))
            # Use a literal JS function call; pywebview will marshal the string.
            self.window.evaluate_js(
                f"window.onTetherEvent && window.onTetherEvent({payload})"
            )
        except Exception:
            pass


def _sanitize_event(evt):
    """Recursively convert non-JSON types (numpy int/float/bool, Path) to
    plain Python types."""
    import numpy as _np
    if isinstance(evt, dict):
        return {k: _sanitize_event(v) for k, v in evt.items()}
    if isinstance(evt, (list, tuple)):
        return [_sanitize_event(x) for x in evt]
    if isinstance(evt, _np.integer):
        return int(evt)
    if isinstance(evt, _np.floating):
        return float(evt)
    if isinstance(evt, _np.bool_):
        return bool(evt)
    if isinstance(evt, Path):
        return str(evt)
    return evt


def _resource_dir() -> Path:
    """Where bundled resources live. Differs between dev and PyInstaller frozen."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.resolve()


def main():
    server = ReportServer()
    api = Api(server)
    html_file = _resource_dir() / "app_index.html"
    if not html_file.exists():
        print(f"Missing UI file: {html_file}")
        sys.exit(1)

    window = webview.create_window(
        title="Duplicate",
        url=str(html_file),
        js_api=api,
        width=1500,
        height=950,
        min_size=(900, 600),
        background_color="#0d0d0f",
    )
    api.window = window
    webview.start(debug=False)


if __name__ == "__main__":
    main()
