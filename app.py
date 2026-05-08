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
import socket
import socketserver
import sys
import threading
from pathlib import Path

import webview

import find_duplicates_local as fdl


# ─── Local HTTP server for serving the report ────────────────────────────────
#
# Edge WebView2 (and most modern webviews) block iframe loads of file:// URLs
# from another file:// page for security. To get around this, we run a tiny
# local HTTP server and point the iframe at http://localhost:PORT/... instead.
# The server's served-directory is updated each time the user picks a folder.

class _DynamicHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that reads its directory from a class attribute
    (so we can change it after the server is created)."""

    served_directory = "."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.__class__.served_directory, **kwargs)

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

    def url_for(self, relative_path: str) -> str:
        return f"http://127.0.0.1:{self.port}/{relative_path}"


class Api:
    """JavaScript ↔ Python bridge exposed to the webview."""

    def __init__(self, server: ReportServer):
        self.window = None
        self.server = server
        self.last_folder = None
        self.last_progress = {"phase": "idle", "n": 0, "total": 0}

    # ── Folder picker ─────────────────────────────────────────────────────
    def pick_folder(self):
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            folder = result[0] if isinstance(result, (list, tuple)) else result
            self.last_folder = folder
            return folder
        return None

    # ── Analysis ──────────────────────────────────────────────────────────
    def analyze(self, folder, threshold):
        """Run the full analysis. Returns dict with file:// url to the report."""
        try:
            self.last_folder = folder

            def progress(phase, n, total):
                self.last_progress = {"phase": phase, "n": n, "total": total}
                # Push update to JS
                self._push_progress(phase, n, total)

            images, groups, stats = fdl.analyze_folder(
                folder, threshold=int(threshold), progress_cb=progress,
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
        title="Magazine Duplicates",
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
