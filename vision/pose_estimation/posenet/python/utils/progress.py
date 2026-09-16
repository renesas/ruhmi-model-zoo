# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Reusable progress helpers (spinner + tqdm wrappers).

Provides a uniform progress UX across the PoseNet download / inference /
validate scripts. Use spinner() for blocking operations whose progress
cannot be measured (model load, CMake build, MERA deploy) and the tqdm
wrappers for everything iterable.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterable, Optional

from tqdm import tqdm

_SPINNER_FRAMES = ["|", "/", "-", "\\"]


@contextmanager
def spinner(message, *, stream=sys.stderr, interval=0.1):
    """Animated spinner that runs alongside a long-running block.

    Prints "[ok] <message> (<sec>s)" on success or "[fail] ..." on failure
    (exception is re-raised). Falls back to plain text when the stream is
    not a TTY (CI / log files).
    """
    interactive = hasattr(stream, "isatty") and stream.isatty()
    t0 = time.time()

    if not interactive:
        stream.write("  ... " + message + "...\n")
        stream.flush()
        try:
            yield
        except Exception:
            stream.write("  [fail] " + message + " ({:.1f}s)\n".format(time.time() - t0))
            stream.flush()
            raise
        stream.write("  [ok] " + message + " ({:.1f}s)\n".format(time.time() - t0))
        stream.flush()
        return

    stop_evt = threading.Event()

    def _spin():
        i = 0
        while not stop_evt.is_set():
            stream.write("\r  " + _SPINNER_FRAMES[i % 4] + "  " + message + "...  ")
            stream.flush()
            i += 1
            time.sleep(interval)

    th = threading.Thread(target=_spin, daemon=True)
    th.start()
    try:
        yield
    except Exception:
        stop_evt.set()
        th.join(timeout=0.5)
        stream.write("\r  [fail] " + message + " ({:.1f}s)".format(time.time() - t0) + " " * 30 + "\n")
        stream.flush()
        raise
    stop_evt.set()
    th.join(timeout=0.5)
    stream.write("\r  [ok] " + message + " ({:.1f}s)".format(time.time() - t0) + " " * 30 + "\n")
    stream.flush()


def progress_bar(total, desc="", unit="it", ncols=80, leave=True):
    """Manual-update tqdm bar with the project's default styling."""
    return tqdm(total=total, desc=desc, unit=unit, ncols=ncols, leave=leave)


def progress_iter(iterable, desc="", unit="it", ncols=80, leave=True, total=None):
    """Iterator-style tqdm bar with the project's default styling."""
    return tqdm(iterable, desc=desc, unit=unit, ncols=ncols, leave=leave, total=total)


def download_with_progress(url, dest_path, desc=None, chunk_size=8192):
    """Stream-download url to dest_path with a tqdm byte bar."""
    import urllib.request
    from pathlib import Path

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest_path.name

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with open(dest_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=label, ncols=80, leave=False) as bar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
