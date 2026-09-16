# SPDX-License-Identifier: Apache-2.0
"""Shared progress / spinner UI utilities.

Provides a tiny dependency-free spinner context manager used across
download_model.py, inference.py and validate.py to give live feedback
during slow blocking operations.
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time

__all__ = ["spinner"]


@contextlib.contextmanager
def spinner(msg: str, status_fn=None):
    """Render a live ``|/-\\``-rotator while the wrapped block runs.

    Parameters
    ----------
    msg
        Short description of the operation.
    status_fn
        Optional zero-arg callable returning a sub-stage label, rendered
        in ``[brackets]`` after the elapsed time.
    """
    stop = threading.Event()
    chars = "|/-\\"
    t0 = time.time()
    last_len = [0]

    def _spin():
        i = 0
        while not stop.is_set():
            elapsed = time.time() - t0
            extra = ""
            if status_fn is not None:
                try:
                    s = status_fn()
                    if s:
                        extra = f"  [{s}]"
                except Exception:
                    pass
            line = f"  {chars[i % len(chars)]} {msg} ... {elapsed:6.1f}s{extra}"
            pad = " " * max(0, last_len[0] - len(line))
            sys.stdout.write(f"\r{line}{pad}")
            sys.stdout.flush()
            last_len[0] = len(line)
            i += 1
            stop.wait(0.1)

    th = threading.Thread(target=_spin, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join()
        elapsed = time.time() - t0
        line = f"  + {msg} ... done ({elapsed:.1f}s)"
        pad = " " * max(0, last_len[0] - len(line))
        sys.stdout.write(f"\r{line}{pad}\n")
        sys.stdout.flush()
