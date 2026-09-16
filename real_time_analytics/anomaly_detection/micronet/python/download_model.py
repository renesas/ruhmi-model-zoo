#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download ad_medium_int8.tflite (ARM MicroNet anomaly detection).

The upstream model is already full INT8, so this script has a single stage:
download the pinned TFLite blob to ``model/ad_medium_int8.tflite``.

Pipeline:
1. Resolve the target path under ``<script_dir>/model``.
2. Skip the download when the file already exists (unless --force-download).
3. Stream the file with a tqdm progress bar (fallback: plain byte counter).

Source (pinned commit, do not change without re-validating inference.py):
    https://github.com/Arm-Examples/ML-zoo/tree/
    7c32b097f7d94aae2cd0b98a8ed5a3ba81e66b18/models/anomaly_detection

Usage:
    python download_model.py                # download if missing

"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# --------------------------------------------------------------------------- #
# Paths and remote URL (pinned ML-zoo commit).                                #
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = SCRIPT_DIR / "model"
INT8_PATH = OUTPUT_DIR / "ad_medium_int8.tflite"

MODEL_URL = (
    "https://github.com/Arm-Examples/ML-zoo/raw/"
    "7c32b097f7d94aae2cd0b98a8ed5a3ba81e66b18/"
    "models/anomaly_detection/micronet_medium/tflite_int8/"
    "ad_medium_int8.tflite?download=true"
)

# SHA-256 of the pinned ML-zoo commit blob. 
EXPECTED_INT8_SHA256 = (
    "a8b1c9037c2a80e6ff770f0a550777cf744a52850bed6545b2bfc9bacf604c98"
)


# --------------------------------------------------------------------------- #
# Download helpers.                                                           #
# --------------------------------------------------------------------------- #

def _sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path`` streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected_hex: str) -> None:
    """Raise RuntimeError if ``path`` does not match ``expected_hex``."""
    actual = _sha256_of(path)
    if actual.lower() != expected_hex.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path}\n"
            f"  expected : {expected_hex}\n"
            f"  actual   : {actual}\n"
            "Delete the file and re-download, or pass --allow-unsupported-model"
            " to skip verification (not recommended for production)."
        )


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a tqdm progress bar (or plain fallback)."""
    print(f"Downloading {url}\n         -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        if _HAS_TQDM:
            class _Reporter:
                def __init__(self):
                    self.pbar = None

                def __call__(self, block_num, block_size, total_size):
                    # total_size < 0 means the server did not send Content-Length.
                    if self.pbar is None:
                        self.pbar = tqdm(
                            total=total_size if total_size > 0 else None,
                            unit="B",
                            unit_scale=True,
                            desc="  download",
                        )
                    self.pbar.update(block_size)
                    if total_size > 0 and block_num * block_size >= total_size:
                        self.pbar.close()

            urllib.request.urlretrieve(url, str(dest), reporthook=_Reporter())
        else:
            def _report(block_num, block_size, total_size):
                current = block_num * block_size
                if total_size > 0:
                    print(
                        f"\r  {current / 1e6:.1f} / {total_size / 1e6:.1f} MB",
                        end="",
                    )
                else:
                    print(f"\r  {current / 1e6:.1f} MB", end="")

            urllib.request.urlretrieve(url, str(dest), reporthook=_report)
            print()
    except OSError as exc:
        # Covers HTTPError, URLError, TimeoutError, ConnectionResetError, ...
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def download_int8_model(force: bool = False,
                        verify: bool = True) -> Path:
    """Fetch the pinned INT8 TFLite model, reusing an existing file when possible.

    Args:
        force:  When True, re-download even if the file already exists.
        verify: When True (default), require the downloaded / cached file to
                match EXPECTED_INT8_SHA256. On mismatch, the file is left in
                place and a RuntimeError is raised so the user can inspect it.

    Returns:
        Path to the downloaded (or previously cached) TFLite file.
    """
    if INT8_PATH.exists() and not force:
        size_mb = INT8_PATH.stat().st_size / (1024 * 1024)
        print(f"[OK] INT8 model already exists: {INT8_PATH} ({size_mb:.2f} MB)")
        if verify:
            _verify_sha256(INT8_PATH, EXPECTED_INT8_SHA256)
            print("[OK] SHA-256 verified against pinned commit.")
        return INT8_PATH

    _download_with_progress(MODEL_URL, INT8_PATH)
    size_mb = INT8_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] Downloaded INT8 model: {INT8_PATH} ({size_mb:.2f} MB)")
    if verify:
        _verify_sha256(INT8_PATH, EXPECTED_INT8_SHA256)
        print("[OK] SHA-256 verified against pinned commit.")
    return INT8_PATH


# --------------------------------------------------------------------------- #
# CLI plumbing.                                                               #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the INT8 model even if it already exists.",
    )
    parser.add_argument(
        "--allow-unsupported-model",
        action="store_true",
        help=(
            "Skip the SHA-256 integrity check. Only use when intentionally"
            " swapping in a non-pinned model; the local inference pipeline"
            " may not match."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: download the INT8 TFLite model."""
    args = parse_args()
    download_int8_model(
        force=args.force_download,
        verify=not args.allow_unsupported_model,
    )
    print("[DONE] Pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
