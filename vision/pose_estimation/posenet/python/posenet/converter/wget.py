# Copyright 2018 Ross Wightman  (rwightman/posenet-pytorch, Apache-2.0)
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""TFJS weight downloader for PoseNet."""
import json
import os
import posixpath

import requests

from posenet.constants import *   # noqa: F401,F403  (just for package init)
from posenet.models.mobilenet_v1 import MOBILENET_V1_CHECKPOINTS  # noqa: F401

GOOGLE_CLOUD_STORAGE_DIR = "https://storage.googleapis.com/tfjs-models/weights/posenet/"


def download_json(checkpoint, filename, base_dir):
    url = posixpath.join(GOOGLE_CLOUD_STORAGE_DIR, checkpoint, filename)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = json.loads(response.content)
    out = os.path.join(base_dir, checkpoint, filename)
    with open(out, "w") as f:
        json.dump(data, f)


def download_file(checkpoint, filename, base_dir):
    url = posixpath.join(GOOGLE_CLOUD_STORAGE_DIR, checkpoint, filename)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    out = os.path.join(base_dir, checkpoint, filename)
    with open(out, "wb") as f:
        f.write(response.content)


def download(checkpoint, base_dir="./weights/"):
    """Download a TFJS PoseNet checkpoint into ``base_dir/<checkpoint>/``."""
    save_dir = os.path.join(base_dir, checkpoint)
    os.makedirs(save_dir, exist_ok=True)

    manifest_name = "manifest.json"
    download_json(checkpoint, manifest_name, base_dir)

    with open(os.path.join(save_dir, manifest_name)) as f:
        manifest = json.load(f)

    # Show a progress bar over manifest variables when tqdm is available.
    try:
        from tqdm import tqdm
        items = tqdm(list(manifest.items()),
                     desc=f"  fetching {checkpoint}",
                     unit="file", ncols=80, leave=True)
    except ImportError:
        items = manifest.items()

    for _var, meta in items:
        filename = meta["filename"]
        if not os.path.exists(os.path.join(save_dir, filename)):
            download_file(checkpoint, filename, base_dir)
