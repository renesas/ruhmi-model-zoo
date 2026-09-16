# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Project-level helpers for the Arm ML-zoo RNNoise INT8 port.

Currently re-exports the (vendored) DSP front-end so callers can write::

    from utils import RNNoisePreProcess
"""
from .rnnoise_preprocessing import RNNoisePreProcess

__all__ = ["RNNoisePreProcess"]
