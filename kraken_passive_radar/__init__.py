"""
Kraken Passive Radar - Passive bistatic radar system using KrakenSDR.

This package provides:
- Real-time radar displays (PPI, Range-Doppler)
- Calibration and monitoring panels
- Integration with GNU Radio signal processing blocks

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

__version__ = "0.1.0"
__author__ = "Dr Robert W McGwier, PhD"

from .radar_display import PPIDisplay, PPIDetection, PPITrack, PPIDisplayParams
from .range_doppler_display import (
    RangeDopplerDisplay,
    Detection,
    Track,
    RDDisplayParams,
)
from .calibration_panel import CalibrationPanel, CalibrationPanelParams, CalibrationStatus
from .radar_gui import RadarGUI

__all__ = [
    # Version
    "__version__",
    # PPI Display
    "PPIDisplay",
    "PPIDetection",
    "PPITrack",
    "PPIDisplayParams",
    # Range-Doppler Display
    "RangeDopplerDisplay",
    "Detection",
    "Track",
    "RDDisplayParams",
    # Calibration
    "CalibrationPanel",
    "CalibrationPanelParams",
    "CalibrationStatus",
    # Main GUI
    "RadarGUI",
]
