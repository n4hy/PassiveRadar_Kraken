"""
Kraken Passive Radar - Passive bistatic radar system using KrakenSDR.

This package provides:
- Real-time radar displays (PPI, Range-Doppler)
- Calibration and monitoring panels
- Integration with GNU Radio signal processing blocks
- GPU acceleration support (optional, runtime selectable)

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

__version__ = "0.2.0"
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

# Remote display modules
from .remote_display import RemoteRadarDisplay
from .local_processing import (
    CfarDetector,
    DetectionClusterer,
    KalmanFilter,
    MultiTargetTracker,
    LocalProcessingPipeline,
    Detection as LocalDetection,
    Track as LocalTrack,
    TrackStatus,
)
from .enhanced_remote_display import EnhancedRemoteRadarDisplay

# GPU Backend (optional - only if GPU libraries installed)
try:
    from .gpu_backend import (
        set_processing_backend,
        get_active_backend,
        is_gpu_available,
        get_gpu_info,
        GPUBackend,
    )
    GPU_AVAILABLE = True
except ImportError:
    # CPU-only mode (RPi5) - provide stub functions for API compatibility
    GPU_AVAILABLE = False

    def set_processing_backend(backend):
        """Stub: GPU not available (CPU-only mode)."""
        if backend != 'cpu':
            import warnings
            warnings.warn("GPU backend requested but not available. Using CPU.", RuntimeWarning)

    def get_active_backend():
        """Stub: Always returns 'cpu' in CPU-only mode."""
        return 'cpu'

    def is_gpu_available():
        """Stub: Always returns False in CPU-only mode."""
        return False

    def get_gpu_info():
        """Stub: Returns empty dict in CPU-only mode."""
        return {}

    class GPUBackend:
        """Stub: Empty GPU backend class for CPU-only mode."""
        @staticmethod
        def is_available():
            return False

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
    # Remote Display
    "RemoteRadarDisplay",
    "EnhancedRemoteRadarDisplay",
    # Local Processing
    "CfarDetector",
    "DetectionClusterer",
    "KalmanFilter",
    "MultiTargetTracker",
    "LocalProcessingPipeline",
    "LocalDetection",
    "LocalTrack",
    "TrackStatus",
    # GPU Backend (available on all platforms, graceful fallback on CPU-only)
    "GPU_AVAILABLE",
    "set_processing_backend",
    "get_active_backend",
    "is_gpu_available",
    "get_gpu_info",
    "GPUBackend",
]
