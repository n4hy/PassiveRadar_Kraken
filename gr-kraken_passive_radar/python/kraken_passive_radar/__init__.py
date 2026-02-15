"""
Kraken Passive Radar GNU Radio Module

Provides signal processing blocks for passive bistatic radar using KrakenSDR.

Python blocks:
    krakensdr_source         - KrakenSDR 5-channel coherent source
    rspduo_source            - SDRplay RSPduo dual-tuner source
    EcaBClutterCanceller     - ECA-B clutter cancellation (ctypes kernel)
    ConditioningBlock        - AGC / signal conditioning (ctypes kernel)
    CafBlock                 - Cross-Ambiguity Function (ctypes kernel)
    BackendBlock             - CFAR + sensor fusion (ctypes kernel)
    TimeAlignmentBlock       - Delay/phase measurement (ctypes kernel)
    vector_zero_pad          - Zero-pad vectors for FFT alignment
    CalibrationController    - Automatic phase calibration

C++ blocks (pybind11):
    eca_canceller            - VOLK-accelerated NLMS clutter canceller
    doppler_processor        - Range-Doppler map via slow-time FFT
    cfar_detector            - CA/GO/SO/OS-CFAR detection
    coherence_monitor        - Automatic calibration triggering
    detection_cluster        - Connected-component target extraction
    aoa_estimator            - Bartlett beamforming AoA estimation
    tracker                  - Multi-target Kalman tracker with GNN
"""

# CRITICAL: Import gnuradio runtime first so pybind11 knows about base types
from gnuradio import gr

# Import Python blocks
from .krakensdr_source import krakensdr_source
try:
    from .rspduo_source import rspduo_source
except ImportError:
    pass  # gr-sdrplay3 not installed; rspduo_source unavailable
from .calibration_controller import CalibrationController, CalibrationState, CalibrationResult
from .eca_b_clutter_canceller import EcaBClutterCanceller
from .custom_blocks import ConditioningBlock, CafBlock, BackendBlock, TimeAlignmentBlock
from .vector_zero_pad import vector_zero_pad

# Import C++ blocks via pybind11 bindings (may not be available if not built/installed)
try:
    from .kraken_passive_radar_python import *
except ImportError:
    import sys
    print("Warning: C++ pybind11 blocks not available (module not built/installed)",
          file=sys.stderr)
