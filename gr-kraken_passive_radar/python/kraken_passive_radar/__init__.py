"""
Kraken Passive Radar GNU Radio Module

Provides signal processing blocks for passive bistatic radar using KrakenSDR.

Blocks:
    krakensdr_source - KrakenSDR 5-channel coherent source
    EcaBClutterCanceller - ECA-B Clutter Cancellation (Python/C++ kernel)
    eca_canceller - ECA Clutter Cancellation (C++ pybind11)
"""

# CRITICAL: Import gnuradio runtime first so pybind11 knows about base types
from gnuradio import gr

# Import Python blocks
from .krakensdr_source import krakensdr_source
from .calibration_controller import CalibrationController, CalibrationState, CalibrationResult
from .eca_b_clutter_canceller import EcaBClutterCanceller
from .custom_blocks import ConditioningBlock, CafBlock, BackendBlock, TimeAlignmentBlock

# Now import C++ blocks via pybind11 bindings
from .kraken_passive_radar_python import *
