"""
Kraken Passive Radar GNU Radio Module

Provides signal processing blocks for passive bistatic radar using KrakenSDR.

Blocks:
    eca_canceller - ECA-B Clutter Cancellation
"""

# CRITICAL: Import gnuradio runtime first so pybind11 knows about base types
from gnuradio import gr

# Now import our blocks
from .kraken_passive_radar_python import *
