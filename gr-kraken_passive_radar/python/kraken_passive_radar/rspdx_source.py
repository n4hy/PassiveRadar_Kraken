"""
SDRplay RSPdx Source Block for GNU Radio
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Wraps gnuradio.soapy.source configured for SDRplay RSPdx (1 RX channel).

The RSPdx supports:
  - 1 kHz to 2 GHz tuning range
  - 62.5 kHz to 10.66 MHz sample rates
  - IFGR (20-59 dB) and RFGR (0-27 dB) gain stages
  - 3 antenna ports: A, B, C (selectable, not simultaneous)
  - HDR mode, Bias-T, RF Notch, DAB Notch filters
  - Bandwidths: 0.2, 0.3, 0.6, 1.536, 5, 6, 7, 8 MHz
"""

import numpy as np
from gnuradio import gr
from gnuradio import soapy
import sys


class rspdx_source(gr.hier_block2):
    """
    SDRplay RSPdx Source Block

    Single-channel source using SDRplay RSPdx via SoapySDR.
    Produces 1 complex output stream.

    For passive radar, use self-referencing mode: split the single output
    to feed both reference and surveillance inputs of the ECA canceller.
    """

    def __init__(self, frequency=100e6, sample_rate=2.4e6,
                 if_gain=40.0, rf_gain=0.0,
                 antenna='Antenna A', bandwidth=0,
                 bias_t=False, rf_notch=True, dab_notch=True,
                 hdr_mode=False):
        gr.hier_block2.__init__(
            self,
            "RSPdx Source",
            gr.io_signature(0, 0, 0),                        # No inputs
            gr.io_signature(1, 1, gr.sizeof_gr_complex)       # 1 complex output
        )

        self.frequency = frequency
        self.sample_rate = sample_rate
        self.if_gain = if_gain
        self.rf_gain = rf_gain
        self.antenna = antenna
        self.bandwidth = bandwidth

        # Create SoapySDR source for RSPdx
        self.soapy_src = soapy.source(
            'driver=sdrplay',   # device string
            'fc32',             # stream format: complex float32
            1,                  # nchan: 1 channel
            '',                 # dev_args
            '',                 # stream_args
            [''],               # tune_args (list per channel)
            ['']                # other_settings (list per channel)
        )

        # Configure channel 0
        self.soapy_src.set_sample_rate(0, self.sample_rate)
        self.soapy_src.set_frequency(0, self.frequency)
        self.soapy_src.set_antenna(0, self.antenna)
        self.soapy_src.set_gain(0, 'IFGR', self.if_gain)
        self.soapy_src.set_gain(0, 'RFGR', self.rf_gain)

        if self.bandwidth > 0:
            self.soapy_src.set_bandwidth(0, self.bandwidth)

        # Apply device-level settings
        self._write_setting('biasT_ctrl', 'true' if bias_t else 'false')
        self._write_setting('rfnotch_ctrl', 'true' if rf_notch else 'false')
        self._write_setting('dabnotch_ctrl', 'true' if dab_notch else 'false')
        self._write_setting('hdr_ctrl', 'true' if hdr_mode else 'false')

        # Connect internal source to hier_block2 output
        self.connect((self.soapy_src, 0), (self, 0))

    def _write_setting(self, key, value):
        """Write a device setting, suppressing errors for unsupported settings."""
        try:
            self.soapy_src.write_setting(key, value)
        except Exception as e:
            print(f"Warning: Failed to set {key}={value}: {e}", file=sys.stderr)

    # --- Runtime callbacks ---

    def set_frequency(self, freq):
        self.frequency = freq
        self.soapy_src.set_frequency(0, freq)

    def set_sample_rate(self, rate):
        self.sample_rate = rate
        self.soapy_src.set_sample_rate(0, rate)

    def set_if_gain(self, gain):
        self.if_gain = gain
        self.soapy_src.set_gain(0, 'IFGR', gain)

    def set_rf_gain(self, gain):
        self.rf_gain = gain
        self.soapy_src.set_gain(0, 'RFGR', gain)

    def set_antenna(self, antenna):
        self.antenna = antenna
        self.soapy_src.set_antenna(0, antenna)

    def set_bandwidth(self, bw):
        self.bandwidth = bw
        if bw > 0:
            self.soapy_src.set_bandwidth(0, bw)

    def set_bias_t(self, enable):
        self._write_setting('biasT_ctrl', 'true' if enable else 'false')

    def set_hdr_mode(self, enable):
        self._write_setting('hdr_ctrl', 'true' if enable else 'false')
