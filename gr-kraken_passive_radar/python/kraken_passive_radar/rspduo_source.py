"""
SDRplay RSPduo Dual-Tuner Source Block for GNU Radio
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Wraps gr-sdrplay3's sdrplay3.rspduo in dual-tuner (diversity reception) mode.
Produces 2 complex outputs: output 0 = Tuner 1 (reference), output 1 = Tuner 2 (surveillance).

The RSPduo dual-tuner mode provides:
  - Two coherent tuners sharing a common clock
  - Proper hardware-separated reference + surveillance channels
  - 1 kHz to 2 GHz tuning range
  - 62.5 kHz to 2 MHz sample rates (dual-tuner mode limit)
  - IF gain (20-59 dB) and RF gain reduction (0-27 dB) per tuner
  - Bias-T, RF Notch, DAB Notch, AM Notch filters
  - Bandwidths: 0.2, 0.3, 0.6, 1.536 MHz

gr-sdrplay3 dual-tuner API notes:
  - set_sample_rate(rate, synchronous_updates) - 2 args
  - set_center_freq(freq, synchronous_updates) - 2 args (shared freq in diversity mode)
  - set_gain(gain0, gain1, type, synchronous_updates) - both tuner gains in one call
  - set_gain_modes(agc0, agc1) - per-tuner AGC
  - set_biasT(bool), set_rf_notch_filter(bool), etc. - dedicated setters
"""

import numpy as np
from gnuradio import gr
from gnuradio import sdrplay3
import sys


class rspduo_source(gr.hier_block2):
    """
    SDRplay RSPduo Dual-Tuner Source Block

    Dual-channel source using SDRplay RSPduo via gr-sdrplay3.
    Produces 2 complex output streams (reference + surveillance).

    Uses dual-tuner diversity reception mode: both tuners locked to the
    same frequency with a shared clock for coherent operation.
    """

    def __init__(self, frequency=100e6, sample_rate=2e6,
                 if_gain=40.0, rf_gain=0.0,
                 bandwidth=0,
                 bias_t=False, rf_notch=False, dab_notch=False,
                 am_notch=False):
        gr.hier_block2.__init__(
            self,
            "RSPduo Source",
            gr.io_signature(0, 0, 0),                        # No inputs
            gr.io_signature(2, 2, gr.sizeof_gr_complex)       # 2 complex outputs
        )

        self.frequency = frequency
        self.sample_rate = sample_rate
        self.if_gain = if_gain
        self.rf_gain = rf_gain
        self.bandwidth = bandwidth

        # Create gr-sdrplay3 RSPduo source in dual-tuner mode
        self.sdrplay_src = sdrplay3.rspduo(
            selector='',
            rspduo_mode='Dual Tuner (diversity reception)',
            antenna='Both Tuners',
            stream_args=sdrplay3.stream_args(output_type='fc32', channels_size=2),
        )

        # Configure using dual-tuner API (all setters need synchronous_updates flag)
        self.sdrplay_src.set_sample_rate(self.sample_rate, False)
        self.sdrplay_src.set_center_freq(self.frequency, False)
        self.sdrplay_src.set_bandwidth(int(self.bandwidth))

        # Dual-tuner mode: set_gain takes (gain0, gain1, type, sync)
        self.sdrplay_src.set_gain_modes(False, False)
        self.sdrplay_src.set_gain(-self.if_gain, -self.if_gain, 'IF', False)
        self.sdrplay_src.set_gain(-self.rf_gain, -self.rf_gain, 'RF', False)

        # Other settings with defaults matching gr-sdrplay3
        self.sdrplay_src.set_freq_corr(0)
        self.sdrplay_src.set_dc_offset_mode(False)
        self.sdrplay_src.set_iq_balance_mode(False)
        self.sdrplay_src.set_agc_setpoint(-30)

        # Device-level settings via dedicated methods
        self.sdrplay_src.set_rf_notch_filter(rf_notch)
        self.sdrplay_src.set_dab_notch_filter(dab_notch)
        self.sdrplay_src.set_am_notch_filter(am_notch)
        self.sdrplay_src.set_biasT(bias_t)

        self.sdrplay_src.set_stream_tags(False)
        self.sdrplay_src.set_debug_mode(False)
        self.sdrplay_src.set_sample_sequence_gaps_check(False)
        self.sdrplay_src.set_show_gain_changes(False)

        # Connect internal source outputs to hier_block2 outputs
        self.connect((self.sdrplay_src, 0), (self, 0))  # Tuner 1 -> ref
        self.connect((self.sdrplay_src, 1), (self, 1))  # Tuner 2 -> surv

    # --- Runtime callbacks ---

    def set_frequency(self, freq):
        self.frequency = freq
        self.sdrplay_src.set_center_freq(freq, False)

    def set_sample_rate(self, rate):
        self.sample_rate = rate
        self.sdrplay_src.set_sample_rate(rate, False)

    def set_if_gain(self, gain):
        self.if_gain = gain
        self.sdrplay_src.set_gain(-gain, -gain, 'IF', False)

    def set_rf_gain(self, gain):
        self.rf_gain = gain
        self.sdrplay_src.set_gain(-gain, -gain, 'RF', False)

    def set_bandwidth(self, bw):
        self.bandwidth = bw
        self.sdrplay_src.set_bandwidth(int(bw))

    def set_bias_t(self, enable):
        self.sdrplay_src.set_biasT(enable)

    def set_rf_notch_filter(self, enable):
        self.sdrplay_src.set_rf_notch_filter(enable)

    def set_dab_notch_filter(self, enable):
        self.sdrplay_src.set_dab_notch_filter(enable)

    def set_am_notch_filter(self, enable):
        self.sdrplay_src.set_am_notch_filter(enable)
