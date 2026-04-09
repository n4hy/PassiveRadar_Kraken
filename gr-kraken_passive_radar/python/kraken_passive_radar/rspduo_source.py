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
import subprocess
import time


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
        """Initialize the RSPduo dual-tuner source with frequency, gain, and filter settings.

        Technique: Opens the SDRplay RSPduo in dual-tuner diversity reception mode
        via gr-sdrplay3, proactively restarting the sdrplay service to clear stale
        locks, then configures both tuners with identical gain and frequency settings.
        """
        gr.hier_block2.__init__(
            self,
            "RSPduo Source",
            gr.io_signature(0, 0, 0),                        # No inputs
            gr.io_signature(2, 2, gr.sizeof_gr_complex)       # 2 complex outputs
        )

        self.frequency = frequency
        self.sample_rate = sample_rate
        self.if_gain = self._clamp_if_gain(if_gain)
        self.rf_gain = rf_gain
        self.bandwidth = bandwidth

        # Create gr-sdrplay3 RSPduo source in dual-tuner mode.
        # If the device is locked from a previous session, automatically
        # restart the sdrplay service and retry.
        self.sdrplay_src = self._open_device(max_retries=3)

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

    # RSPduo IF gain reduction (gRdB) valid range per SDRplay API v3
    IF_GAIN_MIN = 20.0
    IF_GAIN_MAX = 59.0

    @classmethod
    def _clamp_if_gain(cls, gain):
        """Clamp IF gain reduction to RSPduo hardware range [20, 59] dB.

        The SDRplay API rejects out-of-range gRdB values at
        sdrplay_api_Init() time with a generic sdrplay_api_Fail error.
        """
        if gain < cls.IF_GAIN_MIN:
            print(f"  Warning: IF gain {gain} dB below RSPduo minimum ({cls.IF_GAIN_MIN}). "
                  f"Clamping to {cls.IF_GAIN_MIN}.")
            return cls.IF_GAIN_MIN
        if gain > cls.IF_GAIN_MAX:
            print(f"  Warning: IF gain {gain} dB above RSPduo maximum ({cls.IF_GAIN_MAX}). "
                  f"Clamping to {cls.IF_GAIN_MAX}.")
            return cls.IF_GAIN_MAX
        return gain

    @staticmethod
    def _restart_sdrplay_service():
        """Restart the sdrplay systemd service to release a stale device lock."""
        print("  Restarting sdrplay_api service...")
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "sdrplay"],
                timeout=15, capture_output=True, check=True
            )
            # The service needs time to scan USB and register device handles
            # before sdrplay_api_Open/Init will succeed.
            time.sleep(7)
            print("  Service restarted and ready.")
        except subprocess.CalledProcessError:
            print("  Warning: 'sudo systemctl restart sdrplay' failed.")
            print("  Check that sdrplay.service exists and sudo is passwordless.")
        except FileNotFoundError:
            print("  Warning: systemctl not found.")

    def _open_device(self, max_retries=3):
        """Open the RSPduo, ensuring clean API state first.

        Always restarts the sdrplay service before the first attempt to
        clear any stale device locks left by a previous session (crash,
        kill -9, incomplete cleanup).  This guarantees that both the
        constructor (sdrplay_api_Open + SelectDevice) and the later
        start() call (sdrplay_api_Init) operate on a clean API state.
        """
        # Proactive restart — the single most important line for
        # eliminating "sdrplay_api_Init() Error: sdrplay_api_Fail".
        self._restart_sdrplay_service()

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                src = sdrplay3.rspduo(
                    selector='',
                    rspduo_mode='Dual Tuner (diversity reception)',
                    antenna='Both Tuners',
                    stream_args=sdrplay3.stream_args(
                        output_type='fc32', channels_size=2
                    ),
                )
                if attempt > 1:
                    print(f"  RSPduo opened on attempt {attempt}.")
                return src
            except RuntimeError as e:
                last_err = e
                print(f"RSPduo open failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    self._restart_sdrplay_service()
        raise RuntimeError(
            f"Could not open RSPduo after {max_retries} attempts. "
            f"Last error: {last_err}\n"
            f"Check: USB connected? lsusb | grep 1df7"
        )

    # --- Runtime callbacks ---

    def set_frequency(self, freq):
        """Set the center frequency on both RSPduo tuners."""
        self.frequency = freq
        self.sdrplay_src.set_center_freq(freq, False)

    def set_sample_rate(self, rate):
        """Set the sample rate on the RSPduo source."""
        self.sample_rate = rate
        self.sdrplay_src.set_sample_rate(rate, False)

    def set_if_gain(self, gain):
        """Set IF gain reduction on both tuners, clamped to hardware range."""
        self.if_gain = self._clamp_if_gain(gain)
        self.sdrplay_src.set_gain(-self.if_gain, -self.if_gain, 'IF', False)

    def set_rf_gain(self, gain):
        """Set RF gain reduction on both tuners."""
        self.rf_gain = gain
        self.sdrplay_src.set_gain(-gain, -gain, 'RF', False)

    def set_bandwidth(self, bw):
        """Set the analog bandwidth filter on the RSPduo."""
        self.bandwidth = bw
        self.sdrplay_src.set_bandwidth(int(bw))

    def set_bias_t(self, enable):
        """Enable or disable the RSPduo bias-T power supply."""
        self.sdrplay_src.set_biasT(enable)

    def set_rf_notch_filter(self, enable):
        """Enable or disable the RSPduo RF notch filter."""
        self.sdrplay_src.set_rf_notch_filter(enable)

    def set_dab_notch_filter(self, enable):
        """Enable or disable the RSPduo DAB notch filter."""
        self.sdrplay_src.set_dab_notch_filter(enable)

    def set_am_notch_filter(self, enable):
        """Enable or disable the RSPduo AM notch filter."""
        self.sdrplay_src.set_am_notch_filter(enable)
