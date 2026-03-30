#!/usr/bin/env python3
"""
Passive Radar Run - Main Signal Processing Pipeline
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Signal Flow (with automatic phase calibration):

    KrakenSDR Source (5 channels)
           |
           v
    +------------------+
    | Calibration      |  <-- Monitors coherence, triggers calibration
    | Controller       |      when drift exceeds threshold
    +------------------+
           |
           | When calibration needed:
           |   1. Enable noise source (HW switch isolates antennas)
           |   2. Capture noise samples (common to all channels)
           |   3. Compute phase correction phasors
           |   4. Disable noise source (HW switch reconnects antennas)
           |   5. Apply corrections to subsequent samples
           v
    Phase Correction (per channel)  <-- BEFORE ECA
           |
           v
    Conditioning (AGC)
           |
           v
    ECA Clutter Canceller  <-- Requires phase-coherent inputs
           |
           v
    CAF -> Doppler -> CFAR -> Display

IMPORTANT: The KrakenSDR has an internal noise source with a high-isolation
silicon switch. When the noise source is enabled via software, the switch
DISCONNECTS all antennas and routes ONLY the internal noise to all receivers.
This is a hardware feature essential for phase calibration.
"""

import sys
import time
import numpy as np
from gnuradio import gr, blocks, filter, analog
from gnuradio.filter import firdes, window
import os
import argparse
import json
import threading
import signal
import subprocess
import atexit

# Import all blocks from installed gnuradio.kraken_passive_radar
from gnuradio.kraken_passive_radar import (
    # Python blocks
    krakensdr_source,
    ConditioningBlock,
    CafBlock,
    TimeAlignmentBlock,
    CalibrationController,
    # C++ blocks
    eca_canceller,
    doppler_processor,
    cfar_detector,
    detection_cluster,
    aoa_estimator,
    tracker,
    dvbt_reconstructor,
    # Display
    dashboard_sink,
)

# RSPduo source requires gr-sdrplay3 — import lazily so the program
# doesn't crash on machines without it installed
try:
    from gnuradio.kraken_passive_radar import rspduo_source
    HAS_RSPDUO = True
except ImportError:
    HAS_RSPDUO = False

# New Display (optional)
try:
    # Try installed version first
    from gnuradio.kraken_passive_radar.radar_display import PPIDisplay, PPIDetection
    HAS_DISPLAY = True
except ImportError:
    # Fall back to local version
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "gr-kraken_passive_radar/python"))
        from kraken_passive_radar.radar_display import PPIDisplay, PPIDetection
        HAS_DISPLAY = True
    except ImportError:
        HAS_DISPLAY = False

# Range-Doppler Display (optional, for RSPduo mode)
try:
    from kraken_passive_radar.range_doppler_display import RangeDopplerDisplay, RDDisplayParams
    HAS_RD_DISPLAY = True
except ImportError:
    HAS_RD_DISPLAY = False


class BlockDiagnostics:
    """Per-block throughput and timing diagnostics.

    Samples nitems_written from every block every interval_sec and prints
    a table showing actual vs required throughput. The block with the lowest
    ratio (actual/required) is the bottleneck.
    """
    def __init__(self, interval_sec=5.0):
        self.interval = interval_sec
        self.blocks = []  # (name, block, port, expected_rate_sps)
        self._prev = {}   # name -> (time, nitems)
        self._stop = threading.Event()
        self._thread = None

    def add(self, name, block, port=0, expected_rate=0):
        self.blocks.append((name, block, port, expected_rate))

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        # Initial snapshot
        time.sleep(2.0)  # let flowgraph settle
        for name, blk, port, _ in self.blocks:
            try:
                self._prev[name] = (time.time(), blk.nitems_written(port))
            except Exception:
                self._prev[name] = (time.time(), 0)

        while not self._stop.wait(self.interval):
            now = time.time()
            lines = []
            for name, blk, port, expected in self.blocks:
                try:
                    nw = blk.nitems_written(port)
                except Exception:
                    continue
                prev_t, prev_n = self._prev.get(name, (now, nw))
                dt = now - prev_t
                if dt < 0.1:
                    continue
                rate = (nw - prev_n) / dt
                self._prev[name] = (now, nw)
                if expected > 0:
                    pct = 100.0 * rate / expected
                    lines.append(f"  {name:30s}  {rate:10.0f} sps  ({pct:5.1f}% of {expected:.0f})")
                else:
                    lines.append(f"  {name:30s}  {rate:10.0f} items/s")

            if lines:
                print(f"\n{'='*70}")
                print(f"BLOCK DIAGNOSTICS (dt={self.interval:.0f}s)")
                print(f"{'='*70}")
                for l in lines:
                    print(l)
                print(f"{'='*70}\n", flush=True)


class PhaseCorrectorBlock(gr.sync_block):
    """
    GNU Radio block that applies phase corrections from CalibrationController,
    including linear drift rate compensation.

    This block sits between the KrakenSDR source and the conditioning/ECA blocks.
    It ensures phase coherence is maintained by:
    1. Applying stored phase corrections + drift extrapolation during normal operation
    2. Feeding samples to CalibrationController during calibration
    3. Outputting zeros during calibration (antennas are isolated anyway)

    The drift rate compensation is critical: without it, the R820T PLL frequency
    offsets (1-3 deg/s) make calibration useless after ~2 seconds. With linear
    extrapolation, residual error stays <5 deg for 300+ seconds.

    IMPORTANT: This MUST be in the signal path BEFORE ECA processing.
    The ECA clutter canceller requires phase-coherent inputs to work correctly.
    """

    def __init__(self, cal_controller: CalibrationController, channel: int,
                 initial_phase: float = 0.0, drift_rate: float = 0.0,
                 cal_timestamp: float = 0.0, sample_rate: float = 2.4e6,
                 settling_samples: int = 0):
        gr.sync_block.__init__(
            self,
            name=f"Phase Corrector Ch{channel}",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.cal_controller = cal_controller
        self.channel = channel
        self.sample_rate = sample_rate

        # Phase correction state: phase(t) = initial_phase + drift_rate * (t - cal_timestamp)
        self.initial_phase = initial_phase    # radians at cal_timestamp
        self.drift_rate = drift_rate          # radians per second
        self.cal_timestamp = cal_timestamp    # absolute time of calibration
        self._sample_count = 0                # samples processed since calibration
        self._cached_phasor = np.complex64(1.0)
        self._cached_correction = 0.0

        # Settling period: output zeros while R820T PLLs lock and USB buffers
        # stabilize. Downstream blocks process zeros trivially, reducing the
        # startup overflow burst from ~5-10s to ~1-2s.
        self._settling_remaining = settling_samples

    def update_calibration(self, phase_rad: float, drift_rate_rad: float, timestamp: float):
        """Update phase correction parameters after a recalibration."""
        self.initial_phase = phase_rad
        self.drift_rate = drift_rate_rad
        self.cal_timestamp = timestamp
        self._sample_count = 0  # Reset sample count; dt computed from samples, not wall clock

    def work(self, input_items, output_items):
        samples = input_items[0]
        n_samples = len(samples)

        # Settling period: output zeros while PLLs lock
        if self._settling_remaining > 0:
            self._settling_remaining -= n_samples
            output_items[0][:n_samples] = 0
            return n_samples

        if self.cal_controller.is_calibrating:
            # During calibration: feed samples to controller and output zeros
            # (The KrakenSDR HW switch has isolated antennas, so samples are
            # from the internal noise source, not antennas)
            self.cal_controller.process_calibration_samples(self.channel, samples)
            output_items[0][:n_samples] = 0
        else:
            # Normal operation: apply phase correction with drift compensation
            if self.channel == 0 or (self.initial_phase == 0.0 and self.drift_rate == 0.0):
                output_items[0][:n_samples] = samples
            else:
                # Compute time since calibration using sample count (deterministic, no wall-clock jitter)
                dt = self._sample_count / self.sample_rate
                correction = self.initial_phase + self.drift_rate * dt
                # Recompute phasor only when correction changes significantly (> 0.01 rad)
                if abs(correction - self._cached_correction) > 0.01:
                    self._cached_phasor = np.complex64(np.exp(1j * correction))
                    self._cached_correction = correction
                output_items[0][:n_samples] = samples * self._cached_phasor

        self._sample_count += n_samples
        return n_samples


class PassiveRadarTopBlock(gr.top_block):
    def __init__(self, freq=100e6, gain=30, geometry='ULA', calibration_file=None,
                 b3_signal_type='passthrough', b3_fft_size=8192, b3_guard_interval=192,
                 source_type='kraken', if_gain=40.0, rf_gain=0.0,
                 bandwidth=0, sample_rate=2.4e6, skip_aoa=False, cpi_len=2048,
                 signal_bw=500000, pfa=1e-6, min_cluster_size=2):
        gr.top_block.__init__(self, "Passive Radar Run")

        # Parameters
        self.freq = freq
        self.gain = gain
        self.source_type = source_type
        self.source_rate = sample_rate  # ADC sample rate
        self.cpi_len = cpi_len
        self.doppler_len = 256
        self.num_taps = 128
        self.geometry = geometry
        self.b3_signal_type = b3_signal_type
        self.b3_fft_size = b3_fft_size
        self.b3_guard_interval = b3_guard_interval

        # Decimation: rational resampler from source rate to signal bandwidth
        # 2 MHz -> 500 kHz = 4:1 gives ~6 dB processing gain
        self.signal_bw = signal_bw
        self.decimation = max(1, int(sample_rate / signal_bw))
        self.sample_rate = sample_rate / self.decimation  # decimated rate for all downstream
        self.lpf_taps = firdes.low_pass(
            1.0, sample_rate, signal_bw / 2, signal_bw / 10, window.WIN_HAMMING)

        # Source-dependent parameters
        if source_type == 'rspduo':
            self.num_surv = 1   # 1 real surveillance channel
            self.skip_aoa = True
            self.skip_calibration = True
        else:
            self.num_surv = 4
            self.skip_aoa = skip_aoa
            self.skip_calibration = False

        # Derived parameters (all use decimated rate)
        c = 3e8
        wavelength = c / freq
        self.range_res = c / (2 * self.sample_rate)  # meters per range bin
        self.doppler_res = self.sample_rate / (self.cpi_len * self.doppler_len)  # Hz per Doppler bin
        range_res = self.range_res
        doppler_res = self.doppler_res
        frame_period = (self.cpi_len * self.doppler_len) / self.sample_rate

        # Load Calibration (phase offsets AND drift rates)
        self.phase_offsets = np.zeros(5, dtype=np.float64)
        self.drift_rates = np.zeros(5, dtype=np.float64)  # radians/sec
        self.cal_timestamp = 0.0
        if calibration_file and os.path.exists(calibration_file):
            print(f"Loading calibration from {calibration_file}...")
            with open(calibration_file, 'r') as f:
                cal = json.load(f)
                self.cal_timestamp = cal.get("timestamp", 0.0)
                for i in range(5):
                    # krakensdr_calibrate.py saves ch{i}_phase_rad and ch{i}_drift_rad_per_sec
                    self.phase_offsets[i] = cal.get(f"ch{i}_phase_rad", cal.get(f"ch{i}", 0.0))
                    self.drift_rates[i] = cal.get(f"ch{i}_drift_rad_per_sec", 0.0)
            print(f"  Phase offsets (deg): {np.degrees(self.phase_offsets)}")
            print(f"  Drift rates (deg/s): {np.degrees(self.drift_rates)}")
        else:
            print("No calibration loaded. Using default (0).")

        # 1. Source (runs at source_rate, e.g. 2 MHz)
        if self.source_type == 'rspduo':
            self.source = rspduo_source(
                frequency=freq, sample_rate=self.source_rate,
                if_gain=if_gain, rf_gain=rf_gain,
                bandwidth=bandwidth
            )
        else:
            self.source = krakensdr_source(
                frequency=freq, gain=gain,
                sample_rate=self.source_rate
            )

        # 1b. Rational resampler: source_rate -> signal_bw (e.g. 2 MHz -> 500 kHz)
        # Decimation with anti-alias LPF gives ~6 dB processing gain
        # Runs BEFORE phase correction so the Python phase corrector operates
        # at the decimated rate (500 kHz) instead of the source rate (2 MHz).
        self.resamplers = []
        if self.decimation > 1:
            print(f"Decimation: {self.source_rate/1e6:.1f} MHz -> "
                  f"{self.sample_rate/1e3:.0f} kHz ({self.decimation}:1, "
                  f"+{10*np.log10(self.decimation):.1f} dB processing gain)")
            n_ch = 2 if self.source_type == 'rspduo' else 5
            for i in range(n_ch):
                rs = filter.rational_resampler_ccc(
                    interpolation=1,
                    decimation=self.decimation,
                    taps=self.lpf_taps,
                )
                self.resamplers.append(rs)
                self.connect((self.source, i), (rs, 0))

        # Helper: output after resampler (or source if no decimation)
        def _resampled_out(ch):
            if self.resamplers:
                return (self.resamplers[ch], 0)
            return (self.source, ch)

        # 1c. Phase correction using C++ blocks (no Python in signal path)
        # multiply_const_cc: static phase offset exp(-j*phi0)
        # rotator_cc: drift compensation exp(-j*omega*n/fs) per sample
        self.phase_mults = []   # multiply_const_cc per channel
        self.phase_rotators = []  # rotator_cc per channel
        self.cal_controller = None
        self.phase_correctors = None  # legacy (unused)

        if not self.skip_calibration:
            for i in range(5):
                phi0 = float(self.phase_offsets[i])
                omega = float(self.drift_rates[i])  # rad/sec
                phase_inc = -omega / self.sample_rate  # rad/sample (negate to correct)

                mult = blocks.multiply_const_cc(complex(np.exp(-1j * phi0)))
                rot = blocks.rotator_cc(phase_inc)
                self.phase_mults.append(mult)
                self.phase_rotators.append(rot)
                self.connect(_resampled_out(i), (mult, 0))
                self.connect((mult, 0), (rot, 0))

            # CalibrationController for periodic recalibration
            # The callback updates the C++ multiply_const/rotator blocks
            self.cal_controller = CalibrationController(
                source=self.source,
                num_channels=5,
                cal_samples=int(0.01 * self.source_rate),  # 10ms at source rate
                settle_time_ms=50.0,
                on_calibration_complete=self._on_calibration_complete
            )

        # Helper: output after phase correction (or resampler if no cal)
        def _corrected_out(ch):
            if self.phase_rotators:
                return (self.phase_rotators[ch], 0)
            return _resampled_out(ch)

        # 2. Conditioning (operates at decimated rate)
        self.cond_blocks = []
        if self.source_type == 'rspduo':
            for i in range(2):
                blk = ConditioningBlock(rate=1e-5)
                self.cond_blocks.append(blk)
                self.connect(_resampled_out(i), (blk, 0))
        else:
            for i in range(5):
                blk = ConditioningBlock(rate=1e-5)
                self.cond_blocks.append(blk)
                self.connect(_corrected_out(i), (blk, 0))

        # 2c. Block B3: Reference Signal Reconstructor
        # Demod-remod to create clean reference signal for improved passive radar sensitivity
        print(f"Block B3: Initializing {self.b3_signal_type} mode reference reconstructor...")
        if self.b3_signal_type == 'fm':
            self.b3_recon = dvbt_reconstructor.make(
                signal_type="fm",
                fm_deviation=75e3,      # 75 kHz for US, 50 kHz for Europe
                enable_stereo=True,
                enable_pilot_regen=True,
                audio_bw=15e3
            )
            print(f"  FM Radio mode: 75 kHz deviation, stereo with pilot regeneration")
        elif self.b3_signal_type == 'atsc3':
            self.b3_recon = dvbt_reconstructor.make(
                signal_type="atsc3",
                fft_size=self.b3_fft_size,
                guard_interval=self.b3_guard_interval,
                pilot_pattern=0,
                enable_svd=True
            )
            print(f"  ATSC 3.0 mode: {self.b3_fft_size} FFT, GI={self.b3_guard_interval}, SVD enabled")
            print(f"  Note: LDPC FEC is placeholder (works best on strong signals)")
        elif self.b3_signal_type == 'dvbt':
            self.b3_recon = dvbt_reconstructor.make(
                signal_type="dvbt",
                fft_size=self.b3_fft_size,
                guard_interval=self.b3_guard_interval,
                enable_svd=True
            )
            print(f"  DVB-T mode: {self.b3_fft_size} FFT, GI={self.b3_guard_interval}")
            print(f"  Note: DVB-T processing is TODO (skeleton only)")
        else:  # passthrough
            self.b3_recon = dvbt_reconstructor.make(signal_type="passthrough")
            print(f"  Passthrough mode: No reference reconstruction")

        # Connect: Conditioning[0] (reference) -> Block B3
        self.connect((self.cond_blocks[0], 0), (self.b3_recon, 0))

        # Reference channel now comes from Block B3
        self.reference_channel = (self.b3_recon, 0)

        # 2b. Time Alignment Probe — DISABLED
        # These Python sink blocks consumed 500k sps from both ref and surv
        # channels via GIL-serialized work() calls. With 4 probes sharing the
        # reference buffer, GIL contention throttled the entire flowgraph.
        # Startup calibration handles inter-channel delay; these aren't needed.
        self.align_blocks = []

        # 3. ECA Clutter Canceller (C++ VOLK-accelerated)
        # Input: port 0 = reference (from Block B3), ports 1..num_surv = surveillance
        # Output: ports 0..num_surv-1 = clutter-cancelled surveillance
        self.eca = eca_canceller(
            num_taps=self.num_taps,
            reg_factor=0.001,
            num_surv=self.num_surv
        )
        self.connect(self.reference_channel, (self.eca, 0))  # Block B3 output
        if self.source_type == 'rspduo':
            # RSPduo: separate surveillance channel from tuner 2
            self.connect((self.cond_blocks[1], 0), (self.eca, 1))
        else:
            for i in range(self.num_surv):
                self.connect((self.cond_blocks[i+1], 0), (self.eca, i+1))

        # 4. CAF (cross-ambiguity function) per surveillance channel
        self.caf_blocks = []
        for i in range(self.num_surv):
            blk = CafBlock(n_samples=self.cpi_len)
            self.caf_blocks.append(blk)
            self.connect(self.reference_channel, (blk, 0))  # Block B3 output
            self.connect((self.eca, i), (blk, 1))           # cleaned surv

        # 5. Doppler Processor (C++ FFTW-accelerated)
        # Accumulates doppler_len range profiles, applies window + slow-time FFT
        # AoA mode: output complex (for beamforming), then |·|² for CFAR
        # Skip-AoA mode: output power directly
        rd_vlen = self.cpi_len * self.doppler_len
        need_complex_doppler = not self.skip_aoa

        self.doppler_blocks = []
        self.mag_sq_blocks = []  # Only used when AoA is active
        for i in range(self.num_surv):
            blk = doppler_processor(
                num_range_bins=self.cpi_len,
                num_doppler_bins=self.doppler_len,
                window_type=1,       # Hamming
                output_power=(not need_complex_doppler)
            )
            self.doppler_blocks.append(blk)
            self.connect((self.caf_blocks[i], 0), (blk, 0))

        # When AoA is active, insert complex→power conversion for CFAR path
        if need_complex_doppler:
            for i in range(self.num_surv):
                mag_sq = blocks.complex_to_mag_squared(rd_vlen)
                self.mag_sq_blocks.append(mag_sq)
                self.connect((self.doppler_blocks[i], 0), (mag_sq, 0))

        # Power source for CFAR: mag_sq blocks (AoA mode) or doppler blocks (skip-AoA)
        def power_src(i):
            if need_complex_doppler:
                return (self.mag_sq_blocks[i], 0)
            return (self.doppler_blocks[i], 0)

        # 5b. Dashboard Sink (optional, for --visualize with KrakenSDR)
        # Taps the power output from each surveillance channel's Doppler processor
        self.dashboard = None

        # 6. CFAR Detector (C++ accelerated) - per surveillance channel
        self.cfar_blocks = []
        for i in range(self.num_surv):
            blk = cfar_detector(
                num_range_bins=self.cpi_len,
                num_doppler_bins=self.doppler_len,
                guard_cells_range=2,
                guard_cells_doppler=2,
                ref_cells_range=8,
                ref_cells_doppler=8,
                pfa=pfa,
                cfar_type=0          # CA-CFAR
            )
            self.cfar_blocks.append(blk)
            self.connect(power_src(i), (blk, 0))

        # 7. Detection Clustering (C++ accelerated) - per surveillance channel
        self.cluster_blocks = []
        for i in range(self.num_surv):
            blk = detection_cluster(
                num_range_bins=self.cpi_len,
                num_doppler_bins=self.doppler_len,
                min_cluster_size=min_cluster_size,
                max_cluster_extent=50,
                range_resolution_m=range_res,
                doppler_resolution_hz=doppler_res,
                max_detections=85
            )
            self.cluster_blocks.append(blk)
            # Cluster takes CFAR mask (input 0) and power map (input 1)
            self.connect((self.cfar_blocks[i], 0), (blk, 0))
            self.connect(power_src(i), (blk, 1))

        if not self.skip_aoa:
            # 8. AoA Estimator (C++ Bartlett beamformer)
            # Inputs 0..num_surv-1: complex range-Doppler maps (from doppler)
            # Input num_surv: detection list (from cluster[0])
            array_type_val = 0 if geometry == 'ULA' else 1
            self.aoa = aoa_estimator.make(
                num_elements=self.num_surv,
                d_lambda=0.5,
                n_angles=181,
                min_angle_deg=-90.0,
                max_angle_deg=90.0,
                array_type=array_type_val,
                num_range_bins=self.cpi_len,
                num_doppler_bins=self.doppler_len,
                max_detections=85
            )
            # AoA CAF inputs: complex doppler output per surveillance channel
            for i in range(self.num_surv):
                self.connect((self.doppler_blocks[i], 0), (self.aoa, i))
            # AoA detection input: use first surveillance channel's detections
            self.connect((self.cluster_blocks[0], 0), (self.aoa, self.num_surv))
            # Null-sink remaining cluster outputs (must be connected)
            for i in range(1, self.num_surv):
                ns = blocks.null_sink(self.cluster_blocks[i].output_signature().sizeof_stream_item(0))
                self.connect((self.cluster_blocks[i], 0), (ns, 0))

            # 9. Multi-Target Tracker (C++ Kalman + GNN)
            max_tracks = 50
            self.trk = tracker(
                dt=frame_period,
                process_noise_range=50.0,
                process_noise_doppler=5.0,
                meas_noise_range=range_res * 2,
                meas_noise_doppler=doppler_res * 2,
                gate_threshold=9.21,     # chi2(2) @ 99%
                confirm_hits=3,
                delete_misses=5,
                max_tracks=max_tracks,
                max_detections=85
            )
            self.connect((self.aoa, 0), (self.trk, 0))

            # 10. Sink (consumes tracker output to trigger display updates)
            self.sink = RadarSink(
                tracker_block=self.trk,
                display_callback=self.update_display,
                vector_len=self.trk.output_signature().sizeof_stream_item(0) // 4,
            )
            self.connect((self.trk, 0), (self.sink, 0))
        else:
            # Skip AoA and tracker: range-Doppler display mode
            self.aoa = None
            self.trk = None

            # Connect cluster outputs to null sinks (outputs must be connected)
            for i in range(self.num_surv):
                ns = blocks.null_sink(self.cluster_blocks[i].output_signature().sizeof_stream_item(0))
                self.connect((self.cluster_blocks[i], 0), (ns, 0))

            # Sink consumes Doppler power map for display
            max_range_km = 50.0
            max_range_bin = int(max_range_km * 1000.0 / range_res) + 1
            max_range_bin = min(max_range_bin, self.cpi_len)
            self.sink = RadarSink(
                tracker_block=None,
                display_callback=self.update_display,
                vector_len=rd_vlen,
                num_range_bins=self.cpi_len,
                num_doppler_bins=self.doppler_len,
                max_range_bin=max_range_bin,
            )
            self.connect((self.doppler_blocks[0], 0), (self.sink, 0))

    def update_display(self, tracks):
        """Update display with confirmed tracks from the tracker."""
        mapped = []
        for t in tracks:
            mapped.append(PPIDetection(
                azimuth_deg=t.get('aoa_deg', 0.0),
                range_m=t['range_m'],
                power_db=t.get('snr_db', 0.0)
            ))
        if hasattr(self, 'display_ref') and self.display_ref:
            self.display_ref.update_detections(mapped)

    def _on_calibration_complete(self, result):
        """
        Callback when automatic calibration completes.

        Updates phase corrector blocks with new offsets and drift rates,
        and persists calibration to disk.

        Args:
            result: CalibrationResult with phase offsets and correlation
        """
        print(f"Calibration #{self.cal_controller.calibration_count} complete:")
        print(f"  Phase corrections (deg): {np.degrees(result.phase_offsets)}")
        print(f"  Correlations: {result.correlation}")

        now = result.timestamp

        # Update C++ phase correction blocks with new calibration
        if self.phase_mults:
            for i in range(5):
                phi0 = float(result.phase_offsets[i])
                self.phase_mults[i].set_k(complex(np.exp(-1j * phi0)))
                # Keep existing drift rate (measured at startup, stable over hours)
                omega = float(self.drift_rates[i])
                phase_inc = -omega / self.sample_rate
                self.phase_rotators[i].set_phase_inc(phase_inc)

        # Save calibration to file (same format as krakensdr_calibrate.py)
        cal_data = {
            "timestamp": now,
            "freq_hz": int(self.freq),
            "calibration_count": self.cal_controller.calibration_count,
        }
        for i in range(5):
            cal_data[f"ch{i}_phase_rad"] = float(result.phase_offsets[i])
            cal_data[f"ch{i}_drift_rad_per_sec"] = float(self.drift_rates[i])
            cal_data[f"ch{i}_corr"] = float(result.correlation[i])

        try:
            with open("calibration.json", 'w') as f:
                json.dump(cal_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save calibration: {e}")

    def trigger_calibration(self, reason: str = "manual"):
        """
        Manually trigger a calibration cycle.

        This enables the noise source (which isolates antennas via HW switch),
        captures calibration samples, computes phase corrections, and resumes
        normal operation.

        Args:
            reason: Reason for calibration (for logging)
        """
        if self.cal_controller is None:
            print("Calibration not available (single-channel mode)")
            return
        print(f"Triggering calibration: {reason}")
        self.cal_controller.handle_cal_request(reason)

    def get_calibration_status(self) -> dict:
        """Get current calibration status."""
        if self.cal_controller is None:
            return {'state': 'N/A', 'source_type': self.source_type}
        return self.cal_controller.get_status()

    def get_b3_snr(self) -> float:
        """Get Block B3 reference signal SNR estimate in dB."""
        if hasattr(self, 'b3_recon'):
            return self.b3_recon.get_snr_estimate()
        return 0.0

    def start_periodic_recal(self, interval_sec: float = 120.0):
        """
        Start a background thread that triggers periodic recalibration.

        With drift rate compensation, 120s interval keeps residual error <5 deg.
        The noise source is briefly enabled (~1s), capturing calibration samples,
        then disabled. The HW silicon switch isolates antennas during this time.

        Args:
            interval_sec: Seconds between recalibrations (default: 120)
        """
        if self.cal_controller is None:
            return  # No calibration for RSPduo/single-channel

        self._recal_stop = threading.Event()

        def recal_loop():
            while not self._recal_stop.wait(interval_sec):
                try:
                    self.trigger_calibration(reason="periodic")
                except Exception as e:
                    print(f"Periodic recalibration error: {e}")

        self._recal_thread = threading.Thread(target=recal_loop, daemon=True)
        self._recal_thread.start()
        print(f"Periodic recalibration: every {interval_sec:.0f}s")

    def stop_periodic_recal(self):
        """Stop the periodic recalibration thread."""
        if hasattr(self, '_recal_stop'):
            self._recal_stop.set()

    def attach_dashboard(self):
        """
        Wire up the dashboard_sink to receive live CAF power data
        from each surveillance channel's Doppler processor.

        Must be called BEFORE tb.start(). Use external_display=True so
        that the matplotlib GUI is driven from the main thread (required
        by TkAgg on most platforms).
        """
        self.dashboard = dashboard_sink(
            fft_len=self.cpi_len,
            doppler_len=self.doppler_len,
            num_channels=self.num_surv,
            sample_rate=self.sample_rate,
            center_freq=self.freq,
            update_rate=10.0,
            external_display=True,
        )
        # Tap power output (same source as CFAR) into the dashboard
        need_complex = not self.skip_aoa
        for i in range(self.num_surv):
            if need_complex:
                src = (self.mag_sq_blocks[i], 0)
            else:
                src = (self.doppler_blocks[i], 0)
            self.connect(src, (self.dashboard, i))


class RadarSink(gr.sync_block):
    def __init__(self, tracker_block=None, display_callback=None, vector_len=1,
                 num_range_bins=0, num_doppler_bins=0, max_range_bin=0):
        gr.sync_block.__init__(
            self,
            name="Radar Sink",
            in_sig=[(np.float32, vector_len)],
            out_sig=None
        )
        self.tracker_block = tracker_block
        self.callback = display_callback
        self.last_display_time = 0
        self.display_interval = 0.1  # 10 Hz display update
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        self.max_range_bin = max_range_bin
        self.rd_display_callback = None  # Set externally for range-Doppler display

    def work(self, input_items, output_items):
        n = len(input_items[0])
        now = time.time()
        if now - self.last_display_time >= self.display_interval:
            self.last_display_time = now
            if self.tracker_block and self.callback:
                confirmed = self.tracker_block.get_confirmed_tracks()
                if confirmed:
                    tracks = []
                    for t in confirmed:
                        tracks.append({
                            'id': t.id,
                            'range_m': t.range_m,
                            'doppler_hz': t.doppler_hz,
                            'aoa_deg': 0.0,  # From AoA estimator results
                            'snr_db': t.score,
                        })
                    self.callback(tracks)
            elif self.rd_display_callback and self.num_range_bins > 0:
                # RSPduo mode: reshape Doppler power map and send to display
                raw = input_items[0][0]
                power = raw.reshape(self.num_doppler_bins, self.num_range_bins)
                power_sliced = power[:, :self.max_range_bin]
                power_db = 10.0 * np.log10(power_sliced + 1e-20)
                self.rd_display_callback(power_db)
        return n

def main():
    parser = argparse.ArgumentParser(
        description="Passive Bistatic Radar using KrakenSDR or RSPduo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Signal Flow (KrakenSDR):
  KrakenSDR -> Phase Correction -> AGC -> ECA -> CAF -> Doppler -> CFAR -> AoA -> Tracker

Signal Flow (RSPduo, dual-tuner):
  RSPduo [Tuner1=ref, Tuner2=surv] -> AGC -> Block B3 -> ECA -> CAF -> Doppler -> CFAR -> Display
        """
    )
    parser.add_argument("--source", choices=['kraken', 'rspduo'], default='kraken',
                        help="SDR source: kraken (5-ch KrakenSDR) or rspduo (2-ch SDRplay RSPduo)")
    parser.add_argument("--freq", type=float, default=103.7e6,
                        help="Center frequency in Hz (default: 103.7 MHz)")
    parser.add_argument("--gain", type=float, default=30,
                        help="Receiver gain in dB (default: 30, KrakenSDR only)")
    parser.add_argument("--geometry", choices=['ULA', 'URA'], default='ULA',
                        help="Antenna array geometry (default: ULA)")
    parser.add_argument("--no-startup-cal", action="store_true",
                        help="Skip startup calibration (use saved calibration only)")
    parser.add_argument("--recal-interval", type=float, default=120.0,
                        help="Periodic recalibration interval in seconds (default: 120)")
    parser.add_argument("--skip-aoa", action="store_true",
                        help="Skip AoA estimator and tracker (range-Doppler display only)")
    parser.add_argument("--cpi-len", type=int, default=2048, choices=[512, 1024, 2048, 4096],
                        help="CPI length / range bins (default: 2048, range res = c/2/fs per bin)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show GUI display")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: show display with simulated data (no hardware needed)")
    parser.add_argument("--diag", action="store_true",
                        help="Print per-block throughput diagnostics every 5 seconds")

    # RSPduo-specific arguments
    parser.add_argument("--if-gain", type=float, default=40.0,
                        help="RSPduo IF gain in dB (default: 40, range: 20-59)")
    parser.add_argument("--rf-gain", type=float, default=0.0,
                        help="RSPduo RF gain reduction in dB (default: 0, range: 0-27)")
    parser.add_argument("--bandwidth", type=float, default=0,
                        help="RSPduo analog bandwidth in Hz (0=auto)")
    parser.add_argument("--sample-rate", type=float, default=2e6,
                        help="Source sample rate in Hz (default: 2e6)")
    parser.add_argument("--signal-bw", type=float, default=500e3,
                        help="Signal bandwidth in Hz; decimates source to this rate (default: 500e3)")

    # CFAR tuning
    parser.add_argument("--pfa", type=float, default=1e-6,
                        help="CFAR probability of false alarm (default: 1e-6, try 1e-8 or 1e-10 to reduce false alarms)")
    parser.add_argument("--min-cluster-size", type=int, default=2,
                        help="Minimum CFAR detections in a cluster to report (default: 2, raise to reject isolated noise)")

    # Block B3: Reference Signal Reconstructor
    parser.add_argument("--b3-signal", choices=['passthrough', 'fm', 'atsc3', 'dvbt'],
                        default='passthrough',
                        help="Block B3 signal type (default: passthrough, no reconstruction)")
    parser.add_argument("--b3-fft-size", type=int, choices=[2048, 4096, 8192, 16384, 32768],
                        default=8192,
                        help="OFDM FFT size for ATSC3/DVB-T (default: 8192)")
    parser.add_argument("--b3-guard-interval", type=int, default=192,
                        help="OFDM guard interval in samples (default: 192 for ATSC3 8K)")

    args = parser.parse_args()

    if args.source == 'rspduo' and not HAS_RSPDUO:
        print("ERROR: --source rspduo requires gr-sdrplay3 (gnuradio.sdrplay3).")
        print("Install gr-sdrplay3: https://github.com/fventuri/gr-sdrplay3")
        sys.exit(1)

    # Demo mode: simulated display, no hardware needed
    if args.demo:
        if not HAS_RD_DISPLAY:
            print("ERROR: Range-Doppler display not available. Check kraken_passive_radar installation.")
            sys.exit(1)
        from kraken_passive_radar.range_doppler_display import demo_delay_doppler
        print("DEMO MODE: Simulated Delay-Doppler display (no hardware)")
        demo_delay_doppler()
        sys.exit(0)

    # --- OS-level buffer preflight check ---
    def check_os_buffers():
        """Check and warn about OS-level buffer settings critical for 5-dongle SDR."""
        issues = []
        try:
            with open('/sys/module/usbcore/parameters/usbfs_memory_mb') as f:
                usb_mb = int(f.read().strip())
            # 5 dongles × 128 buffers × 64KB = 40 MB minimum
            if usb_mb < 64:
                issues.append(f"  usbfs_memory_mb = {usb_mb} (need >= 64, recommend 256)")
                issues.append("    Fix: sudo sh -c 'echo 256 > /sys/module/usbcore/parameters/usbfs_memory_mb'")
            else:
                print(f"  USB buffer pool: {usb_mb} MB (OK)")
        except Exception:
            pass
        try:
            page_size = os.sysconf('SC_PAGE_SIZE')
            if page_size > 4096:
                print(f"  Page size: {page_size} bytes (16K pages — GR buffers will be padded)")
        except Exception:
            pass
        if issues:
            print("WARNING: OS buffer settings may cause overruns:")
            for i in issues:
                print(i)
            print()

    if args.source == 'kraken':
        check_os_buffers()

    # --- Pre-startup calibration (KrakenSDR only) ---
    # Must run BEFORE constructing the flowgraph because osmosdr and librtlsdr
    # both need exclusive device access.
    cal_file = "calibration.json"
    if args.source == 'kraken' and not args.no_startup_cal:
        from krakensdr_calibrate import calibrate
        print("\n" + "="*60)
        print("STARTUP CALIBRATION (librtlsdr direct)")
        print("="*60 + "\n")
        try:
            calibrate(
                freq_hz=int(args.freq),
                sample_rate=int(args.sample_rate),
                gain_db=0.0,  # Must be 0 dB: noise source clips 8-bit ADC at higher gains
                cal_samples=262144,
                cal_file=cal_file,
            )
        except Exception as e:
            print(f"\nCalibration failed: {e}")
            print("Continuing without calibration.\n")
        print()

    tb = PassiveRadarTopBlock(
        freq=args.freq,
        gain=args.gain,
        geometry=args.geometry,
        calibration_file=cal_file,
        b3_signal_type=args.b3_signal,
        b3_fft_size=args.b3_fft_size,
        b3_guard_interval=args.b3_guard_interval,
        source_type=args.source,
        if_gain=args.if_gain,
        rf_gain=args.rf_gain,
        bandwidth=args.bandwidth,
        sample_rate=args.sample_rate,
        skip_aoa=args.skip_aoa,
        cpi_len=args.cpi_len,
        signal_bw=args.signal_bw,
        pfa=args.pfa,
        min_cluster_size=args.min_cluster_size,
    )

    if args.source == 'rspduo':
        print("\n" + "="*60)
        print("RSPduo MODE: Dual-tuner passive radar (ref + surv)")
        print("="*60)
        print(f"  Tuner 1 (ref) + Tuner 2 (surv), coherent clock")
        print(f"  IF Gain: {args.if_gain} dB, RF Gain Reduction: {args.rf_gain} dB")
        print(f"  Sample Rate: {args.sample_rate/1e6:.3f} MHz")
        print(f"  Phase calibration: SKIPPED (coherent clock)")
        print(f"  AoA estimation: SKIPPED (single surveillance element)\n")
    elif args.no_startup_cal:
        print("Skipping startup calibration (using saved calibration if available)")

    if args.source == 'rspduo':
        print(f"Processing chain: RSPduo [ref|surv] -> AGC -> Block B3 ({args.b3_signal}) -> "
              f"ECA -> CAF -> Doppler -> CFAR -> Cluster -> Display\n")
    elif tb.skip_aoa:
        print(f"\nProcessing chain: Source -> Phase Corr -> AGC -> Block B3 ({args.b3_signal}) -> ECA (C++) -> CAF -> "
              f"Doppler (C++) -> CFAR (C++) -> Cluster (C++) -> Display\n")
        print("  AoA/Tracker: SKIPPED (--skip-aoa)")
    else:
        print(f"\nProcessing chain: Source -> Phase Corr -> AGC -> Block B3 ({args.b3_signal}) -> ECA (C++) -> CAF -> "
              f"Doppler (C++) -> CFAR (C++) -> Cluster (C++) -> AoA (C++) -> Tracker (C++)")
        print(f"  CFAR: pfa={args.pfa:.0e}, min_cluster_size={args.min_cluster_size}\n")

    if args.b3_signal != 'passthrough':
        print(f"Block B3 Reference Reconstruction: {args.b3_signal.upper()}")
        print(f"  Expected SNR improvement: 10-20 dB")
        print(f"  Initial SNR estimate: {tb.b3_recon.get_snr_estimate():.1f} dB\n")

    # --- Clean shutdown handling ---
    # Ensures GNU Radio stops cleanly and SDRplay service is restarted
    # on any exit path: Ctrl-C, window close, crash, or normal exit.
    shutdown_done = threading.Event()

    def cleanup():
        if shutdown_done.is_set():
            return
        shutdown_done.set()
        print("\nShutting down...")
        try:
            tb.stop()
            tb.wait()
            print("GNU Radio flowgraph stopped.")
        except Exception:
            pass
        if args.source == 'rspduo':
            print("Restarting SDRplay service to release device...")
            try:
                subprocess.run(
                    ["sudo", "systemctl", "restart", "sdrplay"],
                    timeout=15, capture_output=True
                )
                time.sleep(5)
                print("SDRplay service restarted. Device ready for next run.")
            except Exception as e:
                print(f"Warning: could not restart sdrplay service: {e}")
                print("Run manually: sudo systemctl restart sdrplay")

    atexit.register(cleanup)

    def sigint_handler(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    # --- Start pipeline ---
    if args.visualize and args.source == 'kraken':
        # 5-channel KrakenSDR dashboard: per-channel CAFs, fused CAF,
        # PPI, channel health, waterfalls, max-hold, detection trails
        tb.attach_dashboard()

        max_delay_km = tb.cpi_len * (3e8 / (2 * tb.sample_rate)) / 1000.0
        max_doppler_hz = tb.doppler_res * (tb.doppler_len // 2)
        print("Starting 5-channel KrakenSDR dashboard...")
        print(f"  Channels: {tb.num_surv} surveillance + 1 reference")
        print(f"  Range: 0 - {max_delay_km:.1f} km ({tb.cpi_len} bins, {tb.range_res:.1f} m/bin)")
        print(f"  Doppler: +/- {max_doppler_hz:.1f} Hz ({tb.doppler_len} bins, {tb.doppler_res:.2f} Hz/bin)")
        print("Press Ctrl+C or close window to stop.\n")

        tb.start()
        tb.start_periodic_recal(interval_sec=args.recal_interval)

        try:
            # Matplotlib must run in main thread (TkAgg requirement)
            tb.dashboard.run_display_blocking()
        finally:
            tb.stop_periodic_recal()
            cleanup()

    elif args.visualize and tb.skip_aoa and HAS_RD_DISPLAY:
        # Range-Doppler heatmap display (RSPduo or KrakenSDR --skip-aoa)
        max_doppler_hz = tb.doppler_res * (tb.doppler_len // 2)
        rd_params = RDDisplayParams(
            n_range_bins=tb.sink.max_range_bin,
            n_doppler_bins=tb.doppler_len,
            range_resolution_m=tb.range_res,
            doppler_resolution_hz=tb.doppler_res,
            max_range_km=50.0,
            min_doppler_hz=-max_doppler_hz,
            max_doppler_hz=max_doppler_hz,
        )
        rd_display = RangeDopplerDisplay(rd_params, update_interval_ms=100)
        tb.sink.rd_display_callback = rd_display.update_caf

        print("Starting Range-Doppler display...")
        print(f"  Range: 0 - 50 km ({tb.sink.max_range_bin} bins, {tb.range_res:.1f} m/bin)")
        print(f"  Doppler: +/- {max_doppler_hz:.1f} Hz ({tb.doppler_len} bins, {tb.doppler_res:.2f} Hz/bin)")
        print("Press Ctrl+C or close window to stop.\n")

        # Start flowgraph in main thread so sdrplay_api_Init errors
        # propagate as exceptions instead of being swallowed in a daemon thread.
        tb.start()
        tb.start_periodic_recal(interval_sec=args.recal_interval)

        def wait_gr():
            tb.wait()

        t = threading.Thread(target=wait_gr, daemon=True)
        t.start()

        try:
            rd_display.start()  # Blocks in main thread (matplotlib needs main thread)
        finally:
            tb.stop_periodic_recal()
            cleanup()

    elif args.visualize and HAS_DISPLAY:
        # KrakenSDR mode: PPI display (fallback if dashboard unavailable)
        print("Starting GUI...")
        disp = PPIDisplay()
        tb.display_ref = disp

        def run_gr():
            tb.start()
            tb.wait()

        t = threading.Thread(target=run_gr, daemon=True)
        t.start()
        tb.start_periodic_recal(interval_sec=args.recal_interval)

        try:
            disp.start()
        finally:
            tb.stop_periodic_recal()
            cleanup()

    else:
        print("Running Headless...")
        print("Press Ctrl+C to stop")
        print(f"Calibration status: {tb.get_calibration_status()}")

        # --- Block diagnostics (--diag) ---
        diag = None
        if args.diag:
            diag = BlockDiagnostics(interval_sec=5.0)
            decimated = tb.sample_rate
            source_r = tb.source_rate
            frame_rate = decimated / (tb.cpi_len * tb.doppler_len)

            # Source outputs at source_rate per channel
            diag.add("source[0]", tb.source, port=0, expected_rate=source_r)

            # Resamplers output at decimated rate (before phase corr)
            if tb.resamplers:
                diag.add("resampler[0]", tb.resamplers[0], port=0, expected_rate=decimated)

            # Phase correction (C++ rotator) at decimated rate
            if tb.phase_rotators:
                diag.add("phase_rot[0]", tb.phase_rotators[0], port=0, expected_rate=decimated)

            # Conditioning at decimated rate
            if tb.cond_blocks:
                diag.add("conditioning[0]", tb.cond_blocks[0], port=0, expected_rate=decimated)

            # ECA at decimated rate (scalar output)
            diag.add("eca", tb.eca, port=0, expected_rate=decimated)

            # CAF: output is vectors of cpi_len complex, at decimated/cpi_len rate
            caf_rate = decimated / tb.cpi_len
            if tb.caf_blocks:
                diag.add("caf[0]", tb.caf_blocks[0], port=0, expected_rate=caf_rate)

            # Doppler: 256:1 decimator, output R-D frames
            if tb.doppler_blocks:
                diag.add("doppler[0]", tb.doppler_blocks[0], port=0, expected_rate=frame_rate)

            # CFAR: same rate as doppler
            if tb.cfar_blocks:
                diag.add("cfar[0]", tb.cfar_blocks[0], port=0, expected_rate=frame_rate)

            # Cluster
            if tb.cluster_blocks:
                diag.add("cluster[0]", tb.cluster_blocks[0], port=0, expected_rate=frame_rate)

            # AoA
            if tb.aoa:
                diag.add("aoa", tb.aoa, port=0, expected_rate=frame_rate)

            # Tracker
            if tb.trk:
                diag.add("tracker", tb.trk, port=0, expected_rate=frame_rate)

        # Periodic recalibration setup (KrakenSDR only)
        # With drift rate compensation, recal every 120s is sufficient
        recal_interval = args.recal_interval
        last_recal_time = time.time()

        tb.start()
        if diag:
            diag.start()

        try:
            while True:
                time.sleep(1.0)

                # Periodic recalibration trigger
                if (tb.cal_controller is not None and
                        time.time() - last_recal_time >= recal_interval):
                    tb.trigger_calibration(reason="periodic")
                    last_recal_time = time.time()

                if not args.diag:
                    if tb.trk is not None:
                        n_tracks = tb.trk.get_num_tracks()
                        n_confirmed = tb.trk.get_num_confirmed_tracks()
                        n_dets = tb.cluster_blocks[0].get_num_detections() if tb.cluster_blocks else 0
                        if n_tracks > 0 or n_dets > 0:
                            print(f"  Dets: {n_dets}  Tracks: {n_confirmed} confirmed / {n_tracks} total")
                    else:
                        print("  Range-Doppler mode: processing...", end='\r')
        except KeyboardInterrupt:
            pass
        finally:
            if diag:
                diag.stop()
            cleanup()


if __name__ == "__main__":
    main()
