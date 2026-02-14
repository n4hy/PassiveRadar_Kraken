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
import signal
import numpy as np
from gnuradio import gr, blocks
import os
import argparse
import json
import threading

# Import all blocks from installed gnuradio.kraken_passive_radar
from gnuradio.kraken_passive_radar import (
    # Python blocks
    krakensdr_source,
    rspdx_source,
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
)

# New Display (optional)
try:
    # Try installed version first
    from gnuradio.kraken_passive_radar.radar_display import PPIDisplay
    HAS_DISPLAY = True
except ImportError:
    # Fall back to local version
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "gr-kraken_passive_radar/python"))
        from kraken_passive_radar.radar_display import PPIDisplay
        HAS_DISPLAY = True
    except ImportError:
        HAS_DISPLAY = False


class PhaseCorrectorBlock(gr.sync_block):
    """
    GNU Radio block that applies phase corrections from CalibrationController.

    This block sits between the KrakenSDR source and the conditioning/ECA blocks.
    It ensures phase coherence is maintained by:
    1. Applying stored phase corrections during normal operation
    2. Feeding samples to CalibrationController during calibration
    3. Outputting zeros during calibration (antennas are isolated anyway)

    IMPORTANT: This MUST be in the signal path BEFORE ECA processing.
    The ECA clutter canceller requires phase-coherent inputs to work correctly.
    """

    def __init__(self, cal_controller: CalibrationController, channel: int):
        gr.sync_block.__init__(
            self,
            name=f"Phase Corrector Ch{channel}",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.cal_controller = cal_controller
        self.channel = channel

    def work(self, input_items, output_items):
        samples = input_items[0]
        n_samples = len(samples)

        if self.cal_controller.is_calibrating:
            # During calibration: feed samples to controller and output zeros
            # (The KrakenSDR HW switch has isolated antennas, so samples are
            # from the internal noise source, not antennas)
            self.cal_controller.process_calibration_samples(self.channel, samples)
            output_items[0][:n_samples] = 0
        else:
            # Normal operation: apply phase correction
            output_items[0][:n_samples] = self.cal_controller.apply_phase_correction(
                self.channel, samples
            )

        return n_samples


class PassiveRadarTopBlock(gr.top_block):
    def __init__(self, freq=100e6, gain=30, geometry='ULA', calibration_file=None, use_ref_aoa=False,
                 b3_signal_type='passthrough', b3_fft_size=8192, b3_guard_interval=192,
                 source_type='kraken', if_gain=40.0, rf_gain=0.0,
                 antenna='Antenna A', bandwidth=0, sample_rate=2.4e6):
        gr.top_block.__init__(self, "Passive Radar Run")

        # Parameters
        self.freq = freq
        self.gain = gain
        self.source_type = source_type
        self.sample_rate = sample_rate
        self.cpi_len = 4096
        self.doppler_len = 64
        self.num_taps = 128
        self.geometry = geometry
        self.use_ref_aoa = use_ref_aoa
        self.b3_signal_type = b3_signal_type
        self.b3_fft_size = b3_fft_size
        self.b3_guard_interval = b3_guard_interval

        # Source-dependent parameters
        if source_type == 'rspdx':
            self.num_surv = 1   # Self-referencing: single channel
            self.skip_aoa = True
            self.skip_calibration = True
        else:
            self.num_surv = 4
            self.skip_aoa = False
            self.skip_calibration = False

        # Derived parameters
        c = 3e8
        wavelength = c / freq
        range_res = c / (2 * self.sample_rate)  # meters per range bin
        doppler_res = self.sample_rate / (self.cpi_len * self.doppler_len)  # Hz per Doppler bin
        frame_period = (self.cpi_len * self.doppler_len) / self.sample_rate

        # Load Calibration
        self.phase_offsets = np.zeros(5, dtype=np.float32)
        if calibration_file and os.path.exists(calibration_file):
            print(f"Loading calibration from {calibration_file}...")
            with open(calibration_file, 'r') as f:
                cal = json.load(f)
                for i in range(5):
                    self.phase_offsets[i] = cal.get(f"ch{i}", 0.0)
        else:
            print("No calibration loaded. Using default (0).")

        # 1. Source
        if self.source_type == 'rspdx':
            self.source = rspdx_source(
                frequency=freq, sample_rate=self.sample_rate,
                if_gain=if_gain, rf_gain=rf_gain,
                antenna=antenna, bandwidth=bandwidth
            )
        else:
            self.source = krakensdr_source(
                frequency=freq, gain=gain,
                sample_rate=self.sample_rate
            )

        # 1b. Calibration Controller (KrakenSDR only)
        if not self.skip_calibration:
            self.cal_controller = CalibrationController(
                source=self.source,
                num_channels=5,
                cal_samples=24000,  # 10ms at 2.4 MHz
                settle_time_ms=50.0,
                on_calibration_complete=self._on_calibration_complete
            )

            # 1c. Phase Corrector Blocks (one per channel)
            self.phase_correctors = []
            for i in range(5):
                blk = PhaseCorrectorBlock(
                    cal_controller=self.cal_controller,
                    channel=i
                )
                self.phase_correctors.append(blk)
                self.connect((self.source, i), (blk, 0))
        else:
            self.cal_controller = None
            self.phase_correctors = None

        # 2. Conditioning
        self.cond_blocks = []
        if self.source_type == 'rspdx':
            # RSPdx: 1 channel, conditioned once
            blk = ConditioningBlock(rate=1e-5)
            self.cond_blocks.append(blk)
            self.connect((self.source, 0), (blk, 0))
        else:
            # KrakenSDR: 5 channels through phase correctors
            for i in range(5):
                blk = ConditioningBlock(rate=1e-5)
                self.cond_blocks.append(blk)
                self.connect((self.phase_correctors[i], 0), (blk, 0))

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

        # 2b. Time Alignment Probe (Calibration)
        self.align_blocks = []
        if self.source_type == 'rspdx':
            # RSPdx: single channel, no time alignment needed (self-referencing)
            pass
        else:
            for i in range(self.num_surv):
                blk = TimeAlignmentBlock(n_samples=self.cpi_len, interval_sec=5.0)
                self.align_blocks.append(blk)
                self.connect(self.reference_channel, (blk, 0))  # Block B3 output
                self.connect((self.cond_blocks[i+1], 0), (blk, 1))

        # 3. ECA Clutter Canceller (C++ VOLK-accelerated)
        # Input: port 0 = reference (from Block B3), ports 1..num_surv = surveillance
        # Output: ports 0..num_surv-1 = clutter-cancelled surveillance
        self.eca = eca_canceller(
            num_taps=self.num_taps,
            reg_factor=0.001,
            num_surv=self.num_surv
        )
        self.connect(self.reference_channel, (self.eca, 0))  # Block B3 output
        if self.source_type == 'rspdx':
            # Self-referencing: same conditioned signal feeds surveillance input
            self.connect((self.cond_blocks[0], 0), (self.eca, 1))
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
        self.doppler_blocks = []
        for i in range(self.num_surv):
            blk = doppler_processor.make(
                num_range_bins=self.cpi_len,
                num_doppler_bins=self.doppler_len,
                window_type=1,       # Hamming
                output_power=True    # Output |X|^2 for CFAR
            )
            self.doppler_blocks.append(blk)
            self.connect((self.caf_blocks[i], 0), (blk, 0))

        # 6. CFAR Detector (C++ accelerated) - per surveillance channel
        self.cfar_blocks = []
        for i in range(self.num_surv):
            blk = cfar_detector.make(
                num_range_bins=self.cpi_len,
                num_doppler_bins=self.doppler_len,
                guard_cells_range=2,
                guard_cells_doppler=2,
                ref_cells_range=8,
                ref_cells_doppler=8,
                pfa=1e-6,
                cfar_type=0          # CA-CFAR
            )
            self.cfar_blocks.append(blk)
            self.connect((self.doppler_blocks[i], 0), (blk, 0))

        # 7. Detection Clustering (C++ accelerated) - per surveillance channel
        self.cluster_blocks = []
        for i in range(self.num_surv):
            blk = detection_cluster.make(
                num_range_bins=self.cpi_len,
                num_doppler_bins=self.doppler_len,
                min_cluster_size=1,
                max_cluster_extent=50,
                range_resolution_m=range_res,
                doppler_resolution_hz=doppler_res,
                max_detections=100
            )
            self.cluster_blocks.append(blk)
            # Cluster takes CFAR mask (input 0) and power map (input 1)
            self.connect((self.cfar_blocks[i], 0), (blk, 0))
            self.connect((self.doppler_blocks[i], 0), (blk, 1))

        if not self.skip_aoa:
            # 8. AoA Estimator (C++ Bartlett beamformer)
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
                max_detections=100
            )
            # AoA takes detection lists from all surveillance channels
            for i in range(self.num_surv):
                self.connect((self.cluster_blocks[i], 0), (self.aoa, i))

            # 9. Multi-Target Tracker (C++ Kalman + GNN)
            self.trk = tracker.make(
                dt=frame_period,
                process_noise_range=50.0,
                process_noise_doppler=5.0,
                meas_noise_range=range_res * 2,
                meas_noise_doppler=doppler_res * 2,
                gate_threshold=9.21,     # chi2(2) @ 99%
                confirm_hits=3,
                delete_misses=5,
                max_tracks=50,
                max_detections=100
            )
            self.connect((self.aoa, 0), (self.trk, 0))

            # 10. Sink
            self.sink = RadarSink(
                tracker_block=self.trk,
                display_callback=self.update_display
            )
            self.connect((self.trk, 0), (self.sink, 0))
        else:
            # RSPdx single-channel mode: skip AoA and tracker
            # (detection_cluster stride=10 vs tracker stride=12 mismatch)
            self.aoa = None
            self.trk = None

            # Sink consumes Doppler power map for display
            self.sink = RadarSink(
                tracker_block=None,
                display_callback=self.update_display
            )
            self.connect((self.doppler_blocks[0], 0), (self.sink, 0))

    def update_display(self, tracks):
        """Update display with confirmed tracks from the tracker."""
        mapped = []
        for t in tracks:
            mapped.append((
                t.get('aoa_deg', 0.0),
                t['range_m'],
                t.get('snr_db', 0.0)
            ))
        if hasattr(self, 'display_ref') and self.display_ref:
            self.display_ref.update_detections(mapped)

    def _on_calibration_complete(self, result):
        """
        Callback when automatic calibration completes.

        Args:
            result: CalibrationResult with phase offsets and correlation
        """
        print(f"Calibration #{self.cal_controller.calibration_count} complete:")
        print(f"  Phase corrections (deg): {np.degrees(result.phase_offsets)}")
        print(f"  Correlations: {result.correlation}")

        # Save calibration to file for persistence
        cal_data = {
            f"ch{i}": float(result.phase_offsets[i])
            for i in range(5)
        }
        cal_data['timestamp'] = result.timestamp
        cal_data['calibration_count'] = self.cal_controller.calibration_count

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


class RadarSink(gr.sync_block):
    def __init__(self, tracker_block=None, display_callback=None):
        gr.sync_block.__init__(
            self,
            name="Radar Sink",
            in_sig=[np.float32],
            out_sig=None
        )
        self.tracker_block = tracker_block
        self.callback = display_callback
        self.last_display_time = 0
        self.display_interval = 0.1  # 10 Hz display update

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
        return n

def main():
    parser = argparse.ArgumentParser(
        description="Passive Bistatic Radar using KrakenSDR or RSPdx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Signal Flow (KrakenSDR):
  KrakenSDR -> Phase Correction -> AGC -> ECA -> CAF -> Doppler -> CFAR -> AoA -> Tracker

Signal Flow (RSPdx, self-referencing):
  RSPdx -> AGC -> ECA (ref=surv) -> CAF -> Doppler -> CFAR -> Display
        """
    )
    parser.add_argument("--source", choices=['kraken', 'rspdx'], default='kraken',
                        help="SDR source: kraken (5-ch KrakenSDR) or rspdx (1-ch SDRplay RSPdx)")
    parser.add_argument("--freq", type=float, default=100e6,
                        help="Center frequency in Hz (default: 100 MHz)")
    parser.add_argument("--gain", type=float, default=30,
                        help="Receiver gain in dB (default: 30, KrakenSDR only)")
    parser.add_argument("--geometry", choices=['ULA', 'URA'], default='ULA',
                        help="Antenna array geometry (default: ULA)")
    parser.add_argument("--include-ref", action="store_true",
                        help="Include Reference antenna in AoA array")
    parser.add_argument("--no-startup-cal", action="store_true",
                        help="Skip startup calibration (use saved calibration only)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show GUI display")

    # RSPdx-specific arguments
    parser.add_argument("--if-gain", type=float, default=40.0,
                        help="RSPdx IF gain IFGR in dB (default: 40, range: 20-59)")
    parser.add_argument("--rf-gain", type=float, default=0.0,
                        help="RSPdx RF gain reduction RFGR in dB (default: 0, range: 0-27)")
    parser.add_argument("--antenna", default='Antenna A',
                        choices=['Antenna A', 'Antenna B', 'Antenna C'],
                        help="RSPdx antenna port (default: Antenna A)")
    parser.add_argument("--bandwidth", type=float, default=0,
                        help="RSPdx analog bandwidth in Hz (0=auto)")
    parser.add_argument("--sample-rate", type=float, default=2.4e6,
                        help="Sample rate in Hz (default: 2.4e6)")

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

    tb = PassiveRadarTopBlock(
        freq=args.freq,
        gain=args.gain,
        geometry=args.geometry,
        calibration_file="calibration.json",
        use_ref_aoa=args.include_ref,
        b3_signal_type=args.b3_signal,
        b3_fft_size=args.b3_fft_size,
        b3_guard_interval=args.b3_guard_interval,
        source_type=args.source,
        if_gain=args.if_gain,
        rf_gain=args.rf_gain,
        antenna=args.antenna,
        bandwidth=args.bandwidth,
        sample_rate=args.sample_rate,
    )

    if args.source == 'rspdx':
        print("\n" + "="*60)
        print("RSPdx MODE: Single-channel self-referencing passive radar")
        print("="*60)
        print(f"  Antenna: {args.antenna}")
        print(f"  IF Gain: {args.if_gain} dB, RF Gain Reduction: {args.rf_gain} dB")
        print(f"  Sample Rate: {args.sample_rate/1e6:.3f} MHz")
        print(f"  Phase calibration: SKIPPED (single channel)")
        print(f"  AoA estimation: SKIPPED (single antenna element)\n")
    elif not args.no_startup_cal:
        # Startup calibration (KrakenSDR only)
        # This uses the integrated CalibrationController which:
        # 1. Enables noise source (HW switch isolates antennas)
        # 2. Captures calibration samples
        # 3. Computes phase corrections
        # 4. Disables noise source (HW switch reconnects antennas)
        print("\n" + "="*60)
        print("STARTUP CALIBRATION")
        print("Enabling noise source (antennas will be isolated by HW switch)")
        print("="*60 + "\n")
        tb.trigger_calibration("startup")
        # Wait for calibration to complete
        while tb.cal_controller.is_calibrating:
            time.sleep(0.1)
        print("\nStartup calibration complete. Normal operation starting.\n")
    else:
        print("Skipping startup calibration (using saved calibration if available)")

    if args.source == 'rspdx':
        print(f"Processing chain: RSPdx -> AGC -> Block B3 ({args.b3_signal}) -> "
              f"ECA (self-ref) -> CAF -> Doppler -> CFAR -> Cluster -> Display\n")
    else:
        print(f"\nProcessing chain: Source -> Phase Corr -> AGC -> Block B3 ({args.b3_signal}) -> ECA (C++) -> CAF -> "
              f"Doppler (C++) -> CFAR (C++) -> Cluster (C++) -> AoA (C++) -> Tracker (C++)\n")

    if args.b3_signal != 'passthrough':
        print(f"Block B3 Reference Reconstruction: {args.b3_signal.upper()}")
        print(f"  Expected SNR improvement: 10-20 dB")
        print(f"  Initial SNR estimate: {tb.b3_recon.get_snr_estimate():.1f} dB\n")

    if args.visualize and HAS_DISPLAY:
        print("Starting GUI...")
        disp = PPIDisplay()
        tb.display_ref = disp

        def run_gr():
            tb.start()
            tb.wait()

        t = threading.Thread(target=run_gr, daemon=True)
        t.start()

        disp.start()
    else:
        print("Running Headless...")
        print("Press Ctrl+C to stop")
        print(f"Calibration status: {tb.get_calibration_status()}")
        tb.start()
        try:
            while True:
                time.sleep(1.0)
                # Print tracker status periodically (if tracker is active)
                if tb.trk is not None:
                    n_tracks = tb.trk.get_num_tracks()
                    n_confirmed = tb.trk.get_num_confirmed_tracks()
                    if n_tracks > 0:
                        print(f"  Tracks: {n_confirmed} confirmed / {n_tracks} total")
                else:
                    print("  RSPdx mode: processing...", end='\r')
        except KeyboardInterrupt:
            pass
        tb.stop()
        tb.wait()

if __name__ == "__main__":
    main()
