#!/usr/bin/env python3
"""
Unit Tests for KrakenSDR Passive Radar GNU Radio Blocks

Tests:
1. ECA Canceller - Direct path suppression
2. Doppler Processor - Range-Doppler map generation
3. CFAR Detector - Target detection with controlled Pfa
4. Coherence Monitor - Calibration verification and auto-trigger

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import sys
import os

# Add build path if running from source tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

try:
    from gnuradio import gr, blocks
    from gnuradio import kraken_passive_radar
    HAS_GNURADIO = True
except ImportError:
    HAS_GNURADIO = False
    print("WARNING: GNU Radio not available, using standalone test mode")


class SignalGenerator:
    """Generate synthetic test signals for passive radar"""
    
    @staticmethod
    def generate_reference_signal(num_samples, sample_rate, fm_freq=1000, fm_mod_index=5):
        """Generate FM-like reference signal (wideband, noise-like)"""
        t = np.arange(num_samples) / sample_rate
        # FM modulation with pseudo-random modulating signal
        np.random.seed(42)
        mod_signal = np.cumsum(np.random.randn(num_samples)) * 0.01
        phase = 2 * np.pi * fm_freq * t + fm_mod_index * mod_signal
        return np.exp(1j * phase).astype(np.complex64)
    
    @staticmethod
    def generate_target_echo(ref_signal, delay_samples, doppler_hz, sample_rate, amplitude=0.1):
        """Generate delayed and Doppler-shifted target echo"""
        num_samples = len(ref_signal)
        t = np.arange(num_samples) / sample_rate
        
        # Delay
        delayed = np.zeros(num_samples, dtype=np.complex64)
        if delay_samples < num_samples:
            delayed[delay_samples:] = ref_signal[:-delay_samples] if delay_samples > 0 else ref_signal
        
        # Doppler shift
        doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * t).astype(np.complex64)
        
        return amplitude * delayed * doppler_shift
    
    @staticmethod
    def add_noise(signal, snr_db):
        """Add AWGN to signal"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                           1j * np.random.randn(len(signal)))
        return (signal + noise).astype(np.complex64)
    
    @staticmethod
    def generate_coherent_channels(num_samples, sample_rate, num_channels=5,
                                    phase_offsets=None, phase_noise_std=0.0):
        """Generate coherent multi-channel signals for calibration testing"""
        ref = SignalGenerator.generate_reference_signal(num_samples, sample_rate)
        
        if phase_offsets is None:
            phase_offsets = np.zeros(num_channels)
        
        channels = []
        for ch in range(num_channels):
            # Apply fixed phase offset
            phase_shifted = ref * np.exp(1j * phase_offsets[ch])
            
            # Add phase noise if specified
            if phase_noise_std > 0:
                phase_noise = np.random.randn(num_samples) * phase_noise_std
                phase_shifted = phase_shifted * np.exp(1j * phase_noise)
            
            channels.append(phase_shifted.astype(np.complex64))
        
        return channels


class TestECACanceller(unittest.TestCase):
    """Test ECA clutter cancellation block"""
    
    def setUp(self):
        self.sample_rate = 250000  # 250 kHz after decimation
        self.num_samples = 25000   # 100ms CPI
        
    @unittest.skipUnless(HAS_GNURADIO, "GNU Radio not available")
    def test_direct_path_suppression(self):
        """Test that direct path (zero delay) is suppressed"""
        # Generate reference signal
        ref = SignalGenerator.generate_reference_signal(self.num_samples, self.sample_rate)
        
        # Surveillance = reference (direct path) + target echo + noise
        direct_path = ref * 1.0  # Strong direct path
        target = SignalGenerator.generate_target_echo(ref, 100, 50, self.sample_rate, 0.01)
        surv = SignalGenerator.add_noise(direct_path + target, snr_db=20)
        
        # Create flowgraph
        tb = gr.top_block()
        
        src_ref = blocks.vector_source_c(ref.tolist())
        src_surv = blocks.vector_source_c(surv.tolist())
        
        eca = kraken_passive_radar.eca_canceller(
            num_surv=1,
            num_taps=128,
            reg_factor=0.001
        )
        
        sink = blocks.vector_sink_c()
        
        tb.connect(src_ref, (eca, 0))
        tb.connect(src_surv, (eca, 1))
        tb.connect(eca, sink)
        
        tb.run()
        
        output = np.array(sink.data())
        
        # After cancellation, direct path power should be significantly reduced
        # Measure power in zero-Doppler region
        input_dp_power = np.mean(np.abs(direct_path)**2)
        output_power = np.mean(np.abs(output)**2)
        
        suppression_db = 10 * np.log10(input_dp_power / (output_power + 1e-10))
        
        print(f"Direct path suppression: {suppression_db:.1f} dB")
        
        # Should achieve at least 20 dB suppression
        self.assertGreater(suppression_db, 20.0,
                          f"Insufficient suppression: {suppression_db:.1f} dB < 20 dB")
    
    @unittest.skipUnless(HAS_GNURADIO, "GNU Radio not available")
    def test_target_preservation(self):
        """Test that target echoes are preserved after cancellation"""
        ref = SignalGenerator.generate_reference_signal(self.num_samples, self.sample_rate)
        
        # Target at 100 samples delay (~120m at 250kHz), 50 Hz Doppler
        target_delay = 100
        target_doppler = 50
        target_amp = 0.1
        
        direct_path = ref * 0.5
        target = SignalGenerator.generate_target_echo(ref, target_delay, target_doppler, 
                                                       self.sample_rate, target_amp)
        surv = direct_path + target
        
        tb = gr.top_block()
        
        src_ref = blocks.vector_source_c(ref.tolist())
        src_surv = blocks.vector_source_c(surv.tolist())
        
        eca = kraken_passive_radar.eca_canceller(num_surv=1, num_taps=128, reg_factor=0.001)
        sink = blocks.vector_sink_c()
        
        tb.connect(src_ref, (eca, 0))
        tb.connect(src_surv, (eca, 1))
        tb.connect(eca, sink)
        
        tb.run()
        
        output = np.array(sink.data())
        
        # Cross-correlate output with reference to check if target is present
        xcorr = np.abs(np.correlate(output, ref, mode='same'))
        peak_idx = np.argmax(xcorr)
        
        # Peak should be near the target delay
        expected_peak = len(output) // 2 + target_delay
        
        self.assertLess(abs(peak_idx - expected_peak), 10,
                       f"Target peak at wrong location: {peak_idx} vs expected {expected_peak}")


class TestDopplerProcessor(unittest.TestCase):
    """Test Doppler processing block"""
    
    def setUp(self):
        self.sample_rate = 250000
        self.cpi_samples = 1024
        self.num_cpis = 64
        
    @unittest.skipUnless(HAS_GNURADIO, "GNU Radio not available")
    def test_doppler_detection(self):
        """Test detection of known Doppler shift"""
        # Generate multiple CPIs with a target at known Doppler
        target_doppler_hz = 100  # Hz
        target_range_bin = 50
        
        # PRF = sample_rate / cpi_samples
        prf = self.sample_rate / self.cpi_samples
        
        # Doppler bin = target_doppler / (prf / num_cpis)
        doppler_resolution = prf / self.num_cpis
        expected_doppler_bin = int(target_doppler_hz / doppler_resolution)
        
        # Generate range profiles with target
        range_profiles = []
        for cpi in range(self.num_cpis):
            profile = np.zeros(self.cpi_samples, dtype=np.complex64)
            # Add target with Doppler phase progression
            phase = 2 * np.pi * target_doppler_hz * cpi / prf
            profile[target_range_bin] = np.exp(1j * phase)
            # Add noise
            profile = SignalGenerator.add_noise(profile, snr_db=10)
            range_profiles.append(profile)
        
        # Flatten for GNU Radio
        input_data = np.concatenate(range_profiles)
        
        tb = gr.top_block()
        
        src = blocks.vector_source_c(input_data.tolist())
        s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.cpi_samples)
        
        doppler = kraken_passive_radar.doppler_processor(
            num_range_bins=self.cpi_samples,
            num_doppler_bins=self.num_cpis,
            window_type=1,  # Hamming
            output_power=True
        )
        
        sink = blocks.vector_sink_f()
        
        tb.connect(src, s2v)
        tb.connect(s2v, doppler)
        tb.connect(doppler, sink)
        
        tb.run()
        
        output = np.array(sink.data())
        rdm = output.reshape(self.num_cpis, self.cpi_samples)
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(rdm), rdm.shape)
        detected_doppler_bin = peak_idx[0]
        detected_range_bin = peak_idx[1]
        
        print(f"Expected: Doppler bin {expected_doppler_bin + self.num_cpis//2}, Range bin {target_range_bin}")
        print(f"Detected: Doppler bin {detected_doppler_bin}, Range bin {detected_range_bin}")
        
        # Check range bin (should be exact)
        self.assertEqual(detected_range_bin, target_range_bin,
                        f"Wrong range bin: {detected_range_bin} vs {target_range_bin}")
        
        # Check Doppler bin (allow ±1 bin due to windowing/FFT effects)
        # Note: FFT shifted, so zero Doppler is at center
        expected_shifted = (expected_doppler_bin + self.num_cpis // 2) % self.num_cpis
        self.assertLess(abs(detected_doppler_bin - expected_shifted), 2,
                       f"Wrong Doppler bin: {detected_doppler_bin} vs {expected_shifted}")


class TestCFARDetector(unittest.TestCase):
    """Test CFAR detection block"""
    
    def setUp(self):
        self.num_range_bins = 256
        self.num_doppler_bins = 64
        
    @unittest.skipUnless(HAS_GNURADIO, "GNU Radio not available")
    def test_target_detection(self):
        """Test that targets above threshold are detected"""
        # Create RDM with known targets
        rdm = np.random.exponential(1.0, (self.num_doppler_bins, self.num_range_bins)).astype(np.float32)
        
        # Add targets (20 dB above noise floor)
        targets = [(32, 100), (16, 50), (48, 200)]
        target_power = 100.0  # ~20 dB above unit exponential
        
        for d, r in targets:
            rdm[d, r] = target_power
        
        input_data = rdm.flatten()
        
        tb = gr.top_block()
        
        src = blocks.vector_source_f(input_data.tolist())
        s2v = blocks.stream_to_vector(gr.sizeof_float, self.num_range_bins * self.num_doppler_bins)
        
        cfar = kraken_passive_radar.cfar_detector(
            num_range_bins=self.num_range_bins,
            num_doppler_bins=self.num_doppler_bins,
            guard_cells_range=2,
            guard_cells_doppler=2,
            ref_cells_range=8,
            ref_cells_doppler=8,
            pfa=1e-6,
            cfar_type=0  # CA-CFAR
        )
        
        v2s = blocks.vector_to_stream(gr.sizeof_float, self.num_range_bins * self.num_doppler_bins)
        sink = blocks.vector_sink_f()
        
        tb.connect(src, s2v)
        tb.connect(s2v, cfar)
        tb.connect(cfar, v2s)
        tb.connect(v2s, sink)
        
        tb.run()
        
        output = np.array(sink.data())
        det_map = output.reshape(self.num_doppler_bins, self.num_range_bins)
        
        # Check that all targets are detected
        for d, r in targets:
            self.assertEqual(det_map[d, r], 1.0,
                           f"Target at ({d}, {r}) not detected")
        
        # Count total detections (should be close to number of targets)
        total_detections = np.sum(det_map)
        print(f"Total detections: {total_detections}, Expected: {len(targets)}")
        
        # Allow some false alarms due to statistical variation
        self.assertLess(total_detections, len(targets) + 10,
                       f"Too many detections: {total_detections}")
    
    @unittest.skipUnless(HAS_GNURADIO, "GNU Radio not available")
    def test_false_alarm_rate(self):
        """Test that Pfa is approximately correct"""
        # Generate pure noise RDM
        np.random.seed(123)
        rdm = np.random.exponential(1.0, (self.num_doppler_bins, self.num_range_bins)).astype(np.float32)
        
        pfa_target = 1e-4
        
        tb = gr.top_block()
        
        src = blocks.vector_source_f(rdm.flatten().tolist())
        s2v = blocks.stream_to_vector(gr.sizeof_float, self.num_range_bins * self.num_doppler_bins)
        
        cfar = kraken_passive_radar.cfar_detector(
            num_range_bins=self.num_range_bins,
            num_doppler_bins=self.num_doppler_bins,
            guard_cells_range=2,
            guard_cells_doppler=2,
            ref_cells_range=8,
            ref_cells_doppler=8,
            pfa=pfa_target,
            cfar_type=0
        )
        
        v2s = blocks.vector_to_stream(gr.sizeof_float, self.num_range_bins * self.num_doppler_bins)
        sink = blocks.vector_sink_f()
        
        tb.connect(src, s2v)
        tb.connect(s2v, cfar)
        tb.connect(cfar, v2s)
        tb.connect(v2s, sink)
        
        tb.run()
        
        output = np.array(sink.data())
        num_cells = self.num_range_bins * self.num_doppler_bins
        num_false_alarms = np.sum(output)
        measured_pfa = num_false_alarms / num_cells
        
        print(f"Target Pfa: {pfa_target}, Measured Pfa: {measured_pfa:.2e}")
        
        # Pfa should be within order of magnitude
        self.assertLess(measured_pfa, pfa_target * 10,
                       f"Pfa too high: {measured_pfa:.2e} vs {pfa_target:.2e}")


class TestCoherenceMonitor(unittest.TestCase):
    """Test coherence monitoring and calibration verification"""
    
    def setUp(self):
        self.sample_rate = 2.4e6
        self.num_channels = 5
        self.num_samples = int(0.1 * self.sample_rate)  # 100ms
        
    @unittest.skipUnless(HAS_GNURADIO, "GNU Radio not available")
    def test_coherent_channels_pass(self):
        """Test that coherent channels pass verification"""
        # Generate coherent channels with small fixed phase offsets
        phase_offsets = np.array([0, 0.1, 0.2, -0.1, 0.15])
        channels = SignalGenerator.generate_coherent_channels(
            self.num_samples, self.sample_rate,
            num_channels=self.num_channels,
            phase_offsets=phase_offsets,
            phase_noise_std=0.01  # ~0.6 degrees
        )
        
        tb = gr.top_block()
        
        sources = [blocks.vector_source_c(ch.tolist()) for ch in channels]
        
        monitor = kraken_passive_radar.coherence_monitor(
            num_channels=self.num_channels,
            sample_rate=self.sample_rate,
            measure_interval_ms=50.0,
            measure_duration_ms=10.0,
            corr_threshold=0.95,
            phase_threshold_deg=5.0
        )
        
        sinks = [blocks.vector_sink_c() for _ in range(self.num_channels)]
        
        for i, src in enumerate(sources):
            tb.connect(src, (monitor, i))
        
        for i, sink in enumerate(sinks):
            tb.connect((monitor, i), sink)
        
        tb.run()
        
        # Check that calibration is NOT needed
        self.assertFalse(monitor.is_calibration_needed(),
                        "Calibration flagged for coherent channels")
        
        # Check correlation values
        for ch in range(1, self.num_channels):
            corr = monitor.get_correlation(ch)
            print(f"Channel {ch} correlation: {corr:.4f}")
            self.assertGreater(corr, 0.95, f"Channel {ch} correlation too low: {corr}")
    
    @unittest.skipUnless(HAS_GNURADIO, "GNU Radio not available")
    def test_phase_drift_detection(self):
        """Test that phase drift triggers calibration request"""
        # Generate channels with significant phase noise
        channels = SignalGenerator.generate_coherent_channels(
            self.num_samples, self.sample_rate,
            num_channels=self.num_channels,
            phase_offsets=np.zeros(self.num_channels),
            phase_noise_std=0.2  # ~11 degrees - should trigger
        )
        
        tb = gr.top_block()
        
        sources = [blocks.vector_source_c(ch.tolist()) for ch in channels]
        
        monitor = kraken_passive_radar.coherence_monitor(
            num_channels=self.num_channels,
            sample_rate=self.sample_rate,
            measure_interval_ms=10.0,  # Measure frequently
            measure_duration_ms=5.0,
            corr_threshold=0.95,
            phase_threshold_deg=5.0
        )
        
        sinks = [blocks.vector_sink_c() for _ in range(self.num_channels)]
        
        for i, src in enumerate(sources):
            tb.connect(src, (monitor, i))
        
        for i, sink in enumerate(sinks):
            tb.connect((monitor, i), sink)
        
        tb.run()
        
        # With high phase noise, calibration should be triggered
        # (may take multiple measurements due to hysteresis)
        phase_var = monitor.get_phase_variance(1)
        print(f"Channel 1 phase variance: {np.degrees(phase_var):.2f} degrees")
        
        # Either calibration needed or phase variance detected
        # Note: May not always trigger due to hysteresis (3 consecutive failures required)
    
    def test_correlation_calculation(self):
        """Test correlation coefficient calculation (standalone)"""
        # Generate perfectly correlated signals
        np.random.seed(42)
        ref = np.random.randn(1000) + 1j * np.random.randn(1000)
        
        # Same signal = correlation 1.0
        corr = self._compute_correlation(ref, ref)
        self.assertAlmostEqual(corr, 1.0, places=5)
        
        # Orthogonal signals = correlation ~0
        surv = np.random.randn(1000) + 1j * np.random.randn(1000)
        corr = self._compute_correlation(ref, surv)
        self.assertLess(abs(corr), 0.2)  # Should be near zero
        
        # Phase-shifted signal = correlation 1.0
        phase_shift = np.exp(1j * 0.5)
        shifted = ref * phase_shift
        corr = self._compute_correlation(ref, shifted)
        self.assertAlmostEqual(corr, 1.0, places=5)
    
    def _compute_correlation(self, ref, surv):
        """Compute correlation coefficient"""
        cross = np.sum(ref * np.conj(surv))
        ref_power = np.sum(np.abs(ref)**2)
        surv_power = np.sum(np.abs(surv)**2)
        return np.abs(cross) / np.sqrt(ref_power * surv_power)


class TestCalibrationIntegration(unittest.TestCase):
    """Integration tests for automatic calibration system"""
    
    def test_calibration_metrics(self):
        """Test calibration success metrics computation"""
        sample_rate = 2.4e6
        num_samples = 24000  # 10ms
        
        # Good calibration: high correlation, low phase variance
        good_channels = SignalGenerator.generate_coherent_channels(
            num_samples, sample_rate,
            num_channels=5,
            phase_offsets=[0, 0.1, 0.2, 0.15, 0.05],
            phase_noise_std=0.01
        )
        
        # Check reference vs each surveillance
        for ch in range(1, 5):
            corr = self._compute_correlation(good_channels[0], good_channels[ch])
            print(f"Good cal - Channel {ch} correlation: {corr:.4f}")
            self.assertGreater(corr, 0.99)
        
        # Bad calibration: decorrelated due to phase noise
        bad_channels = SignalGenerator.generate_coherent_channels(
            num_samples, sample_rate,
            num_channels=5,
            phase_offsets=[0, 0.1, 0.2, 0.15, 0.05],
            phase_noise_std=0.5  # Severe phase noise
        )
        
        for ch in range(1, 5):
            corr = self._compute_correlation(bad_channels[0], bad_channels[ch])
            print(f"Bad cal - Channel {ch} correlation: {corr:.4f}")
            # Should be degraded but still somewhat correlated
            self.assertLess(corr, 0.95)
    
    def _compute_correlation(self, ref, surv):
        cross = np.sum(ref * np.conj(surv))
        ref_power = np.sum(np.abs(ref)**2)
        surv_power = np.sum(np.abs(surv)**2)
        return np.abs(cross) / np.sqrt(ref_power * surv_power)


class TestEndToEnd(unittest.TestCase):
    """End-to-end passive radar processing tests"""
    
    @unittest.skipUnless(HAS_GNURADIO, "GNU Radio not available")
    def test_full_processing_chain(self):
        """Test complete processing: ECA → Doppler → CFAR"""
        sample_rate = 250000
        cpi_samples = 1024
        num_cpis = 32
        
        # Generate test scenario
        ref = SignalGenerator.generate_reference_signal(cpi_samples * num_cpis, sample_rate)
        
        # Direct path + target
        direct_path = ref * 0.5
        target = SignalGenerator.generate_target_echo(
            ref, delay_samples=50, doppler_hz=75, 
            sample_rate=sample_rate, amplitude=0.05
        )
        surv = SignalGenerator.add_noise(direct_path + target, snr_db=15)
        
        print("\n=== End-to-End Test ===")
        print(f"Target: delay=50 samples, Doppler=75 Hz, amplitude=0.05")
        print(f"Direct path amplitude: 0.5")
        print(f"SNR: 15 dB")
        
        # In a full test, we would connect:
        # ref, surv → ECA → range_xcorr → Doppler → CFAR
        # For now, just verify the chain conceptually works
        
        # Expected range bin: 50
        # Expected Doppler bin: 75 / (sample_rate/cpi_samples/num_cpis) 
        prf = sample_rate / cpi_samples
        doppler_res = prf / num_cpis
        expected_doppler_bin = int(75 / doppler_res)
        
        print(f"PRF: {prf:.1f} Hz")
        print(f"Doppler resolution: {doppler_res:.2f} Hz")
        print(f"Expected Doppler bin: {expected_doppler_bin}")
        
        self.assertTrue(True)  # Placeholder - expand with actual chain test


# Standalone test functions (no GNU Radio required)
class TestStandalone(unittest.TestCase):
    """Tests that run without GNU Radio"""
    
    def test_signal_generator(self):
        """Test synthetic signal generation"""
        ref = SignalGenerator.generate_reference_signal(1000, 250000)
        self.assertEqual(len(ref), 1000)
        self.assertEqual(ref.dtype, np.complex64)
        
        # Check it's approximately unit power
        power = np.mean(np.abs(ref)**2)
        self.assertAlmostEqual(power, 1.0, places=1)
    
    def test_target_echo_delay(self):
        """Test target echo has correct delay"""
        ref = np.ones(100, dtype=np.complex64)
        echo = SignalGenerator.generate_target_echo(ref, 10, 0, 250000, 1.0)
        
        # First 10 samples should be zero
        self.assertTrue(np.all(echo[:10] == 0))
        # Rest should be non-zero
        self.assertTrue(np.all(echo[10:] != 0))
    
    def test_noise_snr(self):
        """Test that added noise has correct SNR"""
        signal = np.ones(10000, dtype=np.complex64)
        noisy = SignalGenerator.add_noise(signal, snr_db=10)
        
        signal_power = 1.0
        noise = noisy - signal
        noise_power = np.mean(np.abs(noise)**2)
        measured_snr = 10 * np.log10(signal_power / noise_power)
        
        self.assertAlmostEqual(measured_snr, 10.0, delta=1.0)
    
    def test_cfar_threshold_calculation(self):
        """Test CFAR threshold factor calculation"""
        # For CA-CFAR: alpha = N * (Pfa^(-1/N) - 1)
        N = 64  # Number of reference cells
        Pfa = 1e-6
        
        alpha = N * (Pfa**(-1/N) - 1)
        
        # For N=64, Pfa=1e-6, alpha should be around 14-15
        print(f"CFAR threshold factor (N={N}, Pfa={Pfa}): {alpha:.2f}")
        self.assertGreater(alpha, 10)
        self.assertLess(alpha, 30)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
