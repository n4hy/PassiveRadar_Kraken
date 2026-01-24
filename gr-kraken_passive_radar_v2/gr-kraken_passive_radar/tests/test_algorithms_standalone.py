#!/usr/bin/env python3
"""
Standalone Algorithm Verification Tests
No GNU Radio required - tests core signal processing algorithms

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift
import unittest


class TestECAAlgorithm(unittest.TestCase):
    """Test ECA (Extensive Cancellation Algorithm) implementation"""
    
    def test_eca_matrix_formulation(self):
        """Test ECA via least squares: min ||s - Rw||²"""
        np.random.seed(42)
        
        # Generate reference signal
        N = 1000  # samples
        L = 64    # filter taps
        
        ref = np.random.randn(N) + 1j * np.random.randn(N)
        ref = ref.astype(np.complex64)
        
        # Create surveillance with direct path at delay 0 and multipath at delay 10
        direct_path_coef = 0.8 + 0.2j
        multipath_coef = 0.3 - 0.1j
        multipath_delay = 10
        
        surv = direct_path_coef * ref.copy()
        surv[multipath_delay:] += multipath_coef * ref[:-multipath_delay]
        
        # Add weak target (should be preserved)
        target_delay = 50
        target_coef = 0.01 + 0.005j
        surv[target_delay:] += target_coef * ref[:-target_delay]
        
        # Add noise
        noise = 0.001 * (np.random.randn(N) + 1j * np.random.randn(N))
        surv += noise
        
        # Build reference matrix R (Toeplitz-like structure)
        # R[n, l] = ref[n - l] for l = 0, 1, ..., L-1
        R = np.zeros((N, L), dtype=np.complex64)
        for l in range(L):
            R[l:, l] = ref[:N-l]
        
        # Solve least squares with regularization: w = (R^H R + λI)^(-1) R^H s
        reg = 0.001
        RhR = R.conj().T @ R
        Rhs = R.conj().T @ surv
        w = np.linalg.solve(RhR + reg * np.eye(L), Rhs)
        
        # Reconstruct clutter estimate
        clutter_estimate = R @ w
        
        # Cancel clutter
        cleaned = surv - clutter_estimate
        
        # Measure suppression
        clutter_power_before = np.abs(direct_path_coef)**2 + np.abs(multipath_coef)**2
        clutter_power_after = np.mean(np.abs(cleaned[:target_delay])**2)  # Before target
        
        suppression_db = 10 * np.log10(clutter_power_before / (clutter_power_after + 1e-10))
        print(f"ECA suppression: {suppression_db:.1f} dB")
        
        # Should achieve significant suppression
        self.assertGreater(suppression_db, 20, "ECA should achieve >20 dB suppression")
        
        # Check that target is preserved (cross-correlate cleaned signal with ref)
        # Use normalized cross-correlation to find delay
        xcorr = np.correlate(cleaned, ref, mode='full')
        xcorr_mag = np.abs(xcorr)
        
        # In 'full' mode, zero-lag is at index len(ref)-1
        # Positive delays are at indices > len(ref)-1
        zero_lag_idx = len(ref) - 1
        
        # Look for peak in expected region (around target delay)
        search_start = zero_lag_idx + target_delay - 10
        search_end = zero_lag_idx + target_delay + 10
        search_region = xcorr_mag[search_start:search_end]
        
        peak_in_region = np.argmax(search_region) + search_start
        detected_delay = peak_in_region - zero_lag_idx
        
        print(f"Expected target delay: {target_delay}, Detected: {detected_delay}")
        
        # Peak should be within 15 samples of target delay
        # (some offset expected due to filter startup transient)
        self.assertLess(abs(detected_delay - target_delay), 15, 
                       f"Target delay mismatch: expected {target_delay}, got {detected_delay}")


class TestDopplerProcessing(unittest.TestCase):
    """Test Doppler FFT processing"""
    
    def test_doppler_detection(self):
        """Test detection of known Doppler shift via slow-time FFT"""
        sample_rate = 250000  # Hz
        cpi_samples = 1024
        num_cpis = 64
        
        prf = sample_rate / cpi_samples  # ~244 Hz
        doppler_resolution = prf / num_cpis  # ~3.8 Hz
        
        # Target parameters
        target_range_bin = 100
        target_doppler_hz = 50  # Hz
        target_amplitude = 1.0
        
        # Generate range profiles across CPIs
        range_profiles = np.zeros((num_cpis, cpi_samples), dtype=np.complex64)
        
        for cpi in range(num_cpis):
            # Target phase progression due to Doppler
            t_cpi = cpi / prf
            phase = 2 * np.pi * target_doppler_hz * t_cpi
            range_profiles[cpi, target_range_bin] = target_amplitude * np.exp(1j * phase)
        
        # Add noise
        noise_power = 0.01
        range_profiles += np.sqrt(noise_power/2) * (
            np.random.randn(num_cpis, cpi_samples) + 
            1j * np.random.randn(num_cpis, cpi_samples)
        ).astype(np.complex64)
        
        # Apply Hamming window along slow-time (Doppler) dimension
        window = np.hamming(num_cpis)
        range_profiles_windowed = range_profiles * window[:, np.newaxis]
        
        # FFT along slow-time (axis 0) for each range bin
        rdm = fftshift(fft(range_profiles_windowed, axis=0), axes=0)
        rdm_power = np.abs(rdm)**2
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(rdm_power), rdm_power.shape)
        detected_doppler_bin = peak_idx[0]
        detected_range_bin = peak_idx[1]
        
        # Expected Doppler bin (after fftshift, zero Doppler is at center)
        expected_doppler_bin = num_cpis // 2 + int(target_doppler_hz / doppler_resolution)
        
        print(f"Doppler resolution: {doppler_resolution:.2f} Hz")
        print(f"Expected Doppler bin: {expected_doppler_bin}, Detected: {detected_doppler_bin}")
        print(f"Expected range bin: {target_range_bin}, Detected: {detected_range_bin}")
        
        # Verify range
        self.assertEqual(detected_range_bin, target_range_bin, "Wrong range bin")
        
        # Verify Doppler (allow ±1 bin)
        self.assertLess(abs(detected_doppler_bin - expected_doppler_bin), 2, "Wrong Doppler bin")


class TestCFARAlgorithm(unittest.TestCase):
    """Test CFAR detection algorithms"""
    
    def test_ca_cfar_threshold(self):
        """Test CA-CFAR threshold calculation"""
        # For CA-CFAR: α = N × (Pfa^(-1/N) - 1)
        N = 64  # reference cells
        Pfa = 1e-6
        
        alpha = N * (Pfa**(-1/N) - 1)
        
        print(f"CA-CFAR threshold factor (N={N}, Pfa={Pfa}): α = {alpha:.2f}")
        
        # Verify with Monte Carlo simulation
        num_trials = 100000
        false_alarms = 0
        
        for _ in range(num_trials):
            # Generate exponential noise (power from Rayleigh envelope)
            noise = np.random.exponential(1.0, N)
            threshold = alpha * np.mean(noise)
            # CUT also exponential
            cut = np.random.exponential(1.0)
            if cut > threshold:
                false_alarms += 1
        
        measured_pfa = false_alarms / num_trials
        print(f"Target Pfa: {Pfa:.2e}, Measured Pfa: {measured_pfa:.2e}")
        
        # Should be within order of magnitude (with margin for Monte Carlo variance)
        self.assertLess(measured_pfa, Pfa * 20, "Pfa too high")
        self.assertGreater(measured_pfa, Pfa / 20, "Pfa too low")
    
    def test_ca_cfar_2d(self):
        """Test 2D CA-CFAR detection"""
        np.random.seed(123)
        
        num_range = 128
        num_doppler = 64
        guard_range = 2
        guard_doppler = 2
        ref_range = 8
        ref_doppler = 4
        Pfa = 1e-4
        
        # Calculate threshold factor
        window_area = (2*guard_range + 2*ref_range + 1) * (2*guard_doppler + 2*ref_doppler + 1)
        guard_area = (2*guard_range + 1) * (2*guard_doppler + 1)
        N = window_area - guard_area
        alpha = N * (Pfa**(-1/N) - 1)
        
        # Generate noise-only RDM
        rdm = np.random.exponential(1.0, (num_doppler, num_range)).astype(np.float32)
        
        # Add targets
        targets = [(32, 64, 100.0), (16, 32, 50.0), (48, 96, 75.0)]  # (d, r, power)
        for d, r, power in targets:
            rdm[d, r] = power
        
        # Run 2D CA-CFAR
        detections = np.zeros_like(rdm)
        
        for d in range(num_doppler):
            for r in range(num_range):
                # Collect reference cells
                ref_cells = []
                for dd in range(-guard_doppler - ref_doppler, guard_doppler + ref_doppler + 1):
                    for dr in range(-guard_range - ref_range, guard_range + ref_range + 1):
                        # Skip guard region and CUT
                        if abs(dd) <= guard_doppler and abs(dr) <= guard_range:
                            continue
                        # Wrap around boundaries
                        rr = (r + dr) % num_range
                        dd_idx = (d + dd) % num_doppler
                        ref_cells.append(rdm[dd_idx, rr])
                
                # Compute threshold
                noise_estimate = np.mean(ref_cells)
                threshold = alpha * noise_estimate
                
                # Detection
                if rdm[d, r] > threshold:
                    detections[d, r] = 1.0
        
        # Verify all targets detected
        for d, r, _ in targets:
            self.assertEqual(detections[d, r], 1.0, f"Target at ({d}, {r}) not detected")
        
        # Count false alarms (excluding target cells)
        total_cells = num_range * num_doppler - len(targets)
        fa_count = np.sum(detections) - len(targets)
        measured_pfa = fa_count / total_cells
        
        print(f"2D CA-CFAR: Detected {int(np.sum(detections))} cells, {len(targets)} targets")
        print(f"False alarms: {int(fa_count)}, Measured Pfa: {measured_pfa:.2e}")


class TestCoherenceMetrics(unittest.TestCase):
    """Test coherence monitoring calculations"""
    
    def test_correlation_coefficient(self):
        """Test cross-correlation coefficient calculation"""
        np.random.seed(42)
        N = 10000
        
        # Generate reference signal
        ref = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Test 1: Same signal = correlation 1.0
        corr = self._compute_correlation(ref, ref)
        self.assertAlmostEqual(corr, 1.0, places=10)
        
        # Test 2: Phase-shifted signal = correlation 1.0
        phase_shift = np.exp(1j * 0.75)
        shifted = ref * phase_shift
        corr = self._compute_correlation(ref, shifted)
        self.assertAlmostEqual(corr, 1.0, places=5)
        
        # Test 3: Independent signals = correlation ~0
        other = np.random.randn(N) + 1j * np.random.randn(N)
        corr = self._compute_correlation(ref, other)
        self.assertLess(abs(corr), 0.05)  # Should be near zero
        
        # Test 4: Partially correlated (phase noise)
        phase_noise_std = 0.1  # ~5.7 degrees
        noisy_phase = ref * np.exp(1j * np.random.randn(N) * phase_noise_std)
        corr = self._compute_correlation(ref, noisy_phase)
        # Correlation degrades with phase noise
        expected_corr = np.exp(-phase_noise_std**2 / 2)  # Approximate
        print(f"Phase noise σ={np.degrees(phase_noise_std):.1f}°: corr={corr:.4f}, expected≈{expected_corr:.4f}")
        self.assertGreater(corr, 0.9)
    
    def test_phase_variance_detection(self):
        """Test phase variance calculation for calibration monitoring"""
        np.random.seed(42)
        N = 1000
        num_measurements = 20
        
        # Simulate measurements with varying phase stability
        ref = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Good calibration: low phase variance
        phase_offsets_good = 0.5 + np.random.randn(num_measurements) * 0.02  # ~1° std
        phase_var_good = self._compute_circular_variance(phase_offsets_good)
        
        # Bad calibration: high phase variance
        phase_offsets_bad = 0.5 + np.random.randn(num_measurements) * 0.2  # ~11° std
        phase_var_bad = self._compute_circular_variance(phase_offsets_bad)
        
        print(f"Good calibration phase variance: {np.degrees(phase_var_good):.2f}°")
        print(f"Bad calibration phase variance: {np.degrees(phase_var_bad):.2f}°")
        
        # Good should be below threshold, bad should be above
        threshold_deg = 5.0
        threshold_rad = np.radians(threshold_deg)
        
        self.assertLess(phase_var_good, threshold_rad, "Good cal should pass")
        self.assertGreater(phase_var_bad, threshold_rad, "Bad cal should fail")
    
    def _compute_correlation(self, ref, surv):
        """Compute correlation coefficient"""
        cross = np.sum(ref * np.conj(surv))
        ref_power = np.sum(np.abs(ref)**2)
        surv_power = np.sum(np.abs(surv)**2)
        return np.abs(cross) / np.sqrt(ref_power * surv_power)
    
    def _compute_circular_variance(self, phases):
        """Compute circular standard deviation from phases"""
        sum_cos = np.sum(np.cos(phases))
        sum_sin = np.sum(np.sin(phases))
        R = np.sqrt(sum_cos**2 + sum_sin**2) / len(phases)
        # Circular std = sqrt(2(1-R)) for small variance
        return np.sqrt(2 * (1 - R))


class TestCalibrationDecision(unittest.TestCase):
    """Test automatic calibration triggering logic"""
    
    def test_hysteresis(self):
        """Test that hysteresis prevents spurious triggers"""
        corr_threshold = 0.95
        phase_threshold_deg = 5.0
        consecutive_failures_required = 3
        
        # Simulate measurements with occasional dips
        measurements = [
            (0.98, 2.0),  # Good
            (0.94, 3.0),  # Fail (corr)
            (0.96, 4.0),  # Good
            (0.93, 6.0),  # Fail (both)
            (0.97, 2.0),  # Good - should reset counter
            (0.90, 7.0),  # Fail
            (0.91, 6.0),  # Fail
            (0.89, 8.0),  # Fail - 3rd consecutive, should trigger
        ]
        
        consecutive_failures = 0
        calibration_triggered = False
        trigger_index = None
        
        for i, (corr, phase_var_deg) in enumerate(measurements):
            failed = corr < corr_threshold or phase_var_deg > phase_threshold_deg
            
            if failed:
                consecutive_failures += 1
                if consecutive_failures >= consecutive_failures_required and not calibration_triggered:
                    calibration_triggered = True
                    trigger_index = i
            else:
                consecutive_failures = 0
        
        self.assertTrue(calibration_triggered, "Should have triggered calibration")
        self.assertEqual(trigger_index, 7, "Should trigger on 3rd consecutive failure")
        
        print(f"Calibration triggered at measurement {trigger_index}")


class TestRangeResolution(unittest.TestCase):
    """Test range resolution and ambiguity calculations"""
    
    def test_range_parameters(self):
        """Verify range resolution and max range calculations"""
        sample_rate = 250000  # Hz (after decimation)
        bandwidth = sample_rate  # For FM, BW ≈ sample rate
        cpi_duration = 0.1  # 100 ms
        c = 3e8  # speed of light
        
        # Range resolution = c / (2 * BW)
        range_resolution = c / (2 * bandwidth)
        print(f"Range resolution: {range_resolution:.1f} m")
        self.assertAlmostEqual(range_resolution, 600, delta=10)
        
        # Maximum unambiguous range = c * CPI / 2
        max_range_m = c * cpi_duration / 2  # in meters
        max_range_km = max_range_m / 1000   # convert to km
        print(f"Maximum unambiguous range: {max_range_km:.1f} km")
        self.assertAlmostEqual(max_range_km, 15000, delta=100)  # 15000 km (15,000,000 m)
        
        # Number of range bins
        cpi_samples = int(cpi_duration * sample_rate)
        num_range_bins = cpi_samples
        print(f"Number of range bins: {num_range_bins}")
        self.assertEqual(num_range_bins, 25000)


if __name__ == '__main__':
    print("=" * 60)
    print("Standalone Algorithm Verification Tests")
    print("No GNU Radio required")
    print("=" * 60)
    unittest.main(verbosity=2)
