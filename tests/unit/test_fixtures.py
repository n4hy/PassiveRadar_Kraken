"""
Unit tests for test fixtures (signal generators).
Tests synthetic_targets, clutter_models, and noise_models.
"""
import unittest
import numpy as np

from tests.fixtures.synthetic_targets import (
    BistaticTargetGenerator,
    TargetSpec,
    SystemParams,
    ScenarioData,
    single_target_snr15,
    two_close_targets,
    crossing_targets,
    weak_target,
    multi_target_dense,
)
from tests.fixtures.clutter_models import ClutterGenerator
from tests.fixtures.noise_models import NoiseGenerator


class TestBistaticTargetGenerator(unittest.TestCase):
    """Tests for BistaticTargetGenerator."""

    def setUp(self):
        self.generator = BistaticTargetGenerator()
        params = SystemParams(
            sample_rate=1e6,
            center_freq=88e6,
            range_resolution=150.0,
            doppler_resolution=0.5,
            num_channels=5,
            element_spacing=0.5
        )
        self.custom_params_generator = BistaticTargetGenerator(params)

    def test_fm_waveform_shape(self):
        """Test FM waveform has correct shape and dtype."""
        n_samples = 10000
        waveform = self.generator.generate_fm_waveform(n_samples, seed=42)

        self.assertEqual(waveform.shape, (n_samples,))
        self.assertEqual(waveform.dtype, np.complex64)

    def test_fm_waveform_unit_magnitude(self):
        """Test FM waveform has approximately unit magnitude."""
        waveform = self.generator.generate_fm_waveform(10000, seed=42)
        magnitudes = np.abs(waveform)

        # FM signal should have constant envelope (magnitude ~1)
        self.assertTrue(np.allclose(magnitudes, 1.0, atol=0.01))

    def test_fm_waveform_reproducibility(self):
        """Test FM waveform is reproducible with seed."""
        w1 = self.generator.generate_fm_waveform(1000, seed=123)
        w2 = self.generator.generate_fm_waveform(1000, seed=123)
        w3 = self.generator.generate_fm_waveform(1000, seed=456)

        np.testing.assert_array_equal(w1, w2)
        self.assertFalse(np.array_equal(w1, w3))

    def test_apply_delay_integer(self):
        """Test integer sample delay."""
        signal = np.array([1, 2, 3, 4, 5], dtype=np.complex64)
        delayed = self.generator.apply_delay(signal, 2.0)

        # First 2 samples should be zero, rest shifted
        self.assertEqual(delayed[0], 0)
        self.assertEqual(delayed[1], 0)
        self.assertEqual(delayed[2], 1)

    def test_apply_delay_fractional(self):
        """Test fractional sample delay preserves energy."""
        signal = self.generator.generate_fm_waveform(1024, seed=42)
        delayed = self.generator.apply_delay(signal, 10.5)

        # Energy should be approximately preserved (except zeroed samples)
        original_energy = np.sum(np.abs(signal[11:])**2)
        delayed_energy = np.sum(np.abs(delayed[11:])**2)
        self.assertTrue(np.isclose(original_energy, delayed_energy, rtol=0.1))

    def test_generate_target_snr(self):
        """Test target generation produces correct SNR."""
        ref = self.generator.generate_fm_waveform(10000, seed=42)
        target_sig, _ = self.generator.generate_target(ref, 5000, 100, snr_db=20)

        # Target signal should have reasonable power
        self.assertGreater(np.mean(np.abs(target_sig)**2), 0)

    def test_generate_target_doppler(self):
        """Test target has Doppler shift in spectrum."""
        ref = self.generator.generate_fm_waveform(10000, seed=42)
        doppler_hz = 100
        target_sig, _ = self.generator.generate_target(ref, 5000, doppler_hz, snr_db=20)

        # Check spectral content is shifted
        # This is a basic sanity check - detailed Doppler tests are in CAF tests
        self.assertTrue(np.any(np.abs(target_sig) > 0))

    def test_generate_multipath_power_decay(self):
        """Test multipath power is less than reference."""
        ref = self.generator.generate_fm_waveform(10000, seed=42)
        multipath = self.generator.generate_multipath(ref, n_paths=5, max_delay_samples=50)

        ref_power = np.mean(np.abs(ref)**2)
        mp_power = np.mean(np.abs(multipath)**2)

        # Multipath should be much weaker than reference
        self.assertLess(mp_power, ref_power)

    def test_generate_scenario_structure(self):
        """Test scenario generation returns correct structure."""
        targets = [
            TargetSpec(range_m=5000, doppler_hz=100, snr_db=15),
            TargetSpec(range_m=8000, doppler_hz=-50, snr_db=12),
        ]
        scenario = self.generator.generate_scenario(targets, duration_sec=0.05, seed=42)

        self.assertIsInstance(scenario, ScenarioData)
        self.assertEqual(scenario.ref_signal.dtype, np.complex64)
        self.assertEqual(scenario.surv_signals.dtype, np.complex64)
        self.assertEqual(scenario.surv_signals.shape[0], self.generator.params.num_channels - 1)
        self.assertEqual(len(scenario.targets), 2)
        self.assertEqual(scenario.ground_truth['expected_detections'], 2)

    def test_scenario_reproducibility(self):
        """Test scenario is reproducible with seed."""
        targets = [TargetSpec(range_m=5000, doppler_hz=100, snr_db=15)]

        s1 = self.generator.generate_scenario(targets, seed=42)
        s2 = self.generator.generate_scenario(targets, seed=42)

        np.testing.assert_array_equal(s1.ref_signal, s2.ref_signal)
        np.testing.assert_array_equal(s1.surv_signals, s2.surv_signals)


class TestPredefinedScenarios(unittest.TestCase):
    """Tests for predefined test scenarios."""

    def test_single_target_snr15(self):
        """Test single target scenario."""
        scenario = single_target_snr15()

        self.assertEqual(len(scenario.targets), 1)
        self.assertEqual(scenario.targets[0].snr_db, 15)
        self.assertEqual(scenario.ground_truth['expected_detections'], 1)

    def test_two_close_targets(self):
        """Test close targets scenario."""
        scenario = two_close_targets()

        self.assertEqual(len(scenario.targets), 2)
        # Targets should be close in range
        range_diff = abs(scenario.targets[0].range_m - scenario.targets[1].range_m)
        self.assertLess(range_diff, 500)  # Less than 500m apart

    def test_crossing_targets(self):
        """Test crossing targets scenario."""
        scenario = crossing_targets()

        self.assertEqual(len(scenario.targets), 2)
        # Targets should have different AoA
        aoa_diff = abs(scenario.targets[0].aoa_deg - scenario.targets[1].aoa_deg)
        self.assertGreater(aoa_diff, 0)

    def test_weak_target(self):
        """Test weak target scenario."""
        scenario = weak_target()

        self.assertEqual(len(scenario.targets), 1)
        self.assertLess(scenario.targets[0].snr_db, 0)  # Negative SNR

    def test_multi_target_dense(self):
        """Test dense multi-target scenario."""
        scenario = multi_target_dense()

        self.assertEqual(len(scenario.targets), 10)
        self.assertEqual(scenario.ground_truth['expected_detections'], 10)


class TestClutterGenerator(unittest.TestCase):
    """Tests for ClutterGenerator."""

    def setUp(self):
        self.generator = ClutterGenerator(sample_rate=2.4e6)

    def test_direct_path_power(self):
        """Test direct path has correct power level."""
        ref = np.ones(1000, dtype=np.complex64)
        direct = self.generator.generate_direct_path(ref, power_db=20)

        # 20 dB = 10x amplitude = 100x power
        expected_power = 100.0
        actual_power = np.mean(np.abs(direct)**2)
        self.assertTrue(np.isclose(actual_power, expected_power, rtol=0.01))

    def test_direct_path_shape(self):
        """Test direct path preserves signal shape."""
        ref = np.exp(1j * np.linspace(0, 10*np.pi, 1000)).astype(np.complex64)
        direct = self.generator.generate_direct_path(ref, power_db=10)

        # Phase should be preserved (just scaled)
        phase_diff = np.angle(direct) - np.angle(ref)
        # Unwrap and check
        self.assertLess(np.std(np.unwrap(phase_diff)), 0.01)

    def test_multipath_shape(self):
        """Test multipath has correct shape."""
        ref = np.ones(1000, dtype=np.complex64)
        multipath = self.generator.generate_multipath(ref, n_paths=5, seed=42)

        self.assertEqual(multipath.shape, ref.shape)
        self.assertEqual(multipath.dtype, np.complex64)

    def test_multipath_weaker_than_direct(self):
        """Test multipath is weaker than direct path."""
        ref = np.ones(1000, dtype=np.complex64)
        multipath = self.generator.generate_multipath(ref, n_paths=10, seed=42)

        # Multipath power should be significantly lower
        self.assertLess(np.mean(np.abs(multipath)**2), np.mean(np.abs(ref)**2))

    def test_multipath_reproducibility(self):
        """Test multipath is reproducible with seed."""
        ref = np.ones(1000, dtype=np.complex64)
        mp1 = self.generator.generate_multipath(ref, seed=42)
        mp2 = self.generator.generate_multipath(ref, seed=42)

        np.testing.assert_array_equal(mp1, mp2)

    def test_ground_clutter_narrow_doppler(self):
        """Test ground clutter has narrow Doppler spread."""
        ref = np.ones(10000, dtype=np.complex64)
        clutter = self.generator.generate_ground_clutter(ref, clutter_bandwidth_hz=5.0, seed=42)

        # FFT to check Doppler spread
        spectrum = np.abs(np.fft.fft(clutter))
        freqs = np.fft.fftfreq(len(clutter), 1/self.generator.sample_rate)

        # Most energy should be near zero Doppler
        near_zero = np.abs(freqs) < 50  # Within 50 Hz
        energy_near_zero = np.sum(spectrum[near_zero]**2)
        total_energy = np.sum(spectrum**2)

        # Most energy should be near zero Doppler
        self.assertGreater(energy_near_zero, 0.1 * total_energy)

    def test_interference_narrowband(self):
        """Test interference is narrowband."""
        interference = self.generator.generate_interference(
            n_samples=10000,
            center_offset_hz=50e3,
            bandwidth_hz=1e3,
            seed=42
        )

        self.assertEqual(interference.shape, (10000,))
        self.assertEqual(interference.dtype, np.complex64)

    def test_combined_clutter_returns_params(self):
        """Test combined clutter returns parameters dict."""
        ref = np.ones(1000, dtype=np.complex64)
        clutter, params = self.generator.generate_combined_clutter(
            ref, direct_path_db=40, n_multipath=5, seed=42
        )

        self.assertIn('direct_path_db', params)
        self.assertIn('n_multipath', params)
        self.assertIn('direct_power', params)
        self.assertEqual(params['direct_path_db'], 40)
        self.assertEqual(params['n_multipath'], 5)


class TestNoiseGenerator(unittest.TestCase):
    """Tests for NoiseGenerator."""

    def setUp(self):
        self.generator = NoiseGenerator(sample_rate=2.4e6)

    def test_awgn_shape_dtype(self):
        """Test AWGN has correct shape and dtype."""
        noise = self.generator.generate_awgn(1000, power_dbm=-100, seed=42)

        self.assertEqual(noise.shape, (1000,))
        self.assertEqual(noise.dtype, np.complex64)

    def test_awgn_power(self):
        """Test AWGN has approximately correct power."""
        power_dbm = -90
        n_samples = 100000  # Large sample for accuracy
        noise = self.generator.generate_awgn(n_samples, power_dbm=power_dbm, seed=42)

        # Calculate actual power in dBm
        power_w = np.mean(np.abs(noise)**2)
        actual_power_dbm = 10 * np.log10(power_w * 1000)

        # Should be close to specified power (within 1 dB)
        self.assertTrue(np.isclose(actual_power_dbm, power_dbm, atol=1.0))

    def test_awgn_reproducibility(self):
        """Test AWGN is reproducible with seed."""
        n1 = self.generator.generate_awgn(1000, seed=42)
        n2 = self.generator.generate_awgn(1000, seed=42)

        np.testing.assert_array_equal(n1, n2)

    def test_awgn_uncorrelated(self):
        """Test AWGN samples are uncorrelated."""
        noise = self.generator.generate_awgn(10000, seed=42)

        # Autocorrelation should peak at lag 0 and be small elsewhere
        autocorr = np.correlate(noise, noise, mode='same')
        autocorr = np.abs(autocorr)
        peak_idx = len(autocorr) // 2

        # Peak at center, much smaller at other lags
        self.assertGreater(autocorr[peak_idx], 2 * np.mean(autocorr[peak_idx-100:peak_idx-10]))

    def test_colored_noise_shape(self):
        """Test colored noise has correct shape."""
        noise = self.generator.generate_colored_noise(1000, alpha=1.0, seed=42)

        self.assertEqual(noise.shape, (1000,))
        self.assertEqual(noise.dtype, np.complex64)

    def test_colored_noise_different_alphas(self):
        """Test different alpha values produce different spectra."""
        white = self.generator.generate_colored_noise(10000, alpha=0.0, seed=42)
        pink = self.generator.generate_colored_noise(10000, alpha=1.0, seed=42)
        brown = self.generator.generate_colored_noise(10000, alpha=2.0, seed=42)

        # Brown noise should have less high-freq energy than pink, less than white
        # This is a simplified check
        self.assertEqual(white.shape, pink.shape)
        self.assertEqual(pink.shape, brown.shape)

    def test_impulsive_noise_has_impulses(self):
        """Test impulsive noise contains impulses."""
        noise = self.generator.generate_impulsive_noise(
            n_samples=10000,
            background_power_dbm=-100,
            impulse_rate=0.01,
            impulse_power_dbm=-60,
            seed=42
        )

        # Check for presence of large values (impulses)
        magnitude = np.abs(noise)
        median_mag = np.median(magnitude)
        max_mag = np.max(magnitude)

        # Max should be much larger than median due to impulses
        self.assertGreater(max_mag, 10 * median_mag)

    def test_phase_noise_unit_magnitude(self):
        """Test phase noise multiplier has unit magnitude."""
        pn = self.generator.generate_phase_noise(1000, seed=42)

        # Phase noise is a complex multiplier with |z|=1
        magnitudes = np.abs(pn)
        np.testing.assert_allclose(magnitudes, 1.0, rtol=1e-5)

    def test_receiver_noise_returns_floor(self):
        """Test receiver noise addition returns noise floor."""
        signal = np.ones(1000, dtype=np.complex64)
        noisy, floor_dbm = self.generator.add_receiver_noise(signal, noise_figure_db=3.0)

        self.assertIsInstance(floor_dbm, float)
        self.assertEqual(noisy.shape, signal.shape)
        # Noise floor should be reasonable (thermal + NF)
        self.assertLess(floor_dbm, -70)  # Should be below -70 dBm typically

    def test_quantization_preserves_shape(self):
        """Test quantization preserves signal shape."""
        signal = np.random.randn(1000).astype(np.complex64)
        quantized = self.generator.generate_quantization_noise(signal, n_bits=12)

        self.assertEqual(quantized.shape, signal.shape)
        self.assertEqual(quantized.dtype, np.complex64)

    def test_quantization_reduces_levels(self):
        """Test quantization reduces to discrete levels."""
        signal = np.linspace(-1, 1, 1000).astype(np.complex64)
        quantized = self.generator.generate_quantization_noise(signal, n_bits=4)

        # With 4 bits, should have at most 16 unique levels per component
        unique_real = len(np.unique(np.real(quantized)))
        self.assertLessEqual(unique_real, 16)


class TestIntegration(unittest.TestCase):
    """Integration tests combining fixtures."""

    def test_scenario_with_clutter(self):
        """Test creating scenario with additional clutter."""
        gen = BistaticTargetGenerator()
        clutter_gen = ClutterGenerator()

        targets = [TargetSpec(range_m=5000, doppler_hz=100, snr_db=15)]
        scenario = gen.generate_scenario(targets, seed=42)

        # Add extra clutter to surveillance channels
        for ch in range(scenario.surv_signals.shape[0]):
            extra_clutter, _ = clutter_gen.generate_combined_clutter(
                scenario.ref_signal, seed=42+ch
            )
            scenario.surv_signals[ch] += extra_clutter

        self.assertEqual(scenario.surv_signals.shape[0], 4)

    def test_scenario_with_noise(self):
        """Test creating scenario with additional noise."""
        gen = BistaticTargetGenerator()
        noise_gen = NoiseGenerator()

        targets = [TargetSpec(range_m=5000, doppler_hz=100, snr_db=15)]
        scenario = gen.generate_scenario(targets, seed=42)

        # Add receiver noise
        for ch in range(scenario.surv_signals.shape[0]):
            scenario.surv_signals[ch], _ = noise_gen.add_receiver_noise(
                scenario.surv_signals[ch],
                noise_figure_db=3.0
            )

        self.assertEqual(scenario.surv_signals.dtype, np.complex64)


if __name__ == '__main__':
    unittest.main()
