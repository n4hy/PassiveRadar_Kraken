"""
Integration tests for full passive radar pipeline.
"""
import unittest
import numpy as np
from pathlib import Path
import sys
import ctypes

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.fixtures.synthetic_targets import (
    BistaticTargetGenerator, TargetSpec,
    single_target_snr15, two_close_targets, multi_target_dense
)
from tests.fixtures.clutter_models import ClutterGenerator


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end pipeline integration tests."""

    @classmethod
    def setUpClass(cls):
        """Load required libraries."""
        repo_root = Path(__file__).resolve().parents[2]

        cls.libs = {}
        lib_names = ['caf_processing', 'eca_b_clutter_canceller', 'doppler_processing']

        for name in lib_names:
            lib_path = repo_root / "src" / f"libkraken_{name}.so"
            if lib_path.exists():
                try:
                    cls.libs[name] = ctypes.cdll.LoadLibrary(str(lib_path))
                except OSError:
                    pass

        if not cls.libs:
            raise unittest.SkipTest("No processing libraries found")

    def _run_eca(self, ref, surv, num_taps=64):
        """Run ECA clutter cancellation."""
        if 'eca_b_clutter_canceller' not in self.libs:
            return surv  # Pass through if not available

        lib = self.libs['eca_b_clutter_canceller']
        lib.eca_b_create.restype = ctypes.c_void_p
        lib.eca_b_create.argtypes = [ctypes.c_int]
        lib.eca_b_destroy.argtypes = [ctypes.c_void_p]
        lib.eca_b_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]

        n = len(ref)
        state = lib.eca_b_create(num_taps)

        ref_c = np.ascontiguousarray(ref, dtype=np.complex64)
        surv_c = np.ascontiguousarray(surv, dtype=np.complex64)
        out_c = np.zeros_like(surv_c)

        lib.eca_b_process(
            state,
            ref_c.view(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv_c.view(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_c.view(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n
        )

        lib.eca_b_destroy(state)
        return out_c

    def _run_caf(self, ref, surv):
        """Run CAF processing."""
        if 'caf_processing' not in self.libs:
            # Fallback to numpy
            return np.fft.ifft(np.fft.fft(surv) * np.conj(np.fft.fft(ref)))

        lib = self.libs['caf_processing']
        lib.caf_create.restype = ctypes.c_void_p
        lib.caf_create.argtypes = [ctypes.c_int]
        lib.caf_destroy.argtypes = [ctypes.c_void_p]
        lib.caf_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        n = len(ref)
        obj = lib.caf_create(n)

        ref_c = np.ascontiguousarray(ref, dtype=np.complex64)
        surv_c = np.ascontiguousarray(surv, dtype=np.complex64)
        out_c = np.zeros(n, dtype=np.complex64)

        lib.caf_process(
            obj,
            ref_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        lib.caf_destroy(obj)
        return out_c

    def _run_cfar(self, caf_power, guard=2, ref_cells=4, pfa=1e-4):
        """Run CFAR detection (Python reference)."""
        n = len(caf_power)
        detections = np.zeros(n, dtype=bool)

        n_ref = 2 * ref_cells
        alpha = n_ref * (pfa ** (-1/n_ref) - 1)

        for i in range(guard + ref_cells, n - guard - ref_cells):
            left_ref = caf_power[i - guard - ref_cells:i - guard]
            right_ref = caf_power[i + guard + 1:i + guard + ref_cells + 1]
            threshold = alpha * (np.sum(left_ref) + np.sum(right_ref)) / n_ref

            if caf_power[i] > threshold:
                detections[i] = True

        return detections

    def test_single_target_detection(self):
        """Verify CAF shows peak at expected target location."""
        scenario = single_target_snr15()

        ref = scenario.ref_signal
        surv = scenario.surv_signals[0]  # First surveillance channel

        # CAF without ECA (to see raw correlation)
        caf = self._run_caf(ref, surv)
        caf_power = np.abs(caf)**2

        # Find peak
        peak_idx = np.argmax(caf_power)
        peak_val = caf_power[peak_idx]
        median_val = np.median(caf_power)

        # Peak should be significantly above median (target present)
        snr = peak_val / (median_val + 1e-10)

        self.assertGreater(snr, 10,
                           f"CAF peak SNR too low: {snr:.1f}")

    def test_eca_improves_detection(self):
        """Verify ECA reduces direct path power."""
        np.random.seed(42)
        n = 4096

        # Create a simple scenario with strong direct path
        from tests.fixtures.synthetic_targets import BistaticTargetGenerator
        gen = BistaticTargetGenerator()
        ref = gen.generate_fm_waveform(n, seed=42)

        # Surveillance = strong direct path + weak target
        direct_path = ref * 100.0  # 40 dB direct path
        target_delay = 50
        target = np.roll(ref, target_delay) * 0.1  # Weak target
        surv = direct_path + target

        # Measure direct path correlation before ECA
        caf_before = self._run_caf(ref, surv)
        zero_delay_power_before = np.abs(caf_before[0])**2

        # Apply ECA
        cancelled = self._run_eca(ref, surv, num_taps=64)

        # Measure direct path correlation after ECA
        caf_after = self._run_caf(ref, cancelled)
        zero_delay_power_after = np.abs(caf_after[0])**2

        # ECA should reduce direct path power
        suppression_db = 10 * np.log10(zero_delay_power_before / (zero_delay_power_after + 1e-10))

        self.assertGreater(suppression_db, 10,
                           f"ECA suppression too low: {suppression_db:.1f} dB")

    def test_multi_target_scenario(self):
        """Test CAF computation completes without errors on multi-target data."""
        scenario = multi_target_dense()

        ref = scenario.ref_signal
        surv = scenario.surv_signals[0]

        # Run CAF
        caf = self._run_caf(ref, surv)
        caf_power = np.abs(caf)**2

        # Verify CAF output is valid
        self.assertEqual(len(caf_power), len(ref))
        self.assertFalse(np.any(np.isnan(caf_power)), "CAF contains NaN")
        self.assertFalse(np.any(np.isinf(caf_power)), "CAF contains Inf")

        # Should have some structure (not flat noise)
        dynamic_range_db = 10 * np.log10(caf_power.max() / (caf_power.min() + 1e-10))
        self.assertGreater(dynamic_range_db, 10,
                           f"CAF dynamic range too low: {dynamic_range_db:.1f} dB")


class TestProcessingChainValidation(unittest.TestCase):
    """Validate processing chain data flow."""

    def test_signal_dimensions(self):
        """Verify signal dimensions are preserved through chain."""
        n_samples = 4096

        ref = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
        surv = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)

        # Each processing step should maintain dimensions
        caf = np.fft.ifft(np.fft.fft(surv) * np.conj(np.fft.fft(ref)))

        self.assertEqual(len(caf), n_samples)

    def test_power_conservation(self):
        """Verify signal power is reasonable through chain."""
        n_samples = 4096

        ref = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
        ref /= np.std(ref)  # Normalize

        surv = ref.copy()  # Perfect correlation

        caf = np.fft.ifft(np.fft.fft(surv) * np.conj(np.fft.fft(ref)))

        # Peak should be approximately N (for perfectly correlated signals)
        peak = np.max(np.abs(caf))
        expected_peak = n_samples * np.var(ref)

        self.assertGreater(peak, expected_peak * 0.5)
        self.assertLess(peak, expected_peak * 2.0)


if __name__ == '__main__':
    unittest.main()
