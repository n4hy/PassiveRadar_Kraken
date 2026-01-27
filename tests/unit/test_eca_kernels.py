"""
Unit tests for ECA (Extensive Cancellation Algorithm) clutter cancellation.
"""
import unittest
import numpy as np
import ctypes
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.fixtures.synthetic_targets import BistaticTargetGenerator, TargetSpec
from tests.fixtures.clutter_models import ClutterGenerator


class TestECAClutterCancellation(unittest.TestCase):
    """Test ECA block clutter suppression."""

    @classmethod
    def setUpClass(cls):
        """Load ECA library."""
        repo_root = Path(__file__).resolve().parents[2]
        lib_path = repo_root / "src" / "libkraken_eca_b_clutter_canceller.so"

        if not lib_path.exists():
            raise unittest.SkipTest(f"ECA library not found at {lib_path}")

        cls.lib = ctypes.cdll.LoadLibrary(str(lib_path))

        # Setup function signatures
        cls.lib.eca_b_create.restype = ctypes.c_void_p
        cls.lib.eca_b_create.argtypes = [ctypes.c_int]

        cls.lib.eca_b_destroy.restype = None
        cls.lib.eca_b_destroy.argtypes = [ctypes.c_void_p]

        cls.lib.eca_b_process.restype = None
        cls.lib.eca_b_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

    def _process_eca(self, ref, surv, num_taps=64):
        """Helper to run ECA processing."""
        n_samples = len(ref)
        state = self.lib.eca_b_create(num_taps)
        self.assertIsNotNone(state)

        ref_c = np.ascontiguousarray(ref, dtype=np.complex64)
        surv_c = np.ascontiguousarray(surv, dtype=np.complex64)
        out_c = np.zeros_like(surv_c)

        ref_f = ref_c.view(np.float32)
        surv_f = surv_c.view(np.float32)
        out_f = out_c.view(np.float32)

        self.lib.eca_b_process(
            state,
            ref_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_samples,
        )

        self.lib.eca_b_destroy(state)
        return out_c

    def test_direct_path_cancellation(self):
        """Verify direct path is cancelled to <-40dB."""
        np.random.seed(42)
        n = 4096

        # Generate reference
        gen = BistaticTargetGenerator()
        ref = gen.generate_fm_waveform(n, seed=42)

        # Surveillance = direct path only (strong)
        surv = ref * 100.0  # 40 dB direct path

        output = self._process_eca(ref, surv, num_taps=64)

        suppression_db = 10 * np.log10(np.var(surv) / (np.var(output) + 1e-20))

        self.assertGreater(suppression_db, 30.0,
                           f"Direct path not cancelled: {suppression_db:.2f}dB")

    def test_multipath_cancellation(self):
        """Verify multipath echoes are cancelled."""
        np.random.seed(42)
        n = 4096

        gen = BistaticTargetGenerator()
        ref = gen.generate_fm_waveform(n, seed=42)

        clutter_gen = ClutterGenerator()
        surv, _ = clutter_gen.generate_combined_clutter(ref, direct_path_db=40, n_multipath=5)

        output = self._process_eca(ref, surv, num_taps=128)

        # Measure residual correlation with reference
        corr_before = np.abs(np.correlate(surv, ref, mode='valid')).max()
        corr_after = np.abs(np.correlate(output, ref, mode='valid')).max()

        self.assertLess(corr_after, corr_before * 0.1,
                        "Multipath not sufficiently cancelled")

    def test_target_preservation(self):
        """Verify target signal is NOT cancelled (different Doppler)."""
        np.random.seed(42)
        n = 4096

        gen = BistaticTargetGenerator()
        ref = gen.generate_fm_waveform(n, seed=42)

        # Create target with Doppler shift (won't be cancelled)
        target_sig, surv = gen.generate_target(ref, bistatic_range_m=5000,
                                                doppler_hz=50, snr_db=20)

        # Add strong direct path
        surv = ref * 100.0 + target_sig

        output = self._process_eca(ref, surv, num_taps=64)

        # Target should still be present (correlates with Doppler-shifted ref)
        t = np.arange(n) / 2.4e6
        doppler_ref = ref * np.exp(2j * np.pi * 50 * t)

        target_corr = np.abs(np.correlate(output, doppler_ref, mode='valid')).max()
        noise_level = np.std(output)

        self.assertGreater(target_corr, noise_level * 5,
                           "Target was incorrectly cancelled")

    def test_numerical_stability_zero_input(self):
        """Verify no NaN/Inf with zero input."""
        n = 1024
        ref = np.zeros(n, dtype=np.complex64)
        surv = np.zeros(n, dtype=np.complex64)

        output = self._process_eca(ref, surv)

        self.assertFalse(np.any(np.isnan(output)), "NaN in output with zero input")
        self.assertFalse(np.any(np.isinf(output)), "Inf in output with zero input")

    def test_numerical_stability_large_input(self):
        """Verify no NaN/Inf with large input."""
        np.random.seed(42)
        n = 1024
        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * 1e6
        surv = ref.copy()

        output = self._process_eca(ref, surv)

        self.assertFalse(np.any(np.isnan(output)), "NaN in output with large input")
        self.assertFalse(np.any(np.isinf(output)), "Inf in output with large input")

    def test_numerical_stability_small_input(self):
        """Verify no NaN/Inf with very small input."""
        np.random.seed(42)
        n = 1024
        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * 1e-10
        surv = ref.copy()

        output = self._process_eca(ref, surv)

        self.assertFalse(np.any(np.isnan(output)), "NaN in output with small input")
        self.assertFalse(np.any(np.isinf(output)), "Inf in output with small input")

    def test_varying_tap_lengths(self):
        """Test ECA with different filter lengths."""
        np.random.seed(42)
        n = 4096

        gen = BistaticTargetGenerator()
        ref = gen.generate_fm_waveform(n, seed=42)
        surv = ref * 100.0  # Direct path

        for num_taps in [8, 16, 32, 64, 128]:
            output = self._process_eca(ref, surv, num_taps=num_taps)
            suppression_db = 10 * np.log10(np.var(surv) / (np.var(output) + 1e-20))

            self.assertGreater(suppression_db, 20.0,
                               f"Poor suppression with {num_taps} taps: {suppression_db:.1f}dB")


if __name__ == '__main__':
    unittest.main()
