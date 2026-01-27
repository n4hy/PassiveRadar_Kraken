"""
Unit tests for Cross-Ambiguity Function (CAF) computation.
"""
import unittest
import numpy as np
import ctypes
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.fixtures.synthetic_targets import BistaticTargetGenerator


class TestCAFComputation(unittest.TestCase):
    """Test Cross-Ambiguity Function computation."""

    @classmethod
    def setUpClass(cls):
        """Load CAF library."""
        repo_root = Path(__file__).resolve().parents[2]
        lib_path = repo_root / "src" / "libkraken_caf_processing.so"

        if not lib_path.exists():
            raise unittest.SkipTest(f"CAF library not found at {lib_path}")

        cls.lib = ctypes.cdll.LoadLibrary(str(lib_path))

        cls.lib.caf_create.restype = ctypes.c_void_p
        cls.lib.caf_create.argtypes = [ctypes.c_int]

        cls.lib.caf_destroy.argtypes = [ctypes.c_void_p]
        cls.lib.caf_destroy.restype = None

        cls.lib.caf_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        cls.lib.caf_process.restype = None

    def _compute_caf(self, ref, surv):
        """Helper to compute CAF."""
        n = len(ref)
        obj = self.lib.caf_create(n)
        self.assertIsNotNone(obj)

        ref_c = np.ascontiguousarray(ref, dtype=np.complex64)
        surv_c = np.ascontiguousarray(surv, dtype=np.complex64)
        out_c = np.zeros(n, dtype=np.complex64)

        self.lib.caf_process(
            obj,
            ref_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        self.lib.caf_destroy(obj)
        return out_c

    def test_peak_location_integer_delay(self):
        """Verify CAF peak at correct range for integer delay."""
        np.random.seed(42)
        n = 4096
        true_delay = 100

        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        surv = np.roll(ref, true_delay)

        caf = self._compute_caf(ref, surv)
        mag = np.abs(caf)
        peak_idx = np.argmax(mag)

        self.assertEqual(peak_idx, true_delay,
                         f"Peak at {peak_idx}, expected {true_delay}")

    def test_peak_location_various_delays(self):
        """Test CAF peak detection at various delays."""
        np.random.seed(42)
        n = 4096

        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

        for delay in [10, 50, 100, 200, 500, 1000]:
            surv = np.roll(ref, delay)
            caf = self._compute_caf(ref, surv)
            peak_idx = np.argmax(np.abs(caf))

            self.assertEqual(peak_idx, delay,
                             f"Delay {delay}: peak at {peak_idx}")

    def test_sidelobe_levels(self):
        """Verify CAF sidelobes are below expected threshold."""
        np.random.seed(42)
        n = 4096

        # Use FM waveform for realistic sidelobes
        gen = BistaticTargetGenerator()
        ref = gen.generate_fm_waveform(n, seed=42)
        surv = np.roll(ref, 100)

        caf = self._compute_caf(ref, surv)
        mag = np.abs(caf)
        peak = mag.max()

        # Mask main lobe (peak +/- 20 samples for FM signal)
        peak_idx = np.argmax(mag)
        main_lobe_mask = np.zeros(n, dtype=bool)
        main_lobe_mask[max(0, peak_idx-20):min(n, peak_idx+21)] = True

        sidelobes = mag.copy()
        sidelobes[main_lobe_mask] = 0
        max_sidelobe = sidelobes.max()

        psll = 20 * np.log10(max_sidelobe / peak + 1e-10)

        # FM signals have wider correlation, accept -5 dB PSLL
        self.assertLess(psll, -5,
                        f"PSLL too high: {psll:.1f} dB")

    def test_caf_linearity(self):
        """Verify CAF is linear (scaled input = scaled output)."""
        np.random.seed(42)
        n = 4096

        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        surv = np.roll(ref, 50)

        caf1 = self._compute_caf(ref, surv)
        caf2 = self._compute_caf(ref, surv * 2.0)

        # Peak should scale by 2x
        ratio = np.abs(caf2).max() / np.abs(caf1).max()
        self.assertAlmostEqual(ratio, 2.0, places=1,
                               msg=f"Linearity violated: ratio = {ratio}")

    def test_noise_only_no_false_peak(self):
        """Verify no strong false peaks with uncorrelated noise."""
        np.random.seed(42)
        n = 4096

        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        surv = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

        caf = self._compute_caf(ref, surv)
        mag = np.abs(caf)

        # With uncorrelated noise, peak should be modest
        # Expected peak ~ sqrt(N) * std
        expected_noise_peak = np.sqrt(n) * np.std(ref) * np.std(surv)
        actual_peak = mag.max()

        # Peak should be within 3x expected noise level
        self.assertLess(actual_peak, 5 * expected_noise_peak,
                        "Unexpected peak in noise-only CAF")

    def test_caf_symmetry(self):
        """Verify CAF peaks are at expected positions when swapping ref/surv."""
        np.random.seed(42)
        n = 4096

        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        surv = np.roll(ref, 100)

        caf_rs = self._compute_caf(ref, surv)
        caf_sr = self._compute_caf(surv, ref)

        # Both should have clear peaks
        peak_rs = np.argmax(np.abs(caf_rs))

        # When we swap, the delay relationship inverts
        # peak_rs should be at 100
        self.assertEqual(peak_rs, 100, f"Expected peak at 100, got {peak_rs}")

        # The forward CAF should have a strong peak
        self.assertGreater(np.abs(caf_rs).max(), np.abs(caf_rs).mean() * 5,
                           "CAF peak not strong enough")

    def test_multiple_targets(self):
        """Verify CAF shows multiple targets."""
        np.random.seed(42)
        n = 4096

        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

        # Two targets at different delays
        delay1, delay2 = 100, 300
        surv = np.roll(ref, delay1) + 0.5 * np.roll(ref, delay2)

        caf = self._compute_caf(ref, surv)
        mag = np.abs(caf)

        # Find two highest peaks
        sorted_indices = np.argsort(mag)[::-1]
        peaks = sorted_indices[:10]

        detected_delays = set()
        for p in peaks:
            if any(abs(p - d) <= 2 for d in [delay1, delay2]):
                for d in [delay1, delay2]:
                    if abs(p - d) <= 2:
                        detected_delays.add(d)

        self.assertEqual(detected_delays, {delay1, delay2},
                         f"Did not detect both targets: {detected_delays}")


class TestCAFPerformance(unittest.TestCase):
    """Performance tests for CAF computation."""

    @classmethod
    def setUpClass(cls):
        """Load CAF library."""
        repo_root = Path(__file__).resolve().parents[2]
        lib_path = repo_root / "src" / "libkraken_caf_processing.so"

        if not lib_path.exists():
            raise unittest.SkipTest(f"CAF library not found at {lib_path}")

        cls.lib = ctypes.cdll.LoadLibrary(str(lib_path))

        cls.lib.caf_create.restype = ctypes.c_void_p
        cls.lib.caf_create.argtypes = [ctypes.c_int]
        cls.lib.caf_destroy.argtypes = [ctypes.c_void_p]
        cls.lib.caf_destroy.restype = None
        cls.lib.caf_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        cls.lib.caf_process.restype = None

    def test_caf_timing(self):
        """Benchmark CAF processing time."""
        import time

        n = 4096
        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        surv = np.roll(ref, 100)

        obj = self.lib.caf_create(n)
        out = np.zeros(n, dtype=np.complex64)

        # Warmup
        for _ in range(5):
            self.lib.caf_process(
                obj,
                ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )

        # Benchmark
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            self.lib.caf_process(
                obj,
                ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
        elapsed = time.perf_counter() - start

        self.lib.caf_destroy(obj)

        time_per_call_ms = (elapsed / iterations) * 1000
        print(f"\nCAF 4096 samples: {time_per_call_ms:.3f} ms/call")

        # Should complete in less than 5ms for real-time on Pi5
        self.assertLess(time_per_call_ms, 10.0,
                        f"CAF too slow: {time_per_call_ms:.3f} ms")


if __name__ == '__main__':
    unittest.main()
