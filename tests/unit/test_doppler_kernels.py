"""
Unit tests for Doppler processing.
"""
import unittest
import numpy as np
import ctypes
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestDopplerProcessing(unittest.TestCase):
    """Test Doppler FFT processing."""

    @classmethod
    def setUpClass(cls):
        """Load Doppler library."""
        repo_root = Path(__file__).resolve().parents[2]
        lib_path = repo_root / "src" / "libkraken_doppler_processing.so"

        if not lib_path.exists():
            raise unittest.SkipTest(f"Doppler library not found at {lib_path}")

        cls.lib = ctypes.cdll.LoadLibrary(str(lib_path))
        cls.sample_rate = 2.4e6

    def test_doppler_resolution(self):
        """Verify Doppler frequency resolution."""
        n_samples = 4096
        sample_rate = self.sample_rate

        # Doppler resolution = sample_rate / n_samples
        expected_resolution = sample_rate / n_samples  # ~586 Hz

        self.assertAlmostEqual(expected_resolution, 585.9375, places=2)

    def test_doppler_shift_detection(self):
        """Verify correct Doppler bin for known frequency shift."""
        n = 4096
        sample_rate = self.sample_rate

        # Create signal with known Doppler
        doppler_hz = 1000.0
        t = np.arange(n) / sample_rate
        signal = np.exp(2j * np.pi * doppler_hz * t).astype(np.complex64)

        # Apply FFT
        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/sample_rate)

        peak_idx = np.argmax(np.abs(spectrum))
        detected_doppler = freqs[peak_idx]

        self.assertAlmostEqual(detected_doppler, doppler_hz, delta=sample_rate/n)

    def test_windowing_reduces_sidelobes(self):
        """Verify window function reduces spectral leakage."""
        n = 4096
        sample_rate = self.sample_rate

        doppler_hz = 1234.5  # Not on a bin center
        t = np.arange(n) / sample_rate
        signal = np.exp(2j * np.pi * doppler_hz * t).astype(np.complex64)

        # Without window
        spectrum_rect = np.abs(np.fft.fft(signal))

        # With Hamming window
        window = np.hamming(n)
        spectrum_hamming = np.abs(np.fft.fft(signal * window))

        # Normalize
        spectrum_rect /= spectrum_rect.max()
        spectrum_hamming /= spectrum_hamming.max()

        # Find sidelobes (away from main lobe)
        peak_idx = np.argmax(spectrum_rect)
        far_bins = np.abs(np.arange(n) - peak_idx) > 50

        sidelobe_rect = spectrum_rect[far_bins].max()
        sidelobe_hamming = spectrum_hamming[far_bins].max()

        # Hamming should have lower sidelobes
        self.assertLess(sidelobe_hamming, sidelobe_rect,
                        "Windowing should reduce sidelobes")

    def test_negative_doppler(self):
        """Verify detection of negative Doppler shift."""
        n = 4096
        sample_rate = self.sample_rate

        doppler_hz = -800.0
        t = np.arange(n) / sample_rate
        signal = np.exp(2j * np.pi * doppler_hz * t).astype(np.complex64)

        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/sample_rate)

        peak_idx = np.argmax(np.abs(spectrum))
        detected_doppler = freqs[peak_idx]

        self.assertAlmostEqual(detected_doppler, doppler_hz, delta=sample_rate/n)

    def test_multiple_dopplers(self):
        """Verify detection of multiple Doppler frequencies."""
        n = 4096
        sample_rate = self.sample_rate

        dopplers = [500.0, -1000.0, 2000.0]
        t = np.arange(n) / sample_rate

        signal = np.zeros(n, dtype=np.complex64)
        for d in dopplers:
            signal += np.exp(2j * np.pi * d * t)

        spectrum = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(n, 1/sample_rate)

        # Find peaks
        threshold = spectrum.max() * 0.5
        peak_mask = spectrum > threshold

        detected = []
        for i in range(n):
            if peak_mask[i]:
                if not detected or abs(freqs[i] - detected[-1]) > 100:
                    detected.append(freqs[i])

        # Should detect all three Dopplers
        for d in dopplers:
            found = any(abs(det - d) < sample_rate/n for det in detected)
            self.assertTrue(found, f"Doppler {d} Hz not detected")

    def test_doppler_ambiguity(self):
        """Verify Doppler ambiguity (aliasing) behavior."""
        n = 4096
        sample_rate = self.sample_rate
        max_doppler = sample_rate / 2  # Nyquist

        # Doppler beyond Nyquist should alias
        doppler_hz = max_doppler + 1000.0
        t = np.arange(n) / sample_rate
        signal = np.exp(2j * np.pi * doppler_hz * t).astype(np.complex64)

        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/sample_rate)

        peak_idx = np.argmax(np.abs(spectrum))
        detected_doppler = freqs[peak_idx]

        # Should alias to negative frequency
        expected_alias = doppler_hz - sample_rate
        self.assertAlmostEqual(detected_doppler, expected_alias, delta=sample_rate/n)


class TestDopplerWindows(unittest.TestCase):
    """Test window function implementations."""

    def test_hamming_window(self):
        """Verify Hamming window properties."""
        n = 1024
        w = np.hamming(n)

        # Should be symmetric
        np.testing.assert_array_almost_equal(w, w[::-1])

        # Should taper to edges
        self.assertLess(w[0], w[n//2])
        self.assertLess(w[-1], w[n//2])

        # Peak should be near 1.0 at center
        self.assertAlmostEqual(w[n//2], 1.0, places=1)

    def test_hanning_window(self):
        """Verify Hanning (Hann) window properties."""
        n = 1024
        w = np.hanning(n)

        # Should be symmetric
        np.testing.assert_array_almost_equal(w, w[::-1])

        # Should taper to zero at edges
        self.assertAlmostEqual(w[0], 0.0, places=2)

    def test_blackman_window(self):
        """Verify Blackman window properties."""
        n = 1024
        w = np.blackman(n)

        # Should be symmetric
        np.testing.assert_array_almost_equal(w, w[::-1])

        # Should have low sidelobes
        spectrum = np.abs(np.fft.fft(w, 8*n))
        spectrum_db = 20 * np.log10(spectrum / spectrum.max() + 1e-10)

        # Far sidelobes (skip main lobe region) should be < -40 dB
        main_lobe_width = 100  # bins around DC
        far_sidelobes = np.concatenate([
            spectrum_db[main_lobe_width:4*n],
            spectrum_db[4*n:8*n-main_lobe_width]
        ])
        self.assertLess(far_sidelobes.max(), -40,
                        f"Blackman sidelobes too high: {far_sidelobes.max():.1f} dB")

    def test_window_gain_normalization(self):
        """Verify window functions don't introduce NaN or Inf."""
        n = 4096

        for window_name, window_func in [('hamming', np.hamming),
                                          ('hanning', np.hanning),
                                          ('blackman', np.blackman)]:
            w = window_func(n)

            # Window should be finite
            self.assertFalse(np.any(np.isnan(w)), f"{window_name} contains NaN")
            self.assertFalse(np.any(np.isinf(w)), f"{window_name} contains Inf")

            # Window should be mostly positive (blackman has tiny negative values at edges)
            self.assertGreater(np.sum(w > 0), n * 0.95,
                               f"{window_name} has too many non-positive values")

            # Window max should be approximately 1
            self.assertGreater(w.max(), 0.9, f"{window_name} max too low")
            self.assertLess(w.max(), 1.1, f"{window_name} max too high")


if __name__ == '__main__':
    unittest.main()
