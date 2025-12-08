import os
import sys
import unittest
import numpy as np

# Ensure mock_gnuradio is importable
sys.path.insert(0, os.path.dirname(__file__))
import mock_gnuradio  # noqa: F401

# Ensure we can import the local kraken_passive_radar package
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
python_pkg = os.path.join(repo_root, "gr-kraken_passive_radar", "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from kraken_passive_radar import ClutterCanceller


class TestClutterCancellerNLMS(unittest.TestCase):
    def test_nlms_reduces_clutter_power(self):
        np.random.seed(123)

        num_taps = 8
        mu = 0.5
        n_samples = 4096

        # Generate reference signal x[n]
        x_long = (np.random.randn(n_samples + num_taps - 1) +
                  1j * np.random.randn(n_samples + num_taps - 1)).astype(np.complex64)

        # True clutter filter h (unknown to NLMS)
        h = (np.random.randn(num_taps) +
             1j * np.random.randn(num_taps)).astype(np.complex64)

        # Generate surveillance d[n] = x * h (valid convolution) + small noise
        d_valid = np.convolve(x_long, h, mode="valid").astype(np.complex64)
        d = d_valid[:n_samples]
        d += 0.01 * (np.random.randn(n_samples) +
                     1j * np.random.randn(n_samples)).astype(np.complex64)

        # Align reference with the last n_samples of x_long
        x = x_long[num_taps - 1:num_taps - 1 + n_samples]

        # Instantiate block
        cc = ClutterCanceller(num_taps=num_taps, mu=mu)

        # Prepare input/output buffers for one call to general_work
        input_items = [x.copy(), d.copy()]
        out_err = np.zeros_like(d)
        output_items = [out_err]

        produced = cc.general_work(input_items, output_items)

        assert produced == n_samples

        # Check that the output error power is significantly less than input surveillance power.
        power_in = np.mean(np.abs(d) ** 2)
        power_out = np.mean(np.abs(out_err) ** 2)

        # We expect at least ~10 dB reduction in clutter power in this synthetic scenario.
        reduction_db = 10 * np.log10(power_in / (power_out + 1e-12))

        self.assertGreater(reduction_db, 10.0, msg=f"Expected >10 dB clutter reduction, got {reduction_db:.2f} dB")


if __name__ == "__main__":
    unittest.main()
