import os
import sys
import unittest
import numpy as np

# Ensure we use the mock GNU Radio implementation for tests, not the real C++ GNURadio
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import mock_gnuradio as gnuradio  # noqa: E402

sys.modules["gnuradio"] = gnuradio  # so "from gnuradio import gr" resolves to the mock

from kraken_passive_radar import ClutterCanceller  # noqa: E402


class TestClutterCancellerNLMS(unittest.TestCase):
    """
    Deterministic, numerically sane NLMS test.

    Scenario:
      - Reference x[n]: single complex sinusoid (strong "clutter"/direct path)
      - Surveillance d[n]: clutter + small complex noise
      - NLMS should learn the clutter and reduce its power in the error signal.

    We use conservative mu and moderate data length to avoid pathological
    numerical regimes while still verifying useful clutter suppression.
    """

    def test_nlms_reduces_clutter_power(self):
        rng = np.random.default_rng(0)

        n_samples = 2000
        num_taps = 16
        mu = 0.02  # conservative, stable step size

        # Time index
        n = np.arange(n_samples, dtype=np.float32)

        # Reference: single complex sinusoid (strong clutter/direct path)
        f0 = 0.01  # normalized frequency
        ref = np.exp(1j * 2.0 * np.pi * f0 * n).astype(np.complex64)

        # Surveillance: clutter + small noise
        clutter_gain = np.complex64(0.8 + 0.0j)
        noise_power = 1e-3

        noise = (
            rng.standard_normal(n_samples).astype(np.float32)
            + 1j * rng.standard_normal(n_samples).astype(np.float32)
        ).astype(np.complex64)

        # Normalize noise to desired power
        noise *= np.sqrt(
            noise_power / (np.mean(np.abs(noise) ** 2) + 1e-12)
        ).astype(np.float32)

        d = clutter_gain * ref + noise

        # Instantiate the NLMS clutter canceller
        cc = ClutterCanceller(num_taps=num_taps, mu=mu)

        # Prepare I/O buffers as GNU Radio would present them
        input_items = [ref.copy(), d.copy()]
        out_err = np.zeros_like(d)
        output_items = [out_err]

        produced = cc.general_work(input_items, output_items)
        self.assertEqual(
            produced,
            n_samples,
            msg=f"Expected to process {n_samples} samples, got {produced}",
        )

        power_in = np.mean(np.abs(d) ** 2)
        power_out = np.mean(np.abs(out_err) ** 2)

        # Basic sanity: powers must be finite and positive
        self.assertTrue(
            np.isfinite(power_in) and power_in > 0,
            msg=f"Input power is not finite/positive: {power_in}",
        )
        self.assertTrue(
            np.isfinite(power_out) and power_out > 0,
            msg=f"Output power is not finite/positive: {power_out}",
        )

        reduction_db = 10.0 * np.log10(power_in / (power_out + 1e-12))

        # We do not demand heroic performance in a tiny CPI; 6 dB is fine.
        self.assertGreater(
            reduction_db,
            6.0,
            msg=f"Expected >6 dB clutter reduction, got {reduction_db:.2f} dB",
        )


if __name__ == "__main__":
    unittest.main()
