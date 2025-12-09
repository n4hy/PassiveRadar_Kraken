import os
import sys
import unittest
import numpy as np

# Ensure we use the mock GNU Radio implementation for tests
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Ensure we can find the local 'kraken_passive_radar' OOT python package
OOT_PYTHON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../gr-kraken_passive_radar/python"))
if OOT_PYTHON_DIR not in sys.path:
    sys.path.insert(0, OOT_PYTHON_DIR)

import mock_gnuradio as gnuradio  # noqa: E402
sys.modules["gnuradio"] = gnuradio

from kraken_passive_radar import ClutterCanceller  # noqa: E402

class TestClutterCancellerNLMSDelayed(unittest.TestCase):
    def test_nlms_reduces_delayed_clutter(self):
        rng = np.random.default_rng(0)
        n_samples = 2000
        num_taps = 32
        mu = 0.05
        delay = 5  # Clutter is delayed by 5 samples

        # Reference signal (random complex)
        ref = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)

        # Surveillance = Delayed Reference + Noise
        # surv[n] = ref[n - delay]
        surv = np.zeros_like(ref)
        surv[delay:] = ref[:-delay]

        surv *= 0.8 # Some gain

        # Instantiate NLMS
        cc = ClutterCanceller(num_taps=num_taps, mu=mu)

        input_items = [ref.copy(), surv.copy()]
        out_err = np.zeros_like(surv)
        output_items = [out_err]

        cc.general_work(input_items, output_items)

        # Check power in the steady state (ignore initial convergence)
        start_check = 500
        power_in = np.mean(np.abs(surv[start_check:]) ** 2)
        power_out = np.mean(np.abs(out_err[start_check:]) ** 2)

        reduction_db = 10.0 * np.log10(power_in / (power_out + 1e-12))
        print(f"Delay={delay}, Reduction={reduction_db:.2f} dB")

        self.assertGreater(reduction_db, 6.0, "Filter failed to cancel delayed clutter")

if __name__ == "__main__":
    unittest.main()
