import numpy as np
from gnuradio import gr


class ClutterCanceller(gr.basic_block):
    """
    NLMS Adaptive Clutter Canceller
    --------------------------------
    Inputs:
      - input 0: reference channel (complex)
      - input 1: surveillance channel (complex)

    Output:
      - output 0: clutter-cancelled surveillance (complex error signal)

    Implements classical NLMS:

        w[n+1] = w[n] + mu * e[n] * conj(x[n]) / (||x[n]||^2 + eps)

    where:
        e[n] = d[n] - w^H x[n]

    This implementation includes numerical safety guards so it
    tends not to generate NaN or Inf weights, and it works both
    in real GNU Radio and under the mock_gnuradio test harness.
    """

    def __init__(self, num_taps=16, mu=0.05):
        gr.basic_block.__init__(
            self,
            name="Kraken NLMS Clutter Canceller",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64],
        )

        self.num_taps = int(num_taps)
        self.mu = float(mu)

        # Adaptive filter weights
        self.w = np.zeros(self.num_taps, dtype=np.complex64)

        # Small epsilon to prevent division-by-zero
        self.eps = 1e-12

    def general_work(self, input_items, output_items):
        ref = input_items[0]
        surv = input_items[1]
        out_err = output_items[0]

        n_input = min(len(ref), len(surv))

        if n_input <= 0:
            return 0

        # Zero-pad reference so we can form full tap windows safely
        full_ref = np.zeros(n_input + self.num_taps, dtype=np.complex64)
        full_ref[:n_input] = ref

        for i in range(n_input):
            # If weights ever became non-finite, reset before using
            if not np.all(np.isfinite(self.w)):
                self.w[:] = 0.0

            # Reference vector for this time step
            x = full_ref[i : i + self.num_taps]

            # Estimated clutter: y_hat = w^H x
            y_hat = np.vdot(self.w, x)

            # Error signal (clutter-cancelled output)
            e = surv[i] - y_hat

            # Power of reference vector
            norm_x_sq = float(np.real(np.vdot(x, x)))

            # Guard against degenerate or non-finite reference power
            if (not np.isfinite(norm_x_sq)) or (norm_x_sq <= self.eps):
                out_err[i] = e
                continue

            # Proposed NLMS weight update
            delta = self.mu * e * np.conj(x) / (norm_x_sq + self.eps)

            # If the update itself is non-finite, reset and skip
            if not np.all(np.isfinite(delta)):
                self.w[:] = 0.0
                out_err[i] = e
                continue

            # Apply update
            self.w += delta

            # Final safety check
            if not np.all(np.isfinite(self.w)):
                self.w[:] = 0.0

            out_err[i] = e

        # In real GNU Radio runtime, consume_each exists.
        # In the test harness mock_gnuradio, it does not.
        if hasattr(self, "consume_each"):
            self.consume_each(n_input)

        return n_input
