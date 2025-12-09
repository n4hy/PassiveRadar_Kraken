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

        w[n+1] = w[n] + mu * conj(e[n]) * x[n] / (||x[n]||^2 + eps)

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

        # History for reference channel (to form tap vector at block boundaries)
        # We need num_taps-1 past samples.
        self._ref_history = np.zeros(self.num_taps - 1, dtype=np.complex64)

    def general_work(self, input_items, output_items):
        ref = input_items[0]
        surv = input_items[1]
        out_err = output_items[0]

        n_input = min(len(ref), len(surv))

        if n_input <= 0:
            return 0

        # Construct full reference buffer with history:
        # [hist_0, ..., hist_M-1, ref_0, ..., ref_N-1]
        full_ref = np.concatenate((self._ref_history, ref[:n_input]))

        # Update history for next call
        # We need the last (num_taps - 1) samples from the current full sequence
        if n_input >= self.num_taps - 1:
            self._ref_history = ref[n_input - (self.num_taps - 1):n_input]
        else:
            # If input is shorter than history length, we roll the buffer
            # New history is: old_history_tail + ref
            needed = self.num_taps - 1
            combined = np.concatenate((self._ref_history, ref[:n_input]))
            self._ref_history = combined[-needed:]

        # Check weight validity
        if not np.all(np.isfinite(self.w)):
            self.w[:] = 0.0

        for i in range(n_input):
            # We want x vector corresponding to time i (which corresponds to ref[i] and surv[i])
            # The causal window should be [ref[i-(M-1)], ..., ref[i]]
            # In full_ref:
            # ref[0] is at index M-1.
            # ref[i] is at index i + M - 1.
            # So slice [i : i + M] gives [ref[i-(M-1)] ... ref[i]].

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
            # Use conjugate of error to ensure descent in complex domain: w += mu * conj(e) * x
            delta = self.mu * np.conj(e) * x / (norm_x_sq + self.eps)

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
