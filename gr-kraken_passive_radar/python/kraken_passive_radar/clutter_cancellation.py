from gnuradio import gr
import numpy as np


class ClutterCanceller(gr.basic_block):
    """
    Adaptive Clutter Canceller using NLMS.

    This block implements a complex-valued NLMS adaptive FIR filter where:
      - Input 0 is the reference / predictor signal (x[n])
      - Input 1 is the surveillance / desired signal (d[n])
      - Output 0 is the error signal e[n] = d[n] - y_hat[n],
        where y_hat[n] is the NLMS estimate of the clutter component.

    Typical use in passive radar:
      - Reference: direct-path signal from illuminator
      - Surveillance: antenna looking into the scene (direct + clutter + targets)
      - Output: clutter-suppressed surveillance, with moving targets emphasized.
    """

    def __init__(self, num_taps=64, mu=0.1):
        """
        Parameters
        ----------
        num_taps : int
            Length of the adaptive FIR filter.
        mu : float
            NLMS step size (0 < mu <= 1). Smaller values converge more slowly
            but are more stable; larger values converge faster but may diverge
            if too large.
        """
        gr.basic_block.__init__(
            self,
            name="ClutterCanceller",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64],
        )

        self.num_taps = int(num_taps)
        self.mu = float(mu)

        # Adaptive weight vector (complex)
        self.w = np.zeros(self.num_taps, dtype=np.complex64)

        # History buffer for the reference signal so that we can form
        # tap-length vectors across work() calls.
        self.history = np.zeros(max(self.num_taps - 1, 0), dtype=np.complex64)

    def forecast(self, noutput_items, ninput_items_required):
        """
        GNU Radio scheduler hook: declare how many input items are needed
        to produce noutput_items.
        """
        # We require 1:1 reference/surveillance samples.
        ninput_items_required[0] = noutput_items
        ninput_items_required[1] = noutput_items

    def general_work(self, input_items, output_items):
        """
        Perform NLMS adaptation and clutter cancellation.

        input_items[0] : reference signal (complex64)
        input_items[1] : surveillance signal (complex64)
        output_items[0]: error signal (complex64)
        """
        ref = input_items[0]
        surv = input_items[1]
        out_err = output_items[0]

        n_input = min(len(ref), len(surv), len(out_err))
        if n_input <= 0:
            return 0

        # Concatenate history + new reference samples so we can slide a window
        full_ref = np.concatenate((self.history, ref[:n_input]))

        # NLMS adaptation loop
        for i in range(n_input):
            # Reference vector x[n] of length num_taps, newest on the right
            x = full_ref[i : i + self.num_taps]

            # Estimated clutter y_hat[n] = w^H x
            y_hat = np.vdot(self.w, x)  # vdot does conj(self.w)*x and sums

            # Error e[n] = d[n] - y_hat[n]
            e = surv[i] - y_hat

            # NLMS weight update:
            #   w[n+1] = w[n] + mu * e * conj(x) / (||x||^2 + eps)
            norm_x_sq = float(np.real(np.vdot(x, x))) + 1e-12
            if norm_x_sq > 0.0:
                self.w += self.mu * e * np.conj(x) / norm_x_sq

            # Output the error sample
            out_err[i] = e

        # Update reference history to the last (num_taps - 1) samples
        if self.num_taps > 1:
            self.history = full_ref[n_input:]
        else:
            self.history = np.zeros(0, dtype=np.complex64)

        # Consume the processed input
        self.consume(0, n_input)
        self.consume(1, n_input)
        return n_input
