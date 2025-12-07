
from gnuradio import gr
import numpy as np

class ClutterCanceller(gr.basic_block):
    """
    Adaptive Clutter Canceller using NLMS.

    Inputs:
    0: Reference Signal (Predictor)
    1: Surveillance Signal (Desired)

    Outputs:
    0: Error Signal (Surveillance - Estimated Clutter)
    """
    def __init__(self, num_taps=32, mu=0.1):
        gr.basic_block.__init__(self,
            name="ClutterCanceller",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64])

        self.num_taps = num_taps
        self.mu = mu
        self.w = np.zeros(num_taps, dtype=np.complex64)

        # Buffer for reference history
        # We need at least num_taps history to compute filter output
        # But basic_block gives us chunks. We need to manage history manually or use sync_block with history?
        # sync_block with set_history is easier for FIR operations.
        # But this is an IIR filter? No, FIR adaptive.

        # Let's switch to sync_block if possible?
        # No, adaptive filter updates weights, so it has internal state.
        # basic_block is fine, we just need to keep a small history buffer.
        self.history = np.zeros(num_taps, dtype=np.complex64)

    def general_work(self, input_items, output_items):
        ref_in = input_items[0]
        surv_in = input_items[1]
        out_err = output_items[0]

        n_input = min(len(ref_in), len(surv_in), len(out_err))

        if n_input == 0:
            return 0

        # Process sample by sample (slow in Python, but correct for NLMS)
        # To optimize: use numba or C++. For now, simple loop.

        # We need history of 'ref' to compute dot product w^H * ref_vector

        # Prepare full reference stream including history
        full_ref = np.concatenate((self.history, ref_in[:n_input]))

        # Iterate
        for i in range(n_input):
            # Ref vector at time i (current + past samples)
            # x[n] = [ref[i], ref[i-1], ... ref[i-N+1]]
            # In full_ref: full_ref[i + num_taps - 1 : i - 1 : -1] ?
            # Let's say history is [r[-N], ..., r[-1]].
            # full_ref is [r[-N]...r[-1], r[0]...r[n-1]]
            # At i=0, we need r[0], r[-1], ... r[-N+1]
            # So slice full_ref from i to i + num_taps
            # But we usually define taps such that y[n] = sum(w[k] * x[n-k])
            # So vector x is full_ref[i : i + num_taps][::-1]

            x = full_ref[i : i + self.num_taps][::-1]

            # Filter output (Estimate of clutter)
            y = np.dot(self.w, x)

            # Error (Desired - Estimate)
            # Desired is Surveillance
            d = surv_in[i]
            e = d - y

            # Update weights (NLMS)
            # w[n+1] = w[n] + mu * e * conj(x) / (norm(x)^2 + eps)
            norm_x_sq = np.real(np.dot(x, np.conj(x))) + 1e-12
            self.w += self.mu * e * np.conj(x) / norm_x_sq

            out_err[i] = e

        # Update history
        self.history = full_ref[n_input:]

        self.consume(0, n_input)
        self.consume(1, n_input)
        return n_input
