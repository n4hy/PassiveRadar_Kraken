import os
import ctypes
import numpy as np
from gnuradio import gr
import sys
import time

class EcaBClutterCanceller(gr.sync_block):
    """
    ECA-B-based clutter canceller using an external C++ kernel loaded via ctypes.

    Inputs
    ------
    0: Reference channel (complex64)
    1..N: Surveillance channels (complex64)

    Output
    ------
    0..N-1: Clutter-suppressed surveillance (complex64)
    """

    def __init__(self, num_taps=16, num_surv_channels=1, lib_path=""):
        self.num_surv_channels = int(num_surv_channels)
        if self.num_surv_channels < 1:
            raise ValueError("Number of surveillance channels must be at least 1")

        gr.sync_block.__init__(
            self,
            name="EcaBClutterCanceller",
            in_sig=[np.complex64] * (1 + self.num_surv_channels),
            out_sig=[np.complex64] * self.num_surv_channels,
        )

        self.num_taps = int(num_taps)
        self.chunk_size = 4096 # Larger chunks to reduce overhead, fits in L2

        # Enforce minimum block size
        self.set_output_multiple(self.chunk_size)

        self._lib = self._load_library(lib_path)

        # Configure C API signatures
        self._lib.eca_b_create.restype = ctypes.c_void_p
        self._lib.eca_b_create.argtypes = [ctypes.c_int]

        self._lib.eca_b_destroy.restype = None
        self._lib.eca_b_destroy.argtypes = [ctypes.c_void_p]

        self._lib.eca_b_process.restype = None
        self._lib.eca_b_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        # Create state for each surveillance channel
        self._states = []
        for _ in range(self.num_surv_channels):
            state = self._lib.eca_b_create(self.num_taps)
            if not state:
                # Cleanup already created states before raising
                for s in self._states:
                    self._lib.eca_b_destroy(s)
                raise RuntimeError("Failed to create ECA-B canceller state")
            self._states.append(state)

        # Metrics
        self.last_log_time = time.time()
        self.items_processed = 0
        self.total_proc_time = 0.0
        self.log_interval = 2.0 # Seconds

    def _load_library(self, lib_path: str):
        candidates = []
        if lib_path:
            candidates.append(lib_path)

        base_dir = os.path.dirname(__file__)
        candidates.extend(
            [
                os.path.join(base_dir, "libkraken_eca_b_clutter_canceller.so"),
                os.path.join(base_dir, "kraken_eca_b_clutter_canceller.so"),
                "libkraken_eca_b_clutter_canceller.so",
                "kraken_eca_b_clutter_canceller.so",
            ]
        )

        last_err = None
        for candidate in candidates:
            try:
                return ctypes.cdll.LoadLibrary(candidate)
            except OSError as e:
                last_err = e
                continue

        print(f"DEBUG: EcaBClutterCanceller failed to load library from base_dir: {base_dir}", file=sys.stderr)
        try:
            print(f"DEBUG: Files in base_dir: {os.listdir(base_dir)}", file=sys.stderr)
        except Exception:
            pass
        raise OSError(
            f"Could not load ECA-B library. Tried: {candidates!r}. Last error: {last_err}"
        )

    def work(self, input_items, output_items):
        try:
            ref = input_items[0]
            n = len(ref)

            if n == 0:
                return 0

            # Process in chunks to maintain cache locality and prevent processing spikes
            for offset in range(0, n, self.chunk_size):
                current_n = min(self.chunk_size, n - offset)

                ref_chunk = ref[offset : offset+current_n]
                ref_ptr = ref_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

                for i in range(self.num_surv_channels):
                    if i + 1 >= len(input_items):
                        continue

                    surv_chunk = input_items[1 + i][offset : offset+current_n]
                    out_chunk = output_items[i][offset : offset+current_n]

                    surv_ptr = surv_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    out_ptr = out_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

                    t0 = time.time()
                    self._lib.eca_b_process(
                        self._states[i],
                        ref_ptr,
                        surv_ptr,
                        out_ptr,
                        current_n,
                    )
                    self.total_proc_time += (time.time() - t0)

            # Update stats
            self.items_processed += n
            now = time.time()
            dt = now - self.last_log_time
            if dt > self.log_interval:
                rate_msps = (self.items_processed / dt) / 1e6
                # Average processing time per sample (across all channels)
                # total_proc_time is sum of 4 calls per chunk.
                # Avg time per sample = total_time / items_processed
                avg_us = (self.total_proc_time / self.items_processed) * 1e6

                # CPU Load Estimate: if avg_us > (1/sample_rate), we overflow.
                # At 250kSPS, limit is 4us. At 2MSPS, limit is 0.5us.

                log_msg = f"[ECA] Rate: {rate_msps:.3f} MSPS | Avg Proc: {avg_us:.2f} us/sample | Load: {avg_us * rate_msps * 100:.1f}%"
                print(log_msg, flush=True)
                try:
                    with open("eca_stats.txt", "a") as f:
                        f.write(log_msg + "\n")
                except Exception:
                    pass # Don't crash on logging

                self.last_log_time = now
                self.items_processed = 0
                self.total_proc_time = 0.0

            return n
        except Exception as e:
            print(f"ERROR: Exception in work: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 0

    def __del__(self):
        try:
            if getattr(self, "_states", None) and getattr(self, "_lib", None):
                for state in self._states:
                    if state:
                        self._lib.eca_b_destroy(state)
                self._states = []
        except Exception:
            pass
