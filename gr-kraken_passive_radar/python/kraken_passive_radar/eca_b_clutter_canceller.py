import os
import ctypes
import numpy as np
from gnuradio import gr
import sys

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

    def __init__(self, num_taps=64, num_surv_channels=1, lib_path=""):
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
                # Debug output removed for stable run
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
            # print(f"DEBUG: Enter work. n={n}", file=sys.stderr)

            if n == 0:
                # print("DEBUG: Returning 0 (empty input)", file=sys.stderr)
                return 0

            # Prepare Reference (used for all)
            ref_c = np.ascontiguousarray(ref, dtype=np.complex64)
            ref_f = ref_c.view(np.float32)
            ref_ptr = ref_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # Process each surveillance channel
            for i in range(self.num_surv_channels):
                # Check input bounds
                if i + 1 >= len(input_items):
                    print(f"ERROR: Missing input channel {i+1}", file=sys.stderr)
                    continue

                surv = input_items[1 + i]
                out_err = output_items[i]

                surv_c = np.ascontiguousarray(surv[:n], dtype=np.complex64)
                # Ensure output buffer is contiguous (output_items usually are)
                out_c = np.ascontiguousarray(out_err[:n], dtype=np.complex64)

                surv_f = surv_c.view(np.float32)
                out_f = out_c.view(np.float32)

                self._lib.eca_b_process(
                    self._states[i],
                    ref_ptr,
                    surv_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    n,
                )

                out_err[:n] = out_c

            # print(f"DEBUG: Exit work. Returning {n}", file=sys.stderr)
            return n
        except Exception as e:
            print(f"ERROR: Exception in work: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 0 # Fail gracefully

    def __del__(self):
        try:
            if getattr(self, "_states", None) and getattr(self, "_lib", None):
                for state in self._states:
                    if state:
                        self._lib.eca_b_destroy(state)
                self._states = []
        except Exception:
            # Destructors must never raise
            pass
