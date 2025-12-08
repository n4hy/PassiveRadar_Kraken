import os
import ctypes
import numpy as np
from gnuradio import gr


class EcaBClutterCanceller(gr.basic_block):
    """
    ECA-B-based clutter canceller using an external C++ kernel loaded via ctypes.

    Inputs
    ------
    0: Reference channel (complex64)
    1: Surveillance channel (complex64)

    Output
    ------
    0: Clutter-suppressed surveillance (complex64)
    """

    def __init__(self, num_taps=64, lib_path=""):
        gr.basic_block.__init__(
            self,
            name="EcaBClutterCanceller",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64],
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

        self._state = self._lib.eca_b_create(self.num_taps)
        if not self._state:
            raise RuntimeError("Failed to create ECA-B canceller state")

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

        raise OSError(
            f"Could not load ECA-B library. Tried: {candidates!r}. Last error: {last_err}"
        )

    def forecast(self, noutput_items, ninput_items_required):
        ninput_items_required[0] = noutput_items
        ninput_items_required[1] = noutput_items

    def general_work(self, input_items, output_items):
        ref = input_items[0]
        surv = input_items[1]
        out_err = output_items[0]

        n = min(len(ref), len(surv), len(out_err))
        if n <= 0:
            return 0

        ref_c = np.ascontiguousarray(ref[:n], dtype=np.complex64)
        surv_c = np.ascontiguousarray(surv[:n], dtype=np.complex64)
        out_c = np.ascontiguousarray(out_err[:n], dtype=np.complex64)

        # Reinterpret complex64 as float32[2*n]
        ref_f = ref_c.view(np.float32)
        surv_f = surv_c.view(np.float32)
        out_f = out_c.view(np.float32)

        self._lib.eca_b_process(
            self._state,
            ref_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )

        out_err[:n] = out_c

        self.consume(0, n)
        self.consume(1, n)
        return n

    def __del__(self):
        try:
            if getattr(self, "_state", None) and getattr(self, "_lib", None):
                self._lib.eca_b_destroy(self._state)
                self._state = None
        except Exception:
            # Destructors must never raise
            pass
