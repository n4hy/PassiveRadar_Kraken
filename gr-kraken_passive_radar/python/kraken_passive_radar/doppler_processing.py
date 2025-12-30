import os
import ctypes
import numpy as np
from gnuradio import gr
import sys
import time

class DopplerProcessingBlock(gr.basic_block):
    """
    Doppler Processing Block (C++ Accelerated)

    Accumulates 'doppler_len' vectors of size 'fft_len', performs
    Range-Doppler processing (Windowing + FFT + LogMag), and outputs
    a single flattened vector (image) of size 'fft_len * doppler_len'.
    """

    def __init__(self, fft_len=1024, doppler_len=128):
        gr.basic_block.__init__(
            self,
            name="Doppler Processing (C++)",
            in_sig=[(np.complex64, fft_len)],
            out_sig=[(np.float32, fft_len * doppler_len)],
        )

        self.fft_len = fft_len
        self.doppler_len = doppler_len

        self._lib = self._load_library()

        # Configure C API
        self._lib.doppler_create.restype = ctypes.c_void_p
        self._lib.doppler_create.argtypes = [ctypes.c_int, ctypes.c_int]

        self._lib.doppler_destroy.restype = None
        self._lib.doppler_destroy.argtypes = [ctypes.c_void_p]

        self._lib.doppler_process.restype = None
        self._lib.doppler_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        self._state = self._lib.doppler_create(self.fft_len, self.doppler_len)
        if not self._state:
            raise RuntimeError("Failed to create Doppler processor state")

        # Metrics
        self.last_log_time = time.time()
        self.frames_processed = 0
        self.total_proc_time = 0.0
        self.log_interval = 2.0 # Seconds

    def _load_library(self):
        # Helper to load the shared library
        base_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(base_dir, "libkraken_doppler_processing.so"),
            "libkraken_doppler_processing.so"
        ]

        last_err = None
        for path in candidates:
            try:
                return ctypes.cdll.LoadLibrary(path)
            except OSError as e:
                last_err = e

        raise OSError(f"Could not load Doppler library. Tried {candidates}. Last error: {last_err}")

    def forecast(self, noutput_items, ninput_items_required):
        # We act as a decimator: 1 output requires 'doppler_len' input vectors
        req = noutput_items * self.doppler_len

        if isinstance(ninput_items_required, list):
            ninput_items_required[0] = req
        else:
            pass

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]

        n_in = len(in0)
        n_out = len(out0)

        # Determine how many full blocks we can process
        num_blocks = min(n_out, n_in // self.doppler_len)

        if num_blocks == 0:
            self.consume(0, 0)
            return 0

        # Process blocks
        for i in range(num_blocks):
            in_start = i * self.doppler_len
            in_end = in_start + self.doppler_len

            in_slice = in0[in_start : in_end]
            out_slice = out0[i]

            in_ptr = in_slice.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_ptr = out_slice.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            t0 = time.time()
            self._lib.doppler_process(self._state, in_ptr, out_ptr)
            self.total_proc_time += (time.time() - t0)

        # Consume the inputs we used
        self.consume(0, num_blocks * self.doppler_len)

        # Update stats
        self.frames_processed += num_blocks
        now = time.time()
        dt = now - self.last_log_time
        if dt > self.log_interval:
            rate = self.frames_processed / dt
            # Avg time per frame
            avg_ms = (self.total_proc_time / self.frames_processed) * 1000
            print(f"[Doppler] Rate: {rate:.2f} frames/sec | Avg Proc: {avg_ms:.2f} ms/frame", flush=True)
            self.last_log_time = now
            self.frames_processed = 0
            self.total_proc_time = 0.0

        return num_blocks

    def __del__(self):
        if getattr(self, "_state", None) and getattr(self, "_lib", None):
            self._lib.doppler_destroy(self._state)
