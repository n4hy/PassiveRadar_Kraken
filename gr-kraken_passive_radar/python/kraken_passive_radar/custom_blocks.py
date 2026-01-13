import ctypes
import numpy as np
from gnuradio import gr
import os

class ConditioningBlock(gr.sync_block):
    """
    Applies AGC/Conditioning using C++ kernel.
    """
    def __init__(self, rate=1e-5):
        gr.sync_block.__init__(
            self,
            name="Conditioning",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        self.lib = self._load_lib()
        self.obj = self.lib.cond_create(ctypes.c_float(rate))

    def _load_lib(self):
        # Locate libkraken_conditioning.so
        path = os.path.join(os.path.dirname(__file__), "libkraken_conditioning.so")
        if not os.path.exists(path):
             # Fallback to local src for dev
             path = os.path.abspath("src/libkraken_conditioning.so")

        lib = ctypes.cdll.LoadLibrary(path)
        lib.cond_create.restype = ctypes.c_void_p
        lib.cond_create.argtypes = [ctypes.c_float]
        lib.cond_destroy.argtypes = [ctypes.c_void_p]
        lib.cond_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        return lib

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        n = len(in0)

        np.copyto(out0, in0)

        self.lib.cond_process(
            self.obj,
            out0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n
        )
        return n

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'obj'):
            self.lib.cond_destroy(self.obj)

class CafBlock(gr.basic_block):
    """
    Computes Range Profile using C++ CAF kernel.
    Inputs: 0: Ref, 1: Surv (Stream)
    Output: 0: Range Profile (Vector length N)

    Decimates stream by N to produce 1 vector.
    """
    def __init__(self, n_samples=4096):
        self.n_samples = n_samples
        gr.basic_block.__init__(
            self,
            name="CAF Block",
            in_sig=[np.complex64, np.complex64],
            out_sig=[(np.complex64, n_samples)]
        )

        self.lib = self._load_lib()
        self.obj = self.lib.caf_create(n_samples)

        # We need N input items to produce 1 output item
        self.set_output_multiple(1)

    def _load_lib(self):
        path = os.path.join(os.path.dirname(__file__), "libkraken_caf_processing.so")
        if not os.path.exists(path):
             path = os.path.abspath("src/libkraken_caf_processing.so")

        lib = ctypes.cdll.LoadLibrary(path)
        lib.caf_create.restype = ctypes.c_void_p
        lib.caf_create.argtypes = [ctypes.c_int]
        lib.caf_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        return lib

    def general_work(self, input_items, output_items):
        ref = input_items[0]
        surv = input_items[1]
        out = output_items[0]

        n_in = len(ref)
        n_out_avail = len(out)

        # Calculate how many vectors we can produce
        n_chunks = min(n_out_avail, n_in // self.n_samples)

        if n_chunks == 0:
            # Tell scheduler we need more data
            self.consume_each(0)
            return 0

        for i in range(n_chunks):
            start = i * self.n_samples
            end = start + self.n_samples

            r_ptr = ref[start:end].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            s_ptr = surv[start:end].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            o_ptr = out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            self.lib.caf_process(self.obj, r_ptr, s_ptr, o_ptr)

        self.consume_each(n_chunks * self.n_samples)
        return n_chunks

class BackendBlock(gr.sync_block):
    """
    CFAR + Fusion.
    Input: N channels of Range-Doppler maps (Vectors).
    Output: Detection Map (Vector).
    """
    def __init__(self, rows, cols, num_inputs=1):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.num_inputs = num_inputs

        gr.sync_block.__init__(
            self,
            name="Backend",
            in_sig=[(np.float32, self.size)] * num_inputs,
            out_sig=[(np.float32, self.size)]
        )

        self.lib = self._load_lib()

    def _load_lib(self):
        path = os.path.join(os.path.dirname(__file__), "libkraken_backend.so")
        if not os.path.exists(path):
             path = os.path.abspath("src/libkraken_backend.so")
        lib = ctypes.cdll.LoadLibrary(path)
        lib.fusion_process.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cfar_2d.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        return lib

    def work(self, input_items, output_items):
        n_vecs = len(input_items[0])
        out = output_items[0]

        input_ptrs = (ctypes.POINTER(ctypes.c_float) * self.num_inputs)()

        for i in range(n_vecs):
            # 1. Fusion
            for k in range(self.num_inputs):
                input_ptrs[k] = input_items[k][i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            fused = np.zeros(self.size, dtype=np.float32)
            self.lib.fusion_process(input_ptrs, self.num_inputs, fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.size)

            # 2. CFAR
            self.lib.cfar_2d(
                fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.rows, self.cols,
                2, 4, 15.0 # Guard, Train, Thresh
            )

        return n_vecs

class TimeAlignmentBlock(gr.sync_block):
    """
    Computes delay/phase offset between Ref and Surv.
    Inputs: 0: Ref, 1: Surv
    Output: None (Prints updates)
    """
    def __init__(self, n_samples=4096, interval_sec=1.0, samp_rate=2.4e6):
        gr.sync_block.__init__(
            self,
            name="Time Alignment Probe",
            in_sig=[np.complex64, np.complex64],
            out_sig=None
        )
        self.n_samples = n_samples
        self.interval_samples = int(interval_sec * samp_rate)
        self.processed_samples = 0

        self.lib = self._load_lib()
        self.obj = self.lib.align_create(n_samples)

    def _load_lib(self):
        path = os.path.join(os.path.dirname(__file__), "libkraken_time_alignment.so")
        if not os.path.exists(path):
             path = os.path.abspath("src/libkraken_time_alignment.so")
        lib = ctypes.cdll.LoadLibrary(path)
        lib.align_create.restype = ctypes.c_void_p
        lib.align_create.argtypes = [ctypes.c_int]
        lib.align_destroy.argtypes = [ctypes.c_void_p]
        lib.align_compute.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
        return lib

    def work(self, input_items, output_items):
        ref = input_items[0]
        surv = input_items[1]
        n = len(ref)

        # Check if time to run alignment
        # This is a probe, so we just sample chunks periodically.

        self.processed_samples += n

        if self.processed_samples >= self.interval_samples:
            self.processed_samples = 0

            # Take a chunk
            if n >= self.n_samples:
                r_ptr = ref[:self.n_samples].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                s_ptr = surv[:self.n_samples].ctypes.data_as(ctypes.POINTER(ctypes.c_float))

                delay = ctypes.c_int(0)
                phase = ctypes.c_float(0.0)

                self.lib.align_compute(self.obj, r_ptr, s_ptr, ctypes.byref(delay), ctypes.byref(phase))

                print(f"[Alignment] Delay: {delay.value} samples, Phase: {phase.value:.3f} rad")

        return n

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'obj'):
            self.lib.align_destroy(self.obj)
