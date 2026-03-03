import sys
import unittest
import numpy as np
import ctypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from conftest import find_kernel_lib


class TestCafCpp(unittest.TestCase):
    def test_caf_process(self):
        lib_path = find_kernel_lib("caf_processing")

        if not lib_path.exists():
            self.skipTest(f"CAF library not found at {lib_path}")

        try:
            lib = ctypes.cdll.LoadLibrary(str(lib_path))
        except OSError as e:
            self.skipTest(f"Could not load C++ library: {e}")

        lib.caf_create.restype = ctypes.c_void_p
        lib.caf_create.argtypes = [ctypes.c_int]

        lib.caf_destroy.argtypes = [ctypes.c_void_p]
        lib.caf_destroy.restype = None

        lib.caf_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.caf_process.restype = None

        n = 4096
        # print(f"DEBUG: Creating CAF with n={n}", flush=True)
        obj = lib.caf_create(n)
        # print(f"DEBUG: Created CAF Object: {obj}", flush=True)
        if obj is None:
            self.fail("Failed to create CAF object")

        np.random.seed(42)
        ref = (np.random.randn(n) + 1j*np.random.randn(n)).astype(np.complex64)
        surv = np.roll(ref, 100) # Delay 100

        out = np.zeros(n, dtype=np.complex64)

        # print("DEBUG: Calling Process...", flush=True)
        lib.caf_process(
            obj,
            ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        # print("DEBUG: Process Returned", flush=True)

        mag = np.abs(out)
        peak_idx = np.argmax(mag)

        # print(f"CAF Peak at: {peak_idx}, Max Val: {mag[peak_idx]}", flush=True)

        self.assertEqual(peak_idx, 100)

        lib.caf_destroy(obj)

if __name__ == '__main__':
    unittest.main()
