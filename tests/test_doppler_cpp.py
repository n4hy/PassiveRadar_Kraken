
import sys
import os
import unittest
import numpy as np
import ctypes
from pathlib import Path

# Test the Doppler Processor logic specifically
# We will compile/load the library manually or reuse the logic in top block
# To avoid complex deps, we load the .so directly

class TestDopplerCpp(unittest.TestCase):
    def setUp(self):
        # Locate library
        repo_root = Path(__file__).resolve().parents[1]

        candidates = [
            repo_root / "src" / "libkraken_doppler_processing.so",
            repo_root / "gr-kraken_passive_radar" / "python" / "kraken_passive_radar" / "libkraken_doppler_processing.so",
            Path("libkraken_doppler_processing.so")
        ]

        self.lib_path = None
        for p in candidates:
            if p.exists():
                self.lib_path = p
                break

        if not self.lib_path:
             print(f"DEBUG: Searched for libraries in: {[str(c) for c in candidates]}")
             print("DEBUG: Hint: Ensure 'libfftw3-dev' is installed and run './build_oot.sh'")
             self.skipTest("C++ Library not found (compilation likely failed due to missing FFTW)")

        try:
            self.lib = ctypes.CDLL(str(self.lib_path))
        except OSError:
            self.skipTest("Could not load C++ Library (missing dependencies?)")

        # doppler_create(int fft_len, int doppler_len) -> void*
        self.lib.doppler_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.doppler_create.restype = ctypes.c_void_p

        # doppler_destroy(void*)
        self.lib.doppler_destroy.argtypes = [ctypes.c_void_p]
        self.lib.doppler_destroy.restype = None

        # doppler_process(void*, float* in, float* out)
        self.lib.doppler_process.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.POINTER(ctypes.c_float)]
        self.lib.doppler_process.restype = None

    def test_doppler_processing(self):
        fft_len = 16
        doppler_len = 16 # small power of 2

        obj = self.lib.doppler_create(fft_len, doppler_len)
        self.assertIsNotNone(obj)

        # Create synthetic data
        # Matrix (doppler_len, fft_len)
        # Let's put a DC signal in one Range bin (column)
        # In column 5, put a constant value 1.0
        # In column 8, put an alternating value 1, -1, 1, -1 (Nyquist doppler)

        input_mat = np.zeros((doppler_len, fft_len), dtype=np.complex64)

        for i in range(doppler_len):
            input_mat[i, 5] = 1.0 + 0j
            input_mat[i, 8] = (1.0 if i % 2 == 0 else -1.0) + 0j

        # Flatten
        # Note: Input layout expected by C++ is Row Major (flat array)
        input_flat = input_mat.flatten() # defaults to 'C' (row-major)

        # Output buffer
        output_flat = np.zeros(doppler_len * fft_len, dtype=np.float32)

        p_in = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_out = output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.doppler_process(obj, p_in, p_out)

        # Analyze output
        # Output layout: Row Major.
        # Column 5 should have peak at DC (Doppler index 0 or doppler_len/2 depending on shift)
        # My C++ code does fftshift.
        # DC (freq 0) is usually at index 0 before shift.
        # After shift, DC is at index N/2.

        # Let's check column 5
        col5 = []
        for d in range(doppler_len):
            col5.append(output_flat[d * fft_len + 5])

        peak_idx = np.argmax(col5)
        # Center index is 8
        # print(f"DC Signal Peak Index: {peak_idx} (Expected ~8)")

        # Check column 8 (Nyquist)
        # Frequency is Fs/2.
        # Before shift: Index N/2.
        # After shift: Index 0 (or N-1).
        col8 = []
        for d in range(doppler_len):
            col8.append(output_flat[d * fft_len + 8])

        peak_idx_nyq = np.argmax(col8)
        # print(f"Nyquist Signal Peak Index: {peak_idx_nyq} (Expected ~0 or 15)")

        self.assertEqual(peak_idx, doppler_len // 2, "DC signal should be centered after fftshift")

        # Nyquist might be at 0 because fftshift moves [N/2] to [0]
        self.assertTrue(peak_idx_nyq == 0 or peak_idx_nyq == doppler_len - 1, "Nyquist signal should be at edge after fftshift")

        self.lib.doppler_destroy(obj)

if __name__ == "__main__":
    unittest.main()
