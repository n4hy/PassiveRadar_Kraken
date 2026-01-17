import unittest
import numpy as np
import ctypes
import os
import sysconfig
from pathlib import Path

class TestCafCpp(unittest.TestCase):
    def test_caf_process(self):
        print("DEBUG: Loading Lib...", flush=True)
        # Robust path finding:
        # __file__ is tests/test_caf_cpp.py
        # parents[1] is repo root
        repo_root = Path(__file__).resolve().parents[1]

        # System install path
        site_packages = Path(sysconfig.get_paths()["purelib"])

        # Try finding the lib in multiple places
        candidates = [
            repo_root / "src" / "libkraken_caf_processing.so",
            repo_root / "gr-kraken_passive_radar" / "python" / "kraken_passive_radar" / "libkraken_caf_processing.so",
            site_packages / "kraken_passive_radar" / "libkraken_caf_processing.so",
            Path("libkraken_caf_processing.so") # Fallback to system path
        ]

        lib_path = None
        for p in candidates:
            if p.exists():
                lib_path = p
                break

        if not lib_path:
             print(f"DEBUG: Searched for libraries in: {[str(c) for c in candidates]}")
             print("DEBUG: Hint: Ensure 'libfftw3-dev' is installed and run './build_oot.sh'")
             # If we can't find it, try loading by name (maybe installed)
             lib_path = "libkraken_caf_processing.so"

        try:
            lib = ctypes.cdll.LoadLibrary(str(lib_path))
        except OSError as e:
            self.skipTest(f"Could not load C++ library: {e}. (Searched paths printed above)")
            return

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
