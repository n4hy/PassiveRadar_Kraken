import os
import sys
import unittest
import numpy as np
import ctypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from conftest import find_kernel_lib


class TestConditioningCpp(unittest.TestCase):
    def test_agc_normalizes_level(self):
        lib_path = find_kernel_lib("conditioning")

        if not lib_path.exists():
            self.skipTest(f"Conditioning library not found at {lib_path}")

        lib = ctypes.cdll.LoadLibrary(str(lib_path))

        lib.cond_create.restype = ctypes.c_void_p
        lib.cond_create.argtypes = [ctypes.c_float]

        lib.cond_destroy.restype = None
        lib.cond_destroy.argtypes = [ctypes.c_void_p]

        lib.cond_process.restype = None
        lib.cond_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        # Create test signal with varying amplitude
        n_samples = 4096
        np.random.seed(42)

        # Generate complex signal with low amplitude (needs gain)
        amplitude = 0.1
        signal = amplitude * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        initial_power = np.mean(np.abs(signal) ** 2)

        # Process with AGC
        signal_c = np.ascontiguousarray(signal, dtype=np.complex64)
        signal_f = signal_c.view(np.float32)

        # Higher rate for faster convergence in test
        rate = 1e-3
        state = lib.cond_create(ctypes.c_float(rate))
        self.assertIsNotNone(state)

        lib.cond_process(
            state,
            signal_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_samples,
        )

        lib.cond_destroy(state)

        # Check that output power increased (AGC applied gain)
        final_power = np.mean(np.abs(signal_c) ** 2)

        self.assertGreater(
            final_power,
            initial_power,
            msg=f"AGC should increase power from {initial_power:.6f} to target ~1.0"
        )


if __name__ == "__main__":
    unittest.main()
