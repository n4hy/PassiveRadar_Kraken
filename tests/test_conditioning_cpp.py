import os
import sys
import unittest
import numpy as np
import ctypes
import subprocess
from pathlib import Path


class TestConditioningCpp(unittest.TestCase):
    def test_agc_normalizes_level(self):
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        lib_path = src_dir / "libkraken_conditioning.so"

        # Build the shared library if it does not exist
        if not lib_path.exists():
            cpp_file = src_dir / "conditioning.cpp"
            cmd = [
                "g++",
                "-O3",
                "-march=native",
                "-ffast-math",
                "-fPIC",
                "-shared",
                str(cpp_file),
                "-o",
                str(lib_path),
            ]
            subprocess.check_call(cmd, cwd=str(src_dir))

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
