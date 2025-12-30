import os
import sys
import unittest
import numpy as np
import ctypes
import subprocess
from pathlib import Path


class TestEcaBCpp(unittest.TestCase):
    def test_eca_b_reduces_clutter_power(self):
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        lib_path = src_dir / "libkraken_eca_b_clutter_canceller.so"

        # Build the shared library if it does not exist
        if not lib_path.exists():
            cpp_file = src_dir / "eca_b_clutter_canceller.cpp"
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

        lib.eca_b_create.restype = ctypes.c_void_p
        lib.eca_b_create.argtypes = [ctypes.c_int]

        lib.eca_b_destroy.restype = None
        lib.eca_b_destroy.argtypes = [ctypes.c_void_p]

        lib.eca_b_process.restype = None
        lib.eca_b_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        num_taps = 8
        n_samples = 4096
        np.random.seed(123)

        # Generate reference signal x[n]
        x_long = (np.random.randn(n_samples + num_taps - 1) +
                  1j * np.random.randn(n_samples + num_taps - 1)).astype(np.complex64)

        # True clutter filter h (unknown to ECA-B)
        h = (np.random.randn(num_taps) +
             1j * np.random.randn(num_taps)).astype(np.complex64)

        # Generate surveillance d[n] = x * h (valid convolution) + small noise
        d_valid = np.convolve(x_long, h, mode="valid").astype(np.complex64)
        d = d_valid[:n_samples]
        d += 0.01 * (np.random.randn(n_samples) +
                     1j * np.random.randn(n_samples)).astype(np.complex64)

        # Align reference with the last n_samples of x_long
        x = x_long[num_taps - 1:num_taps - 1 + n_samples]

        # Prepare buffers
        x_c = np.ascontiguousarray(x, dtype=np.complex64)
        d_c = np.ascontiguousarray(d, dtype=np.complex64)
        e_c = np.zeros_like(d_c)

        x_f = x_c.view(np.float32)
        d_f = d_c.view(np.float32)
        e_f = e_c.view(np.float32)

        state = lib.eca_b_create(num_taps)
        self.assertIsNotNone(state)

        lib.eca_b_process(
            state,
            x_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            e_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_samples,
        )

        lib.eca_b_destroy(state)

        power_in = np.mean(np.abs(d_c) ** 2)
        power_out = np.mean(np.abs(e_c) ** 2)
        reduction_db = 10 * np.log10(power_in / (power_out + 1e-12))

        self.assertGreater(reduction_db, 10.0, msg=f"Expected >10 dB clutter reduction, got {reduction_db:.2f} dB")


if __name__ == "__main__":
    unittest.main()
