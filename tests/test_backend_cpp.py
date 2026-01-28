import os
import sys
import unittest
import numpy as np
import ctypes
import subprocess
from pathlib import Path


class TestBackendCpp(unittest.TestCase):
    def test_cfar_detects_target(self):
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        lib_path = src_dir / "libkraken_backend.so"

        # Build the shared library if it does not exist
        if not lib_path.exists():
            cpp_file = src_dir / "backend.cpp"
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

        lib.cfar_2d.restype = None
        lib.cfar_2d.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,                     # rows
            ctypes.c_int,                     # cols
            ctypes.c_int,                     # guard
            ctypes.c_int,                     # train
            ctypes.c_float,                   # threshold
        ]

        # Create test Range-Doppler map with a target
        rows = 64
        cols = 64
        guard = 2
        train = 4
        threshold_db = 10.0

        # Background noise floor at 0 dB
        np.random.seed(42)
        input_map = np.random.randn(rows, cols).astype(np.float32)

        # Insert a strong target at center
        target_row, target_col = rows // 2, cols // 2
        input_map[target_row, target_col] = 25.0  # 25 dB above noise

        output_map = np.zeros_like(input_map)

        input_c = np.ascontiguousarray(input_map, dtype=np.float32)
        output_c = np.ascontiguousarray(output_map, dtype=np.float32)

        lib.cfar_2d(
            input_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            cols,
            guard,
            train,
            threshold_db,
        )

        # Check that target was detected
        self.assertEqual(
            output_c[target_row, target_col],
            1.0,
            msg="CFAR should detect the target"
        )

        # Check that most background is not detected
        total_detections = np.sum(output_c)
        self.assertLess(
            total_detections,
            10,
            msg=f"Expected few false alarms, got {total_detections} detections"
        )


if __name__ == "__main__":
    unittest.main()
