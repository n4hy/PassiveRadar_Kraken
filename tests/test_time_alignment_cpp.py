import os
import sys
import unittest
import numpy as np
import ctypes
import subprocess
from pathlib import Path


class TestTimeAlignmentCpp(unittest.TestCase):
    def test_detects_known_delay(self):
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        lib_path = src_dir / "libkraken_time_alignment.so"

        # Build the shared library if it does not exist
        if not lib_path.exists():
            cpp_file = src_dir / "time_alignment.cpp"
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
                "-lfftw3f",
                "-lfftw3f_threads",
                "-lpthread",
            ]
            subprocess.check_call(cmd, cwd=str(src_dir))

        lib = ctypes.cdll.LoadLibrary(str(lib_path))

        lib.align_create.restype = ctypes.c_void_p
        lib.align_create.argtypes = [ctypes.c_int]

        lib.align_destroy.restype = None
        lib.align_destroy.argtypes = [ctypes.c_void_p]

        lib.align_compute.restype = None
        lib.align_compute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),  # ref
            ctypes.POINTER(ctypes.c_float),  # surv
            ctypes.POINTER(ctypes.c_int),    # delay
            ctypes.POINTER(ctypes.c_float),  # phase
        ]

        # Create test signals with known delay
        n_samples = 4096
        known_delay = 17  # samples
        known_phase = np.pi / 4  # 45 degrees

        np.random.seed(42)

        # Generate reference signal (FM-like)
        t = np.arange(n_samples)
        modulation = np.sin(2 * np.pi * 0.01 * t)
        ref = np.exp(1j * np.cumsum(modulation)).astype(np.complex64)

        # Create surveillance signal with delay and phase shift
        surv = np.roll(ref, known_delay) * np.exp(1j * known_phase)
        surv = surv.astype(np.complex64)

        # Prepare buffers
        ref_c = np.ascontiguousarray(ref, dtype=np.complex64)
        surv_c = np.ascontiguousarray(surv, dtype=np.complex64)
        ref_f = ref_c.view(np.float32)
        surv_f = surv_c.view(np.float32)

        state = lib.align_create(n_samples)
        self.assertIsNotNone(state)

        delay_out = ctypes.c_int(0)
        phase_out = ctypes.c_float(0.0)

        lib.align_compute(
            state,
            ref_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(delay_out),
            ctypes.byref(phase_out),
        )

        lib.align_destroy(state)

        estimated_delay = delay_out.value
        estimated_phase = phase_out.value

        # Check delay is accurate
        self.assertEqual(
            estimated_delay,
            known_delay,
            msg=f"Expected delay {known_delay}, got {estimated_delay}"
        )

        # Check phase is approximately correct (within 0.1 rad)
        phase_error = abs(estimated_phase - known_phase)
        # Handle phase wrapping
        if phase_error > np.pi:
            phase_error = 2 * np.pi - phase_error

        self.assertLess(
            phase_error,
            0.1,
            msg=f"Expected phase ~{known_phase:.3f}, got {estimated_phase:.3f}"
        )


if __name__ == "__main__":
    unittest.main()
