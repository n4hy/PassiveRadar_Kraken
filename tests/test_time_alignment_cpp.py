import os
import sys
import unittest
import numpy as np
import ctypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from conftest import find_kernel_lib


class TestTimeAlignmentCpp(unittest.TestCase):
    def test_detects_known_delay(self):
        lib_path = find_kernel_lib("time_alignment")

        if not lib_path.exists():
            self.skipTest(f"Time alignment library not found at {lib_path}")

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
