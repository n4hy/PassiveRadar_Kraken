import sys
import os
import unittest
import numpy as np
import ctypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from conftest import find_kernel_lib


class TestAoACpp(unittest.TestCase):
    def setUp(self):
        # Locate libraries using centralized find_kernel_lib
        lib_doppler_path = find_kernel_lib("doppler_processing")
        lib_aoa_path = find_kernel_lib("aoa_processing")

        if not lib_doppler_path.exists() or not lib_aoa_path.exists():
            self.skipTest("C++ Libraries not found (compilation likely failed due to missing FFTW)")

        try:
            self.doppler_lib = ctypes.CDLL(str(lib_doppler_path))
            self.aoa_lib = ctypes.CDLL(str(lib_aoa_path))
        except OSError:
            self.skipTest("Could not load C++ Libraries")

        # Doppler signatures
        self.doppler_lib.doppler_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self.doppler_lib.doppler_create.restype = ctypes.c_void_p

        self.doppler_lib.doppler_destroy.argtypes = [ctypes.c_void_p]
        self.doppler_lib.doppler_destroy.restype = None

        self.doppler_lib.doppler_process_complex.argtypes = [ctypes.c_void_p,
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float)]
        self.doppler_lib.doppler_process_complex.restype = None

        # AoA signatures (3 args: n_ant, spacing, array_type)
        self.aoa_lib.aoa_create.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int]
        self.aoa_lib.aoa_create.restype = ctypes.c_void_p

        self.aoa_lib.aoa_destroy.argtypes = [ctypes.c_void_p]
        self.aoa_lib.aoa_destroy.restype = None

        self.aoa_lib.aoa_process.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_float,
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_int]
        self.aoa_lib.aoa_process.restype = None

    def test_aoa_estimation(self):
        # 1. Setup Parameters
        fft_len = 16
        doppler_len = 16
        n_antennas = 5  # KrakenSDR: 5 channels (Ch0=ref, Ch1-4=surv)
        spacing = 0.15 # meters (approx half wave for 1GHz)
        freq_hz = 1e9
        lambda_val = 299792458.0 / freq_hz # ~0.3m

        # 2. Simulate Signal
        # Source at angle theta = 30 degrees
        target_angle_deg = 30.0
        target_angle_rad = np.radians(target_angle_deg)
        k = 2 * np.pi / lambda_val

        # aoa_process() legacy uses Ch1..4 as ULA (start_idx=1, count=0..3)
        # So we need 5 elements where Ch0=ref (unused) and Ch1-4 are the ULA
        # Generate 5 antenna inputs through Doppler processing
        doppler_procs = []
        for _ in range(n_antennas):
            doppler_procs.append(self.doppler_lib.doppler_create(fft_len, doppler_len))

        rd_maps = []

        for m in range(n_antennas):
            if m == 0:
                # Ch0 (ref): no particular phase needed for legacy aoa_process
                phase_shift = 0.0
            else:
                # Ch1..4: ULA with indices 0,1,2,3
                # Plane wave from angle theta: positive phase progression
                phase_shift = k * (m - 1) * spacing * np.sin(target_angle_rad)
            complex_factor = np.exp(1j * phase_shift)

            input_mat = np.zeros((doppler_len, fft_len), dtype=np.complex64)
            for i in range(doppler_len):
                input_mat[i, 5] = complex_factor

            p_in = input_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_mat = np.zeros((doppler_len, fft_len), dtype=np.complex64)
            p_out = out_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            self.doppler_lib.doppler_process_complex(doppler_procs[m], p_in, p_out)
            rd_maps.append(out_mat)

        # 3. Extract Snapshot for AoA (all 5 channels)
        snapshot = np.zeros(n_antennas, dtype=np.complex64)
        for m in range(n_antennas):
            snapshot[m] = rd_maps[m][8, 5]

        # 4. Run AoA
        aoa_obj = self.aoa_lib.aoa_create(n_antennas, ctypes.c_float(spacing), 0)  # 0 = ULA
        n_angles = 181
        spectrum = np.zeros(n_angles, dtype=np.float32)

        p_snap = snapshot.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_spec = spectrum.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.aoa_lib.aoa_process(aoa_obj, p_snap, ctypes.c_float(lambda_val), p_spec, n_angles)

        # 5. Find Peak
        peak_idx = np.argmax(spectrum)
        est_angle = -90.0 + peak_idx # since steps are 1 degree

        print(f"Target Angle: {target_angle_deg}, Estimated: {est_angle}")

        # Allow error due to resolution (1 deg) and discrete sampling
        self.assertTrue(abs(est_angle - target_angle_deg) <= 2.0,
                        f"AoA estimate {est_angle} too far from {target_angle_deg}")

        # Cleanup
        for p in doppler_procs:
            self.doppler_lib.doppler_destroy(p)
        self.aoa_lib.aoa_destroy(aoa_obj)

class TestAoAMusicCpp(unittest.TestCase):
    def setUp(self):
        lib_aoa_path = find_kernel_lib("aoa_processing")

        if not lib_aoa_path.exists():
            self.skipTest("C++ AoA library not found")

        try:
            self.aoa_lib = ctypes.CDLL(str(lib_aoa_path))
        except OSError:
            self.skipTest("Could not load C++ AoA library")

        # aoa_process_music signature
        self.aoa_lib.aoa_process_music.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # snapshots (interleaved complex)
            ctypes.c_int,                     # n_ant
            ctypes.c_int,                     # n_snapshots
            ctypes.c_int,                     # n_sources
            ctypes.c_float,                   # d_spacing
            ctypes.c_float,                   # lambda
            ctypes.POINTER(ctypes.c_float),   # output
            ctypes.c_int                      # n_angles
        ]
        self.aoa_lib.aoa_process_music.restype = None

    def test_aoa_music_estimation(self):
        """Test MUSIC AoA estimation via ctypes with known angle + noise."""
        np.random.seed(42)
        n_ant = 4
        n_snapshots = 50
        n_sources = 1
        freq_hz = 1e9
        lambda_val = 299792458.0 / freq_hz
        d_spacing = 0.5 * lambda_val
        target_angle_deg = 25.0
        target_angle_rad = np.radians(target_angle_deg)

        k = 2 * np.pi / lambda_val

        # Generate multi-snapshot data: n_snapshots x n_ant complex
        # Each snapshot: steering vector * random phase + noise
        snapshots = np.zeros((n_snapshots, n_ant), dtype=np.complex64)
        sv = np.exp(-1j * k * d_spacing * np.arange(n_ant) * np.sin(target_angle_rad))

        for i in range(n_snapshots):
            phase = np.random.uniform(0, 2 * np.pi)
            snapshots[i, :] = sv * np.exp(1j * phase)
            snapshots[i, :] += 0.1 * (np.random.randn(n_ant) + 1j * np.random.randn(n_ant))

        n_angles = 181
        output = np.zeros(n_angles, dtype=np.float32)

        p_snap = snapshots.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_out = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.aoa_lib.aoa_process_music(
            p_snap, n_ant, n_snapshots, n_sources,
            ctypes.c_float(d_spacing), ctypes.c_float(lambda_val),
            p_out, n_angles
        )

        # Find peak
        peak_idx = np.argmax(output)
        est_angle = -90.0 + peak_idx * (180.0 / (n_angles - 1))

        print(f"MUSIC Target Angle: {target_angle_deg}, Estimated: {est_angle:.1f}")
        self.assertTrue(abs(est_angle - target_angle_deg) <= 3.0,
                        f"MUSIC AoA estimate {est_angle:.1f} too far from {target_angle_deg}")


if __name__ == "__main__":
    unittest.main()
