
import sys
import os
import unittest
import numpy as np
import ctypes

class TestAoACpp(unittest.TestCase):
    def setUp(self):
        # Locate libraries (names match CMake targets: kraken_doppler_processing, kraken_aoa_processing)
        lib_doppler_path = os.path.abspath("src/libkraken_doppler_processing.so")
        lib_aoa_path = os.path.abspath("src/libkraken_aoa_processing.so")

        if not os.path.exists(lib_doppler_path) or not os.path.exists(lib_aoa_path):
             self.skipTest("C++ Libraries not found (compilation likely failed due to missing FFTW)")

        try:
            self.doppler_lib = ctypes.CDLL(lib_doppler_path)
            self.aoa_lib = ctypes.CDLL(lib_aoa_path)
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

if __name__ == "__main__":
    unittest.main()
