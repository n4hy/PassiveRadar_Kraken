
import sys
import os
import unittest
import numpy as np
import ctypes

class TestAoACpp(unittest.TestCase):
    def setUp(self):
        # Locate libraries
        self.doppler_lib = ctypes.CDLL(os.path.abspath("src/libdoppler_processing.so"))
        self.aoa_lib = ctypes.CDLL(os.path.abspath("src/libaoa_processing.so"))

        # Doppler signatures
        self.doppler_lib.doppler_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self.doppler_lib.doppler_create.restype = ctypes.c_void_p

        self.doppler_lib.doppler_destroy.argtypes = [ctypes.c_void_p]
        self.doppler_lib.doppler_destroy.restype = None

        self.doppler_lib.doppler_process_complex.argtypes = [ctypes.c_void_p,
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float)]
        self.doppler_lib.doppler_process_complex.restype = None

        # AoA signatures
        self.aoa_lib.aoa_create.argtypes = [ctypes.c_int, ctypes.c_float]
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
        n_antennas = 4
        spacing = 0.15 # meters (approx half wave for 1GHz)
        freq_hz = 1e9
        lambda_val = 299792458.0 / freq_hz # ~0.3m
        # spacing is 0.15m which is lambda/2.

        # 2. Simulate Signal
        # Source at angle theta = 30 degrees
        target_angle_deg = 30.0
        target_angle_rad = np.radians(target_angle_deg)
        k = 2 * np.pi / lambda_val

        # Generate 4 antenna inputs (Matrix: n_ant x (doppler_len * fft_len))
        # But wait, DopplerProcessor takes 1 channel.
        # So we process 4 channels independently to get 4 RD maps.

        # Create 4 Doppler processors
        doppler_procs = []
        for _ in range(n_antennas):
            doppler_procs.append(self.doppler_lib.doppler_create(fft_len, doppler_len))

        # Target is at Range Bin 5, Doppler Bin 8 (DC)
        rd_maps = []

        for m in range(n_antennas):
            # Phase shift for antenna m
            # phase = -k * m * d * sin(theta)
            phase_shift = -k * m * spacing * np.sin(target_angle_rad)
            complex_factor = np.exp(1j * phase_shift)

            # Create input buffer for this antenna
            input_mat = np.zeros((doppler_len, fft_len), dtype=np.complex64)
            # Add signal at bin (8, 5) with correct phase
            for i in range(doppler_len):
                input_mat[i, 5] = complex_factor # Constant (DC doppler)

            # Process
            p_in = input_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_mat = np.zeros((doppler_len, fft_len), dtype=np.complex64)
            p_out = out_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            self.doppler_lib.doppler_process_complex(doppler_procs[m], p_in, p_out)
            rd_maps.append(out_mat)

        # 3. Extract Snapshot for AoA
        # We look at the target bin (Doppler 8, Range 5) across all 4 maps
        # Note: Doppler 8 is the center index (DC) after fftshift if N=16?
        # My FFT shift swaps halves. 0->8.
        # So DC (idx 0) moves to idx 8.

        snapshot = np.zeros(n_antennas, dtype=np.complex64)
        for m in range(n_antennas):
            snapshot[m] = rd_maps[m][8, 5]

        # 4. Run AoA
        aoa_obj = self.aoa_lib.aoa_create(n_antennas, ctypes.c_float(spacing))
        n_angles = 181
        spectrum = np.zeros(n_angles, dtype=np.float32)

        p_snap = snapshot.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_spec = spectrum.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.aoa_lib.aoa_process(aoa_obj, p_snap, ctypes.c_float(lambda_val), p_spec, n_angles)

        # 5. Find Peak
        peak_idx = np.argmax(spectrum)
        est_angle = -90.0 + peak_idx # since steps are 1 degree

        print(f"Target Angle: {target_angle_deg}, Estimated: {est_angle}")

        # Allow small error due to resolution (1 deg)
        self.assertTrue(abs(est_angle - target_angle_deg) <= 1.0)

        # Cleanup
        for p in doppler_procs:
            self.doppler_lib.doppler_destroy(p)
        self.aoa_lib.aoa_destroy(aoa_obj)

if __name__ == "__main__":
    unittest.main()
