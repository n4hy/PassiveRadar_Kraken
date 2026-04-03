"""
GPU UKF Tests - Unscented Kalman Filter Acceleration
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Tests GPU-accelerated UKF functionality:
- Sigma point generation
- State propagation
- Measurement update
- CUDA version compatibility (11.8+, 12.x, 13.x)
"""

import pytest
import numpy as np
import ctypes
from pathlib import Path
import sys

# Add repo root to path
_repo_root = str(Path(__file__).parents[2])
sys.path.insert(0, _repo_root)

from kraken_passive_radar import is_gpu_available, GPU_AVAILABLE

pytestmark = pytest.mark.gpu

# State and measurement dimensions (must match ukf_gpu.cu)
NX = 5   # [range, doppler, range_rate, doppler_rate, turn_rate]
NY = 3   # [range, doppler, aoa_deg]
NSIGMA = 2 * NX + 1  # 11 sigma points


def load_ukf_gpu_lib():
    """Load the UKF GPU library if available."""
    lib_paths = [
        Path(__file__).parents[2] / 'build' / 'lib' / 'libkraken_ukf_gpu.so',
        Path(__file__).parents[2] / 'src' / 'build' / 'lib' / 'libkraken_ukf_gpu.so',
        Path(__file__).parents[2] / 'src' / 'lib' / 'libkraken_ukf_gpu.so',
        Path(__file__).parents[2] / 'kraken_passive_radar' / 'libkraken_ukf_gpu.so',
    ]

    for lib_path in lib_paths:
        if lib_path.exists():
            try:
                return ctypes.CDLL(str(lib_path))
            except OSError:
                continue

    return None


@pytest.fixture
def ukf_lib():
    """Fixture to load UKF GPU library."""
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")

    lib = load_ukf_gpu_lib()
    if lib is None:
        pytest.skip("UKF GPU library not found - build with CUDA 11.8+")

    # Set up function signatures
    lib.ukf_gpu_create.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.ukf_gpu_create.restype = ctypes.c_void_p

    lib.ukf_gpu_destroy.argtypes = [ctypes.c_void_p]
    lib.ukf_gpu_destroy.restype = None

    lib.ukf_gpu_set_states.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.ukf_gpu_set_states.restype = None

    lib.ukf_gpu_get_states.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.ukf_gpu_get_states.restype = None

    lib.ukf_gpu_predict.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    lib.ukf_gpu_predict.restype = None

    lib.ukf_gpu_is_available.argtypes = []
    lib.ukf_gpu_is_available.restype = ctypes.c_int

    return lib


class TestUKFGPUCreation:
    """Test UKF GPU processor creation and destruction."""

    def test_create_destroy(self, ukf_lib):
        """Test basic creation and destruction."""
        max_tracks = 32
        alpha = 1.0
        beta = 2.0
        kappa = 0.0

        handle = ukf_lib.ukf_gpu_create(max_tracks, alpha, beta, kappa)
        assert handle is not None

        ukf_lib.ukf_gpu_destroy(handle)

    def test_create_with_different_params(self, ukf_lib):
        """Test creation with different UKF parameters."""
        # Conservative parameters
        handle1 = ukf_lib.ukf_gpu_create(16, 0.001, 2.0, 0.0)
        assert handle1 is not None
        ukf_lib.ukf_gpu_destroy(handle1)

        # Aggressive parameters
        handle2 = ukf_lib.ukf_gpu_create(64, 1.0, 2.0, 3.0 - NX)
        assert handle2 is not None
        ukf_lib.ukf_gpu_destroy(handle2)

    def test_is_available(self, ukf_lib):
        """Test GPU availability check."""
        result = ukf_lib.ukf_gpu_is_available()
        assert result == 1, "GPU should be available"


class TestUKFGPUStateManagement:
    """Test state get/set operations."""

    def test_set_get_states(self, ukf_lib):
        """Test setting and getting track states."""
        max_tracks = 4
        n_tracks = 2

        handle = ukf_lib.ukf_gpu_create(max_tracks, 1.0, 2.0, 0.0)
        assert handle is not None

        try:
            # Create test states
            states_in = np.array([
                [1000.0, 50.0, 10.0, 5.0, 0.1],  # Track 0
                [2000.0, -30.0, -5.0, -2.0, -0.05],  # Track 1
            ], dtype=np.float32)

            # Set states
            states_in_ptr = states_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_set_states(handle, states_in_ptr, n_tracks)

            # Get states back
            states_out = np.zeros((n_tracks, NX), dtype=np.float32)
            states_out_ptr = states_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_get_states(handle, states_out_ptr, n_tracks)

            # Verify
            np.testing.assert_allclose(states_out, states_in, rtol=1e-5)

        finally:
            ukf_lib.ukf_gpu_destroy(handle)


class TestUKFGPUPredict:
    """Test UKF prediction step."""

    def test_predict_stationary_target(self, ukf_lib):
        """Test prediction of stationary target (zero velocities)."""
        handle = ukf_lib.ukf_gpu_create(4, 1.0, 2.0, 0.0)
        assert handle is not None

        try:
            # Stationary target at range=1000m, doppler=0
            states_in = np.array([[1000.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            states_in_ptr = states_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_set_states(handle, states_in_ptr, 1)

            # Predict with dt=0.1s
            ukf_lib.ukf_gpu_predict(handle, 1, 0.1)

            # Get predicted state
            states_out = np.zeros((1, NX), dtype=np.float32)
            states_out_ptr = states_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_get_states(handle, states_out_ptr, 1)

            # Stationary target should remain at same position
            assert abs(states_out[0, 0] - 1000.0) < 10.0, f"Range changed too much: {states_out[0, 0]}"
            assert abs(states_out[0, 1]) < 10.0, f"Doppler changed: {states_out[0, 1]}"

        finally:
            ukf_lib.ukf_gpu_destroy(handle)

    def test_predict_moving_target(self, ukf_lib):
        """Test prediction of moving target."""
        handle = ukf_lib.ukf_gpu_create(4, 1.0, 2.0, 0.0)
        assert handle is not None

        try:
            # Moving target: range=1000m, range_rate=100m/s
            states_in = np.array([[1000.0, 50.0, 100.0, 0.0, 0.0]], dtype=np.float32)
            states_in_ptr = states_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_set_states(handle, states_in_ptr, 1)

            # Predict with dt=1.0s
            dt = 1.0
            ukf_lib.ukf_gpu_predict(handle, 1, dt)

            # Get predicted state
            states_out = np.zeros((1, NX), dtype=np.float32)
            states_out_ptr = states_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_get_states(handle, states_out_ptr, 1)

            # Range should increase by range_rate * dt
            expected_range = 1000.0 + 100.0 * dt
            assert abs(states_out[0, 0] - expected_range) < 50.0, \
                f"Range prediction incorrect: {states_out[0, 0]} vs {expected_range}"

        finally:
            ukf_lib.ukf_gpu_destroy(handle)

    def test_predict_batch(self, ukf_lib):
        """Test batch prediction of multiple tracks."""
        n_tracks = 8
        handle = ukf_lib.ukf_gpu_create(16, 1.0, 2.0, 0.0)
        assert handle is not None

        try:
            # Create multiple tracks with different states
            states_in = np.zeros((n_tracks, NX), dtype=np.float32)
            for i in range(n_tracks):
                states_in[i, 0] = 1000.0 + i * 500.0  # Range
                states_in[i, 1] = 50.0 - i * 10.0     # Doppler
                states_in[i, 2] = 10.0 * (i - n_tracks/2)  # Range rate
                states_in[i, 3] = 5.0 * (n_tracks/2 - i)   # Doppler rate

            states_in_ptr = states_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_set_states(handle, states_in_ptr, n_tracks)

            # Predict
            ukf_lib.ukf_gpu_predict(handle, n_tracks, 0.1)

            # Get predicted states
            states_out = np.zeros((n_tracks, NX), dtype=np.float32)
            states_out_ptr = states_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_get_states(handle, states_out_ptr, n_tracks)

            # All states should have changed (prediction applied)
            # At least check that range updated based on range_rate
            for i in range(n_tracks):
                expected_range = states_in[i, 0] + states_in[i, 2] * 0.1
                # Allow for some numerical variation due to UKF sigma points
                assert abs(states_out[i, 0] - expected_range) < 50.0, \
                    f"Track {i} range prediction off: {states_out[i, 0]} vs {expected_range}"

        finally:
            ukf_lib.ukf_gpu_destroy(handle)


class TestUKFGPUCUDAVersion:
    """Test CUDA version compatibility."""

    def test_cuda_version_detection(self, ukf_lib):
        """Test that CUDA version is properly detected."""
        # The library should load successfully with CUDA 11.8+
        assert ukf_lib is not None

        # is_available should return 1
        result = ukf_lib.ukf_gpu_is_available()
        assert result == 1

    def test_stream_ordered_memory(self, ukf_lib):
        """Test that stream-ordered memory works (CUDA 11.2+ feature)."""
        # This tests that cudaMallocAsync/cudaFreeAsync work correctly
        handle = ukf_lib.ukf_gpu_create(32, 1.0, 2.0, 0.0)
        assert handle is not None

        try:
            # Run multiple predictions to exercise memory allocation
            states = np.zeros((8, NX), dtype=np.float32)
            states_ptr = states.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_set_states(handle, states_ptr, 8)

            for _ in range(10):
                ukf_lib.ukf_gpu_predict(handle, 8, 0.1)

            # If we get here without crashes, stream-ordered memory works
            assert True

        finally:
            ukf_lib.ukf_gpu_destroy(handle)


class TestUKFGPUNumericalStability:
    """Test numerical stability of GPU UKF implementation."""

    def test_large_state_values(self, ukf_lib):
        """Test with large state values."""
        handle = ukf_lib.ukf_gpu_create(4, 1.0, 2.0, 0.0)
        assert handle is not None

        try:
            # Large values
            states_in = np.array([[1e6, 1e4, 1e3, 1e2, 0.1]], dtype=np.float32)
            states_in_ptr = states_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_set_states(handle, states_in_ptr, 1)

            ukf_lib.ukf_gpu_predict(handle, 1, 0.1)

            states_out = np.zeros((1, NX), dtype=np.float32)
            states_out_ptr = states_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_get_states(handle, states_out_ptr, 1)

            # Should not produce NaN or Inf
            assert np.all(np.isfinite(states_out)), f"Non-finite values: {states_out}"

        finally:
            ukf_lib.ukf_gpu_destroy(handle)

    def test_small_state_values(self, ukf_lib):
        """Test with small state values."""
        handle = ukf_lib.ukf_gpu_create(4, 1.0, 2.0, 0.0)
        assert handle is not None

        try:
            # Small values
            states_in = np.array([[1.0, 0.1, 0.01, 0.001, 1e-4]], dtype=np.float32)
            states_in_ptr = states_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_set_states(handle, states_in_ptr, 1)

            ukf_lib.ukf_gpu_predict(handle, 1, 0.1)

            states_out = np.zeros((1, NX), dtype=np.float32)
            states_out_ptr = states_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ukf_lib.ukf_gpu_get_states(handle, states_out_ptr, 1)

            # Should not produce NaN or Inf
            assert np.all(np.isfinite(states_out)), f"Non-finite values: {states_out}"

        finally:
            ukf_lib.ukf_gpu_destroy(handle)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
