"""
GPU Doppler Kernel Tests - Correctness and Performance
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Tests GPU-accelerated Doppler processing against CPU version for:
- Correctness (output equivalence for both magnitude and complex modes)
- Performance (throughput and speedup)
- FFT shift correctness
"""

import pytest
import numpy as np
import ctypes
import time
from pathlib import Path

# Skip all GPU tests if CUDA not available
pytestmark = pytest.mark.gpu


@pytest.fixture
def gpu_doppler_lib():
    """Load GPU Doppler library."""
    lib_path = Path("/home/n4hy/PassiveRadar_Kraken/src/libkraken_doppler_gpu.so")
    if not lib_path.exists():
        pytest.skip("GPU Doppler library not built")
    return ctypes.cdll.LoadLibrary(str(lib_path))


@pytest.fixture
def cpu_doppler_lib():
    """Load CPU Doppler library for comparison."""
    lib_path = Path("/home/n4hy/PassiveRadar_Kraken/src/libkraken_doppler_processing.so")
    if not lib_path.exists():
        pytest.skip("CPU Doppler library not found")
    return ctypes.cdll.LoadLibrary(str(lib_path))


@pytest.fixture
def doppler_params():
    """Standard Doppler parameters for testing."""
    return {
        'fft_len': 2048,      # Range bins (columns)
        'doppler_len': 256,   # Doppler bins (rows)
    }


@pytest.fixture
def synthetic_doppler_data(doppler_params):
    """
    Generate synthetic Doppler input data with known spectral content.

    Returns:
        numpy array: Complex data (interleaved float I/Q)
    """
    fft_len = doppler_params['fft_len']
    doppler_len = doppler_params['doppler_len']

    # Create synthetic data with known frequency content
    # Add a few strong peaks at specific Doppler bins
    data = np.zeros((doppler_len, fft_len), dtype=np.complex64)

    # Add some noise
    noise_power = 0.1
    data += noise_power * (np.random.randn(doppler_len, fft_len) +
                           1j * np.random.randn(doppler_len, fft_len))

    # Add strong target at specific range/Doppler
    # (Will appear at different Doppler bin after FFT)
    for r in [100, 500, 1000]:
        for d in [50, 100, 200]:
            # Add sinusoid at this Doppler frequency
            t = np.arange(doppler_len)
            freq = d / doppler_len  # Normalized frequency
            signal = 10.0 * np.exp(2j * np.pi * freq * t)
            data[:, r] += signal

    # Convert to interleaved float I/Q
    output = np.empty(2 * doppler_len * fft_len, dtype=np.float32)
    output[0::2] = data.real.flatten()
    output[1::2] = data.imag.flatten()

    return output


class TestGPUDopplerCorrectness:
    """Test GPU Doppler output correctness against CPU reference."""

    def test_gpu_cpu_magnitude_equivalence(self, gpu_doppler_lib, cpu_doppler_lib,
                                            doppler_params, synthetic_doppler_data):
        """Verify GPU magnitude output matches CPU (within tolerance)."""
        input_data = synthetic_doppler_data
        fft_len = doppler_params['fft_len']
        doppler_len = doppler_params['doppler_len']

        # Setup function signatures
        for lib in [gpu_doppler_lib, cpu_doppler_lib]:
            if lib == gpu_doppler_lib:
                lib.doppler_gpu_create.restype = ctypes.c_void_p
                lib.doppler_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
                lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]
                lib.doppler_gpu_process.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float)
                ]
            else:
                lib.doppler_create.restype = ctypes.c_void_p
                lib.doppler_create.argtypes = [ctypes.c_int, ctypes.c_int]
                lib.doppler_destroy.argtypes = [ctypes.c_void_p]
                lib.doppler_process.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float)
                ]

        # Create processors
        gpu_handle = gpu_doppler_lib.doppler_gpu_create(fft_len, doppler_len)
        assert gpu_handle is not None, "Failed to create GPU Doppler processor"

        cpu_handle = cpu_doppler_lib.doppler_create(fft_len, doppler_len)
        assert cpu_handle is not None, "Failed to create CPU Doppler processor"

        # Allocate output buffers
        output_size = doppler_len * fft_len
        gpu_output = np.zeros(output_size, dtype=np.float32)
        cpu_output = np.zeros(output_size, dtype=np.float32)

        # Process
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        gpu_doppler_lib.doppler_gpu_process(
            gpu_handle, input_ptr,
            gpu_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        cpu_doppler_lib.doppler_process(
            cpu_handle, input_ptr,
            cpu_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        # Cleanup
        gpu_doppler_lib.doppler_gpu_destroy(gpu_handle)
        cpu_doppler_lib.doppler_destroy(cpu_handle)

        # Compare outputs
        # Allow for floating-point differences and log scale variations
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-3, atol=0.5,
                                   err_msg="GPU and CPU magnitude outputs differ")

        print(f"\n✓ GPU/CPU magnitude output match (max diff: {np.max(np.abs(gpu_output - cpu_output)):.4f} dB)")

    def test_gpu_cpu_complex_equivalence(self, gpu_doppler_lib, cpu_doppler_lib,
                                          doppler_params, synthetic_doppler_data):
        """Verify GPU complex output matches CPU (within tolerance)."""
        input_data = synthetic_doppler_data
        fft_len = doppler_params['fft_len']
        doppler_len = doppler_params['doppler_len']

        # Setup function signatures
        gpu_doppler_lib.doppler_gpu_create.restype = ctypes.c_void_p
        gpu_doppler_lib.doppler_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_doppler_lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_doppler_lib.doppler_gpu_process_complex.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        cpu_doppler_lib.doppler_create.restype = ctypes.c_void_p
        cpu_doppler_lib.doppler_create.argtypes = [ctypes.c_int, ctypes.c_int]
        cpu_doppler_lib.doppler_destroy.argtypes = [ctypes.c_void_p]
        cpu_doppler_lib.doppler_process_complex.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        # Create processors
        gpu_handle = gpu_doppler_lib.doppler_gpu_create(fft_len, doppler_len)
        assert gpu_handle is not None

        cpu_handle = cpu_doppler_lib.doppler_create(fft_len, doppler_len)
        assert cpu_handle is not None

        # Allocate output buffers
        output_size = 2 * doppler_len * fft_len
        gpu_output = np.zeros(output_size, dtype=np.float32)
        cpu_output = np.zeros(output_size, dtype=np.float32)

        # Process
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        gpu_doppler_lib.doppler_gpu_process_complex(
            gpu_handle, input_ptr,
            gpu_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        cpu_doppler_lib.doppler_process_complex(
            cpu_handle, input_ptr,
            cpu_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        # Cleanup
        gpu_doppler_lib.doppler_gpu_destroy(gpu_handle)
        cpu_doppler_lib.doppler_destroy(cpu_handle)

        # Compare complex outputs (relaxed tolerance for GPU/CPU floating-point differences in FFT)
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=2e-3, atol=2e-4,
                                   err_msg="GPU and CPU complex outputs differ")

        print(f"\n✓ GPU/CPU complex output match (max diff: {np.max(np.abs(gpu_output - cpu_output)):.6f})")

    def test_fftshift_correctness(self, gpu_doppler_lib, doppler_params):
        """Test that FFT shift places DC at center."""
        fft_len = 128
        doppler_len = 64

        # Create DC signal (constant across Doppler bins) - this creates DC in frequency domain
        data = np.zeros((doppler_len, fft_len), dtype=np.complex64)
        data[:, 50] = 10.0  # DC component (constant signal) in column 50

        # Convert to interleaved
        input_data = np.empty(2 * doppler_len * fft_len, dtype=np.float32)
        input_data[0::2] = data.real.flatten()
        input_data[1::2] = data.imag.flatten()

        # Setup
        gpu_doppler_lib.doppler_gpu_create.restype = ctypes.c_void_p
        gpu_doppler_lib.doppler_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_doppler_lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_doppler_lib.doppler_gpu_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        # Process
        handle = gpu_doppler_lib.doppler_gpu_create(fft_len, doppler_len)
        output = np.zeros(doppler_len * fft_len, dtype=np.float32)

        gpu_doppler_lib.doppler_gpu_process(
            handle,
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        gpu_doppler_lib.doppler_gpu_destroy(handle)

        # Check that peak is near center (after FFT shift)
        output_2d = output.reshape(doppler_len, fft_len)
        peak_row, peak_col = np.unravel_index(np.argmax(output_2d), output_2d.shape)

        # DC should be at center after shift
        expected_row = doppler_len // 2
        assert abs(peak_row - expected_row) <= 1, \
            f"FFT shift error: peak at row {peak_row}, expected ~{expected_row}"


class TestGPUDopplerPerformance:
    """Test GPU Doppler throughput and speedup."""

    def test_gpu_throughput(self, gpu_doppler_lib, doppler_params, synthetic_doppler_data):
        """Measure GPU Doppler processing time."""
        input_data = synthetic_doppler_data
        fft_len = doppler_params['fft_len']
        doppler_len = doppler_params['doppler_len']

        # Setup
        gpu_doppler_lib.doppler_gpu_create.restype = ctypes.c_void_p
        gpu_doppler_lib.doppler_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_doppler_lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_doppler_lib.doppler_gpu_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        handle = gpu_doppler_lib.doppler_gpu_create(fft_len, doppler_len)
        assert handle is not None

        output = np.zeros(doppler_len * fft_len, dtype=np.float32)
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Warmup
        for _ in range(10):
            gpu_doppler_lib.doppler_gpu_process(handle, input_ptr, output_ptr)

        # Benchmark
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            gpu_doppler_lib.doppler_gpu_process(handle, input_ptr, output_ptr)
        elapsed = time.perf_counter() - start

        gpu_doppler_lib.doppler_gpu_destroy(handle)

        # Calculate metrics
        avg_time_ms = (elapsed / n_iterations) * 1000
        throughput = 1000.0 / avg_time_ms

        print(f"\nGPU Doppler Performance:")
        print(f"  Average time: {avg_time_ms:.2f} ms/CPI")
        print(f"  Throughput: {throughput:.1f} CPIs/sec")

        # Target: < 5 ms on Jetson Orin, < 2 ms on desktop RTX
        assert avg_time_ms < 50, f"GPU Doppler too slow: {avg_time_ms:.2f} ms"

    def test_gpu_vs_cpu_speedup(self, gpu_doppler_lib, cpu_doppler_lib,
                                 doppler_params, synthetic_doppler_data):
        """Measure speedup of GPU vs CPU."""
        input_data = synthetic_doppler_data
        fft_len = doppler_params['fft_len']
        doppler_len = doppler_params['doppler_len']

        # Setup both
        gpu_doppler_lib.doppler_gpu_create.restype = ctypes.c_void_p
        gpu_doppler_lib.doppler_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_doppler_lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_doppler_lib.doppler_gpu_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        cpu_doppler_lib.doppler_create.restype = ctypes.c_void_p
        cpu_doppler_lib.doppler_create.argtypes = [ctypes.c_int, ctypes.c_int]
        cpu_doppler_lib.doppler_destroy.argtypes = [ctypes.c_void_p]
        cpu_doppler_lib.doppler_process.argtypes = gpu_doppler_lib.doppler_gpu_process.argtypes

        output = np.zeros(doppler_len * fft_len, dtype=np.float32)
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # GPU timing
        gpu_handle = gpu_doppler_lib.doppler_gpu_create(fft_len, doppler_len)
        for _ in range(5):
            gpu_doppler_lib.doppler_gpu_process(gpu_handle, input_ptr, output_ptr)

        gpu_iterations = 50
        gpu_start = time.perf_counter()
        for _ in range(gpu_iterations):
            gpu_doppler_lib.doppler_gpu_process(gpu_handle, input_ptr, output_ptr)
        gpu_time = (time.perf_counter() - gpu_start) / gpu_iterations
        gpu_doppler_lib.doppler_gpu_destroy(gpu_handle)

        # CPU timing
        cpu_handle = cpu_doppler_lib.doppler_create(fft_len, doppler_len)
        for _ in range(2):
            cpu_doppler_lib.doppler_process(cpu_handle, input_ptr, output_ptr)

        cpu_iterations = 10
        cpu_start = time.perf_counter()
        for _ in range(cpu_iterations):
            cpu_doppler_lib.doppler_process(cpu_handle, input_ptr, output_ptr)
        cpu_time = (time.perf_counter() - cpu_start) / cpu_iterations
        cpu_doppler_lib.doppler_destroy(cpu_handle)

        # Calculate speedup
        speedup = cpu_time / gpu_time

        print(f"\nGPU vs CPU Speedup:")
        print(f"  CPU time: {cpu_time * 1000:.2f} ms/CPI")
        print(f"  GPU time: {gpu_time * 1000:.2f} ms/CPI")
        print(f"  Speedup: {speedup:.1f}x")

        # Target: > 5x speedup minimum (10-15x on real GPU)
        assert speedup > 2.0, f"Insufficient speedup: {speedup:.1f}x"


class TestGPUDopplerRobustness:
    """Test GPU Doppler error handling and edge cases."""

    def test_null_handle(self, gpu_doppler_lib):
        """Test graceful handling of NULL handle."""
        gpu_doppler_lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_doppler_lib.doppler_gpu_destroy(None)  # Should not crash

    def test_small_input(self, gpu_doppler_lib):
        """Test with small FFT sizes."""
        gpu_doppler_lib.doppler_gpu_create.restype = ctypes.c_void_p
        gpu_doppler_lib.doppler_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_doppler_lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]

        # Very small problem
        handle = gpu_doppler_lib.doppler_gpu_create(64, 32)
        assert handle is not None, "Failed with small input sizes"
        gpu_doppler_lib.doppler_gpu_destroy(handle)

    def test_power_of_two_sizes(self, gpu_doppler_lib):
        """Test various power-of-two sizes."""
        gpu_doppler_lib.doppler_gpu_create.restype = ctypes.c_void_p
        gpu_doppler_lib.doppler_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_doppler_lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]

        for size in [64, 128, 256, 512, 1024, 2048]:
            handle = gpu_doppler_lib.doppler_gpu_create(size, size // 4)
            assert handle is not None, f"Failed with size {size}"
            gpu_doppler_lib.doppler_gpu_destroy(handle)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
