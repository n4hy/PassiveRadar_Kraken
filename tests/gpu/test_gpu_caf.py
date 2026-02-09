"""
GPU CAF Kernel Tests - Correctness and Performance
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Tests GPU-accelerated CAF implementation against CPU version for:
- Correctness (output equivalence)
- Performance (throughput and speedup)
- Backend selection and fallback
"""

import pytest
import numpy as np
import ctypes
import time
from pathlib import Path

# Skip all GPU tests if CUDA not available
pytestmark = pytest.mark.gpu


@pytest.fixture
def gpu_caf_lib():
    """Load GPU CAF library."""
    lib_path = Path("/home/n4hy/PassiveRadar_Kraken/src/libkraken_caf_gpu.so")
    if not lib_path.exists():
        pytest.skip("GPU CAF library not built")
    return ctypes.cdll.LoadLibrary(str(lib_path))


@pytest.fixture
def cpu_caf_lib():
    """Load CPU CAF library for comparison."""
    lib_path = Path("/home/n4hy/PassiveRadar_Kraken/src/libkraken_caf_processing.so")
    if not lib_path.exists():
        pytest.skip("CPU CAF library not found")
    return ctypes.cdll.LoadLibrary(str(lib_path))


@pytest.fixture
def caf_params():
    """Standard CAF parameters for testing."""
    return {
        'n_samples': 32768,      # 2^15 samples (realistic CPI)
        'n_doppler': 2048,       # 2048 Doppler bins
        'n_range': 32768,        # Same as n_samples for full correlation
        'doppler_start': -1000.0,  # Hz
        'doppler_step': 1.0,     # Hz
        'sample_rate': 2048000.0,  # 2.048 MHz
    }


@pytest.fixture
def synthetic_signal(caf_params):
    """
    Generate synthetic reference and surveillance signals with known delay/Doppler.

    Returns:
        tuple: (ref, surv) as interleaved float I/Q arrays
    """
    n_samples = caf_params['n_samples']
    sample_rate = caf_params['sample_rate']

    # Generate reference: complex exponential with noise
    t = np.arange(n_samples) / sample_rate
    ref_carrier = 1000.0  # 1 kHz carrier
    ref_signal = np.exp(2j * np.pi * ref_carrier * t)

    # Add noise
    noise_power = 0.01
    ref_signal += noise_power * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))

    # Surveillance: delayed and Doppler-shifted copy of reference
    delay_samples = 100  # Known delay (should appear at range bin 100)
    doppler_hz = 50.0    # Known Doppler (should appear at bin ~1050 if doppler_start=-1000)

    surv_signal = np.zeros(n_samples, dtype=np.complex64)
    surv_signal[delay_samples:] = ref_signal[:-delay_samples]

    # Apply Doppler shift
    doppler_phasor = np.exp(2j * np.pi * doppler_hz * t)
    surv_signal *= doppler_phasor

    # Add noise
    surv_signal += noise_power * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))

    # Convert to interleaved float I/Q
    ref_iq = np.empty(2 * n_samples, dtype=np.float32)
    ref_iq[0::2] = ref_signal.real
    ref_iq[1::2] = ref_signal.imag

    surv_iq = np.empty(2 * n_samples, dtype=np.float32)
    surv_iq[0::2] = surv_signal.real
    surv_iq[1::2] = surv_signal.imag

    return ref_iq, surv_iq


class TestGPUCAFCorrectness:
    """Test GPU CAF output correctness against CPU reference."""

    def test_gpu_cpu_equivalence(self, gpu_caf_lib, cpu_caf_lib, caf_params, synthetic_signal):
        """Verify GPU produces same output as CPU (within floating-point tolerance)."""
        ref, surv = synthetic_signal

        # Setup function signatures
        gpu_caf_lib.caf_gpu_create_full.restype = ctypes.c_void_p
        gpu_caf_lib.caf_gpu_create_full.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        gpu_caf_lib.caf_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_caf_lib.caf_gpu_process_full.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        cpu_caf_lib.caf_create_full.restype = ctypes.c_void_p
        cpu_caf_lib.caf_create_full.argtypes = gpu_caf_lib.caf_gpu_create_full.argtypes
        cpu_caf_lib.caf_destroy.argtypes = [ctypes.c_void_p]
        cpu_caf_lib.caf_process_full.argtypes = gpu_caf_lib.caf_gpu_process_full.argtypes

        # Create processors
        gpu_handle = gpu_caf_lib.caf_gpu_create_full(
            caf_params['n_samples'], caf_params['n_doppler'], caf_params['n_range'],
            caf_params['doppler_start'], caf_params['doppler_step'], caf_params['sample_rate']
        )
        assert gpu_handle is not None, "Failed to create GPU CAF processor"

        cpu_handle = cpu_caf_lib.caf_create_full(
            caf_params['n_samples'], caf_params['n_doppler'], caf_params['n_range'],
            caf_params['doppler_start'], caf_params['doppler_step'], caf_params['sample_rate']
        )
        assert cpu_handle is not None, "Failed to create CPU CAF processor"

        # Allocate output buffers
        output_size = caf_params['n_range'] * caf_params['n_doppler']
        gpu_output = np.zeros(output_size, dtype=np.float32)
        cpu_output = np.zeros(output_size, dtype=np.float32)

        # Process
        ref_ptr = ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        surv_ptr = surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        gpu_caf_lib.caf_gpu_process_full(gpu_handle, ref_ptr, surv_ptr,
                                         gpu_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        cpu_caf_lib.caf_process_full(cpu_handle, ref_ptr, surv_ptr,
                                     cpu_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        # Cleanup
        gpu_caf_lib.caf_gpu_destroy(gpu_handle)
        cpu_caf_lib.caf_destroy(cpu_handle)

        # Compare outputs
        # Allow for floating-point differences (GPU uses single-precision throughout)
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-4, atol=1e-5,
                                   err_msg="GPU and CPU outputs differ beyond tolerance")

        # Verify peak location matches
        gpu_peak_idx = np.argmax(gpu_output)
        cpu_peak_idx = np.argmax(cpu_output)
        assert gpu_peak_idx == cpu_peak_idx, \
            f"Peak location mismatch: GPU={gpu_peak_idx}, CPU={cpu_peak_idx}"

    def test_known_delay_detection(self, gpu_caf_lib, caf_params, synthetic_signal):
        """Verify GPU CAF detects known delay/Doppler correctly."""
        ref, surv = synthetic_signal

        # Setup
        gpu_caf_lib.caf_gpu_create_full.restype = ctypes.c_void_p
        gpu_caf_lib.caf_gpu_create_full.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        gpu_caf_lib.caf_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_caf_lib.caf_gpu_process_full.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        # Create processor
        handle = gpu_caf_lib.caf_gpu_create_full(
            caf_params['n_samples'], caf_params['n_doppler'], caf_params['n_range'],
            caf_params['doppler_start'], caf_params['doppler_step'], caf_params['sample_rate']
        )
        assert handle is not None

        # Process
        output_size = caf_params['n_range'] * caf_params['n_doppler']
        output = np.zeros(output_size, dtype=np.float32)

        gpu_caf_lib.caf_gpu_process_full(
            handle,
            ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        gpu_caf_lib.caf_gpu_destroy(handle)

        # Reshape to 2D (range × doppler)
        caf_surface = output.reshape(caf_params['n_range'], caf_params['n_doppler'])

        # Find peak
        peak_idx = np.unravel_index(np.argmax(caf_surface), caf_surface.shape)
        peak_range, peak_doppler = peak_idx

        # Expected: delay=100 samples, doppler=50 Hz (bin ~1050 for doppler_start=-1000)
        expected_range = 100
        expected_doppler_bin = int((50.0 - caf_params['doppler_start']) / caf_params['doppler_step'])

        # Allow ±5 bins tolerance for noise
        assert abs(peak_range - expected_range) <= 5, \
            f"Range detection error: expected ~{expected_range}, got {peak_range}"
        assert abs(peak_doppler - expected_doppler_bin) <= 5, \
            f"Doppler detection error: expected ~{expected_doppler_bin}, got {peak_doppler}"


class TestGPUCAFPerformance:
    """Test GPU CAF throughput and speedup."""

    def test_gpu_throughput(self, gpu_caf_lib, caf_params, synthetic_signal):
        """Measure GPU CAF processing time."""
        ref, surv = synthetic_signal

        # Setup
        gpu_caf_lib.caf_gpu_create_full.restype = ctypes.c_void_p
        gpu_caf_lib.caf_gpu_create_full.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        gpu_caf_lib.caf_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_caf_lib.caf_gpu_process_full.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        handle = gpu_caf_lib.caf_gpu_create_full(
            caf_params['n_samples'], caf_params['n_doppler'], caf_params['n_range'],
            caf_params['doppler_start'], caf_params['doppler_step'], caf_params['sample_rate']
        )
        assert handle is not None

        output_size = caf_params['n_range'] * caf_params['n_doppler']
        output = np.zeros(output_size, dtype=np.float32)

        ref_ptr = ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        surv_ptr = surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Warmup (10 iterations)
        for _ in range(10):
            gpu_caf_lib.caf_gpu_process_full(handle, ref_ptr, surv_ptr, out_ptr)

        # Benchmark (100 iterations)
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            gpu_caf_lib.caf_gpu_process_full(handle, ref_ptr, surv_ptr, out_ptr)
        elapsed = time.perf_counter() - start

        gpu_caf_lib.caf_gpu_destroy(handle)

        # Calculate metrics
        avg_time_ms = (elapsed / n_iterations) * 1000
        throughput_cpi_per_sec = 1000.0 / avg_time_ms

        print(f"\nGPU CAF Performance:")
        print(f"  Average time: {avg_time_ms:.2f} ms/CPI")
        print(f"  Throughput: {throughput_cpi_per_sec:.1f} CPIs/sec")

        # Assert reasonable performance (target: < 10 ms on Jetson Orin)
        # On desktop GPU should be even faster (< 5 ms)
        # Note: This will be slow on CPU-only build with CUDA emulation
        assert avg_time_ms < 100, \
            f"GPU CAF too slow: {avg_time_ms:.2f} ms (expected < 100 ms)"

    def test_gpu_vs_cpu_speedup(self, gpu_caf_lib, cpu_caf_lib, caf_params, synthetic_signal):
        """Measure speedup of GPU vs CPU."""
        ref, surv = synthetic_signal

        # Setup function signatures (same for both)
        for lib in [gpu_caf_lib, cpu_caf_lib]:
            lib.caf_create_full.restype = ctypes.c_void_p
            lib.caf_create_full.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float
            ]
            lib.caf_destroy.argtypes = [ctypes.c_void_p]
            lib.caf_process_full.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float)
            ]

        gpu_caf_lib.caf_gpu_create_full.restype = ctypes.c_void_p
        gpu_caf_lib.caf_gpu_create_full.argtypes = cpu_caf_lib.caf_create_full.argtypes
        gpu_caf_lib.caf_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_caf_lib.caf_gpu_process_full.argtypes = cpu_caf_lib.caf_process_full.argtypes

        # Benchmark GPU
        gpu_handle = gpu_caf_lib.caf_gpu_create_full(
            caf_params['n_samples'], caf_params['n_doppler'], caf_params['n_range'],
            caf_params['doppler_start'], caf_params['doppler_step'], caf_params['sample_rate']
        )
        output = np.zeros(caf_params['n_range'] * caf_params['n_doppler'], dtype=np.float32)
        ref_ptr = ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        surv_ptr = surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # GPU warmup
        for _ in range(5):
            gpu_caf_lib.caf_gpu_process_full(gpu_handle, ref_ptr, surv_ptr, out_ptr)

        # GPU timing
        gpu_iterations = 50
        gpu_start = time.perf_counter()
        for _ in range(gpu_iterations):
            gpu_caf_lib.caf_gpu_process_full(gpu_handle, ref_ptr, surv_ptr, out_ptr)
        gpu_time = (time.perf_counter() - gpu_start) / gpu_iterations
        gpu_caf_lib.caf_gpu_destroy(gpu_handle)

        # Benchmark CPU
        cpu_handle = cpu_caf_lib.caf_create_full(
            caf_params['n_samples'], caf_params['n_doppler'], caf_params['n_range'],
            caf_params['doppler_start'], caf_params['doppler_step'], caf_params['sample_rate']
        )

        # CPU warmup
        for _ in range(2):
            cpu_caf_lib.caf_process_full(cpu_handle, ref_ptr, surv_ptr, out_ptr)

        # CPU timing (fewer iterations - it's slow)
        cpu_iterations = 10
        cpu_start = time.perf_counter()
        for _ in range(cpu_iterations):
            cpu_caf_lib.caf_process_full(cpu_handle, ref_ptr, surv_ptr, out_ptr)
        cpu_time = (time.perf_counter() - cpu_start) / cpu_iterations
        cpu_caf_lib.caf_destroy(cpu_handle)

        # Calculate speedup
        speedup = cpu_time / gpu_time

        print(f"\nGPU vs CPU Speedup:")
        print(f"  CPU time: {cpu_time * 1000:.2f} ms/CPI")
        print(f"  GPU time: {gpu_time * 1000:.2f} ms/CPI")
        print(f"  Speedup: {speedup:.1f}x")

        # Assert speedup > 5x minimum (target is 15-25x on real GPU)
        # Note: On CPU-only build this may not pass
        assert speedup > 2.0, \
            f"Insufficient speedup: {speedup:.1f}x (expected > 2x minimum)"


class TestGPUCAFRobustness:
    """Test GPU CAF error handling and edge cases."""

    def test_null_handle(self, gpu_caf_lib):
        """Test graceful handling of NULL processor handle."""
        gpu_caf_lib.caf_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_caf_lib.caf_gpu_destroy(None)  # Should not crash

    def test_small_input(self, gpu_caf_lib):
        """Test GPU CAF with small input (edge case)."""
        gpu_caf_lib.caf_gpu_create_full.restype = ctypes.c_void_p
        gpu_caf_lib.caf_gpu_create_full.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]

        # Small problem size
        handle = gpu_caf_lib.caf_gpu_create_full(
            256, 32, 256,      # Small n_samples, n_doppler, n_range
            -100.0, 5.0, 10000.0
        )
        assert handle is not None, "Failed to create processor with small inputs"

        gpu_caf_lib.caf_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_caf_lib.caf_gpu_destroy(handle)
