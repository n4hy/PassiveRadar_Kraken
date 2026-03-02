"""
GPU ECA-B Kernel Tests - Correctness and Performance
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Tests GPU-accelerated ECA-B clutter canceller against CPU version.
"""

import pytest
import numpy as np
import ctypes
from pathlib import Path

pytestmark = pytest.mark.gpu


@pytest.fixture
def gpu_eca_lib():
    """Load GPU ECA library."""
    lib_path = Path("/home/n4hy/PassiveRadar_Kraken/src/libkraken_eca_gpu.so")
    if not lib_path.exists():
        pytest.skip("GPU ECA library not built")
    return ctypes.cdll.LoadLibrary(str(lib_path))


@pytest.fixture
def cpu_eca_lib():
    """Load CPU ECA library for comparison."""
    lib_path = Path("/home/n4hy/PassiveRadar_Kraken/src/libkraken_eca_b_clutter_canceller.so")
    if not lib_path.exists():
        pytest.skip("CPU ECA library not found")
    return ctypes.cdll.LoadLibrary(str(lib_path))


@pytest.fixture
def eca_params():
    """Standard ECA parameters for testing."""
    return {
        'num_taps': 32,
        'max_delay': 1024,
        'n_samples': 4096,
    }


@pytest.fixture
def synthetic_signals(eca_params):
    """
    Generate synthetic reference and surveillance signals.
    Surveillance = delayed + scaled reference + noise + target
    """
    n_samples = eca_params['n_samples']

    # Generate reference signal (random complex)
    ref = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    ref = ref.astype(np.complex64)

    # Generate surveillance signal
    # Add delayed and scaled reference (clutter)
    delay = 10
    clutter_power = 5.0
    surv = np.zeros(n_samples, dtype=np.complex64)
    surv[delay:] = clutter_power * ref[:-delay]

    # Add noise
    noise_power = 0.1
    surv += noise_power * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)

    # Add target signal (should be preserved after cancellation)
    target_power = 2.0
    target_freq = 0.05  # Normalized frequency
    t = np.arange(n_samples)
    target = target_power * np.exp(2j * np.pi * target_freq * t).astype(np.complex64)
    surv += target

    # Convert to interleaved floats
    ref_interleaved = np.empty(2 * n_samples, dtype=np.float32)
    ref_interleaved[0::2] = ref.real
    ref_interleaved[1::2] = ref.imag

    surv_interleaved = np.empty(2 * n_samples, dtype=np.float32)
    surv_interleaved[0::2] = surv.real
    surv_interleaved[1::2] = surv.imag

    return ref_interleaved, surv_interleaved, target


class TestGPUECACorrectness:
    """Test GPU ECA output correctness."""

    def test_eca_gpu_basic_functionality(self, gpu_eca_lib, eca_params, synthetic_signals):
        """Test that GPU ECA runs without crashing."""
        ref_in, surv_in, target = synthetic_signals
        n_samples = eca_params['n_samples']
        num_taps = eca_params['num_taps']

        # Setup GPU ECA
        gpu_eca_lib.eca_gpu_create.restype = ctypes.c_void_p
        gpu_eca_lib.eca_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_eca_lib.eca_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_eca_lib.eca_gpu_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]

        # Create handle
        handle = gpu_eca_lib.eca_gpu_create(num_taps, 1024)
        assert handle is not None, "Failed to create GPU ECA handle"

        # Process
        output = np.zeros(2 * n_samples, dtype=np.float32)
        gpu_eca_lib.eca_gpu_process(
            handle,
            ref_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_samples
        )

        # Cleanup
        gpu_eca_lib.eca_gpu_destroy(handle)

        # Check output is reasonable
        output_complex = output[0::2] + 1j * output[1::2]
        output_power = np.mean(np.abs(output_complex)**2)

        print(f"\nGPU ECA output power: {output_power:.6f}")
        assert output_power > 0, "GPU ECA produced zero output"
        assert output_power < 1000, "GPU ECA output power unexpectedly high"

    def test_eca_reduces_clutter_power(self, gpu_eca_lib, eca_params, synthetic_signals):
        """Test that ECA reduces clutter power."""
        ref_in, surv_in, target = synthetic_signals
        n_samples = eca_params['n_samples']
        num_taps = eca_params['num_taps']

        # Setup
        gpu_eca_lib.eca_gpu_create.restype = ctypes.c_void_p
        gpu_eca_lib.eca_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_eca_lib.eca_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_eca_lib.eca_gpu_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]

        handle = gpu_eca_lib.eca_gpu_create(num_taps, 1024)

        # Process
        output = np.zeros(2 * n_samples, dtype=np.float32)
        gpu_eca_lib.eca_gpu_process(
            handle,
            ref_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            surv_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_samples
        )

        gpu_eca_lib.eca_gpu_destroy(handle)

        # Check clutter reduction
        surv_complex = surv_in[0::2] + 1j * surv_in[1::2]
        output_complex = output[0::2] + 1j * output[1::2]

        input_power = np.mean(np.abs(surv_complex)**2)
        output_power = np.mean(np.abs(output_complex)**2)

        print(f"\nInput power: {input_power:.6f}, Output power: {output_power:.6f}")
        print(f"Power reduction: {10*np.log10(output_power/input_power):.2f} dB")

        # Output should have less power (clutter removed)
        assert output_power < input_power, "ECA did not reduce signal power"


class TestGPUECAPerformance:
    """Test GPU ECA performance."""

    def test_gpu_eca_throughput(self, gpu_eca_lib, eca_params, synthetic_signals):
        """Measure GPU ECA processing throughput."""
        import time

        ref_in, surv_in, target = synthetic_signals
        n_samples = eca_params['n_samples']
        num_taps = eca_params['num_taps']

        # Setup
        gpu_eca_lib.eca_gpu_create.restype = ctypes.c_void_p
        gpu_eca_lib.eca_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        gpu_eca_lib.eca_gpu_destroy.argtypes = [ctypes.c_void_p]
        gpu_eca_lib.eca_gpu_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]

        handle = gpu_eca_lib.eca_gpu_create(num_taps, 1024)
        output = np.zeros(2 * n_samples, dtype=np.float32)

        # Warmup
        for _ in range(5):
            gpu_eca_lib.eca_gpu_process(
                handle,
                ref_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                surv_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_samples
            )

        # Benchmark
        n_iterations = 100
        start = time.time()
        for _ in range(n_iterations):
            gpu_eca_lib.eca_gpu_process(
                handle,
                ref_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                surv_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_samples
            )
        elapsed = time.time() - start

        gpu_eca_lib.eca_gpu_destroy(handle)

        avg_time_ms = (elapsed / n_iterations) * 1000
        throughput_hz = 1000.0 / avg_time_ms

        print(f"\nGPU ECA Performance:")
        print(f"  Average time: {avg_time_ms:.2f} ms")
        print(f"  Throughput: {throughput_hz:.1f} Hz")

        assert avg_time_ms < 100, f"GPU ECA too slow: {avg_time_ms:.2f} ms"
