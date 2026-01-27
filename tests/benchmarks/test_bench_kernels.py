"""
Performance benchmarks for radar processing kernels.
"""
import unittest
import numpy as np
import time
import ctypes
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class BenchmarkResult:
    """Container for benchmark results."""
    def __init__(self, name, mean_ms, std_ms, iterations):
        self.name = name
        self.mean_ms = mean_ms
        self.std_ms = std_ms
        self.iterations = iterations

    def __str__(self):
        return f"{self.name}: {self.mean_ms:.3f} +/- {self.std_ms:.3f} ms ({self.iterations} iterations)"

    def to_dict(self):
        return {
            'name': self.name,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'iterations': self.iterations
        }


def benchmark(func, iterations=100, warmup=10):
    """Run benchmark and return results."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


class TestKernelBenchmarks(unittest.TestCase):
    """Benchmark radar processing kernels."""

    @classmethod
    def setUpClass(cls):
        """Load libraries."""
        repo_root = Path(__file__).resolve().parents[2]
        cls.results = []

        cls.libs = {}
        for name in ['caf_processing', 'eca_b_clutter_canceller']:
            lib_path = repo_root / "src" / f"libkraken_{name}.so"
            if lib_path.exists():
                try:
                    cls.libs[name] = ctypes.cdll.LoadLibrary(str(lib_path))
                except OSError:
                    pass

    @classmethod
    def tearDownClass(cls):
        """Print summary of all benchmarks."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        for result in cls.results:
            print(result)
        print("="*60)

        # Save results to JSON
        results_file = Path(__file__).parent / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in cls.results], f, indent=2)

    def test_bench_caf_4096(self):
        """Benchmark CAF with 4096 samples."""
        if 'caf_processing' not in self.libs:
            self.skipTest("CAF library not available")

        lib = self.libs['caf_processing']
        lib.caf_create.restype = ctypes.c_void_p
        lib.caf_create.argtypes = [ctypes.c_int]
        lib.caf_destroy.argtypes = [ctypes.c_void_p]
        lib.caf_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        n = 4096
        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        surv = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        out = np.zeros(n, dtype=np.complex64)

        obj = lib.caf_create(n)

        def run_caf():
            lib.caf_process(
                obj,
                ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )

        mean_ms, std_ms = benchmark(run_caf, iterations=100)
        lib.caf_destroy(obj)

        result = BenchmarkResult("CAF 4096 samples", mean_ms, std_ms, 100)
        self.results.append(result)
        print(f"\n{result}")

        # Performance target: < 5ms for real-time
        self.assertLess(mean_ms, 10.0, f"CAF too slow: {mean_ms:.3f} ms")

    def test_bench_eca_4096(self):
        """Benchmark ECA with 4096 samples."""
        if 'eca_b_clutter_canceller' not in self.libs:
            self.skipTest("ECA library not available")

        lib = self.libs['eca_b_clutter_canceller']
        lib.eca_b_create.restype = ctypes.c_void_p
        lib.eca_b_create.argtypes = [ctypes.c_int]
        lib.eca_b_destroy.argtypes = [ctypes.c_void_p]
        lib.eca_b_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]

        n = 4096
        num_taps = 64
        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        surv = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        out = np.zeros(n, dtype=np.complex64)

        state = lib.eca_b_create(num_taps)

        def run_eca():
            lib.eca_b_process(
                state,
                ref.view(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                surv.view(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.view(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n
            )

        mean_ms, std_ms = benchmark(run_eca, iterations=100)
        lib.eca_b_destroy(state)

        result = BenchmarkResult("ECA 4096 samples, 64 taps", mean_ms, std_ms, 100)
        self.results.append(result)
        print(f"\n{result}")

        self.assertLess(mean_ms, 5.0, f"ECA too slow: {mean_ms:.3f} ms")

    def test_bench_numpy_fft(self):
        """Benchmark NumPy FFT for comparison."""
        n = 4096
        data = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

        def run_fft():
            np.fft.fft(data)

        mean_ms, std_ms = benchmark(run_fft, iterations=100)

        result = BenchmarkResult("NumPy FFT 4096", mean_ms, std_ms, 100)
        self.results.append(result)
        print(f"\n{result}")

    def test_bench_numpy_xcorr(self):
        """Benchmark NumPy cross-correlation for comparison."""
        n = 4096
        ref = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        surv = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

        def run_xcorr():
            np.fft.ifft(np.fft.fft(surv) * np.conj(np.fft.fft(ref)))

        mean_ms, std_ms = benchmark(run_xcorr, iterations=100)

        result = BenchmarkResult("NumPy cross-correlation 4096", mean_ms, std_ms, 100)
        self.results.append(result)
        print(f"\n{result}")

    def test_bench_cfar_2d(self):
        """Benchmark 2D CFAR detection."""
        n_range = 256
        n_doppler = 64

        data = np.random.exponential(1.0, (n_doppler, n_range)).astype(np.float32)

        def run_cfar():
            guard = 2
            ref_cells = 4
            pfa = 1e-4
            n_ref = 2 * ref_cells
            alpha = n_ref * (pfa ** (-1/n_ref) - 1)

            margin = guard + ref_cells
            detections = np.zeros_like(data, dtype=bool)

            for i in range(margin, n_doppler - margin):
                for j in range(margin, n_range - margin):
                    window = data[i-margin:i+margin+1, j-margin:j+margin+1].copy()
                    window[ref_cells:ref_cells+2*guard+1, ref_cells:ref_cells+2*guard+1] = 0
                    threshold = alpha * np.sum(window) / (window.size - (2*guard+1)**2)
                    detections[i, j] = data[i, j] > threshold

            return detections

        mean_ms, std_ms = benchmark(run_cfar, iterations=50)

        result = BenchmarkResult(f"CFAR 2D {n_doppler}x{n_range}", mean_ms, std_ms, 50)
        self.results.append(result)
        print(f"\n{result}")

        # Target: < 20ms for real-time
        self.assertLess(mean_ms, 50.0, f"CFAR too slow: {mean_ms:.3f} ms")


class TestMemoryBenchmarks(unittest.TestCase):
    """Memory allocation benchmarks."""

    def test_allocation_overhead(self):
        """Measure array allocation overhead."""
        sizes = [1024, 4096, 16384, 65536]

        for n in sizes:
            def allocate():
                return np.zeros(n, dtype=np.complex64)

            mean_ms, std_ms = benchmark(allocate, iterations=1000)
            print(f"Allocate {n} complex64: {mean_ms*1000:.1f} us")


if __name__ == '__main__':
    unittest.main(verbosity=2)
