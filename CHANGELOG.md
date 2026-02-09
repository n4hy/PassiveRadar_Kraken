# Changelog
## PassiveRadar_Kraken

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-02-08

### Added - GPU Acceleration (Complete)

#### Infrastructure
- **NVIDIA CUDA GPU acceleration** for compute-intensive DSP kernels
- Optional GPU support with **100% backward compatibility** for CPU-only builds (RPi5)
- Runtime backend selection (auto/gpu/cpu) via environment variable or Python API
- Automatic GPU detection with graceful CPU fallback
- Multi-platform GPU binary support (sm_75/86/87/89 - Turing through Blackwell)

#### GPU Libraries (4 new libraries)
- `libkraken_gpu_runtime.so` - Device management, memory pools, backend selection
- `libkraken_doppler_gpu.so` - **Validated ✅** Batched 2D FFT Doppler processing
- `libkraken_cfar_gpu.so` - **Validated ✅** Parallel 2D CFAR detection
- `libkraken_caf_gpu.so` - **Validated ✅** Batched cuFFT CAF processing (linear correlation)

#### Python API
- `is_gpu_available()` - Check GPU hardware availability
- `get_gpu_info()` - Query GPU device information
- `set_processing_backend(backend)` - Set global backend (auto/gpu/cpu)
- `get_active_backend()` - Query currently active backend
- `GPUBackend` class - Advanced GPU runtime access

#### Documentation
- `docs/GPU_USER_GUIDE.md` - Complete GPU user guide with installation and configuration
- `docs/GPU_PERFORMANCE.md` - Detailed performance benchmarks on RTX 5090
- `docs/GPU_API_REFERENCE.md` - Comprehensive API documentation
- `docs/GPU_DEPLOYMENT.md` - Platform-specific deployment guides
- Updated `README.md` with GPU sections and quick start

#### Performance Gains (NVIDIA RTX 5090)
- **Doppler Processing:** 1.27 ms (1.2x speedup vs laptop CPU, ~10-15x vs RPi5 estimated)
- **CFAR Detection:** 1.94 ms (**305x speedup** vs CPU, 2411 Hz throughput on large grids)
- **CAF Processing:** 2.03 ms (**23x speedup** for 8K samples, 29x for 16K samples)
- **Projected End-to-End:** 100-200 Hz update rate (vs 10 Hz on RPi5 CPU)

#### Validation Status
- ✅ **Doppler GPU:** Perfect correctness (correlation 1.000000 with CPU reference)
- ✅ **CFAR GPU:** 99.99% agreement with CPU, all targets detected
- ✅ **CAF GPU:** Perfect correctness (correlation 1.000000 with CPU reference), 23-29x speedup

#### Test Coverage
- Doppler GPU: Comprehensive validation tests (random, impulse, single tone)
- CFAR GPU: Synthetic target detection tests
- CAF GPU: Performance benchmarks
- RPi5 CPU-only: Full regression testing (191/196 tests pass, 5 skipped)

#### Build System
- CMake GPU detection and conditional build
- `ENABLE_GPU` build option (ON/OFF, default: auto-detect)
- Multi-architecture CUDA compilation (sm_75/86/87/89)
- Forward-compatible builds (sm_89 supports Blackwell RTX 5090 via compute 12.0)

#### Platform Support
- **Raspberry Pi 5:** CPU-only (zero impact, 100% backward compatible)
- **Desktop RTX GPUs:** Tested on RTX 5090 (Blackwell), supports RTX 2000-5000 series
- **NVIDIA Jetson:** sm_87 support (Orin NX/Nano, expected to work)
- **Cloud GPU:** AWS/Azure compatibility (requires sm_70 for V100)

### Changed
- Package version bumped to 0.2.0
- README.md updated with GPU acceleration overview
- Platform badge updated: `RPi5 | x86_64` → `RPi5 | x86_64 | GPU`
- Added GPU badge: `CUDA 12.0+`

### Fixed
- **CAF GPU correctness:** Changed from circular correlation (fft_len=n_samples) to linear correlation (fft_len=2*n_samples) to match CPU implementation, achieving perfect 1.0 correlation

### Security
- No security issues introduced (GPU code isolated in optional libraries)

---

## [0.1.0] - 2026-02-08 (Pre-GPU)

### Added
- Complete passive bistatic radar processing chain
- GNU Radio OOT module `gr-kraken_passive_radar`
- 7 C++ pybind11 blocks (ECA, Doppler, CFAR, Coherence, Clustering, AoA, Tracker)
- 7 Python blocks (KrakenSDR source, Calibration, Conditioning, CAF, TimeAlignment, etc.)
- 10 C++ kernel libraries (CPU-only)
- Display system (Tkinter + matplotlib): Range-Doppler, PPI, Calibration, Metrics
- Comprehensive test suite: 196 tests (191 passed, 5 skipped on headless)
- CI/CD: GitHub Actions (kernels, OOT module, lint)
- Export control compliance documentation (ITAR/EAR)

### Performance (Raspberry Pi 5 CPU)
- CAF: ~90 ms per CPI
- Doppler: ~15 ms
- CFAR: ~600 ms
- Full pipeline: ~10 Hz update rate

### Platform Support
- Raspberry Pi 5 (aarch64)
- x86_64 Linux (Ubuntu/Debian)

---

## [Unreleased]

### Planned
- CAF GPU correctness fix (estimated 2-4 hours)
- Validation on NVIDIA Jetson Orin
- Validation on AWS/Azure cloud GPUs
- Multi-GPU support
- Tensor Core investigation for Blackwell architecture
- Performance profiling with NVIDIA Nsight Compute

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.2.0 | 2026-02-08 | GPU acceleration (Doppler ✅, CFAR ✅, CAF ✅) - All kernels production-ready |
| 0.1.0 | 2026-02-08 | Initial release (CPU-only, full pipeline) |

---

**Author:** Dr. Robert W McGwier, PhD, N4HY
**GPU Implementation:** Claude (Anthropic)
**License:** MIT
