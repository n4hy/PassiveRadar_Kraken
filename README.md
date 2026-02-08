# PassiveRadar_Kraken

**Passive Bistatic Radar System for KrakenSDR**

[![CI](https://github.com/n4hy/PassiveRadar_Kraken/actions/workflows/ci.yml/badge.svg)](https://github.com/n4hy/PassiveRadar_Kraken/actions) [![License](https://img.shields.io/badge/license-MIT-blue)]() [![Platform](https://img.shields.io/badge/platform-RPi5%20%7C%20x86__64-lightgrey)]() [![GNU Radio](https://img.shields.io/badge/GNU%20Radio-3.10+-green)]()

GNU Radio Out-of-Tree (OOT) module for passive bistatic radar using the KrakenSDR 5-channel coherent SDR receiver. Implements the full processing chain from coherent acquisition through clutter cancellation, Doppler processing, CFAR detection, AoA estimation, and multi-target tracking, all in C++ with Python bindings.

---

## Table of Contents

- [Test Results](#test-results)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Signal Processing Chain](#signal-processing-chain)
- [GNU Radio Blocks](#gnu-radio-blocks)
- [C++ Kernel Libraries](#c-kernel-libraries)
- [Export Control Compliance](#export-control-compliance-itar-and-ear)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Running](#running)
- [Testing](#testing)
- [Display System](#display-system)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [References](#references)

---

## Test Results

**Platform**: Raspberry Pi 5 (aarch64), Python 3.13.5, GNU Radio 3.10.12

**Date**: 2026-02-08

```
========================= test session starts =========================
platform linux -- Python 3.13.5, pytest-8.3.5
collected 196 items

tests/benchmarks/test_bench_kernels.py
  TestKernelBenchmarks::test_bench_caf_4096                        PASSED
  TestKernelBenchmarks::test_bench_cfar_2d                         PASSED
  TestKernelBenchmarks::test_bench_eca_4096                        PASSED
  TestKernelBenchmarks::test_bench_numpy_fft                       PASSED
  TestKernelBenchmarks::test_bench_numpy_xcorr                     PASSED
  TestMemoryBenchmarks::test_allocation_overhead                   PASSED

tests/integration/test_full_pipeline.py
  TestPipelineIntegration::test_eca_improves_detection             PASSED
  TestPipelineIntegration::test_multi_target_scenario              PASSED
  TestPipelineIntegration::test_single_target_detection            PASSED
  TestProcessingChainValidation::test_power_conservation           PASSED
  TestProcessingChainValidation::test_signal_dimensions            PASSED

tests/test_aoa_cpp.py
  TestAoACpp::test_aoa_estimation                                  PASSED

tests/test_backend_cpp.py
  TestBackendCpp::test_cfar_detects_target                         PASSED

tests/test_caf_cpp.py
  TestCafCpp::test_caf_process                                     PASSED

tests/test_conditioning_cpp.py
  TestConditioningCpp::test_agc_normalizes_level                   PASSED

tests/test_doppler_cpp.py
  TestDopplerCpp::test_doppler_processing                          PASSED

tests/test_eca_b_cpp.py
  TestEcaBCpp::test_eca_b_reduces_clutter_power                    PASSED

tests/test_end_to_end.py
  TestEndToEndOffline::test_manual_pipeline                        PASSED

tests/test_gr_cpp_blocks.py
  TestEcaCanceller      (4 tests)                                  PASSED
  TestDopplerProcessor  (5 tests)                                  PASSED
  TestCfarDetector      (8 tests)                                  PASSED
  TestCoherenceMonitor  (7 tests)                                  PASSED
  TestDetectionCluster  (8 tests)                                  PASSED
  TestAoaEstimator      (8 tests)                                  PASSED
  TestTracker          (12 tests)                                  PASSED
  TestTrackStatusEnum   (3 tests)                                  PASSED
  TestTrackStruct       (3 tests)                                  PASSED

tests/test_krakensdr_source.py
  TestKrakenSDRSource::test_initialization                         PASSED
  TestKrakenSDRSource::test_setters                                PASSED

tests/test_time_alignment_cpp.py
  TestTimeAlignmentCpp::test_detects_known_delay                   PASSED

tests/unit/test_aoa_kernels.py        (8 tests)                   PASSED
tests/unit/test_caf_kernels.py        (8 tests)                   PASSED
tests/unit/test_cfar_kernels.py       (8 tests)                   PASSED
tests/unit/test_clustering.py        (11 tests)                   PASSED
tests/unit/test_display_modules.py   (12 passed, 5 skipped)
tests/unit/test_doppler_kernels.py   (10 tests)                   PASSED
tests/unit/test_eca_kernels.py        (7 tests)                   PASSED
tests/unit/test_fixtures.py          (28 tests)                   PASSED
tests/unit/test_tracker.py           (10 tests)                   PASSED

=============== 191 passed, 5 skipped in 17.16s ======================
```

### Summary

| Category | Tests | Status |
|----------|------:|--------|
| C++ kernel libraries | 7 | 7 passed |
| GNU Radio C++ blocks (pybind11) | 58 | 58 passed |
| GNU Radio Python blocks | 2 | 2 passed |
| Unit tests (algorithms) | 74 | 74 passed |
| Integration / end-to-end | 6 | 6 passed |
| Benchmarks | 6 | 6 passed |
| Test fixtures | 28 | 28 passed |
| Display module math | 10 | 10 passed |
| Display module imports | 5 | 5 skipped (headless) |
| **Total** | **196** | **191 passed, 5 skipped** |

The 5 skipped tests are display module import checks that require a GUI environment (DISPLAY or WAYLAND_DISPLAY). All algorithmic tests for those modules still pass.

---

## System Architecture

```
KrakenSDR 5-Channel Coherent SDR
  Ch0: Reference (illuminator-facing antenna)
  Ch1-4: Surveillance (target-facing array)
        |
        v
+-------------------+     +-----------------+
| Calibration       |     | Coherence       |
| Controller        |<--->| Monitor (C++)   |
| (noise src ctrl)  |     | (phase drift)   |
+-------------------+     +-----------------+
        |
        v
  Phase Correction (per channel)
        |
        v
  Conditioning / AGC
        |
        v
+-------------------+     +-------------------+     +-------------------+
| ECA Canceller     | --> | CAF               | --> | Doppler Processor |
| (C++/VOLK)        |     | (cross-ambiguity) |     | (C++/FFTW)        |
| NLMS adaptive     |     | range profiles    |     | slow-time FFT     |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
+-------------------+     +-------------------+     +-------------------+
| Tracker           | <-- | AoA Estimator     | <-- | CFAR Detector     |
| (C++ Kalman+GNN)  |     | (C++ Bartlett)    |     | (C++ CA/GO/SO/OS) |
| multi-target      |     | ULA/UCA arrays    |     +-------------------+
+-------------------+     +-------------------+             ^
        |                                                   |
        v                                           +-------------------+
  Display System                                    | Detection Cluster |
  (Tkinter + matplotlib)                            | (C++ 8-connected) |
                                                    +-------------------+
```

### Calibration

The KrakenSDR has an internal wideband noise source with a high-isolation silicon switch. When the noise source is enabled, the switch physically disconnects all antennas and routes only the internal noise to all 5 receivers. This provides a common reference signal for measuring inter-channel phase offsets. The `CalibrationController` automates this cycle and the `coherence_monitor` block triggers it when phase drift is detected.

---

## Project Structure

```
PassiveRadar_Kraken/
|-- gr-kraken_passive_radar/         GNU Radio OOT module (the main module)
|   |-- lib/                         C++ block implementations (7 blocks)
|   |-- include/gnuradio/             Public C++ headers
|   |   +-- kraken_passive_radar/
|   |-- python/kraken_passive_radar/
|   |   |-- bindings/               pybind11 binding files
|   |   |-- __init__.py             Module entry point
|   |   |-- krakensdr_source.py     KrakenSDR source block
|   |   |-- calibration_controller.py
|   |   |-- custom_blocks.py        Conditioning, CAF, TimeAlignment
|   |   |-- vector_zero_pad.py
|   |   |-- eca_b_clutter_canceller.py   (deprecated)
|   |   +-- doppler_processing.py        (deprecated)
|   +-- grc/                         GRC block YAML definitions (12 blocks)
|
|-- src/                             C++ kernel libraries (10 .so)
|   |-- build/                       Out-of-source CMake build directory
|   |-- CMakeLists.txt
|   |-- eca_b_clutter_canceller.cpp
|   |-- conditioning.cpp
|   |-- caf_processing.cpp
|   |-- doppler_processing.cpp
|   |-- backend.cpp
|   |-- aoa_processing.cpp
|   |-- time_alignment.cpp
|   |-- fftw_init.cpp               Centralized FFTW thread init
|   |-- resampler.cpp
|   +-- nlms_clutter_canceller.cpp
|
|-- kraken_passive_radar/            Display system (Tkinter + matplotlib)
|   |-- radar_gui.py
|   |-- range_doppler_display.py
|   |-- radar_display.py
|   |-- calibration_panel.py
|   +-- metrics_dashboard.py
|
|-- tests/                           Test suite (196 tests)
|   |-- conftest.py                  Shared pytest fixtures
|   |-- mock_gnuradio.py            GNU Radio mock for headless testing
|   |-- test_gr_cpp_blocks.py       58 pybind11 block tests
|   |-- test_end_to_end.py          Full pipeline test
|   |-- test_krakensdr_source.py    Source block tests
|   |-- test_*_cpp.py               Per-kernel C++ tests (7 files)
|   |-- unit/                       Algorithm unit tests (9 files)
|   |-- integration/                Pipeline integration tests
|   |-- benchmarks/                 Performance benchmarks
|   +-- fixtures/                   Synthetic targets, clutter, noise
|
|-- .github/workflows/ci.yml        GitHub Actions CI
|-- run_passive_radar.py             Main application script
|-- build_oot.sh                     OOT module build script
|-- rebuild_libs.sh                  Kernel library build script
+-- kraken_passive_radar_103_7MHz.grc  Example GRC flowgraph
```

---

## Signal Processing Chain

The `run_passive_radar.py` script implements the full processing chain using C++ blocks:

```
Source -> PhaseCorr -> AGC -> ECA(C++) -> CAF -> Doppler(C++) ->
  CFAR(C++) -> Cluster(C++) -> AoA(C++) -> Tracker(C++) -> Display
```

| Stage | Block | Language | Description |
|-------|-------|----------|-------------|
| 1 | `krakensdr_source` | Python | 5-channel osmosdr source wrapper |
| 2 | `PhaseCorrectorBlock` | Python | Applies calibration phase corrections |
| 3 | `ConditioningBlock` | Python+ctypes | AGC / signal conditioning |
| 4 | `eca_canceller` | C++ (VOLK) | NLMS adaptive clutter cancellation |
| 5 | `CafBlock` | Python+ctypes | Cross-ambiguity function (range profiles) |
| 6 | `doppler_processor` | C++ (FFTW) | Slow-time FFT for range-Doppler map |
| 7 | `cfar_detector` | C++ | CA/GO/SO/OS-CFAR detection |
| 8 | `detection_cluster` | C++ | 8-connected component target extraction |
| 9 | `aoa_estimator` | C++ | Bartlett beamforming AoA (ULA/UCA) |
| 10 | `tracker` | C++ | Kalman filter + GNN association |

---

## GNU Radio Blocks

The single OOT module `gr-kraken_passive_radar` provides 14 blocks: 7 C++ (pybind11) and 7 Python.

### C++ Blocks (pybind11)

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| `eca_canceller` | VOLK-accelerated NLMS clutter canceller | `num_taps`, `reg_factor`, `num_surv` |
| `doppler_processor` | Range-Doppler map via slow-time FFT | `num_range_bins`, `num_doppler_bins`, `window_type` |
| `cfar_detector` | CA/GO/SO/OS-CFAR detection | `pfa`, `cfar_type`, guard/ref cells |
| `coherence_monitor` | Phase coherence monitoring + cal trigger | `corr_threshold`, `phase_threshold_deg` |
| `detection_cluster` | Connected-component target extraction | `min_cluster_size`, `range_resolution_m` |
| `aoa_estimator` | Bartlett beamforming AoA estimation | `num_elements`, `d_lambda`, `array_type` |
| `tracker` | Multi-target Kalman tracker with GNN | `dt`, process/meas noise, `gate_threshold` |

### Python Blocks

| Block | Description | Status |
|-------|-------------|--------|
| `krakensdr_source` | 5-channel coherent SDR source (osmosdr) | Active |
| `CalibrationController` | Automatic phase calibration manager | Active |
| `ConditioningBlock` | AGC / signal conditioning (ctypes) | Active |
| `CafBlock` | Cross-ambiguity function (ctypes) | Active |
| `TimeAlignmentBlock` | Delay/phase measurement (ctypes) | Active |
| `vector_zero_pad` | Zero-pad vectors for FFT alignment | Active |
| `EcaBClutterCanceller` | ECA-B clutter cancellation (ctypes) | Deprecated - use `eca_canceller` |
| `DopplerProcessingBlock` | Doppler processing (ctypes) | Deprecated - use `doppler_processor` |
| `BackendBlock` | CFAR + fusion (ctypes) | Deprecated - use `cfar_detector` + `detection_cluster` |

---

## C++ Kernel Libraries

Ten shared libraries built from `src/` provide the DSP kernels used by both the OOT module and the Python+ctypes wrappers.

| Library | Description | Dependencies |
|---------|-------------|--------------|
| `libkraken_eca_b_clutter_canceller.so` | ECA-B NLMS clutter cancellation | libm |
| `libkraken_conditioning.so` | Signal conditioning / AGC | libm |
| `libkraken_fftw_init.so` | Centralized FFTW thread init (pthread_once) | fftw3f, fftw3f_threads |
| `libkraken_time_alignment.so` | Cross-correlation time alignment | fftw3f, kraken_fftw_init |
| `libkraken_caf_processing.so` | Cross-ambiguity function | fftw3f, kraken_fftw_init, OptMathKernels (optional) |
| `libkraken_doppler_processing.so` | Range-Doppler map generation | fftw3f, kraken_fftw_init |
| `libkraken_backend.so` | CFAR detection and sensor fusion | libm |
| `libkraken_aoa_processing.so` | Angle-of-arrival processing | libm |
| `libkraken_resampler.so` | Sample rate conversion | libm |
| `libkraken_nlms_clutter_canceller.so` | NLMS adaptive filter | libm |

OptMathKernels (v0.2.1+) provides optional NEON acceleration for `caf_processing` via `neon_complex_mul_f32`. The build auto-detects it via CMake `find_package`.

---

## Export Control Compliance: ITAR and EAR

This passive radar system is ITAR and EAR compliant:

- **Limited bandwidth**: 2.4 MHz max sample rate, well below EAR ECCN 3A001.b.1 threshold (>50 MHz)
- **Low operating frequency**: FM broadcast band (88-108 MHz), civilian illuminators only
- **Passive reception only**: No active transmission capability
- **Open-source**: All algorithms published in open academic literature (NLMS, CFAR, Kalman)
- **COTS hardware**: KrakenSDR is a commercially available consumer SDR (~$400)
- **Performance class**: ~15-20 km range, ~300-600 m resolution, typical of academic research

This software is intended for educational purposes, amateur radio experimentation, and civilian passive radar research. It is not designed for military targeting, fire control, or weapon guidance. Users must consult a qualified export control attorney before exporting this software or derivatives. See the full compliance section in the repository for details.

---

## Prerequisites

### Required

```bash
# Ubuntu/Debian (including Raspberry Pi OS 64-bit)
sudo apt install -y \
    build-essential cmake pkg-config \
    gnuradio gnuradio-dev \
    libfftw3-dev libvolk2-dev pybind11-dev \
    python3-dev python3-numpy python3-pytest
```

### Optional

```bash
# Display system
pip3 install matplotlib

# OptMathKernels for NEON acceleration on Pi 5
# See https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA
```

---

## Building

### 1. Build C++ kernel libraries

```bash
cd src
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..
```

Libraries are output to `src/` (build artifacts stay in `src/build/`). Portable by default; add `-DNATIVE_OPTIMIZATION=ON` for `-march=native` when building and running on the same machine.

### 2. Build and install the OOT module

```bash
./build_oot.sh
```

This builds all 7 C++ blocks with pybind11 bindings, installs them into the gnuradio Python namespace, and copies Python blocks and GRC definitions.

### 3. Verify installation

```bash
python3 -c "
from gnuradio import kraken_passive_radar as kpr
print('eca_canceller:', kpr.eca_canceller)
print('doppler_processor:', kpr.doppler_processor)
print('cfar_detector:', kpr.cfar_detector)
print('tracker:', kpr.tracker)
"
```

---

## Running

### Full processing chain

```bash
python3 run_passive_radar.py --freq 103.7e6 --gain 30
```

Options:
- `--freq` : Center frequency in Hz (default: 100 MHz)
- `--gain` : Receiver gain in dB (default: 30)
- `--geometry` : Array geometry, ULA or URA (default: ULA)
- `--include-ref` : Include reference antenna in AoA array
- `--no-startup-cal` : Skip startup calibration
- `--visualize` : Show GUI display

### GNU Radio Companion

```bash
gnuradio-companion kraken_passive_radar_103_7MHz.grc
```

---

## Testing

### Run all tests

```bash
python3 -m pytest tests/ -v
```

### Run by category

```bash
# C++ pybind11 block tests (58 tests)
python3 -m pytest tests/test_gr_cpp_blocks.py -v

# Kernel library tests (7 tests)
python3 -m pytest tests/test_*_cpp.py -v

# Unit tests (74 tests)
python3 -m pytest tests/unit/ -v

# Integration tests (6 tests)
python3 -m pytest tests/integration/ -v

# Benchmarks (6 tests)
python3 -m pytest tests/benchmarks/ -v

# End-to-end pipeline (1 test)
python3 -m pytest tests/test_end_to_end.py -v
```

### Test file inventory

```
tests/
  conftest.py                        Shared fixtures (sample_rate, complex_noise, etc.)
  mock_gnuradio.py                   GNU Radio mock for headless testing

  test_gr_cpp_blocks.py              58 tests - all 7 C++ pybind11 blocks
  test_krakensdr_source.py            2 tests - KrakenSDR source init/setters
  test_end_to_end.py                  1 test  - full offline pipeline
  test_eca_b_cpp.py                   1 test  - ECA-B kernel clutter reduction
  test_caf_cpp.py                     1 test  - CAF kernel processing
  test_doppler_cpp.py                 1 test  - Doppler kernel processing
  test_backend_cpp.py                 1 test  - CFAR kernel detection
  test_conditioning_cpp.py            1 test  - AGC kernel normalization
  test_time_alignment_cpp.py          1 test  - time alignment kernel
  test_aoa_cpp.py                     1 test  - AoA kernel estimation

  unit/
    test_eca_kernels.py               7 tests - ECA algorithm variants
    test_caf_kernels.py               8 tests - CAF peak location, sidelobes
    test_doppler_kernels.py          10 tests - Doppler shift, windows
    test_cfar_kernels.py              8 tests - CFAR Pfa calibration, variants
    test_aoa_kernels.py               8 tests - AoA accuracy, MUSIC
    test_clustering.py               11 tests - connected components, filtering
    test_tracker.py                  10 tests - Kalman filter, track lifecycle
    test_display_modules.py          17 tests - display math + import checks
    test_fixtures.py                 28 tests - synthetic targets, clutter, noise

  integration/
    test_full_pipeline.py             5 tests - multi-block pipeline validation

  benchmarks/
    test_bench_kernels.py             6 tests - performance benchmarks
```

### CI

GitHub Actions CI runs on every push and PR to `main`:
- **kernels**: Build C++ kernel libraries + run tests (excludes GR block tests)
- **oot-module**: Build OOT module with GNU Radio + run pybind11 block tests
- **lint**: flake8 static analysis

---

## Display System

The `kraken_passive_radar/` package provides Tkinter + matplotlib visualization:

| Module | Description |
|--------|-------------|
| `radar_gui.py` | Integrated multi-panel GUI |
| `range_doppler_display.py` | Range-Doppler heatmap with detection overlays |
| `radar_display.py` | PPI polar display with track trails |
| `calibration_panel.py` | Per-channel SNR, phase offset monitoring |
| `metrics_dashboard.py` | Processing latency and system health metrics |

The display system automatically selects the `Agg` matplotlib backend when no display server is available (headless operation).

---

## API Reference

### C++ blocks (from Python)

```python
from gnuradio.kraken_passive_radar import (
    eca_canceller, doppler_processor, cfar_detector,
    coherence_monitor, detection_cluster, aoa_estimator, tracker,
)

# ECA Clutter Canceller
blk = eca_canceller(num_taps=128, reg_factor=0.001, num_surv=4)
blk.set_num_taps(256)
blk.set_reg_factor(0.01)

# Doppler Processor
blk = doppler_processor.make(
    num_range_bins=4096, num_doppler_bins=64,
    window_type=1,       # 0=rect, 1=hamming, 2=hann, 3=blackman
    output_power=True    # True=|X|^2, False=complex X
)
blk.set_window_type(2)

# CFAR Detector
blk = cfar_detector.make(
    num_range_bins=4096, num_doppler_bins=64,
    guard_cells_range=2, guard_cells_doppler=2,
    ref_cells_range=8, ref_cells_doppler=8,
    pfa=1e-6,            # probability of false alarm
    cfar_type=0           # 0=CA, 1=GO, 2=SO, 3=OS
)
blk.set_pfa(1e-4)
n = blk.get_num_detections()

# Coherence Monitor
blk = coherence_monitor.make(
    num_channels=5, sample_rate=2.4e6,
    corr_threshold=0.95, phase_threshold_deg=5.0
)
needs_cal = blk.is_calibration_needed()

# Detection Clustering
blk = detection_cluster.make(
    num_range_bins=4096, num_doppler_bins=64,
    min_cluster_size=1, max_detections=100,
    range_resolution_m=600.0, doppler_resolution_hz=3.9
)
dets = blk.get_detections()   # list of detection_t

# AoA Estimator
blk = aoa_estimator.make(
    num_elements=4, d_lambda=0.5, n_angles=181,
    min_angle_deg=-90.0, max_angle_deg=90.0,
    array_type=0          # 0=ULA, 1=UCA
)
spectrum = blk.get_spectrum()

# Multi-Target Tracker
blk = tracker.make(
    dt=0.1,
    process_noise_range=50.0, process_noise_doppler=5.0,
    meas_noise_range=100.0, meas_noise_doppler=2.0,
    gate_threshold=9.21,  # chi2(2) @ 99%
    confirm_hits=3, delete_misses=5, max_tracks=50
)
tracks = blk.get_confirmed_tracks()  # list of track_t
blk.reset()
```

### Data structures

```python
from gnuradio.kraken_passive_radar import track_t, track_status_t, detection_t

# track_t fields
t = track_t()
t.id, t.status, t.hits, t.misses, t.age, t.score
t.range_m, t.doppler_hz, t.range_rate, t.doppler_rate  # convenience properties
t.state       # [range_m, doppler_hz, range_rate, doppler_rate]
t.covariance  # 4x4 row-major

# track_status_t enum
track_status_t.TENTATIVE   # 0 - new, needs confirmation
track_status_t.CONFIRMED   # 1 - confirmed track
track_status_t.COASTING    # 2 - predicting, no measurement
```

---

## Troubleshooting

### "C++ pybind11 blocks not available"

The OOT module needs to be built and installed:

```bash
./build_oot.sh
```

### "Could not load libkraken_*.so"

Build the kernel libraries:

```bash
cd src && mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

### Tests fail with MagicMock errors

Some test files inject GNU Radio mocks for headless testing. If running `test_gr_cpp_blocks.py` fails with MagicMock assertions, ensure you run it after the OOT module is installed, or run it in isolation:

```bash
python3 -m pytest tests/test_gr_cpp_blocks.py -v
```

### CFAR benchmark threshold

The CFAR 2D benchmark uses platform-aware thresholds: 150ms on aarch64 (Pi 5), 50ms on x86_64.

### Display tests skip

The 5 display module import tests skip when no DISPLAY or WAYLAND_DISPLAY environment variable is set. This is expected on headless systems. All display algorithm tests still run.

---

## License

MIT License. See [LICENSE](LICENSE).

### Third-Party Licenses

- **GNU Radio**: GPL v3.0
- **FFTW3**: GPL v2.0+ (dynamic linking)
- **VOLK**: LGPL v3.0
- **OptMathKernels**: MIT

---

## References

### Academic

1. M. Cherniakov (Ed.), *Bistatic Radar: Principles and Practice*, Wiley, 2007
2. H. Griffiths and C. Baker, "Passive Coherent Location Radar Systems", IEE Proceedings, 2005
3. R. Tao et al., "ECA-B Clutter Cancellation Algorithm", IEEE Trans. AES, 2012
4. M. Richards, *Fundamentals of Radar Signal Processing*, 2nd Ed., McGraw-Hill, 2014
5. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking Systems*, Artech House, 1999

### Technical

- KrakenSDR: https://www.krakenrf.com/
- GNU Radio: https://www.gnuradio.org/
- FFTW3: http://www.fftw.org/
- VOLK: https://www.libvolk.org/

---

**Author**: Dr. Robert W McGwier, PhD, N4HY

Claude wrote every test and all documentation. The tests enabled diagnosis of code which was written by hand.

Last updated: 2026-02-08
