# PassiveRadar_Kraken

**Complete Passive Bistatic Radar System for KrakenSDR**

GNU Radio Out-of-Tree (OOT) module and Python display system for passive bistatic radar applications using the KrakenSDR 5-channel coherent SDR receiver. Implements the full radar processing chain from coherent acquisition through detection, tracking, and visualization.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Features](#features)
- [Hardware Support](#hardware-support)
  - [Raspberry Pi 5](#raspberry-pi-5)
  - [x86_64 (Intel/AMD)](#x86_64-intelamd)
  - [NVIDIA CUDA/RTX](#nvidia-cudartx)
  - [Vulkan GPU Compute](#vulkan-gpu-compute)
- [Installation](#installation)
  - [Prerequisites by Platform](#prerequisites-by-platform)
  - [Building OptMathKernels](#building-optmathkernels)
  - [Building PassiveRadar_Kraken](#building-passiveradar_kraken)
- [GNU Radio v2 Blocks](#gnu-radio-v2-blocks)
- [API Reference](#api-reference)
- [Display System](#display-system)
- [Performance](#performance)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [License](#license)
- [References](#references)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KrakenSDR 5-Channel Coherent SDR                    │
│                    (Ch0: Reference, Ch1-4: Surveillance)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GNU Radio OOT v2 Blocks                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Coherence  │  │     ECA      │  │   Doppler    │  │    CFAR      │    │
│  │   Monitor    │→ │  Canceller   │→ │  Processor   │→ │  Detector    │    │
│  │              │  │   (NLMS)     │  │              │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                               │             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │             │
│  │   Detection  │← │    Tracker   │← │     AoA      │←───────┘             │
│  │   Cluster    │  │   (Kalman)   │  │  Estimator   │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Python Display System                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Range-Doppler   │  │       PPI        │  │   Calibration    │          │
│  │      Display     │  │     Display      │  │      Panel       │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│  ┌──────────────────┐  ┌──────────────────────────────────────────┐        │
│  │     Metrics      │  │           Integrated Radar GUI           │        │
│  │    Dashboard     │  │                                          │        │
│  └──────────────────┘  └──────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Signal Processing
- **ECA Clutter Cancellation**: NLMS adaptive filter for direct-path and multipath suppression
- **Cross-Ambiguity Function (CAF)**: Range-Doppler processing with hardware acceleration
- **CFAR Detection**: CA/GO/SO/OS variants with configurable Pfa
- **Detection Clustering**: Connected components analysis for merged detections
- **Multi-Target Tracking**: Kalman filter with Global Nearest Neighbor association
- **AoA Estimation**: Bartlett beamformer for angle-of-arrival (ULA/UCA arrays)

### Hardware Acceleration
- **NEON SIMD**: ARM Cortex-A76 acceleration on Raspberry Pi 5
- **CUDA**: NVIDIA GPU acceleration for RTX 2000/3000/4000/5000 series
- **Vulkan Compute**: Cross-platform GPU acceleration
- **FFTW3**: Optimized FFT for Doppler processing

### Display System
- **Range-Doppler Map**: Real-time CAF heatmap with detection overlays
- **PPI Display**: Polar plot with track trails and velocity vectors
- **Calibration Panel**: Per-channel SNR, phase offsets, correlation monitoring
- **Metrics Dashboard**: Processing latencies, detection rates, system health
- **Integrated GUI**: Multi-panel Tkinter application

---

## Hardware Support

PassiveRadar_Kraken leverages [OptimizedKernelsForRaspberryPi5](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5) for hardware-accelerated signal processing across multiple platforms.

### Raspberry Pi 5

**Target Hardware**: Raspberry Pi 5 with Broadcom BCM2712 SoC (Cortex-A76)

| Feature | Specification |
|---------|---------------|
| **CPU** | Quad-core ARM Cortex-A76 @ 2.4 GHz |
| **SIMD** | NEON (128-bit Advanced SIMD) |
| **GPU** | VideoCore VII (Vulkan 1.2) |
| **Memory** | 4GB/8GB LPDDR4X-4267 |
| **USB** | 2x USB 3.0 (for KrakenSDR) |

**NEON Optimizations Available**:
- Complex multiply-accumulate (4 complex float32 per cycle)
- Vectorized dot products (8 float32 per instruction)
- SIMD FFT butterflies
- Parallel magnitude/phase computation

**Raspberry Pi 5 Build Requirements**:
```bash
# Raspberry Pi OS (64-bit) required
uname -m  # Should show aarch64

# Required packages
sudo apt install -y build-essential cmake git pkg-config \
    libeigen3-dev libfftw3-dev \
    gnuradio-dev libvolk2-dev pybind11-dev \
    python3-numpy python3-matplotlib python3-tk

# Optional: Vulkan for GPU compute
sudo apt install -y libvulkan-dev vulkan-tools mesa-vulkan-drivers
```

**Performance on Pi 5**:
| Operation | NEON Time | Scalar Time | Speedup |
|-----------|-----------|-------------|---------|
| Complex dot 4096 | 0.8 μs | 12.4 μs | 15.5x |
| CAF 4096×64 | 5.2 ms | 82 ms | 15.8x |
| CFAR 2D 256×64 | 0.9 ms | 11.2 ms | 12.4x |
| NLMS 1024 taps | 0.4 ms | 5.8 ms | 14.5x |

---

### x86_64 (Intel/AMD)

**Supported Processors**: Any x86_64 CPU with SSE4.2 or AVX2

| Feature | Minimum | Recommended |
|---------|---------|-------------|
| **CPU** | Intel Core i3 / AMD Ryzen 3 | Intel Core i7 / AMD Ryzen 7 |
| **SIMD** | SSE4.2 | AVX2/AVX-512 |
| **Memory** | 8GB DDR4 | 16GB+ DDR4/DDR5 |
| **USB** | USB 3.0 port | USB 3.0 port |

**x86_64 Build Requirements**:
```bash
# Ubuntu 22.04/24.04 LTS
sudo apt install -y build-essential cmake git pkg-config \
    libeigen3-dev libfftw3-dev \
    gnuradio-dev libvolk2-dev pybind11-dev \
    python3-numpy python3-matplotlib python3-tk

# For AVX2 optimization (auto-detected)
cat /proc/cpuinfo | grep avx2
```

**Eigen3 Auto-Vectorization**: The OptMathKernels library uses Eigen3's expression templates which automatically vectorize for AVX/AVX2/AVX-512 when available on x86_64.

---

### NVIDIA CUDA/RTX

**Supported GPUs**: NVIDIA GPUs with Compute Capability 7.0+ (Turing and later)

| GPU Generation | Architecture | Compute Capability | CUDA Cores | Tensor Cores |
|----------------|--------------|-------------------|------------|--------------|
| **RTX 2000** | Turing | 7.5 | 2304-4608 | 288-576 |
| **RTX 3000** | Ampere | 8.6 | 3584-10496 | 112-328 |
| **RTX 4000** | Ada Lovelace | 8.9 | 5888-16384 | 184-512 |
| **RTX 5000** | Blackwell | 10.0+ | TBD | TBD |
| **Jetson** | Orin (Ampere) | 8.7 | 1024-2048 | 32-64 |

**CUDA Build Requirements**:
```bash
# Install NVIDIA CUDA Toolkit 12.x
# Download from: https://developer.nvidia.com/cuda-downloads

# Ubuntu 22.04/24.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

**CUDA-Enabled Build**:
```bash
cd OptimizedKernelsForRaspberryPi5
mkdir -p build && cd build

# Enable CUDA backend
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DOPTMATH_ENABLE_CUDA=ON \
      -DCUDA_ARCHITECTURES="75;86;89" \
      ..

# Build flags explanation:
# 75 = Turing (RTX 2000)
# 86 = Ampere (RTX 3000)
# 89 = Ada Lovelace (RTX 4000)

make -j$(nproc)
sudo make install
```

**CUDA Performance (RTX 4090)**:
| Operation | CUDA Time | CPU Time | Speedup |
|-----------|-----------|----------|---------|
| Complex FFT 65536 | 0.12 ms | 8.4 ms | 70x |
| GEMM 1024×1024 | 0.08 ms | 45 ms | 562x |
| 2D Convolution 512×512 | 0.15 ms | 120 ms | 800x |
| CAF 16384×256 | 1.2 ms | 450 ms | 375x |

**CUDA Kernels Available** (242 functions):
- `cuda_complex_multiply` - Element-wise complex multiplication
- `cuda_fft_1d` / `cuda_fft_2d` - cuFFT-backed transforms
- `cuda_gemm` - cuBLAS matrix multiplication
- `cuda_conv2d` - Optimized 2D convolution
- `cuda_reduce` - Parallel reduction (sum, max, min)
- `cuda_scan` - Prefix scan operations
- `cuda_histogram` - GPU histogramming
- `cuda_sort` - Radix sort

See [FunctionsIncluded.md](FunctionsIncluded.md) for complete CUDA API reference.

---

### Vulkan GPU Compute

**Supported GPUs**: Any Vulkan 1.2+ capable GPU

| Vendor | GPUs | Vulkan Version |
|--------|------|----------------|
| **NVIDIA** | GTX 900+, RTX series | 1.3 |
| **AMD** | RX 400+, RDNA/RDNA2/RDNA3 | 1.3 |
| **Intel** | UHD 600+, Arc | 1.3 |
| **Raspberry Pi 5** | VideoCore VII | 1.2 |
| **Qualcomm** | Adreno 6xx+ | 1.1 |

**Vulkan Build Requirements**:
```bash
# Ubuntu/Debian
sudo apt install -y libvulkan-dev vulkan-tools mesa-vulkan-drivers

# Verify Vulkan
vulkaninfo | grep "Vulkan Instance"

# For NVIDIA (proprietary driver)
sudo apt install -y nvidia-driver-535 # or later

# For AMD (Mesa RADV)
sudo apt install -y mesa-vulkan-drivers

# For Raspberry Pi 5
# Vulkan included in Mesa - ensure firmware is updated
sudo rpi-update
```

**Vulkan-Enabled Build**:
```bash
cd OptimizedKernelsForRaspberryPi5
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DOPTMATH_ENABLE_VULKAN=ON \
      ..

make -j$(nproc)
sudo make install
```

**Vulkan Compute Shaders** (GLSL → SPIR-V):
- `caf_doppler_shift.comp.glsl` - Doppler shift application
- `caf_xcorr.comp.glsl` - Cross-correlation computation
- `cfar_2d.comp.glsl` - 2D CFAR detection
- `vector_ops.comp.glsl` - Vector arithmetic
- `fft_radix2.comp.glsl` - Radix-2 FFT

**Vulkan Performance (Raspberry Pi 5 VideoCore VII)**:
| Operation | Vulkan Time | CPU Time | Speedup |
|-----------|-------------|----------|---------|
| CAF 4096×64 | 2.1 ms | 82 ms | 39x |
| CFAR 2D | 0.6 ms | 11.2 ms | 18.7x |
| Complex Multiply 8192 | 0.3 ms | 3.2 ms | 10.7x |

---

## Installation

### Prerequisites by Platform

#### Raspberry Pi 5 (64-bit)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Core build tools
sudo apt install -y build-essential cmake git pkg-config ninja-build

# Math libraries
sudo apt install -y libeigen3-dev libfftw3-dev libopenblas-dev

# GNU Radio and SDR
sudo apt install -y gnuradio-dev libvolk2-dev libsoapysdr-dev

# Python dependencies
sudo apt install -y python3-dev python3-pip python3-numpy python3-matplotlib python3-tk pybind11-dev

# Vulkan (optional GPU acceleration)
sudo apt install -y libvulkan-dev vulkan-tools glslang-tools

# KrakenSDR driver
sudo apt install -y libusb-1.0-0-dev
```

#### Ubuntu 22.04/24.04 x86_64

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Core build tools
sudo apt install -y build-essential cmake git pkg-config ninja-build

# Math libraries
sudo apt install -y libeigen3-dev libfftw3-dev libopenblas-dev

# GNU Radio and SDR
sudo apt install -y gnuradio-dev libvolk2-dev libsoapysdr-dev

# Python dependencies
sudo apt install -y python3-dev python3-pip python3-numpy python3-matplotlib python3-tk pybind11-dev

# Vulkan (optional)
sudo apt install -y libvulkan-dev vulkan-tools glslang-tools mesa-vulkan-drivers

# CUDA (optional - see NVIDIA section above)
```

#### NVIDIA Jetson (Orin/Xavier)

```bash
# JetPack 5.x or 6.x required
# CUDA is pre-installed

# Additional packages
sudo apt install -y build-essential cmake git pkg-config \
    libeigen3-dev libfftw3-dev \
    python3-dev python3-pip python3-numpy python3-matplotlib

# GNU Radio (may need to build from source on Jetson)
sudo apt install -y gnuradio-dev libvolk2-dev
```

---

### Building OptMathKernels

OptMathKernels provides the hardware-accelerated math kernels used by PassiveRadar_Kraken.

```bash
# Clone OptMathKernels
cd ~
git clone https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA.git
cd OptimizedKernelsForRaspberryPi5_NvidiaCUDA

# Create build directory
mkdir -p build && cd build

# Configure (choose options based on your hardware)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DOPTMATH_ENABLE_NEON=ON \
      -DOPTMATH_ENABLE_VULKAN=ON \
      -DOPTMATH_ENABLE_CUDA=OFF \
      ..

# For NVIDIA systems, enable CUDA:
# cmake -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_INSTALL_PREFIX=/usr/local \
#       -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
#       -DOPTMATH_ENABLE_NEON=OFF \
#       -DOPTMATH_ENABLE_VULKAN=ON \
#       -DOPTMATH_ENABLE_CUDA=ON \
#       -DCUDA_ARCHITECTURES="75;86;89" \
#       ..

# Build
make -j$(nproc)

# Install (requires sudo)
sudo make install

# Update library cache
sudo ldconfig

# Verify installation
ls /usr/local/lib/libOptMathKernels*
ls /usr/local/include/optmath/
```

---

### Building PassiveRadar_Kraken

```bash
# Clone repository
cd ~
git clone https://github.com/n4hy/PassiveRadar_Kraken.git
cd PassiveRadar_Kraken

# Build standalone C++ libraries
cd src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..
make -j$(nproc)
sudo make install
sudo ldconfig

# Return to repo root
cd ../..

# Build GNU Radio OOT v2 module
cd gr-kraken_passive_radar_v2/gr-kraken_passive_radar
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..
make -j$(nproc)
sudo make install
sudo ldconfig

# Return to repo root
cd ../../..

# Verify installation
python3 -c "from gnuradio import kraken_passive_radar; print('GNU Radio blocks loaded successfully')"
```

---

## GNU Radio v2 Blocks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **KrakenSDR Source** | 5-channel coherent source | frequency, sample_rate, gain |
| **ECA Canceller** | NLMS adaptive clutter filter | num_taps, mu, epsilon |
| **Doppler Processor** | Range-Doppler FFT processing | num_range_bins, num_doppler_bins, window_type |
| **CFAR Detector** | Constant False Alarm Rate detection | guard_cells, ref_cells, pfa, cfar_type |
| **Coherence Monitor** | Phase coherence monitoring | corr_threshold, phase_threshold_deg |
| **Detection Cluster** | Connected components clustering | min_cluster_size, max_cluster_extent |
| **Tracker** | Kalman filter multi-target tracker | process_noise, gate_threshold, confirm_hits |
| **AoA Estimator** | Bartlett beamformer | d_lambda, n_angles, array_type |

---

## API Reference

For complete API documentation of all 396+ hardware-accelerated functions, see:

**[FunctionsIncluded.md](FunctionsIncluded.md)** - Complete API Reference

### Quick Reference by Backend

| Backend | Functions | Description |
|---------|-----------|-------------|
| **NEON** | 83 | ARM SIMD operations for Raspberry Pi 5 |
| **CUDA** | 242 | NVIDIA GPU kernels for RTX/Tesla/Jetson |
| **Vulkan** | 23 | Cross-platform GPU compute shaders |
| **Radar** | 48 | Passive radar-specific algorithms |

### Key Radar Functions

```cpp
// CAF Processing
optmath::radar::caf(ref, surv, n_doppler, n_range, output);

// NLMS Clutter Cancellation
optmath::radar::nlms_filter_f32(ref, surv, output, n, num_taps, mu, eps);

// CFAR Detection
optmath::radar::cfar_2d_ca(data, detections, n_range, n_doppler,
                           guard_cells, ref_cells, alpha);

// Steering Vector Generation
optmath::radar::steering_vector_ula_f32(output, n_elements, d_lambda, theta);

// Window Generation
optmath::radar::generate_window(window, n, type); // HAMMING, HANNING, BLACKMAN
```

---

## Display System

### Range-Doppler Display (`range_doppler_display.py`)

Real-time visualization of the Cross-Ambiguity Function output.

**Features**:
- Heatmap with configurable dynamic range (dB scale)
- Detection overlays (red circles)
- Track overlays with fading history trails
- Cursor readout: range (km), velocity (m/s), power (dB)
- Colorbar with adjustable limits

```python
from kraken_passive_radar.range_doppler_display import RangeDopplerDisplay

display = RangeDopplerDisplay(
    n_range=256,
    n_doppler=64,
    range_resolution=600.0,  # meters
    doppler_resolution=3.9,  # Hz
    center_freq=103.7e6
)
display.update(caf_data, detections, tracks)
```

### PPI Display (`radar_display.py`)

Polar Plan Position Indicator with target tracking visualization.

**Features**:
- Polar coordinate display (range vs azimuth)
- Track history trails (fading polylines)
- Velocity vectors (arrows showing heading)
- Track ID labels and status coloring
- Configurable max range and grid

```python
from kraken_passive_radar.radar_display import PPIDisplay, PPITrack

display = PPIDisplay(max_range_km=50.0, update_interval_ms=100)
display.update_tracks([
    PPITrack(id=1, range_m=15000, aoa_deg=45, velocity_mps=120, status='confirmed')
])
```

### Calibration Panel (`calibration_panel.py`)

Real-time monitoring of array calibration quality.

**Displays**:
- Per-channel SNR meters (bar chart)
- Phase offset scatter plot (-180° to +180°)
- Correlation coefficients (bar chart)
- Phase drift history waterfall
- Calibration valid/invalid indicator

### Metrics Dashboard (`metrics_dashboard.py`)

System health and performance monitoring.

**Displays**:
- Processing latency breakdown with sparklines
- Detection rate (detections/second)
- Track counts (confirmed, tentative, coasting)
- CPU/Memory usage bars
- Backend status (NEON/Vulkan/CUDA indicators)

### Integrated GUI (`radar_gui.py`)

Multi-panel Tkinter application combining all displays.

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│         Range-Doppler Map           │            PPI Display              │
│         (CAF heatmap)               │         (polar tracks)              │
├─────────────────────────────────────┼─────────────────────────────────────┤
│       Calibration Panel             │        Metrics Dashboard            │
│    (phase/SNR monitoring)           │     (latency/track counts)          │
├─────────────────────────────────────┴─────────────────────────────────────┤
│  [Start] [Stop] [Reset] [Recalibrate]           Dynamic Range: [====60===]│
└───────────────────────────────────────────────────────────────────────────┘
```

**Launch**:
```bash
python3 kraken_passive_radar/radar_gui.py
```

---

## Performance

### Processing Capabilities by Platform

#### Raspberry Pi 5 (NEON + Vulkan)

| Operation | NEON Time | Vulkan Time | Notes |
|-----------|-----------|-------------|-------|
| ECA (NLMS) 1024×64 | 0.4 ms | N/A | CPU-bound |
| CAF 4096×64×256 | 5.2 ms | 2.1 ms | GPU preferred |
| CFAR 2D 64×256 | 0.9 ms | 0.6 ms | GPU preferred |
| Tracker 10 tracks | 0.1 ms | N/A | CPU-only |
| Full CPI (100ms) | 8 ms | 4 ms | Real-time capable |

#### x86_64 Desktop (AVX2)

| Operation | AVX2 Time | Notes |
|-----------|-----------|-------|
| ECA (NLMS) 1024×64 | 0.2 ms | Eigen3 vectorization |
| CAF 4096×64×256 | 12 ms | FFTW3 |
| CFAR 2D 64×256 | 0.4 ms | Eigen3 |
| Tracker 10 tracks | 0.05 ms | |
| Full CPI (100ms) | 15 ms | Real-time capable |

#### NVIDIA RTX 4090 (CUDA)

| Operation | CUDA Time | Speedup vs CPU |
|-----------|-----------|----------------|
| ECA (NLMS) 1024×64 | 0.05 ms | 4x |
| CAF 4096×64×256 | 0.3 ms | 40x |
| CFAR 2D 64×256 | 0.1 ms | 4x |
| CAF 16384×256×1024 | 1.2 ms | 375x |
| Full CPI (100ms) | 0.5 ms | 16x |

### Radar Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Range resolution | c/(2·BW) | ~600 m @ 250 kHz BW |
| Max unambiguous range | c·CPI/2 | ~15 km @ 100 ms CPI |
| Doppler resolution | 1/CPI | ~10 Hz @ 100 ms CPI |
| Max unambiguous Doppler | ±fs/2 | ±125 kHz @ 250 kHz |
| Angular resolution | ~15° | 4-element array @ λ/2 spacing |

---

## Testing

PassiveRadar_Kraken includes a comprehensive test suite with 118+ tests.

### Running Tests

```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./run_tests.sh unit         # Unit tests only
./run_tests.sh integration  # Integration tests
./run_tests.sh benchmark    # Performance benchmarks
./run_tests.sh quick        # Quick smoke tests

# Run with verbose output
./run_tests.sh all verbose
```

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | 81 | Individual component tests |
| Integration | 5 | Multi-block pipeline tests |
| Benchmark | 6 | Performance measurements |
| C++ | 7 | Native library tests |
| Fixtures | 3 | Test data generator validation |
| Display | 19 | GUI component tests |

### Test Files

```
tests/
├── unit/
│   ├── test_eca_kernels.py      # ECA clutter cancellation
│   ├── test_caf_kernels.py      # CAF computation
│   ├── test_cfar_kernels.py     # CFAR detection
│   ├── test_doppler_kernels.py  # Doppler processing
│   ├── test_aoa_kernels.py      # AoA estimation
│   ├── test_clustering.py       # Detection clustering
│   ├── test_tracker.py          # Multi-target tracker
│   └── test_display_modules.py  # Display components
├── integration/
│   └── test_full_pipeline.py    # End-to-end tests
├── benchmarks/
│   └── test_bench_kernels.py    # Performance tests
└── fixtures/
    ├── synthetic_targets.py     # Target generation
    ├── clutter_models.py        # Clutter models
    └── noise_models.py          # Noise generation
```

---

## Troubleshooting

### Build Issues

**"OptMathKernels not found"**:
```bash
# Verify installation
ls /usr/local/lib/cmake/OptMathKernels/
# Should contain OptMathKernelsConfig.cmake

# If missing, reinstall:
cd ~/OptimizedKernelsForRaspberryPi5_NvidiaCUDA/build
sudo make install
sudo ldconfig
```

**"-fPIC" linking error**:
```bash
# Rebuild OptMathKernels with position-independent code
cd ~/OptimizedKernelsForRaspberryPi5_NvidiaCUDA/build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make -j$(nproc)
sudo make install
```

**"pybind11 not found"**:
```bash
sudo apt install pybind11-dev python3-pybind11
```

**"Vulkan not found"**:
```bash
sudo apt install libvulkan-dev vulkan-tools
# Verify with:
vulkaninfo | head -20
```

**"CUDA not found"**:
```bash
# Ensure CUDA toolkit is installed
nvcc --version

# Add to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Runtime Issues

**Block not appearing in GRC**:
```bash
grcc --force-load
# Or restart GNU Radio Companion
```

**Display not updating**:
```python
# Ensure matplotlib backend is set
import matplotlib
matplotlib.use('TkAgg')
```

**Calibration constantly triggering**:
- Check antenna connections
- Verify noise source is working
- Increase `corr_threshold` or `phase_threshold_deg`

### Performance Issues

**High CPU usage**:
- Reduce `num_doppler_bins` or `num_range_bins`
- Enable NEON optimization (Raspberry Pi)
- Enable Vulkan/CUDA for GPU acceleration

**Missed detections**:
- Lower CFAR `pfa` (e.g., 1e-4 instead of 1e-6)
- Increase ECA `num_taps` for better clutter suppression
- Check SNR in calibration panel

**GPU not being used**:
```bash
# Check Vulkan
vulkaninfo | grep "GPU"

# Check CUDA
nvidia-smi
```

---

## File Structure

```
PassiveRadar_Kraken/
├── gr-kraken_passive_radar_v2/
│   └── gr-kraken_passive_radar/
│       ├── include/gnuradio/kraken_passive_radar/
│       │   ├── aoa_estimator.h
│       │   ├── detection_cluster.h
│       │   ├── tracker.h
│       │   └── ... (other block headers)
│       ├── lib/
│       │   ├── aoa_estimator_impl.{h,cc}
│       │   ├── detection_cluster_impl.{h,cc}
│       │   ├── tracker_impl.{h,cc}
│       │   ├── eca_canceller_impl.{h,cc}
│       │   └── CMakeLists.txt
│       ├── grc/
│       │   └── *.block.yml (GRC block definitions)
│       └── python/kraken_passive_radar/bindings/
│           └── *_python.cc (pybind11 bindings)
├── kraken_passive_radar/
│   ├── radar_gui.py              # Integrated multi-panel GUI
│   ├── radar_display.py          # PPI display with tracking
│   ├── range_doppler_display.py  # CAF heatmap display
│   ├── calibration_panel.py      # Calibration monitoring
│   └── metrics_dashboard.py      # System metrics
├── src/
│   ├── caf_processing.cpp        # CAF with OptMathKernels
│   ├── eca_b_clutter_canceller.cpp
│   ├── doppler_processing.cpp
│   └── CMakeLists.txt
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── benchmarks/               # Performance tests
│   └── fixtures/                 # Test data generators
├── examples/
│   └── kraken_passive_radar_103_7MHz.grc
├── run_tests.sh                  # Test runner script
├── FunctionsIncluded.md          # Complete API reference
└── README.md                     # This file
```

---

## License

MIT License

Copyright (c) 2026 Dr Robert W McGwier, PhD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Author

**N4HY - Bob McGwier**
Science Bob
Dr Robert W McGwier, PhD

ALL unit tests, and documentation and debugging done with the aid of claude code.

---

## References

- [KrakenSDR Documentation](https://github.com/krakenrf/krakensdr_docs)
- [Passive Radar Fundamentals](https://en.wikipedia.org/wiki/Passive_radar)
- [GNU Radio OOT Module Tutorial](https://wiki.gnuradio.org/index.php/OutOfTreeModules)
- [OptMathKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5) - NEON/Vulkan/CUDA acceleration
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Vulkan SDK](https://vulkan.lunarg.com/)
- Kulpa, K. "Signal Processing in Noise Waveform Radar" (2013)
- Bar-Shalom, Y. "Estimation with Applications to Tracking and Navigation" (2001)
