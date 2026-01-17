# KrakenSDR Passive Radar

A high-performance, GPU/CPU-accelerated Passive Radar system for the KrakenSDR 5-channel coherent receiver. This project implements a full signal processing pipeline from raw IQ ingestion to Range-Doppler map generation, capable of detecting aircraft and other moving targets using commercial FM broadcast or TV signals as illuminators of opportunity.

![Status](https://img.shields.io/badge/Status-Operational-green)
![Build](https://img.shields.io/badge/Build-CMake-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📡 Architecture & Signal Flow

The system is built on **GNU Radio** but relies heavily on custom **C++ Optimized OOT (Out-of-Tree)** blocks to achieve real-time performance. The architecture follows a strict policy: **All signal processing mathematics are performed in compiled C++ kernels**, while Python is used only for orchestration (glue logic).

### Signal Processing Pipeline (C++)

1.  **Time Alignment (`src/time_alignment.cpp`)**:
    *   Calibrates sample delay and phase offsets between channels using cross-correlation.
2.  **Conditioning (`src/conditioning.cpp`)**:
    *   Applies AGC (Automatic Gain Control) to normalize Reference and Surveillance channels.
3.  **ECA-B Clutter Canceller (`src/eca_b_clutter_canceller.cpp`)**:
    *   Removes direct-path signal and static ground clutter.
    *   Uses **Fast Covariance Update (Toeplitz)** algorithm ($O(MN)$) for efficiency.
    *   Fixes historical index alignment for precise cancellation.
4.  **CAF Processing (`src/caf_processing.cpp`)**:
    *   Computes Range Profiles using FFT-based Cross-Correlation (Cross-Ambiguity Function).
    *   Uses **Overlap-Save** logic or blocked FFTs.
5.  **Doppler Processing (`src/doppler_processing.cpp`)**:
    *   Accumulates pulses and performs Slow-Time FFT.
    *   Produces Range-Doppler maps.
6.  **Backend (`src/backend.cpp`)**:
    *   **Fusion:** Non-coherently combines maps from 4 surveillance channels.
    *   **CFAR:** Performs 2D Constant False Alarm Rate detection.

---

## 🚀 Running the Radar

The primary way to run the radar is via the **pure Python orchestration script**, which bypasses GNU Radio Companion to ensure strict control over the pipeline execution.

### `run_passive_radar.py`

This script instantiates the full processing chain without GRC.

```bash
# Start the radar
python3 run_passive_radar.py
```

*   **Inputs:** Connects to KrakenSDR via `krakensdr_source`.
*   **Outputs:**
    *   Prints detection statistics to console.
    *   Writes a visualization of the Range-Doppler map to `passive_radar_map.ppm` (viewable with any image viewer supporting PPM).
    *   Logs ECA statistics to `eca_stats.txt`.

---

## 🛠️ Hardware Setup

1.  **Power Supply:** A **Powered USB 3.0 Hub (3A+)** is **REQUIRED**. The KrakenSDR draws ~2.2A, exceeding standard port limits.
2.  **Antennas:** Connect 5 matched antennas to ports CH0-CH4.
3.  **Host:** Linux PC (x86_64) or Raspberry Pi 4/5 (aarch64).

### System Configuration
You must configure USB permissions and memory limits before running the software.

```bash
# Run once
sudo ./setup_krakensdr_permissions.sh
```
*   **Blacklists** `dvb_usb_rtl28xxu` kernel driver.
*   **Sets** `usbfs_memory_mb` to 0 (unlimited) to support 5 concurrent streams.
*   **Reboot** after running.

---

## 📦 Installation

### 1. Install Dependencies
```bash
sudo apt update
sudo apt install -y gnuradio gr-osmosdr python3-numpy python3-pyqt5 g++ cmake make libfftw3-dev soapysdr-tools
```

### 2. Build & Install OOT Module
The project contains custom C++ blocks that must be compiled.

```bash
./build_oot.sh
```
This script will:
1.  Detect your Python installation.
2.  Compile the C++ kernels (`libkraken_eca_b_clutter_canceller.so`, `libkraken_doppler_processing.so`, etc.) with **Release** optimizations (`-O3 -march=native -ffast-math`).
3.  Install the Python blocks and GRC definitions to `/usr/local/`.

**Note:** If you modify the C++ code or GRC block definitions, run `./reinstall_oot.sh` to update the system.

---

## 🧪 Testing

The repository includes a suite of unit tests to verify the signal processing kernels and the full pipeline.

```bash
# Run all tests
./run_tests.sh
```

**Key Tests:**
*   **`tests/test_end_to_end.py` (Offline Pipeline Verification):**
    *   Constructs the full pipeline (Conditioning -> ECA -> CAF -> Doppler -> CFAR) using synthetic data.
    *   Injects a simulated target at specific Range/Doppler bins.
    *   Verifies that the system correctly detects the target and produces a valid CFAR mask.
    *   **Output:** Generates `offline_test_map.ppm` showing the detection heatmap.
*   `tests/test_eca_b_cpp.py`: Verifies the C++ ECA-B kernel achieves >10dB clutter suppression.
*   `tests/test_caf_cpp.py`: Verifies the CAF kernel correctly computes cross-correlation lags.
*   `tests/test_doppler_cpp.py`: Verifies the Doppler processing FFT logic.

---

## ⚡ Performance Optimizations

*   **Fast Covariance Algorithm:** The ECA-B block uses a specialized algorithm to update the covariance matrix $R$ in linear time relative to the number of taps.
*   **Zero-Copy Logic:** The Python wrappers in `kraken_passive_radar/custom_blocks.py` pass pointers directly to C++ to avoid memory copying overhead.
*   **Vectorization:** The build system forces `-O3 -march=native -ffast-math`, allowing the compiler to auto-vectorize loops using AVX (x86) or NEON (ARM) instructions.

---

## ⚠️ Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **"PLL not locked"** | Insufficient Power | Use a powered USB hub (3A+). |
| **"Failed to allocate zero-copy buffer"** | Kernel Limit | Run `setup_krakensdr_permissions.sh`. |
| **"O" (Overflow) printing to console** | CPU Overload | Reduce sample rate or `eca_taps`. Ensure you ran `./build_oot.sh` (Optimized build). |
| **OSError: libkraken_... not found** | Missing Build | Run `./build_oot.sh` to compile C++ libraries. |
