# KrakenSDR Passive Radar

A high-performance, GPU/CPU-accelerated Passive Radar system for the KrakenSDR 5-channel coherent receiver. This project implements a full signal processing pipeline from raw IQ ingestion to Range-Doppler map generation, capable of detecting aircraft and other moving targets using commercial FM broadcast or TV signals as illuminators of opportunity.

![Status](https://img.shields.io/badge/Status-Operational-green)
![Build](https://img.shields.io/badge/Build-CMake-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📡 Architecture & Signal Flow

The system is built on **GNU Radio** but relies heavily on custom **C++ Optimized OOT (Out-of-Tree)** blocks to achieve real-time performance. The flowgraph `kraken_passive_radar_system.grc` implements the following pipeline:

```mermaid
graph TD
    ANT[Antennas 0-4] --> KRAKEN[KrakenSDR Source]

    subgraph "Preprocessing"
        KRAKEN -- CH0 (Ref) --> FIR0[Freq Xlating FIR] --> DC0[DC Blocker]
        KRAKEN -- CH1..4 (Surv) --> FIR1[Freq Xlating FIR] --> DC1[DC Blocker]
    end

    subgraph "Clutter Cancellation"
        DC0 --> ECA_REF[ECA-B Reference In]
        DC1 --> ECA_SURV[ECA-B Surveillance In]

        ECA_REF & ECA_SURV --> ECA_BLOCK[ECA-B Clutter Canceller]

        note1[C++ Accelerated\nFast Covariance Algo\nO(MN) Complexity]
        ECA_BLOCK -.- note1
    end

    subgraph "Cross-Ambiguity Function (CAF)"
        ECA_BLOCK -- Clean Surv --> S_VEC[To Vector] --> S_FFT[FFT]
        DC0 -- Ref Copy --> R_VEC[To Vector] --> R_FFT[FFT]

        S_FFT & R_FFT --> MULT[Multiply Conjugate]
        MULT --> IFFT[IFFT (Range Profile)]
    end

    subgraph "Doppler Processing"
        IFFT --> DOPPLER[Doppler Processor]
        DOPPLER --> RASTER[Range-Doppler Map]

        note2[Accumulates M pulses\nPerforms Slow-Time FFT\nLog-Magnitude Output]
        DOPPLER -.- note2
    end
```

### Key Components

#### 1. KrakenSDR Source (`krakensdr_source`)
*   **Type:** Hierarchical Python Block
*   **Function:** Wraps the standard `osmosdr_source`. It explicitly maps the 5 physical ports (Serial 1000-1004) to logical channels 0-4.
*   **Optimization:** Uses large buffers (128 x 64kB) to prevent USB packet drops.

#### 2. ECA-B Clutter Canceller (`kraken_passive_radar_eca_b_clutter_canceller`)
*   **Type:** C++ Accelerated Block
*   **Function:** Removes the direct-path signal and static ground clutter from the surveillance channels.
*   **Algorithm:** **Extensive Cancellation Algorithm (Batch)** using a **Fast Covariance Update**.
    *   Instead of the naive $O(M^2 N)$ matrix calculation, it uses a recursive Toeplitz update $O(MN)$ to compute the autocorrelation matrix $R$.
    *   Solves the Wiener-Hopf equation $Rw = p$ using Cholesky Decomposition.
    *   Compiles with `-O3 -march=native` to utilize AVX/NEON vector instructions automatically.

#### 3. Doppler Processor (`kraken_passive_radar_doppler_processing`)
*   **Type:** C++ Accelerated Block
*   **Function:** Takes a stream of Range Profiles (output of IFFT), accumulates a block of pulses (Slow Time), and performs an FFT across the pulse dimension.
*   **Output:** A flattened vector representing the Range-Doppler map (Log Magnitude), suitable for visualization.

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
2.  Compile the C++ kernels (`libkraken_eca_b_clutter_canceller.so`, `libkraken_doppler_processing.so`) with **Release** optimizations.
3.  Install the Python blocks and GRC definitions to `/usr/local/`.

**Note:** If you modify the C++ code or GRC block definitions, run `./reinstall_oot.sh` to update the system.

---

## 🚀 Usage

### 1. Monitor Input Signals
Verify that all 5 channels are active and not saturated.

1.  Open `kraken_sdr_5ch_monitor.grc` in GNU Radio Companion.
2.  Run the flowgraph.
3.  Adjust **RF Gain** (default 10-14 dB). Ensure the Reference signal (CH0) is strong but not clipping.

### 2. Run Passive Radar
1.  Open `kraken_passive_radar_system.grc`.
2.  **Parameters:**
    *   `freq`: Tuning frequency (e.g., 98.5 MHz for FM).
    *   `eca_taps`: Number of filter taps (Default: 16).
    *   `doppler_len`: Number of pulses for Doppler integration (Default: 128).
3.  Run the flowgraph.
4.  **Visualization:** A "Range-Doppler (CAF)" window will appear.
    *   **X-Axis:** Range Bins (Distance).
    *   **Y-Axis:** Doppler Bins (Velocity).
    *   **Bright Spots:** Detected targets.

---

## 🧪 Testing

The repository includes a suite of unit tests to verify the signal processing kernels.

```bash
# Run all tests
./run_tests.sh
```

**Key Tests:**
*   `tests/test_eca_b_cpp.py`: Verifies the C++ ECA-B kernel achieves >10dB clutter suppression on synthetic data.
*   `tests/test_doppler_cpp.py`: Verifies the Doppler processing FFT logic.
*   `tests/test_krakensdr_source.py`: Checks the Python source block logic.

---

## ⚡ Performance Optimizations

*   **Fast Covariance Algorithm:** The ECA-B block uses a specialized algorithm to update the covariance matrix $R$ in linear time relative to the number of taps, enabling real-time processing even with large tap counts (e.g., 64).
*   **Zero-Copy Logic:** The Python wrappers (`eca_b_clutter_canceller.py`) pass pointers directly to C++ to avoid memory copying overhead.
*   **Vectorization:** The build system forces `-O3 -march=native -ffast-math`, allowing the compiler to auto-vectorize loops using AVX (x86) or NEON (ARM) instructions.

---

## ⚠️ Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **"PLL not locked"** | Insufficient Power | Use a powered USB hub (3A+). |
| **"Failed to allocate zero-copy buffer"** | Kernel Limit | Run `setup_krakensdr_permissions.sh`. |
| **"O" (Overflow) printing to console** | CPU Overload | Reduce sample rate or `eca_taps`. Ensure you ran `./build_oot.sh` (Optimized build). |
| **FlowGraph Error: 'file_format'** | Old GRC | Ensure GRC is 3.10+. |

