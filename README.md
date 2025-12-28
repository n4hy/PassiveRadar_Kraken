# PassiveRadar_Kraken

PassiveRadar_Kraken is a GNU Radio–based passive radar processing chain designed for use with the KrakenSDR (5 coherent channels: 4 surveillance + 1 reference).  
The primary use case is **aircraft detection and tracking** using high‑power illuminators of opportunity (FM broadcast and digital/analog TV transmitters) as the non‑cooperative transmitters.

The project provides:

- A GNU Radio OOT module: **`gr-kraken_passive_radar`**
- Python blocks for:
  - **NLMS adaptive clutter cancellation** (`ClutterCanceller`)
  - **ECA‑B clutter cancellation** (`EcaBClutterCanceller`, driven by a C++ kernel)
  - Doppler and angle‑of‑arrival (AoA) processing (via C++ backends)
- GNU Radio Companion (GRC) blocks under the category **`[Kraken Passive Radar]`**

Once installed, the user can construct a passive radar processing chain in GRC using the Kraken-specific blocks and standard GNU Radio / gr‑osmosdr source blocks.

---

## 1. Getting Started (Hardware & Setup)

### 1.1 Hardware Requirements

- **KrakenSDR**: 5-channel coherent RTL-SDR array.
- **Powered USB Hub**: A high-quality hub with at least a **3A power adapter** is **MANDATORY**. The KrakenSDR draws ~2.2A, which exceeds the limit of standard laptop USB ports. Insufficient power causes "PLL not locked" errors and signal drops.
- **Antennas**: 5 antennas connected to SMA ports CH0-CH4.
- **Host Computer**: Linux machine (x86_64 or Raspberry Pi 4/5).

### 1.2 Channel Mapping

The software assumes a standard 1-to-1 mapping based on the KrakenSDR's factory-programmed serial numbers:

*   **Software Channel 0** <--> **Physical Port CH0** (Serial 1000)
*   **Software Channel 1** <--> **Physical Port CH1** (Serial 1001)
*   **Software Channel 2** <--> **Physical Port CH2** (Serial 1002)
*   **Software Channel 3** <--> **Physical Port CH3** (Serial 1003)
*   **Software Channel 4** <--> **Physical Port CH4** (Serial 1004)

### 1.3 System Configuration Script

A setup script is provided to configure permissions and USB limits. You must run this once:

```bash
cd ~/PassiveRadar_Kraken
sudo ./setup_krakensdr_permissions.sh
```

This script performs three critical actions:
1.  **Blacklists Kernel Driver**: Prevents `dvb_usb_rtl28xxu` from claiming the device.
2.  **Sets udev Rules**: Grants permission to access the USB device without root.
3.  **Increases USB Memory Limit**: Sets `usbfs_memory_mb` to 0 (unlimited). **This is required** to stream 5 channels simultaneously without "Failed to allocate zero-copy buffer" errors.

**Reboot** after running the script to ensure all changes take effect.

---

## 2. Software Installation

The project is designed for GNU Radio 3.10+ installed from distribution packages.

### 2.1 Install Dependencies

```bash
sudo apt update
sudo apt install -y gnuradio gr-osmosdr python3-numpy python3-pyqt5 python3-pyqt5.qtsvg g++ cmake make libfftw3-dev soapysdr-tools
```

### 2.2 Build and Install OOT Module

Use the helper script to build and install the custom Kraken blocks:

```bash
cd ~/PassiveRadar_Kraken
./build_oot.sh
sudo make install
sudo ldconfig
```

---

## 3. Running the Radar

### 3.1 Verify 5-Channel Input

Before running the full radar, verify your signals using the 5-channel monitor flowgraph.

1.  Open **GNU Radio Companion**:
    ```bash
    gnuradio-companion kraken_sdr_5ch_monitor.grc
    ```
2.  Click the **Generate** button (F5) to create the Python script.
3.  Run the monitor:
    ```bash
    python3 kraken_sdr_5ch_monitor.py
    ```
4.  **Tuning**: Use the "RF Gain" variable to adjust gain (Default 10dB). If you are near a strong FM station, keep gain low to avoid overloading Channel 0 (Reference).

### 3.2 Running the Passive Radar

The main radar application computes Range-Doppler maps and cancels clutter.

```bash
python3 kraken_passive_radar_top_block.py
```

---

## 4. Troubleshooting Common Issues

### 4.1 "PLL not locked" or "Flat Line" Signal
*   **Cause**: Insufficient power.
*   **Fix**: Use a **Powered USB Hub** (3A+). Laptop ports are not strong enough.

### 4.2 "Failed to allocate zero-copy buffer"
*   **Cause**: Linux kernel USB memory limit is too low for 5 channels.
*   **Fix**: Run `sudo ./setup_krakensdr_permissions.sh` to increase `usbfs_memory_mb`.

### 4.3 "Channel 0 is dead" (but others work)
*   **Cause**: Signal overload (saturation) from a strong local transmitter.
*   **Fix**: Reduce the **RF Gain** to 10 dB or lower.

### 4.4 GRC Warning: "Flow graph may not have flow control"
*   **Cause**: GRC static analysis doesn't see inside the custom `krakensdr_source` hierarchical block.
*   **Fix**: Ignore this warning. As long as you don't see "O" (Overflow) or "U" (Underrun) in the console, the flowgraph is timing correctly.

---

## 5. Architecture Overview

A typical FM/TV‑based passive radar chain using the KrakenSDR looks like:

1. **KrakenSDR Source (gr‑osmosdr)**
   - 5 coherent channels: 4 surveillance + 1 reference
   - Common sample rate (e.g., 2.048 Msps) and center frequency tuned to the illuminator

2. **Per‑channel preprocessing**
   - Complex gain adjustment and DC offset removal (if needed)
   - Optional band‑limiting around the selected FM/TV carrier

3. **Polyphase resampling**
   - High‑rate input (e.g., 2.048 Msps) decimated via a polyphase resampler to a lower analysis rate (e.g., 175 kHz) suitable for CAF / range–Doppler processing

4. **Adaptive clutter cancellation**
   - **NLMS**: `Kraken NLMS Clutter Canceller`
   - **ECA‑B**: `Kraken ECA-B Clutter Canceller`
   - Each surveillance channel uses the reference channel as input and outputs a clutter‑reduced surveillance (error signal)

5. **Cross‑ambiguity and Doppler processing**
   - Range–Doppler map computation (C++ backend + Python front‑end)
   - AoA processing across multiple surveillance channels (C++ backend)

6. **Display and logging**
   - Range–Doppler visualization
   - Track extraction and export to downstream systems (e.g., ADS‑B fusion, custom track server)

---

## 6. Optimizations (NEON)

This repository includes NEON-accelerated kernels for ARM platforms (e.g., Raspberry Pi 4/5). These optimizations significantly improve the performance of:

1.  **ECA-B Clutter Cancellation**: The core matrix operations (autocorrelation and cross-correlation) utilize NEON SIMD instructions to accelerate complex number arithmetic.
2.  **Polyphase Resampler**: The FIR filtering process is optimized using NEON vector operations for high-throughput decimation. Note: The resampler implementation has been switched to `float` precision to maximize performance on ARM hardware.

---

## 7. ITAR / Export Control Considerations (Informational, Not Legal Advice)

This repository provides **software signal processing components** for passive radar using broadcast FM/TV illuminators and KrakenSDR hardware. As implemented here:

- The code:
  - Operates on **publicly available broadcast waveforms** (FM radio, TV)
  - Uses standard adaptive filtering and correlation methods widely known in the open literature
  - Is not coupled to any specific weapons system or classified sensor

- Typical use:
  - Aircraft detection and tracking for situational awareness, research, and spectrum monitoring
  - Operation with **commercially available hardware** and **commercial broadcast emitters**

Based on these characteristics, the software is **likely to be treated as EAR99** (catch‑all classification) under U.S. export control, and in many cases can be shared publicly (e.g., on GitHub) without a specific export license. However:

- This is **not legal advice**.
- Final jurisdiction and classification determinations rest with the relevant export control authorities (e.g., U.S. BIS/DDTC).
- If you intend to deploy this system in a defense, government, or international context, you should seek an official classification (e.g., a Commodity Jurisdiction request) and/or consult with export control counsel.

The repository is intended for **research and commercial non‑military applications**, and it is licensed under a permissive open‑source license (e.g., MIT or BSD, as declared in the LICENSE file).

---

## 8. License

See the `LICENSE` file in the repository for the full license text. In typical use, this project is distributed under a permissive license (e.g., MIT), allowing:

- Use, modification, and redistribution
- Inclusion in larger systems
- Commercial and non‑commercial usage

subject to the conditions specified in the LICENSE file.
