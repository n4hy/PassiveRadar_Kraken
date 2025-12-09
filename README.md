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

## 1. Architecture Overview

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

The Kraken-specific blocks are intended to be **drop‑in GRC components** at steps (4) and (5).

---

## 2. System Dependencies

### 2.1 Hardware

- Raspberry Pi 4/5 **or** x86_64 host with enough CPU and RAM for FFT‑heavy processing
- **KrakenSDR** 5‑channel coherent SDR
- Stable RF front‑end (LNAs, antenna array, appropriate band filters)
- Network connectivity (for remote control / data export, optional)

### 2.2 Software (Debian / Raspberry Pi OS)

The project is designed for GNU Radio 3.10+ installed from distribution packages.

Install core dependencies:

```bash
sudo apt update
sudo apt install -y   gnuradio   gr-osmosdr   python3-numpy   python3-pyqt5   python3-pyqt5.qtsvg   g++ cmake make
```

You should already have KrakenSDR tooling and calibration utilities installed separately (not part of this repo).

---

## 3. Building and Installing the OOT Module

The OOT module lives in:

```text
gr-kraken_passive_radar/
```

### 3.1 Build and install into the system GNU Radio prefix

You can use the provided helper script:

```bash
cd ~/PassiveRadar_Kraken
./build_oot.sh
sudo make install
sudo ldconfig
```

Or manually:

```bash
cd ~/PassiveRadar_Kraken/gr-kraken_passive_radar
rm -rf build
mkdir build
cd build

# Use the same prefix that GNU Radio uses (typically /usr)
cmake -DCMAKE_INSTALL_PREFIX=/usr ..
make -j"$(nproc)"

sudo make install
sudo ldconfig
```

This installs:

- The Python package `kraken_passive_radar` into the system Python site‑packages (via GNU Radio’s CMake helpers, if configured)
- The GRC block definition files into:

  ```text
  /usr/share/gnuradio/grc/blocks
  ```

### 3.2 Ensuring Python can import `kraken_passive_radar`

On some systems, the Python site‑packages directory is under `/usr/local` instead of `/usr`. To confirm:

```bash
python3 - << 'EOF'
import sys, sysconfig
print("Python:", sys.executable)
print("platlib:", sysconfig.get_paths()["platlib"])
EOF
```

If the `platlib` path does not match where CMake installed the package, you can explicitly set the Python install directory when configuring:

```bash
PY_SITE=$(python3 - << 'EOF'
import sysconfig
print(sysconfig.get_paths()["platlib"])
EOF
)

cd ~/PassiveRadar_Kraken/gr-kraken_passive_radar
rm -rf build
mkdir build
cd build

cmake   -DCMAKE_INSTALL_PREFIX=/usr   -DGR_PYTHON_DIR="$PY_SITE"   ..

make -j"$(nproc)"
sudo make install
sudo ldconfig
```

You should then verify:

```bash
python3 -c "import sys, kraken_passive_radar; print(sys.executable); print(kraken_passive_radar.__file__)"
```

If this succeeds (no `ModuleNotFoundError`), GNU Radio Companion will also be able to import the Kraken blocks.

---

## 4. GNU Radio Companion (GRC) Blocks

After a successful install, start GRC:

```bash
gnuradio-companion
```

In the block tree, you should see a category:

```text
[Kraken Passive Radar]
```

**Note:** OOT modules often appear at the bottom of the block list in GNU Radio Companion. You may need to scroll down to find it.

Under this category, the key blocks are:

1. **Kraken NLMS Clutter Canceller**  
   - ID: `kraken_passive_radar_clutter_canceller`
   - Inputs:
     - `ref` (complex) – reference channel
     - `surv` (complex) – surveillance channel
   - Output:
     - `err` (complex) – clutter‑reduced surveillance (error signal)
   - Parameters:
     - `num_taps` – number of adaptive filter taps (e.g., 16–64)
     - `mu` – NLMS step size (e.g., 0.01–0.1; smaller is more stable, larger is faster)

2. **Kraken ECA-B Clutter Canceller**  
   - ID: `kraken_passive_radar_eca_b_clutter_canceller`
   - Inputs:
     - `ref` (complex) – reference channel
     - `surv` (complex) – surveillance channel
   - Output:
     - `err` (complex) – clutter‑reduced surveillance
   - Parameters:
     - `num_taps` – number of taps per segment
     - `num_segments` – number of historical segments in the batch (lookback)

These block definitions live in:

```text
/usr/share/gnuradio/grc/blocks/
  kraken_passive_radar_clutter_canceller.block.yml
  kraken_passive_radar_eca_b_clutter_canceller.block.yml
```

You can inspect or edit these YAML files directly if you need to customize labels, categories, or defaults.

---

## 5. Example GRC Wiring

A minimal passive radar chain in GRC using Kraken blocks:

1. **Sources**

   - One **KrakenSDR/OSMOSDR Source** configured for 5 coherent channels (4 surveillance + 1 reference)
   - Set sample rate to **2.048 Msps** (for example) and center frequency to the chosen FM/TV illuminator

2. **Per‑channel processing**

   - For each channel, add:
     - `Complex Multiply Const` (optional, for per‑channel gain)
     - `DC Blocker` (optional)

3. **Resampling to analysis rate**

   - Use a polyphase resampler or rational resampler to convert from 2.048 Msps to, e.g., **175 kHz** for analysis
   - The **same resampling chain** must be used for the reference and each surveillance channel so that all streams remain time‑aligned

4. **NLMS clutter cancellation**

   - Insert **Kraken NLMS Clutter Canceller** block
   - Connect:
     - `ref` ← resampled reference channel
     - `surv` ← resampled surveillance channel
     - `err` → clutter‑reduced output
   - Start with conservative parameters, e.g.:
     - `num_taps = 32`
     - `mu = 0.02`

5. **ECA‑B clutter cancellation (optional)**

   - As an alternative, insert **Kraken ECA-B Clutter Canceller**
   - Use the same `ref`/`surv` wiring
   - Typical starting values:
     - `num_taps = 64`
     - `num_segments = 8`

6. **Downstream processing**

   - Feed the clutter‑reduced outputs into:
     - Range–Doppler processing (CFAR / detection)
     - AoA estimation and track extraction
   - These components will live in the same OOT family and can be added as needed.

---

## 6. Running the Unit Tests (Optional)

If you have the development dependencies installed and want to run the tests:

```bash
cd ~/PassiveRadar_Kraken
./run_tests.sh
```

This will execute:

- C++ kernel tests for AoA, Doppler, and ECA‑B
- NLMS clutter canceller tests using a mock GNU Radio environment

Note: the tests are designed primarily for development and CI and are not required for operational deployment of the GRC blocks.

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
