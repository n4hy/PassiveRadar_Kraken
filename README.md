# PassiveRadar_Kraken

This repository implements a passive radar processing chain using a five-channel
KrakenSDR front-end, GNU Radio, adaptive clutter cancellation (NLMS and ECA-B),
and Range–Doppler processing and visualization.

The primary use case is **tracking aircraft using broadcast FM or TV
illuminators-of-opportunity** in a fully passive, receive-only configuration.

---

## 1. System Overview

At a high level, the processing chain is:

1. **Multi-channel acquisition (KrakenSDR)**  
   - 5 coherent RF channels at 2.048 Msps complex baseband
   - 1 reference channel pointed toward the broadcast transmitter
   - 4 surveillance channels pointed toward the monitored airspace

2. **Polyphase resampling**  
   - Front-end rate: **2.048 Msps**
   - Polyphase resampler: `interp = 175`, `decim = 2048`
   - Effective baseband processing rate: **175 kHz** per channel

3. **Clutter cancellation (two interchangeable modes)**  
   - **NLMS (Python GNU Radio block)**  
     - Adaptive FIR canceller, streaming, low-latency
   - **ECA-B (C++ kernel exposed via Python GNU Radio block)**  
     - Batch least-squares canceller, high dynamic range, research-grade

4. **Range–Doppler processing**  
   - Cross-ambiguity processing over coherent processing intervals (CPIs)
   - Slow-time FFT for Doppler
   - Magnitude / log-power Range–Doppler map

5. **Visualization**  
   - Qt-based Range–Doppler widget for real-time display

The codebase is structured to be compatible with a **Python-only GNU Radio OOT
module** (`gr-kraken_passive_radar`) that exposes both clutter cancellers into
GNU Radio Companion (GRC) for direct comparison.

---

## 2. Repository Layout

Key directories and files:

- `kraken_passive_radar_top_block.py`  
  Example Python top block (skeleton) showing how to wire the pieces.

- `gr-kraken_passive_radar/`  
  Python-only GNU Radio OOT module:
  - `python/kraken_passive_radar/`
    - `clutter_cancellation.py` — NLMS adaptive clutter canceller (GR block)
    - `eca_b_clutter_canceller.py` — ECA-B clutter canceller (GR block via ctypes)
    - `doppler_processing.py` — Doppler processing / Range–Doppler construction
    - `range_doppler_widget.py` — Qt Range–Doppler visualization widget
    - `__init__.py` — exports the above blocks/classes
  - `grc/`
    - `kraken_passive_radar_clutter_canceller.block.yml`
    - `kraken_passive_radar_eca_b_clutter_canceller.block.yml`
  - `CMakeLists.txt` — CMake entry for the OOT module

- `src/`
  - `eca_b_clutter_canceller.cpp` — C++ ECA-B kernel (built as a shared library)
  - `nlms_clutter_canceller.cpp` — C++ NLMS kernel (optional/reference)
  - `doppler_processing.cpp` — C++ Doppler kernel (used in tests / acceleration)
  - `aoa_processing.cpp` — C++ AoA kernel (used in tests / acceleration)
  - `resampler.cpp` — Polyphase resampler implementation
  - `CMakeLists.txt` — builds `libkraken_eca_b_clutter_canceller.so`

- `tests/`
  - `mock_gnuradio.py` — mock for GNU Radio APIs (for unit testing)
  - `test_instantiation.py` — basic instantiation checks
  - `test_doppler_cpp.py` — Doppler kernel correctness test
  - `test_aoa_cpp.py` — AoA kernel correctness test
  - `test_clutter_nlms.py` — NLMS clutter canceller regression test
  - `test_eca_b_cpp.py` — ECA-B clutter canceller regression test

- `.gitignore`, `.gitattributes`  
  - Repository hygiene, CI safety, and cross-platform consistency

---

## 3. Software Requirements

### 3.1 System Dependencies (Linux recommended)

- **GNU Radio 3.10+** (runtime and development headers)
- **gr-osmosdr** with SoapySDR support
- **g++** with C++17 support
- **CMake ≥ 3.8**
- **Qt5** (for Range–Doppler widget)

Typical install on Debian/Ubuntu-like systems (example):

```bash
sudo apt update
sudo apt install -y   gnuradio   gr-osmosdr   g++   cmake   qtbase5-dev   python3-pyqt5   python3-pyqt5.qtsvg   python3-venv
```

> Note: package names may vary slightly by distribution. Adjust as necessary.

### 3.2 Python Dependencies

These are captured in `requirements.txt`:

```text
numpy>=1.20
PyQt5>=5.15
PyQt5-sip>=12.0
```

---

## 4. Python Virtual Environment Setup

It is strongly recommended to use an isolated Python environment.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

This venv is used for:

- Running the Python-based unit tests
- Running analysis tools / scripts
- Using the Python blocks directly

> GNU Radio itself is typically installed at the system level and imported into
> this venv via the system Python if installed that way.

---

## 5. Building and Installing the OOT Module

The OOT module is located at `gr-kraken_passive_radar/`. It is a **Python-only**
module with an additional C++ shared library for the ECA-B kernel.

From the repository root:

```bash
cd gr-kraken_passive_radar
mkdir -p build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig   # on Linux, as needed
```

This will:

- Install the `kraken_passive_radar` Python package into GNU Radio’s Python dir
- Install the GRC block definitions into `${GR_PKG_GRC_BLOCKS_DIR}`
- Build and install the `libkraken_eca_b_clutter_canceller.so` shared library
  next to the Python package

After this, start **GNU Radio Companion** and look for the category:

> `[Kraken Passive Radar]`

You should see two blocks:

- `Kraken Passive Radar NLMS Clutter Canceller`
- `Kraken Passive Radar ECA-B Clutter Canceller`

Each block has two complex inputs (reference, surveillance) and a single complex
output (clutter-suppressed surveillance).

---

## 6. Using NLMS and ECA-B in GRC

1. Open **gnuradio-companion**.
2. Place your KrakenSDR (or file source) blocks providing:
   - Reference stream (direct-path illuminator)
   - Surveillance stream(s)
3. Under `[Kraken Passive Radar]`, drop:
   - `Kraken Passive Radar NLMS Clutter Canceller`, or
   - `Kraken Passive Radar ECA-B Clutter Canceller`
4. Wire:
   - Input 0 ← reference
   - Input 1 ← surveillance
   - Output → your CAF / Doppler / Range–Doppler processing chain

For **direct comparison** between NLMS and ECA-B, you can:

- Use a `Selector` block to switch between NLMS and ECA-B outputs feeding a
  common Range–Doppler chain, or
- Run two parallel chains and compare Range–Doppler outputs visually.

---

## 7. Running Unit Tests

From the repository root, with the Python venv activated:

```bash
python -m unittest discover -v tests
```

This runs:

- NLMS clutter cancellation test (`test_clutter_nlms.py`)
- ECA-B kernel test (`test_eca_b_cpp.py`)
- Doppler and AoA kernel tests
- Basic instantiation and mock GNU Radio tests

> Note: `test_eca_b_cpp.py` will compile a local
> `src/libkraken_eca_b_clutter_canceller.so` if it has not already been built.

---

## 8. Sample Rate and Resampling Configuration

The system explicitly enforces:

- **KrakenSDR front-end sample rate:**

```python
samp_rate = 2_048_000  # 2.048 Msps
```

- **Polyphase resampler configuration:**

```python
interp = 175
decim = 2048
```

Thus, the effective processing rate is:

\[
f_{out} = 2.048 \times 10^6 \cdot \frac{175}{2048} = 175\,000 \; \text{Hz}.
\]

This fixed 2.048 Msps → 175 kHz channelization is part of the explicit
engineering basis for export classification (see below).

---

## 9. Export Control & ITAR Compliance

This repository has been designed and reviewed with explicit attention to U.S.
export control regulations, including ITAR (22 CFR §120–130) and EAR
(15 CFR §730–774).

### 9.1 Governing Regulations

- **ITAR** (International Traffic in Arms Regulations) — 22 CFR §120–130  
  Administered by the U.S. Department of State (DDTC). Controls defense
  articles, defense services, and technical data on the U.S. Munitions List
  (USML).

- **EAR** (Export Administration Regulations) — 15 CFR §730–774  
  Administered by the U.S. Department of Commerce (BIS). Controls commercial
  and dual-use items not on the USML.

### 9.2 ITAR Applicability Determination

This project is **not ITAR-controlled** because:

- It is **receive-only** (no RF transmission, no waveform generation).
- It uses **civilian broadcast FM and TV transmitters** as illuminators.
- It uses **commercial, off-the-shelf SDR hardware** (KrakenSDR).
- It does **not** implement or exploit any military-only, classified, or
  proprietary waveforms.
- It is not designed for weapons guidance, fire control, or combat systems.
- It is openly published, unclassified, and research-oriented.

Therefore, this software is **not a defense article** and is not subject to
ITAR under the definitions of 22 CFR §120.10 and §121.

### 9.3 EAR Classification

Under the EAR, this software is appropriately classified as:

> **EAR99 – No License Required (NLR)**

Basis:

- It is open-source signal processing code using standard academic algorithms
  (NLMS, ECA-B, FFT, polyphase resampling).
- It does not contain controlled encryption features.
- It does not implement weapons guidance, tracking, or fire control functions.
- It is designed for civil passive radar / airspace awareness using broadcast
  illuminators.

### 9.4 Engineering Steps Taken to Ensure Civil Classification

- **Receive-only architecture**: No transmit capability, no waveform generation.
- **Civil illuminators**: Explicit use of public FM and TV broadcasts only.
- **Fixed sample rates**:
  - KrakenSDR operated at **2.048 Msps**.
  - Polyphase resampler converts to **175 kHz** channels.
- **Open algorithms**:
  - NLMS adaptive filtering.
  - ECA-B clutter cancellation.
  - CAF and FFT-based Doppler / Range–Doppler processing.
- **Open-source license**: MIT License, ensuring public-domain visibility and
  preventing proprietary military use claims.

### 9.5 Export Statement for the Repository

This software is classified as **EAR99** under the U.S. Export Administration
Regulations (15 CFR §730–774). It is **not subject to ITAR** (22 CFR §120–130).

It is authorized for public release. No export license is required for most
destinations. Users are responsible for ensuring compliance with applicable
U.S. embargoes and sanctions, including restrictions on exports to:

- Iran
- North Korea
- Syria
- Cuba
- Crimea / Donetsk / Luhansk regions (and any other comprehensively sanctioned
  jurisdictions as updated by U.S. law).

---

© Released under the MIT License. All export compliance responsibilities for
deployment, redistribution, or operational use remain with the end user and/or
deploying organization.
