
---

## Requirements

### Hardware
- **KrakenSDR (5-channel coherent SDR)**
- **Reference antenna**: directed toward a strong local FM or TV transmitter  
- **Surveillance antennas (4)**: spaced and oriented toward the monitored airspace  
- **Stable clock and calibrated cables** (phase coherence required)

### Software Dependencies
- [GNU Radio 3.10+](https://wiki.gnuradio.org/)
- [gr-osmosdr](https://osmocom.org/projects/gr-osmosdr/wiki) (with SoapySDR support)
- `g++` (for C++ acceleration)
- **SoapySDR**: Required for KrakenSDR connectivity (via `gr-osmosdr`).

#### System Packages (Ubuntu/Debian/Raspberry Pi)
```bash
sudo apt update
sudo apt install gnuradio gr-osmosdr \
    python3-pyqt5 python3-sip \
    soapysdr-tools soapysdr-module-all \
    build-essential python3-venv git
```

### Virtual Environment Setup (Raspberry Pi / Debian)
Modern Linux distributions (like Raspberry Pi OS) often enforce "externally managed environments" (PEP 668), preventing global pip installs. It is recommended to use a Python virtual environment.

1. **Create the virtual environment**:
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```
   *(Note: You must activate the environment every time you open a new terminal to run this software.)*

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `gnuradio` and `PyQt5` bindings are often best installed via system packages (`apt`) and inherited using `--system-site-packages` if creating the venv manually, or by relying on the system python path. However, `requirements.txt` is provided for standard pip-based environments.*

   To inherit system packages (recommended for GNU Radio):
   ```bash
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Signal Flow Diagram

The following diagram illustrates the processing chain for the Passive Radar, showing the Reference and Surveillance paths, Cross-Ambiguity Function (CAF) processing, and Range-Doppler display.

```mermaid
graph TD
    subgraph Hardware
        A[KrakenSDR] -->|Ch 0| Ref[Reference Signal]
        A -->|Ch 1| Surv[Surveillance Signal]
    end

    subgraph Pre-Processing
        Ref -->|Resampler (2.048M->175k)| RefRes[Polyphase Resampler]
        Surv -->|Resampler (2.048M->175k)| SurvRes[Polyphase Resampler]
        RefRes --> RefDC[DC Blocker]
        SurvRes --> SurvDC[DC Blocker]
    end

    subgraph "Clutter Cancellation (NLMS)"
        RefDC -->|Predictor| NLMS[Adaptive Filter]
        SurvDC -->|Desired| NLMS
        NLMS -->|Error/Clean| CleanSurv[Cleaned Surv]
    end

    subgraph "CAF Processing (Fast-Time)"
        RefDC --> RefFFT[FFT]
        CleanSurv --> SurvFFT[FFT]
        RefFFT --> Mult[Multiply Conj]
        SurvFFT --> Mult
        Mult --> IFFT[IFFT]
    end

    subgraph "Doppler Processing (Slow-Time)"
        IFFT -->|Range Profiles| Buff["Buffer (Slow-Time)"]
        Buff -->|Window & FFT| DoppFFT[Doppler FFT]
        DoppFFT -->|Magnitude| Mag[Log Magnitude]
    end

    subgraph Display
        Mag --> RDMap["Range-Doppler Map (Qt Widget)"]
        IFFT -->|Mag Squared| RTMap[Range-Time Raster]
    end
```

## Performance Acceleration (C++)

This project includes C++ implementations for computationally intensive blocks to improve real-time performance.

1.  **NLMS Clutter Canceller**: `src/nlms_clutter_canceller.cpp`
2.  **Doppler Processing**: `src/doppler_processing.cpp`
3.  **AoA Processing**: `src/aoa_processing.cpp`
4.  **Polyphase Resampling**: `src/resampler.cpp`

The Python top block (`kraken_passive_radar_top_block.py`) automatically detects if these sources are present and attempts to compile them into shared libraries (`.so`) using `g++`. If compilation succeeds, the optimized C++ logic is loaded via `ctypes`. If `g++` is missing or compilation fails, the system seamlessly falls back to the pure Python/NumPy implementation.

**Note on Doppler Processing:** The C++ optimization for Doppler processing currently supports power-of-two Doppler lengths (e.g., 64, 128, 256). If a non-power-of-two length is selected, the system will use the Python fallback.

## REFERENCES
Griffiths, H. D., et al. “Passive Coherent Location RRadar Systems.” IEEE Aerospace & Electronic Systems Magazine, 2017.

Jahangir, M., Baker, C. J. “Performance Evaluation of Passive Radar with FM Radio Signals for Air Traffic Control.” IET Radar, Sonar & Navigation, 2016.

Melvin, W. L., Scheer, J. A. Principles of Modern Radar: Advanced Techniques. SciTech Publishing, 2013.

Fishler, E., Haimovich, A., Blum, R. “High-Resolution Passive Radar Imaging Using Television Broadcast Signals.” IEEE TAES, 2015.
