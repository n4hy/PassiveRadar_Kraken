
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
- `numpy`, `PyQt5`, `sip`
- Optional: `SoapySDR` driver for KrakenRF

Install dependencies (Ubuntu/Debian):
```bash
sudo apt install gnuradio gr-osmosdr python3-pyqt5 python3-sip
pip install numpy

## System Architecture & Signal Flow

The diagram below illustrates a KrakenSDR-based passive radar with **five coherent channels**: one **Reference** channel aimed at the illuminator (FM/TV) and **four Surveillance** channels covering the airspace. Each surveillance branch performs an FFT-domain **Cross-Ambiguity Function (CAF)** with the common Reference FFT, followed by IFFT and magnitude-squared to form a **range profile**; successive profiles form a **range–time** display.
