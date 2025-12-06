
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

