# gr-kraken_passive_radar

GNU Radio Out-of-Tree (OOT) module for KrakenSDR passive bistatic radar applications.

## Overview

This module provides GNU Radio blocks for passive radar signal processing using the KrakenSDR 5-channel coherent SDR receiver. It implements the complete passive radar processing chain from coherent data acquisition through clutter cancellation, range-Doppler processing, and target detection.

## Blocks

| Block | Description |
|-------|-------------|
| **KrakenSDR Source** | 5-channel coherent source for KrakenSDR hardware |
| **ECA Canceller** | Extensive Cancellation Algorithm for direct-path/multipath suppression |
| **Doppler Processor** | Range-Doppler map generation via slow-time FFT |
| **CFAR Detector** | Constant False Alarm Rate detection (CA/GO/SO/OS variants) |
| **Coherence Monitor** | Automatic calibration verification and triggering |

## Dependencies

- GNU Radio 3.10+
- FFTW3 (for Doppler processing FFTs)
- VOLK (vectorized operations)
- KrakenSDR hardware (5-channel coherent RTL-SDR)
- Python 3.8+
- NumPy
- pybind11

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/gr-kraken_passive_radar.git
cd gr-kraken_passive_radar
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Verify Installation

```bash
# Check blocks are installed
ls /usr/local/share/gnuradio/grc/blocks/*kraken*

# Verify Python import
python3 -c "from gnuradio import kraken_passive_radar; print(dir(kraken_passive_radar))"
```

## Block Reference

### KrakenSDR Source

5-channel coherent source block for KrakenSDR hardware.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| frequency | float | 100e6 | Center frequency (Hz) |
| sample_rate | float | 2.4e6 | Sample rate (Hz) |
| gain | float | 30.0 | RF gain (dB) |

**Outputs:**
- 5 complex streams (ch0: reference, ch1-ch4: surveillance)

### ECA Canceller

Extensive Cancellation Algorithm block for removing direct-path interference and multipath clutter from surveillance channels using the reference channel.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_surv | int | 4 | Number of surveillance channels |
| num_taps | int | 128 | Filter length (delay spread coverage) |
| reg_factor | float | 0.001 | Regularization factor for matrix inversion |

**Inputs:**
- Port 0: Reference channel (complex stream)
- Ports 1-N: Surveillance channels (complex streams)

**Outputs:**
- Ports 0-(N-1): Cleaned surveillance channels (complex streams)

### Doppler Processor

Accumulates range profiles across multiple Coherent Processing Intervals (CPIs) and computes FFT along slow-time dimension to extract Doppler information.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_range_bins | int | 1024 | Range bins per CPI (input vector length) |
| num_doppler_bins | int | 64 | CPIs to accumulate (Doppler FFT size) |
| window_type | int | 1 | 0=rect, 1=hamming, 2=hann, 3=blackman |
| output_power | bool | True | Output \|X\|² if True, complex if False |

**Input:** Complex vector of length num_range_bins
**Output:** 2D Range-Doppler map as vector [doppler_bins × range_bins]

### CFAR Detector

Constant False Alarm Rate detector with multiple algorithm variants:

| Algorithm | Description |
|-----------|-------------|
| CA-CFAR | Cell Averaging - uses mean of reference cells |
| GO-CFAR | Greatest-Of - max of leading/lagging windows |
| SO-CFAR | Smallest-Of - min of leading/lagging windows |
| OS-CFAR | Order Statistics - k-th ordered sample |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_range_bins | int | - | Number of range bins |
| num_doppler_bins | int | - | Number of Doppler bins |
| guard_cells_range | int | 2 | Guard cells (range, each side) |
| guard_cells_doppler | int | 2 | Guard cells (Doppler, each side) |
| ref_cells_range | int | 8 | Reference cells (range, each side) |
| ref_cells_doppler | int | 8 | Reference cells (Doppler, each side) |
| pfa | float | 1e-6 | Probability of false alarm |
| cfar_type | int | 0 | 0=CA, 1=GO, 2=SO, 3=OS |

**Input:** Power Range-Doppler map (float vector)
**Output:** Binary detection map (1.0 = detection)

### Coherence Monitor

Monitors inter-channel phase coherence and automatically detects when recalibration is needed. Measurement is periodic to minimize CPU overhead.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_channels | int | 5 | Number of coherent channels |
| sample_rate | float | 2.4e6 | Sample rate (Hz) |
| measure_interval_ms | float | 1000 | Interval between measurements |
| measure_duration_ms | float | 10 | Length of measurement window |
| corr_threshold | float | 0.95 | Minimum correlation coefficient |
| phase_threshold_deg | float | 5.0 | Maximum phase std dev |

**Calibration Trigger Criteria:**
- Correlation coefficient drops below threshold
- Phase variance exceeds threshold
- 3 consecutive failures (hysteresis to avoid spurious triggers)

**Message Ports:**
- `cal_request` (output): PMT dict when calibration needed
- `cal_complete` (input): Acknowledge calibration done

## Unit Testing

Run the test suite:

```bash
cd gr-kraken_passive_radar
python3 -m pytest tests/test_passive_radar.py -v
```

Tests include:
- ECA direct-path suppression (≥20 dB expected)
- Doppler detection accuracy
- CFAR Pfa verification
- Coherence monitor threshold detection

## Example Flowgraph

A complete passive radar flowgraph is provided: `examples/kraken_passive_radar_103_7MHz.grc`

### Signal Flow

```
KrakenSDR (5ch) → Coherence Monitor → DC Block → Decimate → AGC → ECA
                                                                   ↓
Display ← dB ← CFAR ← Σ ← |·|² ← Doppler Processor ← Range Correlation
```

## Passive Radar Performance

| Parameter | Value | Notes |
|-----------|-------|-------|
| Range resolution | c/(2·BW) | ~600m for 250 kHz BW |
| Max unambiguous range | c·CPI/2 | ~15 km for 100ms CPI |
| Doppler resolution | PRF/N_doppler | ~3.9 Hz for 64 CPIs |
| Max unambiguous Doppler | ±PRF/2 | ±125 Hz at 250 kHz |

## Troubleshooting

### Block not appearing in GRC
```bash
grcc --force-load
```

### Import errors
```bash
python3 -c "import sys; print([p for p in sys.path if 'gnuradio' in p])"
```

### "Blacklisted" block ID error
Block instance names cannot shadow imported class names.

### Calibration constantly triggering
- Check antenna connections
- Verify noise source is working
- Increase `corr_threshold` or `phase_threshold_deg`

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

## Author

N4HY - Bob McGwier  
Science Bob  
Dr Robert W McGwier, PhD

## References

- [KrakenSDR Documentation](https://github.com/krakenrf/krakensdr_docs)
- [Passive Radar Fundamentals](https://en.wikipedia.org/wiki/Passive_radar)
- [GNU Radio OOT Module Tutorial](https://wiki.gnuradio.org/index.php/OutOfTreeModules)
- Kulpa, K. "Signal Processing in Noise Waveform Radar" (2013)
