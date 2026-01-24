# gr-kraken_passive_radar

GNU Radio Out-of-Tree (OOT) module for KrakenSDR passive bistatic radar applications.

## Overview

This module provides GNU Radio blocks for passive radar signal processing using the KrakenSDR 5-channel coherent SDR receiver. It implements the complete passive radar processing chain from coherent data acquisition through clutter cancellation and cross-ambiguity function computation.

## Blocks

| Block | Description |
|-------|-------------|
| **KrakenSDR Source** | 5-channel coherent source for KrakenSDR hardware |
| **ECA Canceller** | Extensive Cancellation Algorithm for direct-path/multipath suppression |
| **ECA-B Clutter Canceller** | Batched ECA implementation for improved performance |
| **Clutter Canceller** | General clutter cancellation block |
| **Doppler Processing** | Doppler shift estimation and compensation |

## Dependencies

- GNU Radio 3.10+
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
python3 -c "from kraken_passive_radar import krakensdr_source; print('OK')"
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

## Example Flowgraph

A complete passive radar flowgraph is provided: `kraken_passive_radar_103_7MHz.grc`

This flowgraph implements:
1. KrakenSDR 5-channel coherent acquisition at 103.7 MHz (FM broadcast)
2. DC blocking and decimation (2.4 MHz → 250 kHz)
3. AGC conditioning
4. ECA clutter cancellation
5. Cross-Ambiguity Function (CAF) computation via FFT
6. Non-coherent fusion of 4 surveillance channels
7. Range profile display

### Signal Flow

```
KrakenSDR (5ch) → DC Block → Decimate → AGC → ECA → S2V → FFT
                                              ↓
                    Display ← dB ← Σ ← |·|² ← IFFT ← Surv×conj(Ref)
```

## Passive Radar Basics

Passive bistatic radar uses existing RF transmitters (FM, DVB-T, LTE, etc.) as illuminators of opportunity. The KrakenSDR provides:

- **Reference channel (ch0)**: Points toward the transmitter to capture the direct signal
- **Surveillance channels (ch1-4)**: Point toward the surveillance area to capture target echoes

The Cross-Ambiguity Function correlates each surveillance channel with the reference to detect targets in range-Doppler space.

## Troubleshooting

### Block not appearing in GRC
```bash
# Refresh GRC block cache
grcc --force-load
# Or restart GRC
```

### Import errors
```bash
# Check Python path
python3 -c "import sys; print([p for p in sys.path if 'gnuradio' in p])"

# Verify installation location
python3 -c "import kraken_passive_radar; print(kraken_passive_radar.__file__)"
```

### "Blacklisted" block ID error
Block instance names cannot shadow imported class names. Rename the block instance (e.g., `krakensdr_source` → `kraken_src`).

## License

GPLv3

## Author

N4HY - Bob McGwier

## References

- [KrakenSDR Documentation](https://github.com/krakenrf/krakensdr_docs)
- [Passive Radar Fundamentals](https://en.wikipedia.org/wiki/Passive_radar)
- [GNU Radio OOT Module Tutorial](https://wiki.gnuradio.org/index.php/OutOfTreeModules)
