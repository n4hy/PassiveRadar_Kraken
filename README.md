# PassiveRadar_Kraken

**Complete Passive Bistatic Radar System for KrakenSDR**

GNU Radio Out-of-Tree (OOT) module and Python display system for passive bistatic radar applications using the KrakenSDR 5-channel coherent SDR receiver. Implements the full radar processing chain from coherent acquisition through detection, tracking, and visualization.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KrakenSDR 5-Channel Coherent SDR                    │
│                    (Ch0: Reference, Ch1-4: Surveillance)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GNU Radio OOT v2 Blocks                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Coherence  │  │     ECA      │  │   Doppler    │  │    CFAR      │    │
│  │   Monitor    │→ │  Canceller   │→ │  Processor   │→ │  Detector    │    │
│  │              │  │   (NLMS)     │  │              │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                               │             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │             │
│  │   Detection  │← │    Tracker   │← │     AoA      │←───────┘             │
│  │   Cluster    │  │   (Kalman)   │  │  Estimator   │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Python Display System                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Range-Doppler   │  │       PPI        │  │   Calibration    │          │
│  │      Display     │  │     Display      │  │      Panel       │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│  ┌──────────────────┐  ┌──────────────────────────────────────────┐        │
│  │     Metrics      │  │           Integrated Radar GUI           │        │
│  │    Dashboard     │  │                                          │        │
│  └──────────────────┘  └──────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Signal Processing
- **ECA Clutter Cancellation**: NLMS adaptive filter for direct-path and multipath suppression
- **Cross-Ambiguity Function (CAF)**: Range-Doppler processing with NEON optimization
- **CFAR Detection**: CA/GO/SO/OS variants with configurable Pfa
- **Detection Clustering**: Connected components analysis for merged detections
- **Multi-Target Tracking**: Kalman filter with Global Nearest Neighbor association
- **AoA Estimation**: Bartlett beamformer for angle-of-arrival (ULA/UCA arrays)

### Hardware Acceleration
- **OptMathKernels Integration**: NEON SIMD acceleration on Raspberry Pi 5
- **Vulkan GPU**: Available for large-scale CAF computations
- **FFTW3**: Optimized FFT for Doppler processing

### Display System
- **Range-Doppler Map**: Real-time CAF heatmap with detection overlays
- **PPI Display**: Polar plot with track trails and velocity vectors
- **Calibration Panel**: Per-channel SNR, phase offsets, correlation monitoring
- **Metrics Dashboard**: Processing latencies, detection rates, system health
- **Integrated GUI**: Multi-panel Tkinter application

---

## GNU Radio v2 Blocks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **KrakenSDR Source** | 5-channel coherent source | frequency, sample_rate, gain |
| **ECA Canceller** | NLMS adaptive clutter filter | num_taps, mu, epsilon |
| **Doppler Processor** | Range-Doppler FFT processing | num_range_bins, num_doppler_bins, window_type |
| **CFAR Detector** | Constant False Alarm Rate detection | guard_cells, ref_cells, pfa, cfar_type |
| **Coherence Monitor** | Phase coherence monitoring | corr_threshold, phase_threshold_deg |
| **Detection Cluster** | Connected components clustering | min_cluster_size, max_cluster_extent |
| **Tracker** | Kalman filter multi-target tracker | process_noise, gate_threshold, confirm_hits |
| **AoA Estimator** | Bartlett beamformer | d_lambda, n_angles, array_type |

---

## Installation

### Prerequisites

```bash
# Ubuntu/Debian/Raspberry Pi OS
sudo apt update
sudo apt install -y \
    build-essential cmake git pkg-config \
    gnuradio-dev libfftw3-dev libeigen3-dev \
    python3-numpy python3-matplotlib python3-tk \
    pybind11-dev libvolk2-dev

# Optional: OptMathKernels for NEON acceleration
cd /path/to/OptimizedKernelsForRaspberryPi5
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc) && sudo make install
```

### Build GNU Radio OOT v2

```bash
cd gr-kraken_passive_radar_v2/gr-kraken_passive_radar
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Build Standalone C++ Libraries

```bash
cd src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..
make -j$(nproc)
sudo make install
```

### Verify Installation

```bash
# Check GNU Radio blocks
python3 -c "from gnuradio import kraken_passive_radar; print(dir(kraken_passive_radar))"

# Expected output includes:
# aoa_estimator, cfar_detector, coherence_monitor, detection_cluster,
# doppler_processor, eca_canceller, tracker

# Test the GUI
python3 kraken_passive_radar/radar_gui.py
```

---

## Block Reference

### ECA Canceller (NLMS Adaptive Filter)

Removes direct-path and multipath clutter using Normalized Least Mean Squares adaptive filtering.

**Algorithm**: For each sample, estimates clutter as weighted sum of reference signal delays, adapts weights to minimize residual power.

```
y[n] = surv[n] - w^H * ref_delayed[n]
w[n+1] = w[n] + mu * e[n] * ref_delayed[n] / (||ref_delayed||² + eps)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_surv | int | 4 | Number of surveillance channels |
| num_taps | int | 64 | Filter length (covers multipath spread) |
| mu | float | 0.1 | Adaptation step size (0.01-0.5) |
| epsilon | float | 1e-6 | Regularization for stability |

**Performance**: Achieves >40 dB direct-path suppression, >30 dB multipath suppression.

### Detection Cluster

Merges adjacent CFAR detections into single targets using 8-connectivity connected components.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_range_bins | int | 256 | Range bins in input |
| num_doppler_bins | int | 64 | Doppler bins in input |
| min_cluster_size | int | 2 | Minimum cells to form detection |
| max_cluster_extent | int | 10 | Maximum cluster size before split |

**Output**: Vector of detections with centroid range/Doppler, peak SNR, cluster size.

### Tracker (Kalman Filter)

Multi-target tracker with constant velocity model and GNN data association.

**State Vector**: [range, doppler, range_rate, doppler_rate]

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| process_noise_range | float | 10.0 | Range process noise (m) |
| process_noise_doppler | float | 1.0 | Doppler process noise (Hz) |
| meas_noise_range | float | 50.0 | Range measurement noise (m) |
| meas_noise_doppler | float | 5.0 | Doppler measurement noise (Hz) |
| gate_threshold | float | 9.21 | Chi-squared gate (99% confidence) |
| confirm_hits | int | 3 | Hits to confirm track |
| delete_misses | int | 5 | Misses to delete track |
| max_tracks | int | 100 | Maximum concurrent tracks |

**Track States**: TENTATIVE → CONFIRMED → COASTING → (deleted)

### AoA Estimator (Bartlett Beamformer)

Estimates angle-of-arrival for each detection using spatial spectrum analysis.

**Algorithm**: P(θ) = |a(θ)^H * x|² where a(θ) is the steering vector.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| d_lambda | float | 0.5 | Element spacing in wavelengths |
| n_angles | int | 181 | Angular resolution (-90° to +90°) |
| array_type | enum | ULA | ULA (linear) or UCA (circular) |
| n_elements | int | 4 | Number of array elements |

**Output**: Detection augmented with AoA in radians and confidence metric.

---

## Display System

### Range-Doppler Display (`range_doppler_display.py`)

Real-time visualization of the Cross-Ambiguity Function output.

**Features**:
- Heatmap with configurable dynamic range (dB scale)
- Detection overlays (red circles)
- Track overlays with fading history trails
- Cursor readout: range (km), velocity (m/s), power (dB)
- Colorbar with adjustable limits

```python
from kraken_passive_radar.range_doppler_display import RangeDopplerDisplay

display = RangeDopplerDisplay(
    n_range=256,
    n_doppler=64,
    range_resolution=600.0,  # meters
    doppler_resolution=3.9,  # Hz
    center_freq=103.7e6
)
display.update(caf_data, detections, tracks)
```

### PPI Display (`radar_display.py`)

Polar Plan Position Indicator with target tracking visualization.

**Features**:
- Polar coordinate display (range vs azimuth)
- Track history trails (fading polylines)
- Velocity vectors (arrows showing heading)
- Track ID labels and status coloring
- Configurable max range and grid

```python
from kraken_passive_radar.radar_display import PPIDisplay, PPITrack

display = PPIDisplay(max_range_km=50.0, update_interval_ms=100)
display.update_tracks([
    PPITrack(id=1, range_m=15000, aoa_deg=45, velocity_mps=120, status='confirmed')
])
```

### Calibration Panel (`calibration_panel.py`)

Real-time monitoring of array calibration quality.

**Displays**:
- Per-channel SNR meters (bar chart)
- Phase offset scatter plot (-180° to +180°)
- Correlation coefficients (bar chart)
- Phase drift history waterfall
- Calibration valid/invalid indicator

```python
from kraken_passive_radar.calibration_panel import CalibrationPanel

panel = CalibrationPanel(n_channels=5)
panel.update(
    snr_db=[25.0, 24.5, 24.8, 25.2, 24.9],
    phase_offsets_deg=[0.0, 5.2, -3.1, 2.8, -1.5],
    correlations=[1.0, 0.98, 0.97, 0.99, 0.96]
)
```

### Metrics Dashboard (`metrics_dashboard.py`)

System health and performance monitoring.

**Displays**:
- Processing latency breakdown with sparklines
- Detection rate (detections/second)
- Track counts (confirmed, tentative, coasting)
- CPU/Memory usage bars
- Backend status (NEON/Vulkan indicators)

### Integrated GUI (`radar_gui.py`)

Multi-panel Tkinter application combining all displays.

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│         Range-Doppler Map           │            PPI Display              │
│         (CAF heatmap)               │         (polar tracks)              │
├─────────────────────────────────────┼─────────────────────────────────────┤
│       Calibration Panel             │        Metrics Dashboard            │
│    (phase/SNR monitoring)           │     (latency/track counts)          │
├─────────────────────────────────────┴─────────────────────────────────────┤
│  [Start] [Stop] [Reset] [Recalibrate]           Dynamic Range: [====60===]│
└───────────────────────────────────────────────────────────────────────────┘
```

**Launch**:
```bash
python3 kraken_passive_radar/radar_gui.py
```

---

## Performance

### Processing Capabilities (Raspberry Pi 5)

| Operation | Size | Time | Throughput |
|-----------|------|------|------------|
| ECA (NLMS) | 1024 samples, 64 taps | 0.5 ms | 2000 CPI/s |
| CAF (NEON) | 4096 samples, 64×256 | 5 ms | 200 CPI/s |
| CAF (Vulkan) | 4096 samples, 64×256 | 2 ms | 500 CPI/s |
| CFAR 2D | 64×256 bins | 1 ms | 1000 maps/s |
| Tracker | 10 tracks | 0.1 ms | 10000 updates/s |

### Radar Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Range resolution | c/(2·BW) | ~600 m @ 250 kHz BW |
| Max unambiguous range | c·CPI/2 | ~15 km @ 100 ms CPI |
| Doppler resolution | 1/CPI | ~10 Hz @ 100 ms CPI |
| Max unambiguous Doppler | ±fs/2 | ±125 kHz @ 250 kHz |
| Angular resolution | ~15° | 4-element array @ λ/2 spacing |

---

## OptMathKernels Integration

When OptMathKernels is installed, the following optimizations are enabled:

| Component | Optimization |
|-----------|--------------|
| CAF Processing | NEON complex multiply for Doppler shifts |
| ECA Canceller | NEON dot products for filter operations |
| CFAR Detector | NEON reductions for threshold computation |
| AoA Estimator | NEON complex dot for beamformer output |

Build with OptMathKernels:
```bash
# Ensure OptMathKernels is installed
cmake -DCMAKE_PREFIX_PATH=/usr/local ..

# Verify detection
# CMake output should show: "OptMathKernels found - enabling NEON optimization"
```

---

## Example Flowgraph

A complete passive radar flowgraph is provided: `examples/kraken_passive_radar_103_7MHz.grc`

### Signal Flow
```
KrakenSDR → Coherence Monitor → DC Block → Decimate → AGC
                                                        ↓
                                    ECA Canceller (NLMS)
                                                        ↓
                                    CAF Processing
                                                        ↓
                                    CFAR Detector
                                                        ↓
                                    Detection Cluster
                                                        ↓
                                    AoA Estimator
                                                        ↓
                                    Tracker
                                                        ↓
                                    Display System
```

---

## Troubleshooting

### Build Issues

**"OptMathKernels not found"**:
```bash
# Verify installation
ls /usr/local/lib/cmake/OptMathKernels/
# Should contain OptMathKernelsConfig.cmake
```

**"pybind11 not found"**:
```bash
sudo apt install pybind11-dev python3-pybind11
```

### Runtime Issues

**Block not appearing in GRC**:
```bash
grcc --force-load
# Or restart GNU Radio Companion
```

**Display not updating**:
```python
# Ensure matplotlib backend is set
import matplotlib
matplotlib.use('TkAgg')
```

**Calibration constantly triggering**:
- Check antenna connections
- Verify noise source is working
- Increase `corr_threshold` or `phase_threshold_deg`

### Performance Issues

**High CPU usage**:
- Reduce `num_doppler_bins` or `num_range_bins`
- Enable OptMathKernels NEON optimization
- Use Vulkan for large CAF computations

**Missed detections**:
- Lower CFAR `pfa` (e.g., 1e-4 instead of 1e-6)
- Increase ECA `num_taps` for better clutter suppression
- Check SNR in calibration panel

---

## File Structure

```
PassiveRadar_Kraken/
├── gr-kraken_passive_radar_v2/
│   └── gr-kraken_passive_radar/
│       ├── include/gnuradio/kraken_passive_radar/
│       │   ├── aoa_estimator.h
│       │   ├── detection_cluster.h
│       │   ├── tracker.h
│       │   └── ... (other block headers)
│       ├── lib/
│       │   ├── aoa_estimator_impl.{h,cc}
│       │   ├── detection_cluster_impl.{h,cc}
│       │   ├── tracker_impl.{h,cc}
│       │   ├── eca_canceller_impl.{h,cc}
│       │   └── CMakeLists.txt
│       ├── grc/
│       │   └── *.block.yml (GRC block definitions)
│       └── python/kraken_passive_radar/bindings/
│           └── *_python.cc (pybind11 bindings)
├── kraken_passive_radar/
│   ├── radar_gui.py              # Integrated multi-panel GUI
│   ├── radar_display.py          # PPI display with tracking
│   ├── range_doppler_display.py  # CAF heatmap display
│   ├── calibration_panel.py      # Calibration monitoring
│   └── metrics_dashboard.py      # System metrics
├── src/
│   ├── caf_processing.cpp        # CAF with OptMathKernels
│   ├── eca_b_clutter_canceller.cpp
│   ├── doppler_processing.cpp
│   └── CMakeLists.txt
└── examples/
    └── kraken_passive_radar_103_7MHz.grc
```

---

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

---

## Author

**N4HY - Bob McGwier**
Science Bob
Dr Robert W McGwier, PhD

---

## References

- [KrakenSDR Documentation](https://github.com/krakenrf/krakensdr_docs)
- [Passive Radar Fundamentals](https://en.wikipedia.org/wiki/Passive_radar)
- [GNU Radio OOT Module Tutorial](https://wiki.gnuradio.org/index.php/OutOfTreeModules)
- [OptMathKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5) - NEON/Vulkan acceleration
- Kulpa, K. "Signal Processing in Noise Waveform Radar" (2013)
- Bar-Shalom, Y. "Estimation with Applications to Tracking and Navigation" (2001)
