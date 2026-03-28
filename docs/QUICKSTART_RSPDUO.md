# RSPduo Quick Start Guide

This guide covers setting up the PassiveRadar_Kraken system with an SDRplay RSPduo in dual-tuner mode.

## Hardware Overview

The SDRplay RSPduo is a dual-tuner SDR with excellent dynamic range:
- **Tuner A**: Reference channel (illuminator-facing antenna)
- **Tuner B**: Surveillance channel (target-facing antenna)
- **IF Bandwidth**: Up to 10 MHz per tuner
- **Dynamic Range**: 14-bit ADC, excellent for passive radar

### Hardware Requirements

- SDRplay RSPduo (~$250)
- 2 antennas (one for reference, one for surveillance)
- USB 2.0/3.0 connection
- Host computer: RPi5, x86 PC, or GPU-equipped workstation
- **gr-sdrplay3** GNU Radio OOT module (required)

## Installation

### 1. Install SDRplay API

Download and install the SDRplay API from [sdrplay.com](https://www.sdrplay.com/downloads/):

```bash
# Download SDRplay API 3.x
wget https://www.sdrplay.com/software/SDRplay_RSP_API-Linux-3.14.0.run
chmod +x SDRplay_RSP_API-Linux-3.14.0.run
sudo ./SDRplay_RSP_API-Linux-3.14.0.run

# Start the SDRplay service
sudo systemctl start sdrplay
sudo systemctl enable sdrplay
```

### 2. Install gr-sdrplay3

```bash
# Clone and build gr-sdrplay3
git clone https://github.com/fventuri/gr-sdrplay3.git
cd gr-sdrplay3
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 3. Install PassiveRadar_Kraken Dependencies

```bash
sudo apt update
sudo apt install -y cmake libfftw3-dev libeigen3-dev gnuradio-dev \
    python3-pybind11 python3-pip

pip install numpy matplotlib scipy pytest
```

### 4. Build PassiveRadar_Kraken

```bash
cd PassiveRadar_Kraken
mkdir build && cd build

# CPU-only build
cmake .. -DCMAKE_BUILD_TYPE=Release

# GPU-accelerated build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON

make -j$(nproc)
sudo make install && sudo ldconfig
```

## RSPduo Configuration

### Dual-Tuner Mode

The RSPduo operates in dual-tuner (diversity) mode for passive radar:

| Parameter | Tuner A (Reference) | Tuner B (Surveillance) |
|-----------|---------------------|------------------------|
| Function | Captures illuminator | Captures echoes |
| Antenna | Point at transmitter | Point at target area |
| Sample Rate | 2 MHz (shared) | 2 MHz (shared) |
| IF Gain | 20-59 dB | 20-59 dB |
| LNA State | 0-9 | 0-9 |

### Sample Rate Limitations

In dual-tuner mode, the RSPduo has specific sample rate constraints:
- Maximum: 2 MHz per tuner (6 MHz ADC shared)
- Minimum: 62.5 kHz
- Recommended: 1-2 MHz for passive radar

## Basic Usage

### Running the RSPduo Flowgraph

```bash
# Start passive radar with RSPduo
python3 rspduo_pbr_flowgraph.py --freq 98.1e6 --sample-rate 2e6

# With specific gain settings
python3 rspduo_pbr_flowgraph.py --freq 98.1e6 --sample-rate 2e6 \
    --if-gain 40 --lna-state 3
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--freq` | 98.1e6 | Center frequency (Hz) |
| `--sample-rate` | 2e6 | Sample rate (Hz, max 2 MHz dual-tuner) |
| `--if-gain` | 40 | IF gain (20-59 dB) |
| `--lna-state` | 3 | LNA attenuation state (0-9) |
| `--cpi-size` | 32768 | Coherent processing interval |
| `--num-doppler` | 128 | Doppler FFT bins |
| `--num-range` | 2048 | Range bins |

## Dashboard Usage

### Local Display

```bash
# Start with integrated dashboard
python3 rspduo_pbr_flowgraph.py --freq 98.1e6 --dashboard

# Or use standalone display
python3 kraken_passive_radar/multi_display_dashboard.py
```

### Dashboard Panels

The RSPduo dashboard provides 5 panels:

1. **Delay Profile**: Range-compressed pulse showing direct path and multipath
2. **Doppler Spectrum**: Frequency offset distribution
3. **Range-Doppler Map**: 2D heatmap with CFAR detections
4. **Detection List**: Table of confirmed targets (range, Doppler, SNR)
5. **Waterfall**: Time history of range-Doppler activity

### Remote Display

**On the RSPduo host (server):**
```bash
python3 rspduo_pbr_flowgraph.py --freq 98.1e6 --server --port 5556
```

**On the display computer (client):**
```bash
python3 kraken_passive_radar/remote_display.py --host 192.168.1.100 --port 5556
```

## Signal Processing Flow

```
RSPduo Dual-Tuner
├── Tuner A (Reference) ──┐
│   (illuminator)         │
└── Tuner B (Surveillance)┴──> Phase Alignment
                                    │
                                    v
                              Conditioning (AGC)
                                    │
                                    v
                              Block B3 Reconstruct
                              (FM demod-remod)
                                    │
                                    v
                              ECA-B Canceller
                              (NLMS adaptive)
                                    │
                                    v
                              CAF Processing ──> Doppler FFT
                              (2D x-corr)        (slow-time)
                                    │                 │
                                    v                 v
                              CFAR Detection <────────┘
                              (CA/GO/SO)
                                    │
                                    v
                              Clustering ──> Display
                              (8-connected)
```

## Performance Considerations

### RSPduo vs KrakenSDR

| Feature | RSPduo | KrakenSDR |
|---------|--------|-----------|
| Channels | 2 | 5 |
| AoA Capability | No | Yes (4-element array) |
| Dynamic Range | Excellent (14-bit) | Good (8-bit RTL) |
| Sample Rate | Up to 2 MHz (dual) | Up to 2.4 MHz |
| Price | ~$250 | ~$400 |
| Use Case | 2-antenna bistatic | Direction finding |

### Optimizing Performance

**CPU-Only (RPi5):**
```bash
python3 rspduo_pbr_flowgraph.py --freq 98.1e6 \
    --sample-rate 1e6 \
    --cpi-size 16384 \
    --num-doppler 64 \
    --num-range 1024
```
Expected: ~15 Hz update rate

**GPU-Accelerated:**
```bash
export KRAKEN_GPU_BACKEND=gpu
python3 rspduo_pbr_flowgraph.py --freq 98.1e6 \
    --sample-rate 2e6 \
    --cpi-size 32768 \
    --num-doppler 128 \
    --num-range 2048
```
Expected: ~50-100 Hz update rate

## Troubleshooting

### RSPduo Not Detected

```bash
# Check if SDRplay service is running
sudo systemctl status sdrplay

# Restart service
sudo systemctl restart sdrplay

# Verify device is visible
SoapySDRUtil --find="driver=sdrplay"
```

### gr-sdrplay3 Import Errors

```bash
# Verify installation
python3 -c "from gnuradio import sdrplay3; print('OK')"

# If import fails, check library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Sample Rate Errors

The RSPduo in dual-tuner mode requires sample rates <= 2 MHz:
```bash
# Error: "Sample rate too high for dual-tuner mode"
# Solution: Reduce sample rate
python3 rspduo_pbr_flowgraph.py --freq 98.1e6 --sample-rate 2e6
```

### Phase Alignment Issues

Unlike KrakenSDR, the RSPduo doesn't have an internal calibration source. Phase alignment uses:
1. Cross-correlation peak finding
2. Sample-accurate delay compensation
3. Phase offset estimation from correlation peak

If alignment fails:
- Verify both tuners are receiving the illuminator signal
- Check antenna connections
- Ensure similar cable lengths for both antennas

## GNU Radio Companion (GRC) Usage

The RSPduo flowgraph can also be created in GRC:

1. Open GNU Radio Companion
2. Add blocks:
   - `sdrplay3: RSPduo Source` (dual-tuner mode)
   - `Kraken: Conditioning`
   - `Kraken: ECA Canceller`
   - `Kraken: CAF Processing`
   - `Kraken: Doppler Processor`
   - `Kraken: CFAR Detector`
   - `Kraken: Dashboard Sink`

3. Connect blocks according to the signal flow diagram
4. Save as `.grc` file
5. Generate and run

## Example GRC Flowgraph

```yaml
# rspduo_passive_radar.grc
options:
  parameters:
    freq: 98.1e6
    sample_rate: 2e6
    if_gain: 40

blocks:
  - name: rspduo_source
    type: sdrplay3_rspduo
    parameters:
      center_freq: freq
      sample_rate: sample_rate
      if_agc_enabled: false
      if_gain_db: if_gain

  - name: eca_canceller
    type: kraken_passive_radar_eca_canceller
    parameters:
      num_taps: 128

  - name: cfar_detector
    type: kraken_passive_radar_cfar_detector
    parameters:
      cfar_type: 0  # CA-CFAR
      guard_cells: 2
      training_cells: 8
      threshold_db: 10.0
```

## Next Steps

- [Dashboard Guide](DASHBOARD_GUIDE.md) - Visualization options
- [GPU Deployment](GPU_DEPLOYMENT.md) - CUDA acceleration
- [Block Reference](BLOCKS.md) - Detailed API documentation
- [KrakenSDR Guide](QUICKSTART_KRAKENSDR.md) - 5-channel alternative
