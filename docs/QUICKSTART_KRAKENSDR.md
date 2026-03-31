# KrakenSDR Quick Start Guide

This guide covers setting up the PassiveRadar_Kraken system with a KrakenSDR 5-channel coherent SDR.

## Hardware Overview

The KrakenSDR is a 5-channel coherent RTL-SDR array:
- **Channel 0**: Reference channel (illuminator-facing antenna)
- **Channels 1-4**: Surveillance channels (target-facing array, ULA or UCA)
- **Noise source**: Internal calibration signal with GPIO-controlled switch

### Hardware Requirements

- KrakenSDR 5-channel SDR (~$400)
- 5 antennas (dipoles or Yagi for FM band)
- USB 3.0 connection (power and data)
- Host computer: RPi5, x86 PC, or GPU-equipped workstation

## Installation

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y cmake libfftw3-dev libeigen3-dev gnuradio-dev \
    python3-pybind11 python3-pip libusb-1.0-0-dev

# Install Python dependencies
pip install numpy matplotlib scipy pytest
```

### 2. Build the Project

```bash
cd PassiveRadar_Kraken

# Step 1: Build C++ signal processing kernels
cd src && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..

# Step 2: Build and install the GNU Radio OOT module
cd gr-kraken_passive_radar && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install && sudo ldconfig
cd ../..
```

### 3. Verify Installation

```bash
# Test kernel libraries (requires OOT module installed)
python3 -c "from gnuradio.kraken_passive_radar.custom_blocks import CafBlock; print('OK')"

# Test C++ blocks
python3 -c "from gnuradio import kraken_passive_radar; print('OOT module OK')"
```

## Basic Usage

### Running the Main Application

```bash
# Default configuration (FM broadcast at 103.7 MHz)
python3 run_passive_radar.py --freq 103.7e6 --visualize

# With specific illuminator frequency and gain
python3 run_passive_radar.py --freq 101.9e6 --gain 40 --visualize
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--freq` | 103.7e6 | Illuminator frequency (Hz) |
| `--gain` | 30 | RF gain (dB) |
| `--geometry` | ULA | Array geometry: ULA or URA |
| `--cpi-len` | 2048 | CPI length / range bins: 512, 1024, 2048, 4096 |
| `--sample-rate` | 1e6 | Sample rate (Hz) |
| `--recal-interval` | 120 | Recalibration interval (seconds) |
| `--no-startup-cal` | off | Skip initial phase calibration |
| `--skip-aoa` | off | Skip AoA estimation (range-Doppler only) |
| `--visualize` | off | Show live 5-channel dashboard GUI |
| `--demo` | off | Demo mode with simulated data (no hardware) |
| `--b3-signal` | passthrough | Reference reconstruction: passthrough, fm, atsc3, dvbt |
| `--b3-fft-size` | 8192 | DVB-T FFT size: 2048, 4096, 8192, 16384, 32768 |
| `--b3-guard-interval` | 192 | DVB-T guard interval samples |

## Phase Calibration

### Automatic Calibration

The system performs automatic phase calibration using the internal noise source:

1. **Noise source activation**: GPIO control via direct libusb vendor request
2. **Correlation measurement**: Cross-correlate all channels against reference
3. **Phase correction**: Apply measured offsets to subsequent data
4. **Settling period**: 1 second after correction before processing resumes

### Manual Calibration

```bash
# Run dedicated calibration script
python3 krakensdr_calibrate.py --freq 103.7e6

# View calibration results
cat calibration.json
```

### Calibration Tips

- Run calibration when ambient temperature changes >5°C
- Monitor coherence panel for phase drift warnings
- Recalibrate if phase drift exceeds 0.1 radians over 1 minute
- The system triggers automatic recalibration when coherence drops below threshold

## 5-Channel Dashboard

### Local Display

```bash
# Start with local dashboard
python3 run_passive_radar.py --freq 103.7e6 --visualize

# Or use the dedicated dashboard script
python3 kraken_passive_radar/five_channel_dashboard.py
```

### Dashboard Panels

1. **Channel Status**: Per-channel SNR, phase, and health indicators
2. **CAF View**: Cross-ambiguity function for each surveillance channel
3. **Range-Doppler**: Combined range-Doppler map with CFAR detections
4. **PPI Display**: Plan Position Indicator with AoA tracks
5. **Health Monitor**: Processing latency, buffer status, calibration state

### Remote Display

The dashboard supports remote viewing for headless deployments:

**On the KrakenSDR host (server):**
```bash
# Start radar processing (dashboard streams data for remote display)
python3 run_passive_radar.py --freq 103.7e6 --visualize
```

**On the display computer (client):**
```bash
# Connect to remote radar
python3 kraken_passive_radar/remote_display.py --host 192.168.1.100 --port 5555

# Enhanced display with local post-processing
python3 kraken_passive_radar/enhanced_remote_display.py --host 192.168.1.100 --port 5555
```

### Demo Mode

Test the dashboard without hardware:

```bash
# Run simulated demo with moving targets
python3 kraken_passive_radar/five_channel_demo.py
```

## Signal Flow

```
KrakenSDR 5-Ch      Phase          Conditioning     Block B3        ECA-B
USB Source    -->  Calibration -->    AGC      --> Reconstruct --> Canceller
(1 MHz)           (noise src)     (decay mode)    (FM/ATSC3)   (batch Toeplitz)
     |                                                                |
     v                                                                v
Multi-panel   <--   Tracker   <--   AoA Est   <--  Clustering  <--  CFAR
Dashboard         (Kalman+GNN)     (MUSIC)      (8-connected)    (CA/GO/SO)
                                                                      ^
                                                                      |
                                       CAF Processing  -->  Doppler FFT
                                       (2D x-corr)       (slow-time)
```

## Performance Tuning

### CPU-Only (RPi5)

```bash
# Smaller CPI for real-time on RPi5
python3 run_passive_radar.py --freq 103.7e6 --cpi-len 1024 --visualize

# Range-Doppler only (skip AoA for lower CPU load)
python3 run_passive_radar.py --freq 103.7e6 --cpi-len 1024 --skip-aoa --visualize
```

### GPU-Accelerated (x86_64 with CUDA)

```bash
# Larger CPI for higher resolution (GPU handles the load)
python3 run_passive_radar.py --freq 103.7e6 --cpi-len 4096 --visualize
```

## Troubleshooting

### USB Permission Issues

```bash
# Add udev rules for KrakenSDR
sudo tee /etc/udev/rules.d/99-krakensdr.rules << 'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Noise Source Not Activating

The noise source uses direct libusb vendor control transfers:
```python
# Verify libusb can access device
import usb.core
dev = usb.core.find(idVendor=0x0bda, idProduct=0x2838)
print(f"Found {dev.product}")
```

If the noise source doesn't toggle, check:
1. USB permissions (see above)
2. KrakenSDR firmware version (requires latest)
3. No other program holding the USB device

### Phase Drift

If phase drifts rapidly:
1. Ensure adequate warm-up time (5 minutes)
2. Check antenna cable lengths are matched
3. Verify thermal stability
4. Increase calibration frequency

### Dashboard X11 Errors

For remote SSH connections:
```bash
# Forward X11
ssh -X user@krakensdr-host

# Or use virtual framebuffer
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
python3 run_passive_radar.py --visualize
```

## Next Steps

- [Dashboard Guide](DASHBOARD_GUIDE.md) - Advanced visualization options
- [GPU Deployment](GPU_DEPLOYMENT.md) - CUDA setup and optimization
- [Block Reference](BLOCKS.md) - Detailed API documentation
