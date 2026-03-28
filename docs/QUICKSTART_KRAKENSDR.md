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
mkdir build && cd build

# CPU-only build (RPi5)
cmake .. -DCMAKE_BUILD_TYPE=Release

# GPU-accelerated build (desktop with NVIDIA GPU)
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON

make -j$(nproc)
sudo make install && sudo ldconfig
```

### 3. Verify Installation

```bash
# Test kernel libraries
python3 -c "from kraken_passive_radar.custom_blocks import CAFBlock; print('OK')"

# Test GPU (if enabled)
python3 -c "from kraken_passive_radar.gpu_backend import is_gpu_available; print(is_gpu_available())"
```

## Basic Usage

### Running the Main Application

```bash
# Default configuration (FM broadcast band)
python3 run_passive_radar.py --freq 98.1e6 --sample-rate 2.4e6

# With specific illuminator frequency
python3 run_passive_radar.py --freq 101.9e6 --sample-rate 2.4e6 --gain 40
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--freq` | 98.1e6 | Illuminator frequency (Hz) |
| `--sample-rate` | 2.4e6 | Sample rate (Hz) |
| `--gain` | 40 | RF gain (dB) |
| `--cpi-size` | 65536 | Coherent processing interval |
| `--num-doppler` | 256 | Doppler FFT bins |
| `--num-range` | 4096 | Range bins |
| `--eca-taps` | 128 | ECA filter taps |
| `--cfar-guard` | 2 | CFAR guard cells |
| `--cfar-train` | 8 | CFAR training cells |
| `--cfar-threshold` | 10.0 | CFAR threshold (dB) |

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
python3 krakensdr_calibrate.py --freq 98.1e6 --duration 10

# View calibration results
cat calibration_results.json
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
python3 run_passive_radar.py --freq 98.1e6 --dashboard

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
# Start radar processing with network server
python3 run_passive_radar.py --freq 98.1e6 --server --port 5555
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
(2.4 MHz)         (noise src)     (decay mode)    (FM/ATSC3)      (NLMS)
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
# Optimize for real-time on RPi5
python3 run_passive_radar.py --freq 98.1e6 \
    --cpi-size 32768 \
    --num-doppler 128 \
    --num-range 2048
```

Expected: ~10 Hz update rate

### GPU-Accelerated (RTX 5090)

```bash
# Enable GPU backend
export KRAKEN_GPU_BACKEND=gpu

python3 run_passive_radar.py --freq 98.1e6 \
    --cpi-size 65536 \
    --num-doppler 256 \
    --num-range 4096
```

Expected: 100-200 Hz update rate

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
python3 run_passive_radar.py --dashboard
```

## Next Steps

- [Dashboard Guide](DASHBOARD_GUIDE.md) - Advanced visualization options
- [GPU Deployment](GPU_DEPLOYMENT.md) - CUDA setup and optimization
- [Block Reference](BLOCKS.md) - Detailed API documentation
