# Configuration Reference

Complete reference for all configuration options in PassiveRadar_Kraken.

## Environment Variables

### GPU Backend Control

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `KRAKEN_GPU_BACKEND` | auto, gpu, cpu | auto | Processing backend selection |
| `CUDA_VISIBLE_DEVICES` | 0, 1, etc. | all | Limit visible GPU devices |

**Examples**:
```bash
# Force CPU mode
export KRAKEN_GPU_BACKEND=cpu

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
export KRAKEN_GPU_BACKEND=gpu
```

### Library Paths

| Variable | Purpose | Example |
|----------|---------|---------|
| `LD_LIBRARY_PATH` | Kernel library location | `/usr/local/lib` |
| `PYTHONPATH` | Python module location | `/path/to/PassiveRadar_Kraken` |

**Development Setup**:
```bash
export LD_LIBRARY_PATH="$(pwd)/build/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Display Settings

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `DISPLAY` | :0, :99, etc. | system | X11 display for matplotlib |
| `MPLBACKEND` | TkAgg, Agg, etc. | TkAgg | Matplotlib backend |

**Headless Mode**:
```bash
# Use virtual framebuffer
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
```

## CMake Build Options

### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | string | Release | Build type (Release, Debug, RelWithDebInfo) |
| `CMAKE_INSTALL_PREFIX` | path | /usr/local | Installation prefix |
| `BUILD_KERNELS` | bool | ON | Build C++ signal processing kernels |
| `BUILD_OOT_MODULE` | bool | ON | Build GNU Radio OOT module |

### GPU Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ENABLE_GPU` | bool | ON | Build GPU-accelerated kernels |
| `CMAKE_CUDA_ARCHITECTURES` | string | 75;86;87;89 | Target GPU architectures |

**Architecture Reference**:
| Architecture | GPUs |
|--------------|------|
| sm_75 | RTX 2000 series, GTX 1660 |
| sm_86 | RTX 3000 series, A100 |
| sm_87 | Jetson Orin |
| sm_89 | RTX 4000 series |
| sm_120 | RTX 5000 series (Blackwell) |

### Optimization Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `NATIVE_OPTIMIZATION` | bool | OFF | Build with -march=native |

**Note**: `NATIVE_OPTIMIZATION=ON` produces faster binaries but non-portable.

### Example Build Commands

```bash
# Minimal CPU-only build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=OFF

# Full GPU build for RTX 5090
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89;86;75"

# Development build with debugging
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_GPU=ON

# Optimized for local machine
cmake .. -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIMIZATION=ON
```

## Python Configuration

### Module Configuration

Configuration is typically done via function arguments, but can also use a config dict:

```python
config = {
    # Processing parameters
    'cpi_size': 65536,
    'num_doppler_bins': 256,
    'num_range_bins': 4096,
    'sample_rate': 2.4e6,
    'center_freq': 98.1e6,

    # ECA parameters
    'eca_num_taps': 128,
    'eca_delay': 0,
    'eca_diagonal_loading': 1e-6,

    # CFAR parameters
    'cfar_type': 0,  # CA-CFAR
    'cfar_guard_cells': 2,
    'cfar_training_cells': 8,
    'cfar_threshold_db': 10.0,

    # Display parameters
    'update_rate': 10,  # Hz
    'colormap': 'viridis',
    'dynamic_range_db': 60,

    # Network parameters
    'server_enabled': False,
    'server_port': 5555,
    'server_bind': '0.0.0.0',
}
```

### Logging Configuration

```python
import logging

# Set log level for passive radar modules
logging.getLogger('kraken_passive_radar').setLevel(logging.INFO)

# Verbose debugging
logging.getLogger('kraken_passive_radar').setLevel(logging.DEBUG)
```

## GNU Radio Configuration

### Block Parameters in GRC

All blocks expose parameters in GNU Radio Companion:

```yaml
# Example: CFAR detector block
parameters:
  - id: cfar_type
    label: CFAR Type
    dtype: enum
    options: ['CA-CFAR', 'GO-CFAR', 'SO-CFAR', 'OS-CFAR']
    default: 'CA-CFAR'

  - id: guard_cells
    label: Guard Cells
    dtype: int
    default: 2

  - id: training_cells
    label: Training Cells
    dtype: int
    default: 8

  - id: threshold_db
    label: Threshold (dB)
    dtype: real
    default: 10.0
```

### Runtime Parameter Updates

Some parameters can be changed at runtime via callback:

```python
# In flowgraph
self.cfar_detector.set_threshold_db(15.0)
self.eca_canceller.set_num_taps(128)  # Default is 128
self.doppler_processor.set_window_type(2)  # Hann
```

## Hardware Configuration

### KrakenSDR

```python
krakensdr_config = {
    'device_serial': None,  # Auto-detect
    'center_freq': 98.1e6,
    'sample_rate': 2.4e6,
    'gain': 40,
    'num_channels': 5,

    # Calibration
    'enable_calibration': True,
    'calibration_interval': 60,  # seconds
    'calibration_settling_time': 1.0,  # seconds

    # Noise source GPIO
    'noise_source_gpio_pin': 0,
    'noise_source_active_high': True,
}
```

### RSPduo

```python
rspduo_config = {
    'center_freq': 98.1e6,
    'sample_rate': 2e6,  # Max 2 MHz in dual-tuner

    # Tuner A (Reference)
    'tuner_a_if_gain': 40,
    'tuner_a_lna_state': 3,
    'tuner_a_agc_enabled': False,

    # Tuner B (Surveillance)
    'tuner_b_if_gain': 40,
    'tuner_b_lna_state': 3,
    'tuner_b_agc_enabled': False,
}
```

## Performance Tuning

### CPU Optimization

```python
# For RPi5 / low-power CPU
low_power_config = {
    'cpi_size': 32768,
    'num_doppler_bins': 128,
    'num_range_bins': 2048,
    'eca_num_taps': 64,
    'update_rate': 5,
}
```

### GPU Optimization

```python
# For RTX 5090 / high-end GPU
high_perf_config = {
    'cpi_size': 131072,
    'num_doppler_bins': 512,
    'num_range_bins': 8192,
    'eca_num_taps': 256,
    'update_rate': 30,
}
```

### Memory Tuning

```bash
# Increase GNU Radio buffer sizes for high throughput
export GR_CONF_BUFFER_SIZE_KB=16384

# Pin CUDA memory for faster transfers
export KRAKEN_GPU_PINNED_MEMORY=1
```

## Signal Type Configuration

### FM Broadcast (88-108 MHz)

```python
fm_config = {
    'center_freq': 98.1e6,  # Local strong station
    'sample_rate': 2.4e6,
    'signal_type': 'fm',
    'block_b3_enabled': True,
}
```

### ATSC 3.0 Digital TV

```python
atsc3_config = {
    'center_freq': 533e6,  # Channel 27
    'sample_rate': 6.912e6,  # ATSC 3.0 bandwidth
    'signal_type': 'atsc3',
    'fft_size': 16384,  # 16K OFDM mode
}
```

### DVB-T Digital TV

```python
dvbt_config = {
    'center_freq': 506e6,  # Example UK channel
    'sample_rate': 7.61e6,  # 8 MHz DVB-T
    'signal_type': 'dvbt',
    'fft_size': 8192,  # 8K OFDM mode
}
```

## File Formats

### Calibration Data

```json
{
    "timestamp": "2026-03-27T10:15:30Z",
    "center_freq": 98100000,
    "sample_rate": 2400000,
    "phase_offsets": [0.0, 0.127, -0.054, 0.231, -0.089],
    "snr_db": [45.2, 38.1, 39.5, 37.8, 40.1],
    "coherence": 0.97
}
```

### Detection Log

```csv
timestamp,range_km,doppler_hz,velocity_mps,snr_db,aoa_deg
2026-03-27T10:15:31.123,12.34,45.6,15.2,18.5,32.1
2026-03-27T10:15:31.223,8.12,-23.4,-7.8,14.2,128.7
```

### Track Log

```csv
timestamp,track_id,x_km,y_km,vx_mps,vy_mps,age_s,state
2026-03-27T10:15:31.123,1,10.2,5.3,25.0,12.5,5.2,confirmed
2026-03-27T10:15:31.123,2,15.8,-3.2,-18.0,5.0,1.5,tentative
```
