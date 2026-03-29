# Dashboard Guide

Comprehensive guide to the PassiveRadar_Kraken visualization system.

## Overview

The dashboard system provides real-time visualization of passive radar processing:

| Dashboard | Use Case | Features |
|-----------|----------|----------|
| `five_channel_dashboard.py` | KrakenSDR 5-ch | Per-channel CAF, AoA-PPI, health |
| `multi_display_dashboard.py` | RSPduo 2-ch | Delay/Doppler/trail waterfalls |
| `remote_display.py` | Headless deployment | Network client |
| `enhanced_remote_display.py` | Remote + local | Server data + local CFAR |
| `dashboard_sink.py` | GNU Radio integration | FuncAnimation block |

## Five-Channel Dashboard (KrakenSDR)

### Starting the Dashboard

```bash
# With hardware
python3 run_passive_radar.py --freq 103.7e6 --visualize

# Demo mode (simulated targets)
python3 kraken_passive_radar/five_channel_demo.py
```

### Panel Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Channel Status                                 │
│  Ch0: REF SNR=45dB φ=0.00°  │  Ch1: SNR=38dB φ=12.3°  │  Ch2: SNR=...  │
├───────────────┬───────────────┬───────────────┬───────────────┬─────────┤
│   CAF Ch0     │   CAF Ch1     │   CAF Ch2     │   CAF Ch3     │ CAF Ch4 │
│  (Reference)  │               │               │               │         │
├───────────────┴───────────────┼───────────────┴───────────────┴─────────┤
│     Range-Doppler Map         │            PPI Display                  │
│     (Combined CFAR)           │         (AoA + Tracks)                  │
├───────────────────────────────┴─────────────────────────────────────────┤
│                         Health Monitor                                   │
│  Processing: 15ms │ Buffer: 98% │ Cal: OK │ GPU: RTX5090 │ Rate: 67Hz  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel Descriptions

#### Channel Status Bar
- **SNR**: Per-channel signal-to-noise ratio (green >30dB, yellow >20dB, red <20dB)
- **Phase (φ)**: Phase offset relative to reference channel
- **Health**: Indicates calibration state and coherence quality

#### CAF Panels (Per-Channel)
- **Ch0 (Reference)**: Shows autocorrelation (direct path + multipath)
- **Ch1-4 (Surveillance)**: Cross-ambiguity function with reference
- **Colormap**: Log magnitude (dB), dynamic range 60 dB
- **Markers**: CFAR detections shown as white circles

#### Range-Doppler Map
- **Combined view**: Fused detection map from all channels
- **X-axis**: Range (km or bistatic range bins)
- **Y-axis**: Doppler velocity (m/s or Hz)
- **CFAR overlay**: Detected cells highlighted
- **Threshold line**: Current detection threshold

#### PPI Display
- **Polar format**: Range vs bearing
- **AoA estimation**: MUSIC or Bartlett beamforming
- **Track display**: Confirmed tracks with velocity vectors
- **Range rings**: Every 5 km
- **Bearing markers**: Every 30°

#### Health Monitor
- **Processing latency**: End-to-end processing time
- **Buffer utilization**: GNU Radio buffer fill level
- **Calibration state**: Last calibration time, coherence trend
- **GPU status**: Backend and utilization (if available)
- **Update rate**: Actual frames per second

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Pause/resume display |
| `r` | Reset view to defaults |
| `+`/`-` | Adjust colormap range |
| `c` | Trigger manual calibration |
| `s` | Save screenshot |
| `q` | Quit |

## Multi-Display Dashboard (RSPduo)

### Starting the Dashboard

```bash
python3 kraken_passive_radar/multi_display_dashboard.py
```

### Panel Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Delay Profile                                     │
│  [Direct path pulse, multipath echoes, target returns]                  │
├─────────────────────────────────────────────────────────────────────────┤
│                        Doppler Spectrum                                  │
│  [FFT magnitude across slow-time, velocity distribution]                │
├─────────────────────────────────────────────────────────────────────────┤
│                      Range-Doppler Map                                   │
│  [2D heatmap, CFAR detections marked]                                   │
├──────────────────────────────┬──────────────────────────────────────────┤
│      Detection Table         │            Waterfall                      │
│  Range  Doppler  SNR  Age    │  [Time history scrolling down]           │
│  12.3km +45m/s  18dB  0.2s  │                                          │
│   8.1km -12m/s  14dB  0.5s  │                                          │
└──────────────────────────────┴──────────────────────────────────────────┘
```

## Remote Display

### Server Setup (Radar Host)

The radar server exposes a JSON HTTP API that remote displays connect to.

### Client Connection

```bash
# Basic remote display (default: https://radar3.retnode.com)
python3 -m kraken_passive_radar.remote_display

# Connect to a specific server
python3 -m kraken_passive_radar.remote_display --url https://your-radar-server.com

# With custom poll interval
python3 -m kraken_passive_radar.remote_display --url https://your-radar-server.com --interval 0.5

# Enhanced display with local post-processing
python3 -m kraken_passive_radar.enhanced_remote_display --local

# Enhanced display with custom CFAR/tracker parameters
python3 -m kraken_passive_radar.enhanced_remote_display --url https://your-radar-server.com --local \
    --cfar-guard 2 --cfar-train 8 --cfar-threshold 10 \
    --track-confirm 2 --track-delete 3 --track-gate 150
```

### HTTP JSON API

The remote display fetches data from the radar server via HTTP JSON endpoints:

| Endpoint | Returns |
|----------|---------|
| `/api/map` | 2D CAF matrix (delay, doppler, data arrays) |
| `/api/detection` | CFAR detection list (delay, doppler, snr) |
| `/api/timing` | Processing timing (cpi_ms, uptime_days, nCpi) |

### Bandwidth Requirements

| Data Type | Size | Rate | Bandwidth |
|-----------|------|------|-----------|
| RD Map (256x4096) | 4 MB | 1 Hz | 4 MB/s |
| Detections | ~1 KB | 1 Hz | 1 KB/s |
| Timing | ~100 B | 1 Hz | 100 B/s |
| **Total (default)** | | | **~4 MB/s** |

Bandwidth scales with poll interval (default 1.0s). For lower bandwidth, increase the interval:
```bash
python3 -m kraken_passive_radar.remote_display --url https://your-radar-server.com --interval 2.0
```

## GNU Radio Dashboard Sink

### Integration with Flowgraphs

The `dashboard_sink` block integrates with GNU Radio flowgraphs:

```python
from kraken_passive_radar.dashboard_sink import DashboardSink

# Create dashboard sink
dashboard = DashboardSink(
    num_range_bins=4096,
    num_doppler_bins=256,
    sample_rate=2.4e6,
    update_rate=10  # Hz
)

# Connect to flowgraph
self.connect((cfar_detector, 0), (dashboard, 0))

# Start dashboard (must be in main thread)
dashboard.start_display()
```

### FuncAnimation Implementation

The dashboard uses `matplotlib.animation.FuncAnimation` for efficient updates:

```python
class DashboardSink:
    def __init__(self, ...):
        self.fig, self.axes = plt.subplots(2, 2)
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=100,  # 10 Hz
            blit=True
        )

    def _update_frame(self, frame):
        # Update all artists
        self.rd_image.set_data(self.rd_map)
        self.det_scatter.set_offsets(self.detections)
        return [self.rd_image, self.det_scatter]
```

### X11 Considerations

For remote X11 sessions, the dashboard handles timeouts gracefully:

```python
# No more plt.pause() - use FuncAnimation interval
# X11 timeout fix applied in dashboard_sink.py:143
```

If running headless:
```bash
# Use virtual framebuffer
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
python3 run_passive_radar.py --visualize
```

## Customization

### Color Maps

```python
# In dashboard configuration
COLORMAP = 'viridis'  # Options: viridis, plasma, magma, inferno, jet
DYNAMIC_RANGE_DB = 60
BACKGROUND_COLOR = '#1a1a2e'
```

### Display Scales

```python
# Range display
RANGE_UNITS = 'km'  # or 'bins', 'samples'
MAX_RANGE_KM = 50

# Doppler display
DOPPLER_UNITS = 'm/s'  # or 'Hz', 'bins'
MAX_VELOCITY_MPS = 300

# Bistatic conversion
c = 3e8  # Speed of light
range_km = range_bins * c / (2 * sample_rate) / 1000
velocity_mps = doppler_hz * c / (2 * center_freq)
```

### Track Display

```python
# Track colors by state
TRACK_COLORS = {
    'tentative': 'yellow',
    'confirmed': 'green',
    'coasting': 'orange',
    'deleted': 'red'
}

# Velocity vector scale
VELOCITY_ARROW_SCALE = 0.1  # Arrow length = velocity * scale
```

## Performance Tips

### Reduce Update Rate for Slow Connections

```bash
python3 run_passive_radar.py --visualize
```

### Disable Unused Panels

```python
# In dashboard configuration
ENABLE_CAF_PANELS = True
ENABLE_PPI = True
ENABLE_WATERFALL = False  # Disable for performance
ENABLE_HEALTH = True
```

### GPU Rendering (Experimental)

For systems with NVIDIA GPU and proper drivers:

```bash
# Enable GPU-accelerated matplotlib backend
export MPLBACKEND=module://matplotlib.backends.backend_agg
python3 run_passive_radar.py --visualize
```

## Troubleshooting

### Dashboard Not Updating

1. Check that radar is running and producing data
2. Verify flowgraph connections
3. Check for buffer overflows in health panel

### Blank Panels

1. Verify data flow to dashboard sink
2. Check colormap range (may need adjustment)
3. Ensure signal is present (check SNR)

### High CPU Usage

1. Use smaller CPI: `--cpi-len 1024`
2. Skip AoA: `--skip-aoa`
3. Disable waterfall panel in dashboard configuration

### Remote Display Lag

1. Check network bandwidth
2. Use `enhanced_remote_display.py` with `--local` for local post-processing
4. Consider local post-processing with `enhanced_remote_display.py`

## Screenshots

### Five-Channel Dashboard
![Five-Channel Dashboard](images/five_channel_dashboard.png)

### Range-Doppler View
![Range-Doppler](images/range_doppler.png)

### PPI with Tracks
![PPI Display](images/ppi_tracks.png)

## Next Steps

- [KrakenSDR Quick Start](QUICKSTART_KRAKENSDR.md)
- [RSPduo Quick Start](QUICKSTART_RSPDUO.md)
- [Block Reference](BLOCKS.md)
- [GPU User Guide](GPU_USER_GUIDE.md)
