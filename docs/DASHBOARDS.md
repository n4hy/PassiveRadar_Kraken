# Dashboard and Display Reference

Complete reference for all display systems in the PassiveRadar_Kraken project.

---

## Table of Contents

1. [KrakenSDR Qt5 Flowgraph GUI](#1-krakensdr-qt5-flowgraph-gui)
2. [RSPduo Qt5 Flowgraph GUI](#2-rspduo-qt5-flowgraph-gui)
3. [5-Channel Dashboard Sink](#3-5-channel-dashboard-sink)
4. [Remote Radar Display](#4-remote-radar-display)
5. [Enhanced Remote Display](#5-enhanced-remote-display)
6. [5-Channel Dashboard](#6-5-channel-dashboard)
7. [Multi-Display Dashboard](#7-multi-display-dashboard)
8. [Calibration Panel](#8-calibration-panel)
9. [Metrics Dashboard](#9-metrics-dashboard)
10. [Range-Doppler Display](#10-range-doppler-display)
11. [PPI Polar Display](#11-ppi-polar-display)
12. [Integrated Radar GUI](#12-integrated-radar-gui)
13. [5-Channel Demo](#13-5-channel-demo)
14. [Remote Display Configuration](#14-remote-display-configuration)

---

## 1. KrakenSDR Qt5 Flowgraph GUI

**File:** `kraken_pbr_flowgraph.py`
**Technology:** PyQt5 + matplotlib (Qt5Agg) + GNU Radio Qt widgets
**For:** KrakenSDR 5-channel passive radar

### Launch

```bash
python3 kraken_pbr_flowgraph.py
```

### Layout (5 rows x 2 columns)

```
Row 0: Range-Doppler Heatmap (matplotlib)           [1x2]
Row 1: Freq Sink Ref (ch0) [1x1] | Freq Sink Surv (ch1) [1x1]
Row 2: Phase Drift History (matplotlib)              [1x2]
Row 3: Calibration Status Label                      [1x2]
Row 4: Recalibration Interval Control                [1x2]
```

### Range-Doppler Map (Row 0)

| Property | Value |
|----------|-------|
| Figure size | 8 x 4 inches, 100 dpi |
| Colormap | `inferno` |
| X-axis | Bistatic Range (km), 0 to `rd_display_bins * c/fs/1000` |
| Y-axis | Doppler (Hz), `-doppler_max` to `+doppler_max` |
| Dynamic range | vmin=-10 dB, vmax=50 dB |
| Interpolation | nearest |
| Origin | lower |
| Colorbar | "dB", shrink=0.8 |
| Update rate | 400 ms timer (2.5 Hz) |

Doppler max is computed as `1 / (2 * cpi_dur)` where `cpi_dur = cpi_samples / decimated_rate`.
Range resolution: `c / decimated_rate / 1000` km/bin. Display truncated to first 200 bins.

### Frequency Sinks (Row 1)

Two GNU Radio `qtgui.freq_sink_c` widgets, one for reference (ch0, blue) and one for surveillance (ch1, red).

| Property | Value |
|----------|-------|
| FFT size | 2048 |
| Window | Blackman-Harris |
| Bandwidth | 2.4 MHz (full sample rate, pre-decimation) |
| Y-axis | Relative Gain (dB), -140 to 10 |
| Update time | 100 ms |
| FFT average | 0.2 |
| Autoscale | enabled |

### Phase Drift History (Row 2)

| Property | Value |
|----------|-------|
| Figure size | 8 x 2.5 inches, 100 dpi |
| Y-axis | Phase Offset (deg), -180 to +180 |
| X-axis | Time (s) from flowgraph start |
| Measured points | Cyan circles with lines, markersize=5 |
| Frozen trend | Red dashed, linewidth=1.5, alpha=0.7 |
| Extrapolation | Orange dashed, linewidth=1.2, alpha=0.8 |
| Title | "Phase Drift: X.XXX deg/s (curvature a=X.XXXXX)" |

The trend line is built piecewise: each calibration interval produces one segment using its polynomial coefficients. Once frozen, segments never change. The orange extrapolation extends 25% beyond the current time span.

### Calibration Status Label (Row 3)

Shows per-channel results after each calibration cycle:
```
ch1=+149 deg/0.80  ch2=+126 deg/0.58  ch3=-62 deg/0.72  ch4=-65 deg/0.92
```

Color: green (good), orange (waiting), red (failed).

### Recalibration Interval Control (Row 4)

| Widget | Type | Range | Default |
|--------|------|-------|---------|
| Interval | QSpinBox | 1-600 s | 60 s |
| Execute | QPushButton | — | Applies new interval immediately |

### Processing Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 2.4 MHz |
| Signal bandwidth | 250 kHz |
| Decimation | 9 (2.4M / 250K, truncated) |
| Decimated rate | 266,666 Hz |
| FFT size | 2048 |
| CPI samples | 2048 |
| Doppler bins | 64 |
| ECA taps | 256 |
| ECA regularization | 0.0001 |
| Center frequency | 103.7 MHz |
| RF gain | 49.6 dB |
| Calibration vector size | 65536 samples (245 ms) |
| Pipeline settle time | 1.5 s |

### Remote Display

Auto-detects display environment for SSH X11 forwarding:
- Probes SSH_CONNECTION/SSH_CLIENT environment variables
- Scans X11 forwarding ports 6010-6070 on loopback
- Falls back to Wayland compositor sockets
- Sets XAUTHORITY for authentication
- Configures QT_QPA_PLATFORM for Qt5

---

## 2. RSPduo Qt5 Flowgraph GUI

**File:** `rspduo_pbr_flowgraph.py`
**Technology:** PyQt5 + GNU Radio Qt widgets
**For:** SDRplay RSPduo (coherent dual-tuner)

### Launch

```bash
python3 rspduo_pbr_flowgraph.py
```

### Layout (2 rows x 2 columns)

```
Row 0: Range-Doppler Vector Sink (qtgui)   [1x2]
Row 1: Freq Sink Ref [1x1] | Freq Sink Surv [1x1]
```

### Range-Doppler Vector Sink (Row 0)

| Property | Value |
|----------|-------|
| Type | `qtgui.vector_sink_f` |
| Size | 131,072 (64 Doppler x 2048 FFT) |
| X-axis | Range-Doppler Bin |
| Y-axis | Power (dB), initial -20 to 60 |
| Update time | 400 ms |
| Autoscale | enabled |
| Line color | blue |

### Frequency Sinks (Row 1)

| Property | Reference (Tuner 1) | Surveillance (Tuner 2) |
|----------|--------------------|-----------------------|
| FFT size | 2048 | 2048 |
| Window | Blackman-Harris | Blackman-Harris |
| Bandwidth | 2.0 MHz | 2.0 MHz |
| Y-axis | -140 to 10 dB | -140 to 10 dB |
| Update time | 100 ms | 100 ms |
| Line color | blue | red |

### Key Differences from KrakenSDR

- No phase calibration (RSPduo has coherent clock between tuners)
- No phase drift plot or calibration controls
- Uses `rspduo_source` instead of `krakensdr_source`
- 2 channels only (1 ref + 1 surv)

### Processing Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 2.0 MHz |
| Signal bandwidth | 250 kHz |
| Decimation | 8 |
| FFT size | 2048 |
| Doppler bins | 64 |
| ECA taps | 256 |
| ECA regularization | 0.0001 |
| IF gain | 40.0 dB |
| RF gain reduction | 0.0 dB |

---

## 3. 5-Channel Dashboard Sink

**File:** `gr-kraken_passive_radar/python/kraken_passive_radar/dashboard_sink.py`
**Technology:** Tkinter + matplotlib (TkAgg)
**For:** KrakenSDR 5-channel system (GNU Radio block)

### Launch

```bash
python3 run_passive_radar.py --visualize
```

### Main Figure: 20 x 14 inches

### Layout (4 rows x 3 columns, GridSpec)

```
+------------------+------------------+------------------+
| CAF Ch1 (Surv1)  | CAF Ch2 (Surv2)  | PPI Display      |
| [viridis, 0-15dB]| [viridis, 0-15dB]| [polar, 0-50km]  |
+------------------+------------------+                  |
| CAF Ch3 (Surv3)  | CAF Ch4 (Surv4)  |                  |
| [viridis, 0-15dB]| [viridis, 0-15dB]|                  |
+------------------+------------------+------------------+
| Fused CAF        | Channel Health   | Detection Trails |
| [viridis, 0-15dB]| [bars, 0-40dB]   | [red/orange pts] |
+------------------+------------------+------------------+
| Delay vs Time    | Doppler vs Time  | Max-Hold CAF     |
| [waterfall]      | [waterfall]      | [viridis, 0-15dB]|
+------------------+------------------+------------------+
```

### Per-Channel CAF Panels (4 panels, 2x2 grid)

| Property | Value |
|----------|-------|
| Colormap | viridis |
| X-axis | Delay (km) |
| Y-axis | Doppler (Hz) |
| Color range | 0.0 to 15.0 dB (adjustable via slider) |
| Aspect | auto |
| Origin | lower |

Range resolution: `c / (2 * sample_rate)` = 62.5 m/bin at 2.4 MHz.
Doppler resolution: `sample_rate / fft_len / doppler_len`.

### PPI Display

| Property | Value |
|----------|-------|
| Projection | polar |
| Theta zero | North ('N') |
| Theta direction | clockwise (-1) |
| Radial range | 0 to 50 km (adjustable via slider) |
| Detection markers | Red circles, size=100, alpha=0.8 |

### Fused CAF

Average of all 4 surveillance channel CAFs. Same colormap as individual panels.
CFAR detection markers overlaid as red hollow circles (size=80, linewidth=2).

### Channel Health

| Property | Value |
|----------|-------|
| X-labels | Ref, S1, S2, S3, S4 |
| Y-axis | SNR (dB), 0-40 |
| Green threshold | 25 dB |
| Yellow threshold | 15 dB |
| Bar colors | Dynamic: green (>=25), yellow (>=15), red (<15), gray (inactive) |

### Detection Waterfalls

**Delay vs Time:**
- X-axis: Time (seconds ago), 60 to 0
- Y-axis: Delay (km)
- Markers: red circles, size=20, alpha=0.7

**Doppler vs Time:**
- X-axis: Time (seconds ago), 60 to 0
- Y-axis: Doppler (Hz)
- Markers: red circles, size=20, alpha=0.7

### Max-Hold CAF

Same colormap as individual panels. Decay factor: 0.995 (adjustable via slider).
Max-hold formula: `max_hold = max(max_hold * decay, current_caf)`.

### Control Panel (separate figure, 5 x 10 inches)

| Slider | Range | Default | Units |
|--------|-------|---------|-------|
| CFAR Threshold | 5-25 | 12.0 | dB |
| Color Min | -10 to 10 | 0.0 | dB |
| Color Max | 5-30 | 15.0 | dB |
| Max-Hold Decay | 0.9-1.0 | 0.995 | — |
| PPI Range | 10-100 | 50.0 | km |

### Animation

- Update rate: 10 Hz (100 ms interval)
- FuncAnimation with `cache_frame_data=False`

---

## 4. Remote Radar Display

**File:** `kraken_passive_radar/remote_display.py`
**Technology:** matplotlib (FuncAnimation)
**For:** Viewing remote radar server data

### Launch

```bash
python3 -m kraken_passive_radar.remote_display [--url URL] [--interval SEC]
# Default: https://radar3.retnode.com, 1.0s poll
```

### Figure: 14 x 8 inches

### Delay-Doppler Map

| Property | Value |
|----------|-------|
| X-axis | Bistatic Delay (km) |
| Y-axis | Doppler Shift (Hz) |
| Colormap | viridis |
| Interpolation | bilinear |
| Color range | 0 to 15 dB |
| Colorbar | "Power (dB)", fraction=0.03, pad=0.02 |

### Detection Markers

| Property | Value |
|----------|-------|
| Shape | Hollow circles |
| Size | 60 |
| Edge color | red |
| Edge width | 1.5 |
| Z-order | 5 |

### Info Overlay (top-left)

```
monospace, size=9, white on black (alpha=0.6)
Shows: frame count, error count, latency
```

### Cursor Readout (bottom-left)

```
monospace, size=9, white on black (alpha=0.6)
Format: "Delay: X.XX km  Doppler: Y.Y Hz  Power: Z.Z dB"
```

### Server Endpoints

| Endpoint | Returns |
|----------|---------|
| `/api/map` | 2D CAF matrix (delay, doppler, data arrays) |
| `/api/detection` | CFAR detection list (delay, doppler, snr) |
| `/api/timing` | Processing timing (cpi_ms, uptime_days, nCpi) |

### Animation

- Poll interval: configurable (default 1.0 s)
- Update interval: max(200, poll_interval * 1000) ms
- Background polling thread with threading.Lock

---

## 5. Enhanced Remote Display

**File:** `kraken_passive_radar/enhanced_remote_display.py`
**Technology:** matplotlib (FuncAnimation)
**For:** Remote radar with local CFAR/tracking overlay

### Launch

```bash
# Server detections only
python3 -m kraken_passive_radar.enhanced_remote_display --url URL

# With local processing
python3 -m kraken_passive_radar.enhanced_remote_display --local \
    --cfar-guard 2 --cfar-train 8 --cfar-threshold 10 \
    --track-confirm 2 --track-delete 3 --track-gate 150
```

### Figure: 14 x 8 inches

Extends the remote display with three overlay layers:

| Layer | Marker | Color | Size | Description |
|-------|--------|-------|------|-------------|
| Server detections | Hollow circle | Red | 60 | From `/api/detection` |
| Local CFAR detections | Hollow circle | Green | 50 | Computed locally |
| Confirmed tracks | Diamond (D) | Yellow | 80 | Multi-target tracker |

### Track History Trails

| Property | Value |
|----------|-------|
| Confirmed color | yellow |
| Coasting color | orange |
| Line width | 1.0 |
| Alpha | 0.6 |

### Local Processing Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| CFAR guard cells | 2 | 1-8 |
| CFAR training cells | 4 | 2-16 |
| CFAR threshold | 12.0 dB | 5-25 |
| CFAR type | CA | CA/GO/SO |
| Tracker confirm hits | 3 | 1-10 |
| Tracker delete misses | 5 | 1-20 |
| Tracker gate | 100.0 | 10-500 |

---

## 6. 5-Channel Dashboard

**File:** `kraken_passive_radar/five_channel_dashboard.py`
**Technology:** matplotlib (FuncAnimation)
**For:** KrakenSDR full 5-channel visualization (remote or local)

### Launch

```bash
# Remote server
python3 -m kraken_passive_radar.five_channel_dashboard --url URL

# Local ZMQ
python3 -m kraken_passive_radar.five_channel_dashboard --local --zmq-addr tcp://localhost:5555
```

### Main Figure: 20 x 14 inches

### Layout

```
+------------------+------------------+------------------+
| CAF Ch1          | CAF Ch2          | PPI Display      |
| [viridis, 0-15dB]| [viridis, 0-15dB]| [polar, 50km]   |
+------------------+------------------+                  |
| CAF Ch3          | CAF Ch4          |                  |
+------------------+------------------+------------------+
| Fused CAF                           | Detection Table  |
+-------------------------------------+------------------+
| Delay-Time Waterfall                | Doppler Waterfall|
+-------------------------------------+------------------+
```

### Control Panel (separate figure, 5 x 10 inches)

| Slider | Range | Default |
|--------|-------|---------|
| CFAR Guard | 1-8 | 2 |
| CFAR Train | 2-16 | 4 |
| CFAR Threshold | 5-25 dB | 12.0 dB |
| Color Min | -10 to 10 dB | 0.0 dB |
| Color Max | 5-30 dB | 15.0 dB |
| History Duration | 10-300 s | 60.0 s |
| Max-Hold Decay | 0.9-1.0 | 0.995 |
| PPI Max Range | 10-100 km | 50.0 km |
| Tracker Confirm | 1-10 | 3 |
| Tracker Delete | 1-10 | 5 |
| Tracker Gate | 10-500 | 100.0 |

---

## 7. Multi-Display Dashboard

**File:** `kraken_passive_radar/multi_display_dashboard.py`
**Technology:** matplotlib (FuncAnimation)
**For:** RSPduo or single-channel focus with multiple views

### Launch

```bash
python3 -m kraken_passive_radar.multi_display_dashboard [--url URL] [--interval SEC]
```

### Figure: 20 x 12 inches

### 5-Panel Layout (GridSpec)

| Panel | Position | X-axis | Y-axis | Colormap/Markers |
|-------|----------|--------|--------|-----------------|
| Live Delay-Doppler | Top-left | Delay (km) | Doppler (Hz) | viridis, 0-15 dB |
| Max-Hold Map | Top-right | Delay (km) | Doppler (Hz) | viridis, 0-15 dB, decay=0.995 |
| Detections vs Delay | Mid-left | Time (s ago) | Delay (km) | Red circles, size=20 |
| Detections vs Doppler | Mid-right | Time (s ago) | Doppler (Hz) | Red circles, size=20 |
| Detection Trails | Bottom (full) | Delay (km) | Doppler (Hz) | Red(current)/orange(history) |

### Processing Parameters

| Parameter | Default |
|-----------|---------|
| CFAR guard | 2 |
| CFAR train | 4 |
| CFAR threshold | 12.0 dB |
| Tracker confirm | 3 |
| Tracker delete | 5 |
| Tracker gate | 100.0 |
| Tracker process noise | 10.0 |
| Tracker measurement noise | 50.0 |
| History duration | 60.0 s |

---

## 8. Calibration Panel

**File:** `kraken_passive_radar/calibration_panel.py`
**Technology:** matplotlib (FuncAnimation)
**For:** Per-channel calibration health monitoring

### Launch

```bash
python3 -m kraken_passive_radar.calibration_panel
```

### Figure: 14 x 8 inches

### Layout (GridSpec 3x4)

```
+------------------+-----+------------------+-----------+
| SNR Meters       |     |                  | Status    |
| [bars, 0-40 dB]  |     |                  | Indicator |
+------------------+-----+------------------+-----------+
| Phase Offsets          | Correlation Coefficients     |
| [diamonds, +/-180 deg]| [bars, 0-1.0]               |
+------------------------+------------------------------+
| Phase Drift History (all 4 channels, 60s window)     |
+------------------------------------------------------+
```

### SNR Meters

| Property | Value |
|----------|-------|
| Channels | Ref, Surv 1, Surv 2, Surv 3, Surv 4 |
| Y-range | 0 to 40 dB |
| Green line | 25 dB |
| Yellow line | 15 dB |

### Phase Offsets

| Property | Value |
|----------|-------|
| Marker | Diamond (d), size=200 |
| Y-range | -180 to +180 degrees |
| Zero reference | White line, alpha=0.3 |

### Correlation Coefficients

| Property | Value |
|----------|-------|
| Y-range | 0.0 to 1.0 |
| Green line | 0.95 |
| Yellow line | 0.90 |

### Phase Drift History

| Property | Value |
|----------|-------|
| X-range | -60 to 0 seconds (rolling window) |
| Y-range | -10 to +10 degrees (dynamic) |
| Channel colors | #FF6B6B, #4ECDC4, #45B7D1, #96CEB4 |
| Line width | 1.5 |
| Alpha | 0.8 |

### Color Coding

| State | Color | SNR Threshold | Correlation Threshold |
|-------|-------|--------------|----------------------|
| Good | #00FF00 (green) | >= 25 dB | >= 0.95 |
| Warning | #FFFF00 (yellow) | >= 15 dB | >= 0.90 |
| Bad | #FF0000 (red) | < 15 dB | < 0.90 |
| Inactive | #404040 (gray) | — | — |

### Update Rate: 200 ms (5 Hz)

---

## 9. Metrics Dashboard

**File:** `kraken_passive_radar/metrics_dashboard.py`
**Technology:** matplotlib (FuncAnimation)
**For:** System performance monitoring

### Launch

```bash
python3 -m kraken_passive_radar.metrics_dashboard
```

### Figure: 16 x 9 inches, background #1a1a2e (dark)

### Layout (GridSpec 4x4)

| Panel | Position | Content |
|-------|----------|---------|
| Latency Breakdown | Rows 0-1, Cols 0-1 | Per-block latency with sparklines |
| Detection Counts | Row 0, Cols 2-3 | Detections, confirmed/tentative/coasting tracks |
| Detection Sparkline | Row 1, Cols 2-3 | Detection rate history (100 samples) |
| System Resources | Row 2, Cols 0-1 | CPU %, memory %, sample rate |
| Backend Status | Row 2, Cols 2-3 | NEON/Vulkan enabled, device name |
| Timing Info | Row 3, All cols | Frame period, frames processed, drops, det/sec |

### Latency Components Monitored

ECA, CAF, CFAR, Clustering, Tracker, Total (end-to-end)

| Threshold | Value | Color |
|-----------|-------|-------|
| Normal | < 80 ms | Green |
| Warning | 80-100 ms | Yellow |
| Critical | > 100 ms | Red |

### Sparklines: 100-sample rolling history per metric

### Update Rate: 200 ms (5 Hz)

---

## 10. Range-Doppler Display

**File:** `kraken_passive_radar/range_doppler_display.py`
**Technology:** matplotlib
**For:** Clean standalone Range-Doppler heatmap

### Launch

```bash
python3 -m kraken_passive_radar.range_doppler_display
```

### Figure: 14 x 8 inches

| Property | Value |
|----------|-------|
| X-axis | Bistatic Range (km), 0 to 15.0 |
| Y-axis | Doppler Shift (Hz), -125.0 to +125.0 |
| Colormap | viridis (configurable) |
| Interpolation | bilinear |
| Dynamic range | 60 dB |
| Colorbar | "Intensity (dB)" |
| Range resolution | 600 m/bin |
| Doppler resolution | 3.9 Hz/bin |
| Range bins | 256 |
| Doppler bins | 64 |

### Overlays

- Detection markers: white circles
- Track history: fading trails
- Velocity arrows (configurable)

### Cursor Readout

```
monospace, size=9, white on black (alpha=0.6)
Position: top-left (0.01, 0.99)
```

### Update Rate: 100 ms (10 Hz)

---

## 11. PPI Polar Display

**File:** `kraken_passive_radar/radar_display.py`
**Technology:** matplotlib polar projection
**For:** Azimuth vs Range (Plan Position Indicator)

### Launch

```bash
python3 -m kraken_passive_radar.radar_display
```

### Figure: 10 x 10 inches

| Property | Value |
|----------|-------|
| Projection | polar |
| Theta zero | North ('N') |
| Theta direction | Clockwise (-1) |
| Radial range | 0 to 15.0 km |
| Range rings | 5.0, 10.0, 15.0 km |
| Azimuth ticks | N, NE, E, SE, S, SW, W, NW |
| Grid | Enabled, alpha=0.3 |

### Detection Markers

| Property | Value |
|----------|-------|
| Colormap | hot |
| Size | 80 |
| Alpha | 0.6 |
| Color range | -10 to 30 dB (power) |
| Edge | white, width=0.5 |

### Track Markers

| Property | Value |
|----------|-------|
| Color | lime green |
| Size | 120 |
| Shape | Diamond (D) |
| Edge | white, width=1.5 |

### Track Status Colors

| Status | Color |
|--------|-------|
| Tentative | #FFFF00 (yellow) |
| Confirmed | #00FF00 (green) |
| Coasting | #FFA500 (orange) |

### Additional Features

- Velocity arrows: scale = 0.1 km per m/s
- Track ID labels displayed
- History trails with fading (max 50 points)
- Update rate: 100 ms

---

## 12. Integrated Radar GUI

**File:** `kraken_passive_radar/radar_gui.py`
**Technology:** Tkinter + embedded matplotlib (TkAgg)
**For:** Combined multi-panel display

### Launch

```bash
python3 -m kraken_passive_radar.radar_gui
```

### Window

- Title: "KrakenSDR Passive Radar"
- Size: min(1600, screen_width-100) x min(900, screen_height-100)

### Layout (2x2 embedded canvases)

```
+---------------------+---------------------+
| Range-Doppler Map   | PPI Display         |
| (6x4", dpi=100)     | (6x4", dpi=100)     |
+---------------------+---------------------+
| Calibration Panel   | Metrics Dashboard   |
| (6x3", dpi=100)     | (6x3", dpi=100)     |
+---------------------+---------------------+
```

Each panel includes NavigationToolbar2Tk for zoom/pan/save.

### Update Rate: 100 ms (10 Hz)

---

## 13. 5-Channel Demo

**File:** `kraken_passive_radar/five_channel_demo.py`
**Technology:** matplotlib (TkAgg) + synthetic data
**For:** Testing and demonstration without hardware

### Launch

```bash
python3 -m kraken_passive_radar.five_channel_demo
```

### Simulated Targets

| Target | Delay | Doppler | SNR | Azimuth | Motion |
|--------|-------|---------|-----|---------|--------|
| 1 | 15 km | +50 Hz | 12 dB | 45 deg | +0.1 km/s delay, -0.5 Hz/s Doppler |
| 2 | 25 km | -80 Hz | 15 dB | 120 deg | -0.05 km/s, +0.3 Hz/s |
| 3 | 35 km | +120 Hz | 10 dB | 220 deg | +0.08 km/s, -0.2 Hz/s |
| 4 | 8 km | -30 Hz | 18 dB | 300 deg | +0.15 km/s, +0.1 Hz/s |

### Simulated Channel Health

| Channel | SNR | Phase | Correlation |
|---------|-----|-------|-------------|
| Ref | 28 dB | 0 deg | 1.00 |
| Surv 1 | 25 dB | +5 deg | 0.96 |
| Surv 2 | 24 dB | -3 deg | 0.95 |
| Surv 3 | 26 dB | +8 deg | 0.97 |
| Surv 4 | 23 dB | -2 deg | 0.94 |

### CAF Generation

- Delay bins: 200 (0 to 60 km)
- Doppler bins: 150 (-300 to +300 Hz)
- Target Gaussian blob: sigma_range=1.5, sigma_doppler=15
- Background: exponential noise, scale=1.0, amplitude=0.5

### Controls (separate figure, 4 x 8 inches)

- Add Target button
- Clear Targets button
- Speed slider: 0.0 to 2.0x

---

## 14. Remote Display Configuration

### SSH X11 Forwarding (Qt5 GUIs)

The flowgraph GUIs auto-detect the display environment at startup:

1. Checks `DISPLAY` and `WAYLAND_DISPLAY` environment variables
2. In SSH sessions (`SSH_CONNECTION` set), probes X11 forwarding ports 6010-6070
3. Falls back to local Wayland sockets in `$XDG_RUNTIME_DIR`
4. Sets `XAUTHORITY` from `~/.Xauthority` or Mutter XWayland auth files
5. Configures `QT_QPA_PLATFORM` for Qt5

**Usage:** Connect with `ssh -Y user@host` then run the flowgraph.

### FuncAnimation vs plt.pause()

All dashboards use `FuncAnimation` (not `plt.pause()`). The old `plt.pause()` approach caused X11 timeout after ~2 minutes over SSH forwarding. FuncAnimation uses the backend's native event loop timer, which properly maintains the X11 connection.

### Headless Mode

When no display is available, matplotlib falls back to the Agg backend:
```python
if not (DISPLAY or WAYLAND_DISPLAY):
    matplotlib.use('Agg')
```

### Network Bandwidth (Remote Displays)

| Mode | Data per Frame | At 10 Hz |
|------|---------------|----------|
| Full RD map (256x4096) | ~4 MB | ~40 MB/s |
| Detection-only | ~1 KB | ~10 KB/s |
| Reduced map (64x256) | ~64 KB | ~640 KB/s |

For low-bandwidth connections, use detection-only mode or reduce map resolution.
